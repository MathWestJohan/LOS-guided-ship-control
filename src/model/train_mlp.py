"""
MLP Training Pipeline for Ship Docking Controller
==================================================
Two parallel networks trained on LOS-PID demonstration data:

Speed network:   [dx, dy, u, u_r, e_ct, dist_dock]  → τ_x
Steering network: [sin(ψ), cos(ψ), e_ct, e_ψ, v, r, dx, dy] → τ_y, τ_ψ

Usage:
    1. Place your CSV episode files in data/episodes/
    2. Run: python train_mlp.py
    3. Trained models saved to models/
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import pickle

CONFIG = {
    # Data
    "data_dir": "data/training_data.csv",        # folder with CSV episode files
    "output_dir": "models",

    "speed_hidden": [64, 32],
    "steer_hidden": [64, 32],

    # Data preprocessing
    "subsample_step": 1,                # set >1 if logging at high freq (e.g. 6 for 60Hz→10Hz)

    # Training
    "batch_size": 256,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,               # L2 regularization
    "epochs": 200,
    "patience": 20,                     # early stopping patience
    "val_split": 0.15,                  # 15% validation
    "test_split": 0.15,                 # 15% test

    # Reproducibility
    "seed": 42,
}

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MLP input features from raw logged columns.

    Expected raw columns from your data collector:
        t, x, y, psi, u, v, r, u_r, e_ct, e_psi,
        tau_x, tau_y, tau_n, dist_dock, ...

    If your CSV column names differ, adjust the mapping below.
    """
    feat = pd.DataFrame()

    # Shared features
    feat["dx"]       = df["x"]          # position relative to dock (already in dock frame)
    feat["dy"]       = df["y"]
    feat["u"]        = df["u"]          # surge velocity
    feat["u_r"]      = df["u_r"]        # reference surge speed from guidance
    feat["e_ct"]     = df["e_ct"]       # cross-track error
    feat["v"]        = df["v"]          # sway velocity
    feat["r"]        = df["r"]          # yaw rate
    feat["e_psi"]    = df["e_psi"]      # heading error
    feat["sin_psi"]  = np.sin(df["psi"])
    feat["cos_psi"]  = np.cos(df["psi"])
    feat["dist_dock"] = df["dist_dock"] # distance to dock

    # Targets
    feat["tau_x"]    = df["tau_x"]      # surge force (speed net target)
    feat["tau_y"]    = df["tau_y"]      # sway force (steer net target)
    feat["tau_n"]    = df["tau_psi"]      # yaw moment (steer net target)

    return feat.dropna()

# Input column names for each network
SPEED_INPUTS  = ["dx", "dy", "u", "u_r", "e_ct", "dist_dock"]
STEER_INPUTS  = ["sin_psi", "cos_psi", "e_ct", "e_psi", "v", "r", "dx", "dy"]
SPEED_TARGETS = ["tau_x"]
STEER_TARGETS = ["tau_y", "tau_n"]

class DockingDataset(Dataset):
    """PyTorch dataset for one of the two networks."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class DockingMLP(nn.Module):
    """
    Multi-layer perceptron with configurable hidden layers.
    Uses ReLU activations + optional dropout.
    """

    def __init__(self, n_inputs: int, hidden_sizes: list, n_outputs: int,
                 dropout: float = 0.0):
        super().__init__()

        layers = []
        prev = n_inputs
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_outputs))  # linear output

        self.net = nn.Sequential(*layers)

        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        return self.net(x)

def train_network(model, train_loader, val_loader, config, name="model"):
    """Train one network with early stopping and learning rate scheduling."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config["epochs"]):
        # Train
        model.train()
        train_losses = []
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, Y_batch)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [{name}] Epoch {epoch+1:3d}/{config['epochs']}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"lr={current_lr:.2e}  patience={patience_counter}/{config['patience']}")

        if patience_counter >= config["patience"]:
            print(f"  [{name}] Early stopping at epoch {epoch+1}")
            break

    # Restore best weights
    model.load_state_dict(best_state)
    model = model.to(device)

    return model, history

def evaluate_model(model, test_loader, target_names, scaler_y):
    """Compute test metrics in original (unscaled) units."""

    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(Y_batch.numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    # Inverse transform to original scale
    preds_orig = scaler_y.inverse_transform(preds)
    targets_orig = scaler_y.inverse_transform(targets)

    metrics = {}
    for i, name in enumerate(target_names):
        mse = np.mean((preds_orig[:, i] - targets_orig[:, i]) ** 2)
        mae = np.mean(np.abs(preds_orig[:, i] - targets_orig[:, i]))
        rmse = np.sqrt(mse)
        # R² score
        ss_res = np.sum((targets_orig[:, i] - preds_orig[:, i]) ** 2)
        ss_tot = np.sum((targets_orig[:, i] - np.mean(targets_orig[:, i])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        metrics[name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
        print(f"    {name}: RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

    return metrics, preds_orig, targets_orig

def plot_training_curves(history_speed, history_steer, save_path):
    """Plot training/validation loss curves for both networks."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, hist, title in zip(axes,
                                [history_speed, history_steer],
                                ["Speed Network (τ_x)", "Steering Network (τ_y, τ_ψ)"]):
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], label="Train", linewidth=1.5)
        ax.plot(epochs, hist["val_loss"], label="Validation", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss (normalized)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved to {save_path}")

def plot_predictions(preds, targets, target_names, save_path, n_points=500):
    """Plot predicted vs actual for a time window."""

    n = min(n_points, len(preds))
    fig, axes = plt.subplots(len(target_names), 1, figsize=(14, 4 * len(target_names)))
    if len(target_names) == 1:
        axes = [axes]

    for ax, i, name in zip(axes, range(len(target_names)), target_names):
        ax.plot(targets[:n, i], label="LOS-PID (ground truth)", alpha=0.8, linewidth=1)
        ax.plot(preds[:n, i], label="MLP prediction", alpha=0.8, linewidth=1, linestyle="--")
        ax.set_ylabel(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    plt.suptitle("MLP Predictions vs LOS-PID Ground Truth (test set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Prediction plots saved to {save_path}")

def load_all_episodes(data_path: str) -> pd.DataFrame:
    """Load and concatenate CSV data from a single file or a directory."""

    if os.path.isfile(data_path):
        csv_files = [data_path]
    else:
        csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found at {data_path}\n"
            f"Run your data collection first to generate CSV data."
        )

    dfs = []
    multi_file = len(csv_files) > 1

    for f in csv_files:
        df = pd.read_csv(f)
        file_tag = os.path.basename(f)

        if "episode" in df.columns:
            if multi_file:
                df["episode"] = df["episode"].astype(str).map(lambda ep: f"{file_tag}::{ep}")
            else:
                df["episode"] = df["episode"].astype(str)
        else:
            df["episode"] = file_tag

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(csv_files)} CSV file(s), {len(combined)} total timesteps")
    return combined

def keep_successful_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep all rows from episodes whose final row has finished == 1."""
    if "episode" not in df.columns:
        raise ValueError("Expected an 'episode' column for episode-level filtering.")
    if "finished" not in df.columns:
        raise ValueError("Expected a 'finished' column for success filtering.")

    episode_finished = (
        df.groupby("episode", sort=False)["finished"]
        .last()
        .astype(int)
    )

    successful_episodes = episode_finished[episode_finished == 1].index
    filtered = df[df["episode"].isin(successful_episodes)].reset_index(drop=True)

    print(
        f"Kept {len(successful_episodes)}/{len(episode_finished)} successful episodes "
        f"({len(filtered)} rows)"
    )
    return filtered


def subsample_data(df: pd.DataFrame, step: int) -> pd.DataFrame:
    """Subsample by taking every step-th row per episode."""
    if step <= 1:
        return df
    if "episode" in df.columns:
        return df.groupby("episode").apply(
            lambda g: g.iloc[::step], include_groups=False
        ).reset_index(drop=True)
    return df.iloc[::step].reset_index(drop=True)


def split_by_episode(df: pd.DataFrame, config: dict):
    """Split raw rows by episode so no rollout leaks across train/val/test."""
    if "episode" not in df.columns:
        raise ValueError("Expected an 'episode' column for episode-based splitting.")

    episode_ids = df["episode"].drop_duplicates().to_numpy()
    n_episodes = len(episode_ids)

    if n_episodes < 3:
        raise ValueError(
            f"Need at least 3 successful episodes for train/val/test split, got {n_episodes}."
        )

    rng = np.random.default_rng(config["seed"])
    rng.shuffle(episode_ids)

    n_test = max(1, int(round(n_episodes * config["test_split"])))
    n_val = max(1, int(round(n_episodes * config["val_split"])))

    while n_test + n_val >= n_episodes:
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            raise ValueError("Not enough episodes to create non-empty splits.")

    test_eps = episode_ids[:n_test]
    val_eps = episode_ids[n_test:n_test + n_val]
    train_eps = episode_ids[n_test + n_val:]

    train_df = df[df["episode"].isin(train_eps)].reset_index(drop=True)
    val_df = df[df["episode"].isin(val_eps)].reset_index(drop=True)
    test_df = df[df["episode"].isin(test_eps)].reset_index(drop=True)

    print(
        f"Episode split: {len(train_eps)} train / {len(val_eps)} val / {len(test_eps)} test"
    )
    print(
        f"Row split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test"
    )

    return train_df, val_df, test_df

def prepare_data(train_df, val_df, test_df, input_cols, target_cols, config):
    """Scale data using train only, then build DataLoaders."""

    X_train = train_df[input_cols].values
    Y_train = train_df[target_cols].values
    X_val = val_df[input_cols].values
    Y_val = val_df[target_cols].values
    X_test = test_df[input_cols].values
    Y_test = test_df[target_cols].values

    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(Y_train)

    X_train_scaled = scaler_x.transform(X_train)
    Y_train_scaled = scaler_y.transform(Y_train)
    X_val_scaled = scaler_x.transform(X_val)
    Y_val_scaled = scaler_y.transform(Y_val)
    X_test_scaled = scaler_x.transform(X_test)
    Y_test_scaled = scaler_y.transform(Y_test)

    train_ds = DockingDataset(X_train_scaled, Y_train_scaled)
    val_ds = DockingDataset(X_val_scaled, Y_val_scaled)
    test_ds = DockingDataset(X_test_scaled, Y_test_scaled)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"])

    print(f"  Split: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    return train_loader, val_loader, test_loader, scaler_x, scaler_y

def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "plots"), exist_ok=True)

    # Load data
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    raw_df = load_all_episodes(CONFIG["data_dir"])
    raw_df = keep_successful_episodes(raw_df)
    raw_df = subsample_data(raw_df, CONFIG["subsample_step"])
    print(f"After subsampling (step={CONFIG['subsample_step']}): {len(raw_df)} samples")

    train_raw, val_raw, test_raw = split_by_episode(raw_df, CONFIG)

    train_features = compute_features(train_raw)
    val_features = compute_features(val_raw)
    test_features = compute_features(test_raw)

    print(
        f"After feature engineering: "
        f"{len(train_features)} train / {len(val_features)} val / {len(test_features)} test samples"
    )
    print(f"Features: {list(train_features.columns)}\n")

    # Speed network
    print("=" * 60)
    print("SPEED NETWORK  [dx, dy, u, u_r, e_ct, dist_dock] → τ_x")
    print("=" * 60)
    train_s, val_s, test_s, scaler_sx, scaler_sy = prepare_data(
        train_features, val_features, test_features, SPEED_INPUTS, SPEED_TARGETS, CONFIG
    )

    speed_model = DockingMLP(
        n_inputs=len(SPEED_INPUTS),
        hidden_sizes=CONFIG["speed_hidden"],
        n_outputs=len(SPEED_TARGETS),
    )
    print(f"  Architecture: {len(SPEED_INPUTS)} → {CONFIG['speed_hidden']} → {len(SPEED_TARGETS)}")
    print(f"  Parameters: {speed_model.n_params:,}\n")

    speed_model, hist_speed = train_network(speed_model, train_s, val_s, CONFIG, "Speed")
    print("\n  Test set evaluation:")
    metrics_speed, preds_s, targets_s = evaluate_model(
        speed_model, test_s, SPEED_TARGETS, scaler_sy
    )

    # Steering network
    print("\n" + "=" * 60)
    print("STEERING NETWORK  [sin(ψ), cos(ψ), e_ct, e_ψ, v, r, dx, dy] → τ_y, τ_ψ")
    print("=" * 60)
    train_r, val_r, test_r, scaler_rx, scaler_ry = prepare_data(
        train_features, val_features, test_features, STEER_INPUTS, STEER_TARGETS, CONFIG
    )

    steer_model = DockingMLP(
        n_inputs=len(STEER_INPUTS),
        hidden_sizes=CONFIG["steer_hidden"],
        n_outputs=len(STEER_TARGETS),
    )
    print(f"  Architecture: {len(STEER_INPUTS)} → {CONFIG['steer_hidden']} → {len(STEER_TARGETS)}")
    print(f"  Parameters: {steer_model.n_params:,}\n")

    steer_model, hist_steer = train_network(steer_model, train_r, val_r, CONFIG, "Steer")
    print("\n  Test set evaluation:")
    metrics_steer, preds_r, targets_r = evaluate_model(
        steer_model, test_r, STEER_TARGETS, scaler_ry
    )

    # Save models and scalers
    print("\n" + "=" * 60)
    print("SAVING")
    print("=" * 60)

    torch.save(speed_model.state_dict(), os.path.join(CONFIG["output_dir"], "speed_net.pt"))
    torch.save(steer_model.state_dict(), os.path.join(CONFIG["output_dir"], "steer_net.pt"))

    with open(os.path.join(CONFIG["output_dir"], "scaler_speed_x.pkl"), "wb") as f:
        pickle.dump(scaler_sx, f)
    with open(os.path.join(CONFIG["output_dir"], "scaler_speed_y.pkl"), "wb") as f:
        pickle.dump(scaler_sy, f)
    with open(os.path.join(CONFIG["output_dir"], "scaler_steer_x.pkl"), "wb") as f:
        pickle.dump(scaler_rx, f)
    with open(os.path.join(CONFIG["output_dir"], "scaler_steer_y.pkl"), "wb") as f:
        pickle.dump(scaler_ry, f)

    # Save config and metrics
    results = {
        "config": CONFIG,
        "speed_metrics": metrics_speed,
        "steer_metrics": metrics_steer,
        "speed_params": speed_model.n_params,
        "steer_params": steer_model.n_params,
    }
    with open(os.path.join(CONFIG["output_dir"], "training_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  Models saved to {CONFIG['output_dir']}/")

    # Plots
    plot_training_curves(
        hist_speed, hist_steer,
        os.path.join(CONFIG["output_dir"], "plots", "training_curves.png")
    )
    plot_predictions(
        preds_s, targets_s, SPEED_TARGETS,
        os.path.join(CONFIG["output_dir"], "plots", "speed_predictions.png")
    )
    plot_predictions(
        preds_r, targets_r, STEER_TARGETS,
        os.path.join(CONFIG["output_dir"], "plots", "steer_predictions.png")
    )

    print("\nDone!")

if __name__ == "__main__":
    main()