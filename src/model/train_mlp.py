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
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import pickle

CONFIG = {
    # Data
    "data_dir": "data/training_data",        # folder with CSV episode files
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
    feat["tau_n"]    = df["tau_n"]      # yaw moment (steer net target)

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

def load_all_episodes(data_dir: str) -> pd.DataFrame:
    """Load and concatenate all CSV episode files."""

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}/\n"
            f"Run your data collection first to generate episodes."
        )

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["episode"] = os.path.basename(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(csv_files)} episodes, {len(combined)} total timesteps")
    return combined

def subsample_data(df: pd.DataFrame, step: int) -> pd.DataFrame:
    """Subsample by taking every step-th row per episode."""
    if step <= 1:
        return df
    if "episode" in df.columns:
        return df.groupby("episode").apply(
            lambda g: g.iloc[::step], include_groups=False
        ).reset_index(drop=True)
    return df.iloc[::step].reset_index(drop=True)

def prepare_data(features_df, input_cols, target_cols, config):
    """Scale data and split into train/val/test DataLoaders."""

    X = features_df[input_cols].values
    Y = features_df[target_cols].values

    # Fit scalers on everything first, then split
    scaler_x = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(Y)
    X_scaled = scaler_x.transform(X)
    Y_scaled = scaler_y.transform(Y)

    dataset = DockingDataset(X_scaled, Y_scaled)

    # Split
    n = len(dataset)
    n_test = int(n * config["test_split"])
    n_val = int(n * config["val_split"])
    n_train = n - n_val - n_test

    torch.manual_seed(config["seed"])
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"])
    test_loader  = DataLoader(test_ds,  batch_size=config["batch_size"])

    print(f"  Split: {n_train} train / {n_val} val / {n_test} test")

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
    raw_df = subsample_data(raw_df, CONFIG["subsample_step"])
    print(f"After subsampling (step={CONFIG['subsample_step']}): {len(raw_df)} samples")
    features_df = compute_features(raw_df)
    print(f"After feature engineering: {len(features_df)} samples")
    print(f"Features: {list(features_df.columns)}\n")

    # Speed network
    print("=" * 60)
    print("SPEED NETWORK  [dx, dy, u, u_r, e_ct, dist_dock] → τ_x")
    print("=" * 60)
    train_s, val_s, test_s, scaler_sx, scaler_sy = prepare_data(
        features_df, SPEED_INPUTS, SPEED_TARGETS, CONFIG
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
        features_df, STEER_INPUTS, STEER_TARGETS, CONFIG
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