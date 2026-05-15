import os
import pickle
import numpy as np
import torch
import torch.nn as nn

"""
MLP Inference controller for LOS-guided ship control.
Replacement for the simple PID controller, used to evaluate the trained model

"""

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
      
class MLPController:
  """
  Wraps the two trained MLP networks.
  Handles input scaling, network forward pass, and output unscaling.
  """
  
  def __init__(self, model_dir: str = "models", 
               speed_hidden = (64, 32), steer_hidden = (64, 32)):
    
    self.device = torch.device("cpu")
    
    with open(os.path.join(model_dir, "scaler_speed_x.pkl"), "rb") as f:
            self.scaler_sx = pickle.load(f)
    with open(os.path.join(model_dir, "scaler_speed_y.pkl"), "rb") as f:
            self.scaler_sy = pickle.load(f)
    with open(os.path.join(model_dir, "scaler_steer_x.pkl"), "rb") as f:
            self.scaler_rx = pickle.load(f)
    with open(os.path.join(model_dir, "scaler_steer_y.pkl"), "rb") as f:
            self.scaler_ry = pickle.load(f)
            
    self.speed_model = DockingMLP(6, list(speed_hidden), 1)
    self.speed_model.load_state_dict(
      torch.load(os.path.join(model_dir, "mlp_speed.pt"),
                 map_location=self.device, weights_only=True)
    )
    self.speed_model.eval()
    
    self.steer_model = DockingMLP(12, list(steer_hidden), 2)
    self.steer_model.load_state_dict(
      torch.load(os.path.join(model_dir, "mlp_steer.pt"),
                 map_location=self.device, weights_only=True)
    )
    self.steer_model.eval()
    
    if getattr(self.scaler_sx, "n_features_in_", 6) != 6:
      raise RuntimeError("Speed scaler does not match the expected 6-input model.")
    if getattr(self.scaler_rx, "n_features_in_", 12) != 12:
      raise RuntimeError("Steering scaler does not match the expected 12-input model.")
  
    self.reset()
    
  def reset(self):
    self.prev_tau_y = 0.0
    self.prev_tau_psi = 0.0
    
  @torch.inference_mode()
  def predict(self, x, y, psi, u, v, r, u_r, e_ct, e_psi, dist_dock, psi_r, r_r):
    """
    Compute control forces from current ship state.
    """
    
    # speed network
    x_speed = np.array([[x, y, u, u_r, e_ct, dist_dock]])
    x_speed_scaled = self.scaler_sx.transform(x_speed)
    x_speed_tensor = torch.tensor(x_speed_scaled, dtype=torch.float32)
    tau_x_scaled = self.speed_model(x_speed_tensor).numpy()
    tau_x = self.scaler_sy.inverse_transform(tau_x_scaled)[0, 0]
    
    # steering network
    x_steer = np.array([[np.sin(psi), np.cos(psi), e_ct, e_psi, v, r, x, y, psi_r, r_r, self.prev_tau_y, self.prev_tau_psi]])
    x_steer_scaled = self.scaler_rx.transform(x_steer)
    x_steer_tensor = torch.tensor(x_steer_scaled, dtype=torch.float32, device=self.device)
    tau_yn_scaled = self.steer_model(x_steer_tensor).cpu().numpy()
    tau_yn = self.scaler_ry.inverse_transform(tau_yn_scaled)[0]
    tau_y, tau_psi = tau_yn[0], tau_yn[1]
    
    self.prev_tau_y = tau_y
    self.prev_tau_psi = tau_psi
    
    return float(tau_x), tau_y, tau_psi
  
# Example usage
if __name__ == "__main__":
    ctrl = MLPController("models/")
    tx, ty, tn = ctrl.predict(
        x=100, y=300, psi=np.deg2rad(280),
        u=0.5, v=0.01, r=0.001,
        u_r=0.8, e_ct=2.5, e_psi=np.deg2rad(5),
        dist_dock=350, psi_r=np.deg2rad(280), r_r=0.001
    )
    print(f"τ_x = {tx:.1f} N,  τ_y = {ty:.1f} N,  τ_ψ = {tn:.1f} Nm")
    