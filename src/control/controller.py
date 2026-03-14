# control/controller.py
"""
LOS PID controller for 3-DOF vessel waypoint following.

This module provides:
- LOSPIDGains: Gains for surge speed tracking, heading tracking, and sway damping.
- LOSPIDController: PID controller designed for LOS guidance with three channels:
    - Surge: PI on speed error + damping feedforward
    - Heading: PD + I on heading error
    - Sway: pure damping on lateral velocity

Control law:
    tau_x   = D_u * u_r + Kp_u * (u_r - u_hat) + integral
    tau_psi = D_r * r_r + Kp_psi * (psi_r - psi_hat) + Kd_psi * (r_r - r_hat) + integral
    tau_y   = -Kd_v * v_hat
"""

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional
import math

@dataclass
class LOSPIDGains:
    """
    Gains for the LOS PID controller.

    Surge (speed-tracking):
        Kp_u   [N / (m/s)]   proportional on speed error
        Ki_u   [N / m]       integral on speed error
    Heading (heading-tracking):
        Kp_psi [Nm / rad]    proportional on heading error
        Kd_psi [Nm / (rad/s)] derivative (yaw-rate error)
        Ki_psi [Nm / (rad·s)] integral on heading error
    Sway (damping only):
        Kd_v   [N / (m/s)]   damps lateral drift
    """
    Kp_u: float = 50_000.0
    Ki_u: float = 2_000.0
    Kp_psi: float = 500_000.0
    Kd_psi: float = 2_000_000.0
    Ki_psi: float = 1_000.0
    Kd_v: float = 100_000.0
    tau_surge_max: float = 400_000.0
    tau_sway_max: float = 200_000.0
    tau_yaw_max: float = 1_600_000.0

class LOSPIDController:
    """
    PID controller designed for LOS guidance.

    Channels
    --------
    - Surge : PID on speed error (u_r - u_hat) + damping feedforward
    - Heading : PD + I on heading error (psi_r - psi_hat), yaw-rate derivative
    - Sway : pure damping on v_hat (no sway reference in LOS)
    """

    def __init__(self, M_diag: Sequence[float], D_diag: Sequence[float],
                 gains: LOSPIDGains):
        self.M = list(M_diag)   # [m, m, Iz]
        self.D = list(D_diag)   # [Xu, Yv, Nr]
        self.g = gains
        self.sigma_u = 0.0      # surge integral
        self.sigma_psi = 0.0    # heading integral

    @staticmethod
    def _wrap(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    @staticmethod
    def _sat(val: float, lim: float) -> float:
        return max(-lim, min(lim, val))

    def reset(self):
        self.sigma_u = 0.0
        self.sigma_psi = 0.0

    def step(self, dt: float,
             u_r: float, psi_r: float, r_r: float,
             u_hat: float, v_hat: float, r_hat: float,
             psi_hat: float) -> Tuple[float, float, float]:
        """
        Compute body-frame control forces.

        Parameters
        ----------
        u_r, psi_r, r_r : reference speed, heading, yaw-rate
        u_hat, v_hat, r_hat, psi_hat : estimated body velocities and heading

        Returns
        -------
        (tau_x, tau_y, tau_psi)
        """
        g = self.g

        e_u = u_r - u_hat
        tau_ff_x = self.D[0] * u_r                # damping feedforward
        tau_x_raw = tau_ff_x + g.Kp_u * e_u + self.sigma_u
        tau_x = self._sat(tau_x_raw, g.tau_surge_max)

        if abs(tau_x_raw) < g.tau_surge_max:
            self.sigma_u += g.Ki_u * e_u * dt
        else:
            self.sigma_u *= (1.0 - 0.1 * dt)
        self.sigma_u = self._sat(self.sigma_u, g.tau_surge_max * 0.3)

        e_psi = self._wrap(psi_r - psi_hat)
        e_r = r_r - r_hat
        tau_ff_psi = self.D[2] * r_r              # damping feedforward
        tau_psi_raw = tau_ff_psi + g.Kp_psi * e_psi + g.Kd_psi * e_r + self.sigma_psi
        tau_psi = self._sat(tau_psi_raw, g.tau_yaw_max)

        if abs(tau_psi_raw) < g.tau_yaw_max:
            self.sigma_psi += g.Ki_psi * e_psi * dt
        else:
            self.sigma_psi *= (1.0 - 0.5 * dt)
        self.sigma_psi = self._sat(self.sigma_psi, g.tau_yaw_max * 0.2)

        tau_y = self._sat(-g.Kd_v * v_hat, g.tau_sway_max)

        return tau_x, tau_y, tau_psi