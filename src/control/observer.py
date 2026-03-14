# control/observer.py
from dataclasses import dataclass
import math

@dataclass
class ObsGains:
    """
    Observer gains for the Luenberger-style observer.
    
    Attributes
    ----------
    L_eta : float
        Proportional gain for pose correction (x, y, psi).
    L_nu_xy : float
        Proportional gain for velocity correction (u, v).
    L_nu_psi : float
        Proportional gain for yaw rate correction (r).
    filter_alpha : float
        Low-pass filter coefficient for wave rejection (0 < alpha <= 1).
        Lower values = more filtering (slower response but better noise rejection).
    """
    L_eta: float = 1.0
    L_nu_xy: float = 1.0
    L_nu_psi: float = 1.0
    filter_alpha: float = 0.15


class SimpleObserver:
    """
    Minimal planar Luenberger-style observer for a 3-DOF vessel (x, y, psi) and body velocities (u, v, r).

    Purpose
    -------
    Provide a lightweight state estimator suitable for control loops where only noisy pose
    measurements (x,y,psi) are available at a lower rate than the controller. The observer:
      - Predicts pose by integrating current estimated body velocities.
      - Predicts body accelerations from current estimated velocities and applied generalized forces:
            M * nudot ≈ tau - D * nu  =>  nudot ≈ inv(M) * (tau - D*nu)
      - Uses measured pose to correct both pose and velocities via proportional gains (L gains).
      - Contains a simple angle-wrapping utility for yaw (psi).
      - Includes low-pass filtering for wave rejection.

    Assumptions & Conventions
    -------------------------
    - Frames:
        * eta = [x, y, psi] is expressed in the world (NED/planar) frame.
        * nu  = [u, v, r] are body-fixed velocities: u forward, v lateral (positive to port),
          r yaw-rate (rad/s).
    - M and D are expected to be diagonal-like sequences [M_x, M_y, M_psi] and [D_x, D_y, D_psi].
      The observer uses component-wise inverses of M.
    - Inputs to step(...) are:
        dt: time step in seconds,
        meas_x, meas_y, meas_psi: measured pose (may be noisy),
        tau_x, tau_y, tau_n: applied generalized forces/torque (used for feedforward acceleration estimate).
    - This observer is intentionally simple and not intended to replace a full Kalman filter
      when measurement noise statistics and process noise are known.

    Public API
    ----------
    - reset(x, y, psi): initialize internal state
    - step(dt, meas_x, meas_y, meas_psi, tau_x, tau_y, tau_n, M, D):
        perform one predict-correct iteration and return estimated pose and body velocities.
    """
    
    def __init__(self, gains: ObsGains):
        self.g = gains
        # Estimated states
        self.eta = [0.0, 0.0, 0.0]  # [x, y, psi]
        self.nu = [0.0, 0.0, 0.0]   # [u, v, r]
        # Low-pass filter states for wave rejection
        self.eta_filt = [0.0, 0.0, 0.0]
        self.alpha = gains.filter_alpha

    @staticmethod
    def _wrap(angle: float) -> float:
        """Wrap angle to [-pi, pi] using trig identity (robust for any magnitude)."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def reset(self, x: float, y: float, psi: float):
        """Initialize observer state."""
        self.eta = [x, y, psi]
        self.nu = [0.0, 0.0, 0.0]
        self.eta_filt = [x, y, psi]

    def step(self, dt: float, meas_x: float, meas_y: float, meas_psi: float,
             tau_x: float, tau_y: float, tau_n: float,
             M: list, D: list) -> tuple:
        """
        Perform one predict-correct iteration.
        
        Parameters
        ----------
        dt : float
            Time step in seconds.
        meas_x, meas_y, meas_psi : float
            Measured pose (possibly noisy).
        tau_x, tau_y, tau_n : float
            Applied generalized forces/torque in body frame.
        M : list
            Diagonal inertia [m_x, m_y, I_z].
        D : list
            Diagonal damping [Xu, Yv, Nr].
            
        Returns
        -------
        tuple
            (eta_hat, nu_hat) where eta_hat = [x, y, psi] and nu_hat = [u, v, r]
        """
        # Low-pass filter measurements for wave rejection
        self.eta_filt[0] += self.alpha * (meas_x - self.eta_filt[0])
        self.eta_filt[1] += self.alpha * (meas_y - self.eta_filt[1])
        psi_err_filt = math.atan2(math.sin(meas_psi - self.eta_filt[2]),
                                   math.cos(meas_psi - self.eta_filt[2]))
        self.eta_filt[2] += self.alpha * psi_err_filt
        self.eta_filt[2] = self._wrap(self.eta_filt[2])
        
        meas_x_f = self.eta_filt[0]
        meas_y_f = self.eta_filt[1]
        meas_psi_f = self._wrap(self.eta_filt[2])
        
        # Current estimates
        x_hat, y_hat, psi_hat = self.eta
        u_hat, v_hat, r_hat = self.nu
        
        # Rotation matrix (body to world)
        c = math.cos(psi_hat)
        s = math.sin(psi_hat)
        
        # Pose prediction: eta_dot = R(psi) @ nu
        x_dot_pred = c * u_hat - s * v_hat
        y_dot_pred = s * u_hat + c * v_hat
        psi_dot_pred = r_hat
        
        # Velocity prediction: M * nu_dot = tau - D * nu
        # => nu_dot = inv(M) * (tau - D * nu)
        u_dot_pred = (tau_x - D[0] * u_hat) / M[0]
        v_dot_pred = (tau_y - D[1] * v_hat) / M[1]
        r_dot_pred = (tau_n - D[2] * r_hat) / M[2]
        
        # Euler integration for prediction
        x_pred = x_hat + dt * x_dot_pred
        y_pred = y_hat + dt * y_dot_pred
        psi_pred = self._wrap(psi_hat + dt * psi_dot_pred)
        
        u_pred = u_hat + dt * u_dot_pred
        v_pred = v_hat + dt * v_dot_pred
        r_pred = r_hat + dt * r_dot_pred
        
        # Pose error (using filtered measurements)
        e_x = meas_x_f - x_pred
        e_y = meas_y_f - y_pred
        e_psi = math.atan2(math.sin(meas_psi_f - psi_pred),
                           math.cos(meas_psi_f - psi_pred))
        
        # Correct pose
        self.eta[0] = x_pred + self.g.L_eta * e_x
        self.eta[1] = y_pred + self.g.L_eta * e_y
        self.eta[2] = self._wrap(psi_pred + self.g.L_eta * e_psi)
        
        # Correct velocities (transform pose error to body frame for velocity correction)
        c_new = math.cos(self.eta[2])
        s_new = math.sin(self.eta[2])
        e_u = c_new * e_x + s_new * e_y
        e_v = -s_new * e_x + c_new * e_y
        
        self.nu[0] = u_pred + self.g.L_nu_xy * e_u / dt if dt > 0 else u_pred
        self.nu[1] = v_pred + self.g.L_nu_xy * e_v / dt if dt > 0 else v_pred
        self.nu[2] = r_pred + self.g.L_nu_psi * e_psi / dt if dt > 0 else r_pred
        
        return tuple(self.eta), tuple(self.nu)