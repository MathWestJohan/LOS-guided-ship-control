# control/allocation.py
import numpy as np
from dataclasses import dataclass

# Geometry parameters for two-thruster configuration. 
@dataclass
class Geometry2Thrusters:
    """
    Geometry parameters for a symmetric/asymmetric two-thruster vessel layout.

    Attributes
    ----------
    lx1 : float
        Longitudinal offset (x) of thruster 1 (port) relative to the vessel CoM [m].
        Convention: positive forward, negative aft.
    ly1 : float
        Lateral offset (y) of thruster 1 [m]. Convention: positive = port side.
    lx2 : float
        Longitudinal offset (x) of thruster 2 (starboard) relative to CoM [m].
    ly2 : float
        Lateral offset (y) of thruster 2 [m]. Convention: negative = starboard.
    biasFy : float, optional
        Optional small lateral-force bias applied when redistributing lateral thrust.
        Useful to nudge actuators towards a preferred sector. Units: Newtons.
        Default is 0.0 (no bias).
    """
    lx1: float  # x offset of thruster 1 (port), relative to CoM (m). Negative if behind CoM
    ly1: float  # y offset of thruster 1 (port): positive to starboard is +y (port is +)
    lx2: float  # x offset of thruster 2 (starboard)
    ly2: float  # y offset of thruster 2 (starboard is -)
    biasFy: float = 0.0    # small sway bias to keep jets in a preferred sector

# Two-thruster allocator using pseudo-inverse method
class TwoThrusterAllocator:
    """
    Two-thruster linear allocator.

    Maps a desired generalized force/torque vector (tau = [Fx, Fy, N]) onto individual
    thruster force components f = [Fx1, Fy1, Fx2, Fy2] using a (left) pseudo-inverse
    of the linear mapping T (tau = T @ f). The allocator supports simple per-axis
    symmetric saturation and an optional lateral-bias re-distribution.

    Coordinate & sign conventions
    - Body frame: x forward, y to port (left), positive yaw N is CCW (right-hand rule).
    - Thruster forces are expressed in the body frame at the thruster local positions.
    - The torque (N) row of T is constructed from the moment arms (ly, -lx) such that
      the third row computes the out-of-plane moment produced by the 2D forces.

    Notes on allocation algorithm
    1. Compute a least-squares solution using the pseudo-inverse of T:
         f = T_pinv @ tau
    2. Optionally apply a simple lateral-bias redistribution to encourage jets to
       cooperate (uses Geometry2Thrusters.biasFy).
    3. Apply symmetric per-axis saturation to each element of f (clipping to [-Tmax,Tmax]).
       This is a very simple actuator model — more advanced allocation would include
       thrust-vector magnitude limits, directional constraints or quadratic programming.

    Parameters
    ----------
    geom : Geometry2Thrusters
        Geometry description of the two thrusters.
    Tmax : float, optional
        Per-axis symmetric saturation limit [N]. Default 80_000.0.
    """
    def __init__(self, geom: Geometry2Thrusters, Tmax=80_000.0):
        """
        Initialize allocator and compute the configuration matrix and its pseudo-inverse.

        The configuration matrix T maps thruster axis forces to generalized forces:
            tau = [Fx; Fy; N] = T @ [Fx1, Fy1, Fx2, Fy2]^T

        The rows of T are:
            [ 1, 0, 1, 0 ]     -> Fx contribution (sum of longitudinal components)
            [ 0, 1, 0, 1 ]     -> Fy contribution (sum of lateral components)
            [ ly1, -lx1, ly2, -lx2 ] -> yaw moment from forces (z moment arms)

        Using numpy.linalg.pinv provides a robust pseudo-inverse (works even if
        T is not full row-rank numerically).
        """
        self.g = geom # Geometry parameters
        self.Tmax = float(Tmax) # Max thrust per axis (N)

        self.T = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [-self.g.ly1, self.g.lx1, -self.g.ly2, self.g.lx2]
        ], dtype=float) # Thruster configuration matrix
        self.Tpinv = self.T.T @ np.linalg.inv(self.T @ self.T.T) # Pseudo-inverse of T

    # Saturate values symmetrically
    @staticmethod
    def _sat_symmetric(v, vmax):
        return np.clip(v, -vmax, vmax) # Clip values to [-vmax, vmax]

    # Allocate thrusts given desired forces/torque
    def allocate(self, tau_x, tau_y, tau_n):
        """
        Allocate thruster axis forces for a desired generalized wrench.

        Parameters
        ----------
        tau_x : float
            Desired surge force [N].
        tau_y : float
            Desired sway force [N].
        tau_n : float
            Desired yaw moment/torque [N*m].

        Returns
        -------
        (Fx1, Fy1, Fx2, Fy2) : Tuple[float, float, float, float]
            Per-thruster axis force components (each clipped to [-Tmax, Tmax]).
            These are the longitudinal (x) and lateral (y) components for thruster 1 and 2.

        Behavior details
        ----------------
        - Uses the precomputed pseudo-inverse to obtain a minimum-norm least-squares solution.
        - If Geometry2Thrusters.biasFy != 0.0, a simple redistribution is applied to the
          lateral components (f[1], f[3]) in order to bias the solution towards a
          cooperative pattern (useful for preventing one actuator from taking all lateral load).
        - Finally, each element is clipped independently. This does NOT conserve torque/force
          after clipping; a secondary re-distribution step could be added if conservation is required.
        """
        tau = np.array([tau_x, tau_y, tau_n], dtype=float) # Desired forces/torque vector

        # Base pseudo-inverse
        f = self.Tpinv @ tau  # [Fx1, Fy1, Fx2, Fy2]

        # Optional simple sway biasing
        if self.g.biasFy != 0.0:
            Fy_sum = f[1] + f[3] # Sum of lateral forces
            if Fy_sum > 0: # positive sway
                f[1] = Fy_sum + self.g.biasFy  # port
                f[3] = self.g.biasFy           # starboard
            elif Fy_sum < 0: # negative sway
                f[1] = self.g.biasFy           # port
                f[3] = Fy_sum - self.g.biasFy  # starboard

        # Saturate thrust magnitudes per-axis
        f = self._sat_symmetric(f, self.Tmax) # Saturate thrusts
        return float(f[0]), float(f[1]), float(f[2]), float(f[3]) # Return thruster forces
