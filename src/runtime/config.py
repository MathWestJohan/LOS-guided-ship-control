from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class VesselParams:
    mass: float = 350_000.0
    Iz:   float = 20.0e6
    Xu:   float = 50_000.0
    Yv:   float = 80_000.0
    Nr:   float = 2_000_000.0

    half_length: float = 10.0
    half_width: float = 3.75
    half_height: float = 0.9
    cm_shift_x: float = -0.2

    thruster_z_offset: float = -2.5
    stern_x_offset: float = None

    thr_port_x: float = -10.0
    thr_port_y: float = +2.76
    thr_star_x: float = -10.0
    thr_star_y: float = -2.76

    Tmax_thruster: float = 500_000.0
    alloc_bias_Fy: float = 0.0

@dataclass
class LOSConfig:
    """LOS guidance parameters."""
    Delta_min: float = 20.0            # minimum look-ahead distance [m]
    Delta_k: float = 8.0               # speed gain: Delta = Delta_min + Delta_k * |u|
    switch_radius: float = 8.0         # waypoint acceptance radius [m]
    u_desired: float = 12.0            # cruise speed [m/s]
    u_approach: float = 0.3            # final-approach speed [m/s]
    approach_dist: float = 150.0       # deceleration start distance [m]

@dataclass
class SceneParams:
    wave_height: float = 0.0

    # Heading reference filter
    ref_head_wn: float = 0.60
    ref_head_zeta: float = 1.0
    ref_head_rmax: float = 0.50

    # Speed reference filter
    ref_speed_wn: float = 0.25
    ref_speed_zeta: float = 1.0
    ref_speed_umax: float = 16.0

    # LOS PID controller gains
    Kp_u: float = 80_000.0
    Ki_u: float = 2_000.0
    Kp_psi: float = 5_000_000.0
    Kd_psi: float = 16_000_000.0
    Ki_psi: float = 5_000.0
    Kd_v: float = 100_000.0
    tau_surge_max: float = 800_000.0
    tau_sway_max: float = 200_000.0
    tau_yaw_max: float = 1_600_000.0

    # Observer
    obs_L_eta: float = 0.50
    obs_L_nu_xy: float = 0.50
    obs_L_nu_psi: float = 0.50
    obs_filter_alpha: float = 0.80

@dataclass
class Route:
    """Waypoint route — at least 2 points."""
    waypoints: List[Tuple[float, float]] = None

    def __post_init__(self):
        if self.waypoints is None:
            self.waypoints = [(0.0, 0.0), (200.0, 100.0), (400.0, -50.0)]
@dataclass
class GNSSNoise:
    sigma_pos: float = 0.30
    sigma_psi: float = 0.01
    disable_noise: bool = True

vessel = VesselParams()
los    = LOSConfig()
scene  = SceneParams()
route  = Route()
gnss   = GNSSNoise()