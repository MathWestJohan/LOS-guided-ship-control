""" 
Classic Line-Of-Sight (LOS) guidance for waypoint following.

LOS law:
    chi_los = phi_p - atan2(e_ct, Delta)
    
where:
    phi_p = path-tangent angle of the current leg
    e_ct = signed cross-track error (positive = port side of path)
    Delta = look-ahead distance (tunable)
"""
import math
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class LOSParams:
    Delta_min: float = 20.0       # Minimum look-ahead distance (m)
    Delta_k: float = 8.0          # speed gain: Delta = Delta_min + Delta_k * |u|
    switch_radius: float = 8.0    # Waypoint acceptance radius (m)
    u_desired: float = 12.0       # Cruise surge speed (m/s)
    u_approach: float = 0.3       # Speed near final waypoint (m/s)
    approach_dist: float = 150.0  # Distance at which to start decelerating (m)
    

class LOSGuidance:
  """ 
  Multi-leg LOS guidance
  
  Given a list of waypoints [(x0, y0), (x1, y1), ...], computes the desired
  heading chi_los and speed u_d at each time step so the vessel converges
  to and follows the straight-line path between consecutive waypoints.
  """
  
  def __init__(self, waypoints: List[Tuple[float, float]], params: LOSParams):
    if len(waypoints) < 2:
      raise ValueError("Need at least 2 waypoints")
    self.wps = list(waypoints)
    self.p = params
    self.leg = 0         # Index of "start" waypoint of current leg
    self.finished = False
    
  @staticmethod
  def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))
  
  def _phi(self) -> float:
    """Path-tangent angle of current leg"""
    x0, y0 = self.wps[self.leg]
    x1, y1 = self.wps[self.leg + 1]
    return math.atan2(y1 - y0, x1 - x0)
  
  def _cross_track(self, x: float, y: float) -> float:
    """Signed cross-track error (positive = port side of path)"""
    x0, y0 = self.wps[self.leg]
    phi = self._phi()
    return -(x - x0) * math.sin(phi) + (y - y0) * math.cos(phi)
  
  def _dist_to_next(self, x: float, y: float) -> float:
    x1, y1 = self.wps[self.leg + 1]
    return math.hypot(x1 - x, y1 - y)
  
  def _leg_len(self) -> float:
    x0, y0 = self.wps[self.leg]
    x1, y1 = self.wps[self.leg + 1]
    return math.hypot(x1 - x0, y1 - y0)
  
  def _turn_angle_at_next(self) -> float:
    """Absolute heading change at the next waypoint (rad, 0 to pi)"""
    if self.leg >= len(self.wps) - 2:
      return 0.0
    x0, y0 = self.wps[self.leg]
    x1, y1 = self.wps[self.leg + 1]
    x2, y2 = self.wps[self.leg + 2]
    phi_curr = math.atan2(y1 - y0, x1 - x0)
    phi_next = math.atan2(y2 - y1, x2 - x1)
    return abs(self._wrap(phi_next - phi_curr))
  
  def step(self, x: float, y: float, u: float = 0.0) -> dict:
    """
    Compute LOS guidance outputs.
    
    Returns
    -------
    dict with keys:
        chi_los : desired heading (radians)
        u_d     : desired surge speed (m/s)
        e_ct    : cross-track error (m)
        e_at    : remaining distance to next waypoint (m)
        phi_p   : path tangent angle (radians)
        leg     : current leg index (int)
        finished: True when final waypoint reached (bool)
        turn_angle: Absolute heading change at the next waypoint (radians)
    """
    if self.finished:
      xf, yf = self.wps[-1]
      x0, y0 = self.wps[-2]
      phi_last = math.atan2(yf - y0, xf - x0)
      return dict(chi_los=phi_last, u_d=0.0,
                  e_ct=0.0, e_at=math.hypot(xf - x, yf - y),
                  phi_p=phi_last, leg=self.leg, finished=True,
                  turn_angle=0.0)
      
    rem = self._dist_to_next(x, y)
    last_leg = (self.leg >= len(self.wps) - 2)
    
    if rem < self.p.switch_radius:
      if last_leg:
        self.finished = True
        return dict(chi_los=self._phi(), u_d=0.0,
                    e_ct=self._cross_track(x, y),
                    e_at=rem, phi_p=self._phi(),
                    leg=self.leg, finished=True,
                    turn_angle=self._turn_angle_at_next())
      else:
        self.leg += 1
        
    phi_p = self._phi()
    e_ct = self._cross_track(x, y)
    Delta = self.p.Delta_min + self.p.Delta_k * abs(u)
    chi_los = self._wrap(phi_p - math.atan2(e_ct, Delta))
    
    rem = self._dist_to_next(x, y)
    if last_leg:
      if rem < self.p.approach_dist:
        frac = rem / self.p.approach_dist
        u_d = self.p.u_approach + frac * (self.p.u_desired - self.p.u_approach)
      else:
        u_d = self.p.u_desired
    else:
      turn = self._turn_angle_at_next()
      turn_factor = min(turn / (math.pi / 2), 1.0)
      u_turn = self.p.u_desired * (1.0 - turn_factor) + self.p.u_approach * turn_factor
      eff_approach = self.p.approach_dist * (1.0 + turn_factor)
      if rem < eff_approach:
        frac = rem / eff_approach
        u_d = u_turn + frac * (self.p.u_desired - u_turn)
      else:
        u_d = self.p.u_desired
      
    # Cross track speed governor, limit speed when far off the path
    ct_limit = self.p.Delta_min
    u_recover = max(self.p.u_approach, 0.25 * self.p.u_desired)
    if abs(e_ct) > ct_limit:
      u_d = min(u_d, u_recover)
    elif abs(e_ct) > ct_limit * 0.15:
      blend = (abs(e_ct) - ct_limit * 0.15) / (ct_limit * 0.85)
      u_d = min(u_d, self.p.u_desired * (1.0 - blend) + u_recover * blend)

    return dict(chi_los=chi_los, u_d=u_d, e_ct=e_ct,
                    e_at=rem, phi_p=phi_p, leg=self.leg, finished=False,
                    turn_angle=self._turn_angle_at_next())
    
  def reset(self):
    self.leg = 0
    self.finished = False