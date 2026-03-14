# control/reference.py
"""
2nd-order reference filters for LOS guidance.

Smooths LOS guidance commands into controller references:
- Heading channel: chi_los -> (psi_r, r_r) via critically-damped 2nd-order dynamics
- Speed channel: u_d -> u_r via critically-damped 2nd-order dynamics with saturation
"""
import math
from dataclasses import dataclass

def sat(val, vmin, vmax):
    return max(vmin, min(vmax, val))

@dataclass
class HeadRefParams:
    omega: float = 0.5
    zeta: float  = 1.0
    rmax: float  = 0.5      # rad/s cap
    
@dataclass
class SpeedRefParams:
    """2nd-order speed reference filter parameters"""
    omega: float = 0.15     # bandwidth (rad/s)
    zeta: float = 1.0       # damping ratio
    umax: float = 0.5       # speed saturation (m/s)
    

class LOSReferenceFilter:
    """
    Smooths LOS guidance commands (chi_los, u_d) into controller references.
    
    Channels
    --------
    - Heading : 2nd-order filter chi_los -> (psi_r, r_r)
    - Speed   : 2nd-order filter u_d -> u_r
    
    """
    
    def __init__(self, head_params: HeadRefParams, speed_params: SpeedRefParams):
        self.hp = head_params
        self.sp = speed_params
        self.psir = 0.0
        self.rr = 0.0
        self.ur = 0.0
        self.ar = 0.0
    
    @staticmethod
    def _wrap_pi(a):
        return math.atan2(math.sin(a), math.cos(a))
    
    def reset(self, psi_now=0.0, u_now=0.0):
        self.psir = psi_now
        self.rr = 0.0
        self.ur = u_now
        self.ar = 0.0
        
    def step(self, dt, chi_los, u_d):
        """
        Advance filter one time step
        
        Returns
        -------
        (u_r, psi_r, r_r) : filtered speed, heading, yaw-rate references
        """
        epsi = self._wrap_pi(chi_los - self.psir)
        alpha = self.hp.omega ** 2 * epsi - 2 * self.hp.zeta * self.hp.omega * self.rr
        self.rr += alpha * dt
        self.rr = sat(self.rr, -self.hp.rmax, self.hp.rmax)
        self.psir = self._wrap_pi(self.psir + self.rr * dt)
        
        eu = u_d - self.ur
        accel = self.sp.omega ** 2 * eu - 2 * self.sp.zeta * self.sp.omega * self.ar
        self.ar += accel * dt
        self.ur += self.ar * dt
        self.ur = sat(self.ur, 0.0, self.sp.umax)
        if self.ur <= 0.0:
            self.ar = max(self.ar, 0.0)
            
        return self.ur, self.psir, self.rr