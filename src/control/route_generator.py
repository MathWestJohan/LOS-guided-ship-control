import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

"""
Programmatic route generation for LOS-guided ship control.

Generates a smooth waypoint path from an arbitrary pose 
(x, y, heading) to a fixed dock position, ensuring the ship
always arrives along a consistent approach corridor.

The route has two segments:
1. Connecting curve: cubic hermite spline from the start pose to
the entry of the approach corridor, sampled into discrete waypoints.
2. Approach corridor: fixed straight line segment into the dock,
always the same regardless of the ship initial position.

The Hermite spline guarantees C1 continuity: the path is tangent to the
ship's initial heading at the start, and tangent to the approach corridor
direction at the entry point. LOS guidnace never sees
a large heading discontinuity at the junction.
"""

@dataclass
class DockConfig:
    """Defines the docking target and approach geometry."""
    dock_x: float = 400.0
    dock_y: float = -50.0
    approach_heading: float = None 
    
    approach_length: float = 120.0
    
    def __post_init__(self):
      if self.approach_heading is None:
        # Default approach from upper-left towards lower right
        self.approach_heading = math.atan2(-150, 200) # -36.9 degrees

@dataclass
class InitialArea:
  """Defines the randomized spawn region for the ship."""
  x_min: float = -50.0
  y_min: float = -20.0
  x_max: float = 100.0
  y_max: float = 100.0
  psi_min: float = -0.5
  psi_max: float = 0.5

@dataclass
class RouteGeneratorConfig:
    """Parameters controlling the generated path shape."""
    n_connect_points: int = 8
    tangent_scale: float = 0.55
    min_leg_length: float = 15.0
    approach_midpoints: int = 1

def _wrap_pi(a: float) -> float:
  return math.atan2(math.sin(a), math.cos(a))

def _hermite_spline(P0: np.ndarray, T0: np.ndarray,
                    P1: np.ndarray, T1: np.ndarray,
                    n_points: int) -> List[Tuple[float, float]]:
  """
    Sample a cubic Hermite spline between P0 and P1.
 
    P0, P1: endpoint position vectors (2D)
    T0, T1: tangent vectors at each endpoint (direction + magnitude)
    n_points: number of interior + endpoint samples (total = n_points + 1)
 
    Returns list of (x, y) tuples including both endpoints.
    """
  points = []
  for i in range(n_points + 1):
      t = i / n_points
      # Hermite basis functions
      h00 = 2*t**3 - 3*t**2 + 1
      h10 = t**3 - 2*t**2 + t
      h01 = -2*t**3 + 3*t**2
      h11 = t**3 - t**2
 
      pt = h00 * P0 + h10 * T0 + h01 * P1 + h11 * T1
      points.append((float(pt[0]), float(pt[1])))
  return points

def _prune_short_legs(waypoints: List[Tuple[float, float]],
                      min_distance: float) -> List[Tuple[float, float]]:
  """Remove waypoints that create legs shorter than min_distance,
  keeping the first and last points always.
  """
  if len(waypoints) <= 2:
    return waypoints  # Nothing to prune
  
  pruned = [waypoints[0]]
  for wp in waypoints[1: -1]:
    dx = wp[0] - pruned[-1][0]
    dy = wp[1] - pruned[-1][-1]
    if math.hypot(dx, dy) >= min_distance:
      pruned.append(wp)
  pruned.append(waypoints[-1])
  return pruned

def compute_approach_corridor(dock: DockConfig) -> Tuple[float, float]:
  """
  Build the fixed approach corridor waypoints. 

  Returns a list starting at the corridor entry and ending at the dock.
  The entry point is "approach length" metres "behind" the dock
  (opposite to the approach heading).
  """
  entry_x = dock.dock_x - dock.approach_length * math.cos(dock.approach_heading)
  entry_y = dock.dock_y - dock.approach_length * math.sin(dock.approach_heading)
  
  corridor = [(entry_x, entry_y)]
  
  n_mid = max(0, dock.approach_length // 60 - 1)  # Add midpoints every 60m if approach is long enough
  for i in range(1, int(n_mid) + 1):
    frac = i / (n_mid + 1)
    mx = entry_x + frac * (dock.dock_x - entry_x)
    my = entry_y + frac * (dock.dock_y - entry_y)
    corridor.append((mx, my))
    
  corridor.append((dock.dock_x, dock.dock_y))
  return corridor

def generate_route(start_x: float, start_y: float, start_psi: float,
                   dock: DockConfig = None,
                   params: RouteGeneratorConfig = None) -> List[Tuple[float, float]]:
  
  """Generate a complete waypoint route from an arbitrary start pose to the dock."""
  
  if dock is None:
    dock = DockConfig()
  if params is None:
    params = RouteGeneratorConfig()
    
  corridor = compute_approach_corridor(dock)
  entry_x, entry_y = corridor[0]
  
  # Distance from start to corridor entry
  distance_to_entry = math.hypot(entry_x - start_x, entry_y - start_y)
  
  # Heading from start to entry
  bearing_to_entry = math.atan2(entry_y - start_y, entry_x - start_x)
  
  # Adaptive tangent scaling based on initial heading difference to approach corridor
  # Increase for large
  heading_diff = abs(_wrap_pi(start_psi - dock.approach_heading))
  scale = params.tangent_scale
  
  if heading_diff > math.radians(90):
    scale = min(0.85, scale + 0.15)
  elif heading_diff < math.radians(20):
    scale = max(0.35, scale - 0.1)
    
  mag = distance_to_entry * scale
  
  P0 = np.array([start_x, start_y])
  P1 = np.array([entry_x, entry_y])
  T0 = np.array([math.cos(start_psi) * mag, math.sin(start_psi) * mag])
  T1 = np.array([math.cos(dock.approach_heading) * mag,
                    math.sin(dock.approach_heading) * mag])
  
  # Number of sample points for the connecting curve
  number_of_sample_points = max(params.n_connect_points, int(distance_to_entry // 40))
  connecting = _hermite_spline(P0, T0, P1, T1, number_of_sample_points)
  
  # Prune waypoints that create very short legs, which can cause erratic LOS behavior
  connecting = _prune_short_legs(connecting, params.min_leg_length)
  
  route = connecting[:-1] + corridor # Connect the two segments, avoiding duplicate entry point
  
  return route

def random_start_pose(area: InitialArea = None, seed: int = None) -> Tuple[float, float, float]:
  """Sample a random initial pose for the ship within the defined area."""
  if area is None:
    area = InitialArea()
  if seed is not None:
    random.seed(seed)
    
  x = random.uniform(area.x_min, area.x_max)
  y = random.uniform(area.y_min, area.y_max)
  
  psi = random.uniform(area.psi_min, area.psi_max)
  return x, y, psi

def generate_random_route(dock: DockConfig = None,
                          params: RouteGeneratorConfig = None,
                          area: InitialArea = None,
                          seed: int = None) -> Tuple[List[Tuple[float, float]], Tuple[float, float, float]]:
  
  """Generates a random initial pose and the corresponding route to the dock."""
  
  sx, sy, spsi = random_start_pose(area, seed)
  route = generate_route(sx, sy, spsi, dock, params)
  return route, (sx, sy, spsi)

def analyze_route(waypoints: List[Tuple[float, float]]) -> None:
  """Print leg-by-leg analysis of the route for debugging"""
  
  print(f"Route: {len(waypoints)} waypoints")
  total_length = 0.0
  max_turn = 0.0
  
  # Analyze each leg
  for i in range(len(waypoints) - 1):
    x0, y0 = waypoints[i]
    x1, y1 = waypoints[i + 1]
    phi = math.atan2(y1 - y0, x1 - x0)
    leg_length = math.hypot(x1 - x0, y1 - y0)
    total_length += leg_length
    
    if i > 0:
      # Compute turn angle from previous leg
      x_prev, y_prev = waypoints[i - 1]
      phi_prev = math.atan2(y0 - y_prev, x0 - x_prev)
      turn = abs(_wrap_pi(phi - phi_prev))
    else: 
      turn = 0.0
    max_turn = max(max_turn, turn)
    
    print(f"  Leg {i:2d}: ({x0:7.1f},{y0:7.1f}) -> ({x1:7.1f},{y1:7.1f})  "
              f"heading={math.degrees(phi):+7.1f}°  len={leg_length:6.1f}m  turn={turn:5.1f}°")
 
  print(f"  Total path length: {total_length:.1f} m")
  print(f"  Max turn angle:    {max_turn:.1f}°")
  
if __name__ == "__main__":
  dock = DockConfig()
  params = RouteGeneratorConfig()
  area = InitialArea()
  
  for seed in [42, 123, 7, 999, 2024]:
        print(f"\n{'='*60}")
        print(f"Seed {seed}:")
        route, (sx, sy, spsi) = generate_random_route(area, dock, params, seed)
        print(f"  Start: ({sx:.1f}, {sy:.1f}, {math.degrees(spsi):.1f}°)")
        analyze_route(route)