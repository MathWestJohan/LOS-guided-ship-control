import math
import random
import csv
import os
import time
import argparse
import agx
import agxSDK
import agxCollide
import agxModel
import agxUtil
import numpy as np
from agxPythonModules.utils.numpy_utils import wrap_vector_as_numpy_array

from control.los_guidance import LOSGuidance, LOSParams
from control.reference import LOSReferenceFilter, HeadRefParams, SpeedRefParams
from control.controller import LOSPIDController, LOSPIDGains
from control.observer import SimpleObserver, ObsGains
from control.allocation import TwoThrusterAllocator, Geometry2Thrusters
from control.route_generator import generate_random_route, analyze_route
from runtime.config import (vessel as VCFG, los as LCFG, scene as SCFG,
                             gnss as NCFG, dock as DOCK_CFG, init_area, route_gen)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ship_shape_name = os.path.join(_THIS_DIR, "..", "assets", "Gunnerus.obj")

def create_ship(sim: agxSDK.Simulation,
                mass_kg: float = 350_000.0,
                cm_shift_x: float = -0.2,
                thr_port_x: float = -10.0,
                thr_port_y: float = +2.76,
                thr_star_x: float = -10.0,
                thr_star_y: float = -2.76,
                thruster_z_offset: float = -2.5):
  
  mesh = agxUtil.createTrimesh(
    ship_shape_name,
    agxCollide.Trimesh.REMOVE_DUPLICATE_VERTICES,
    agx.Matrix3x3(agx.Vec3(1.0))
  )
  
  geom = agxCollide.Geometry(mesh)
  geom.setMaterial(agx.Material("steel"))
  
  body = agx.RigidBody(geom)
  body.getMassProperties().setMass(mass_kg)
  body.setPosition(agx.Vec3(0, 0, 0))
  
  mesh_rot = agx.EulerAngles(math.radians(90), 0, math.radians(-90))
  mesh_quat = agx.Quat(mesh_rot)
  mesh_quat_inv = mesh_quat.inverse()
  body.setRotation(mesh_rot)
  
  body.getCmFrame().setLocalTranslate(agx.Vec3(cm_shift_x, 0, 0))
  
  assembly = agxSDK.Assembly()
  assembly.add(body)
  
  sim.add(assembly)
  
  return {
    "body": body,
    "assembly": assembly,
    "mesh_quat": mesh_quat,
    "thr_port": agx.Vec3(thr_port_x, thr_port_y, thruster_z_offset),
    "thr_star": agx.Vec3(thr_star_x, thr_star_y, thruster_z_offset),
  }
  
def create_ocean_headless(sim: agxSDK.Simulation, wave_height: float = 0.0):
  """ Create a simple ocean plane for visualless simulation. """
  hf = agxCollide.HeightField(100, 100, 1000, 1000, 20.0)
  water_geom = agxCollide.Geometry(hf)
  water_geom.setMaterial(agx.Material("waterMaterial"))
  
  sim.add(water_geom)
  
  wwc = agxModel.WindAndWaterController()
  wwc.addWater(water_geom)
  sim.add(wwc)
  
  heights = agx.RealVector(hf.getResolutionX() * hf.getResolutionY())
  for _ in range(hf.getResolutionX() * hf.getResolutionY()):
    heights.append(0.0)
    
  np_heights = wrap_vector_as_numpy_array(heights, np.float64).reshape(
    (hf.getResolutionX(), hf.getResolutionY())
  )
  jj = np.stack((np.arange(hf.getResolutionY()),) * hf.getResolutionX())
  ii = np.stack((np.arange(hf.getResolutionX()),) * hf.getResolutionY(), axis=1)
  amp = float(wave_height)
  
  def update_waves(t: float):
    if amp > 0.0:
      np_heights[:, :] = amp * (
        0.40 * np.sin(1.00 * jj + 0.60 * t) + 
        0.10 * np.sin(1.20 * ii + 0.60 * jj + 1.45 * t)
      )
      hf.setHeights(heights)
      
  return hf, wwc, update_waves

def get_ship_state(body):
  """Return (x, y, heading) where heading is from the ship's
  geometric forward (body -z) projected onto the world XY plane.
  """
  p = body.getPosition()
  x, y = float(p.x()), float(p.y())
  
  q = body.getRotation()
  fwd = q * agx.Vec3(0, 0, -1)
  psi = math.atan2(float(fwd.y()), float(fwd.x()))
  
  return x, y, psi

def apply_thruster_force(body, Fx1, Fy1, Fx2, Fy2, thr_port, thr_star):
  q = body.getRotation()
  
  f1_world = q * agx.Vec3(-Fy1, 0, -Fx1)
  p1_body = agx.Vec3(-float(thr_port.y()),
                         float(thr_port.z()),
                        -float(thr_port.x()))
  body.addForceAtLocalPosition(f1_world, p1_body)
  
  f2_world = q * agx.Vec3(-Fy2, 0, -Fx2)
  p2_body = agx.Vec3(-float(thr_star.y()),
                         float(thr_star.z()),
                        -float(thr_star.x()))
  body.addForceAtLocalPosition(f2_world, p2_body)

def wrap_pi(a: float) -> float:
  return math.atan2(math.sin(a), math.cos(a))

# Data collection loop
def collect_data(n_episodes: int = 50, base_seed: int = 0,
                 max_ep_time: float = 300.0, dt: float = 1.0/60.0,
                 output_path: str = "data/training_data.csv"):
  
  os.makedirs(os.path.dirname(output_path), exist_ok = True)
  
  sim = agxSDK.Simulation()
  sim.setTimeStep(dt)
  
  hf, wwc, update_waves = create_ocean_headless(sim, wave_height=SCFG.wave_height)
  ship = create_ship(
        sim,
        mass_kg=VCFG.mass,
        cm_shift_x=VCFG.cm_shift_x,
        thr_port_x=VCFG.thr_port_x,
        thr_port_y=VCFG.thr_port_y,
        thr_star_x=VCFG.thr_star_x,
        thr_star_y=VCFG.thr_star_y,
        thruster_z_offset=VCFG.thruster_z_offset,
    )
  
  body = ship["body"]
  thr_port = ship["thr_port"]
  thr_star = ship["thr_star"]
  mesh_quat = ship["mesh_quat"]
  
  alloc = TwoThrusterAllocator(
    Geometry2Thrusters(
            lx1=float(thr_port.x()), ly1=float(thr_port.y()),
            lx2=float(thr_star.x()), ly2=float(thr_star.y()),
            biasFy=VCFG.alloc_bias_Fy,
    ),
    Tmax = VCFG.Tmax_thruster,
  )
  
  M = [VCFG.mass, VCFG.mass, VCFG.Iz]
  D = [VCFG.Xu, VCFG.Yv, VCFG.Nr]
  
  csv_file = open(output_path, "w", newline="")
  writer = csv.writer(csv_file)
  writer.writerow([
      "episode", "seed",
      "t", "x", "y", "psi", "u", "v", "r",
      "chi_los", "psi_r", "u_r",
      "e_ct", "e_psi",
      "tau_x", "tau_y", "tau_psi",
      "Fx1", "Fy1", "Fx2", "Fy2",
      "leg", "finished",
      "dx", "dy", "dist_dock",
  ])

  total_start = time.time()
  total_rows = 0
  
  for ep in range(n_episodes):
    
    seed = base_seed + ep
    
    waypoints, (sx, sy, psi) = generate_random_route(
      area=init_area, dock=DOCK_CFG, params=route_gen, seed=seed
    )
    
    body.setPosition(agx.Vec3(sx, sy, 2.0))
    body.setRotation(agx.Quat(agx.EulerAngles(0, 0, psi)) * mesh_quat)
    body.setVelocity(agx.Vec3(0, 0, 0))
    body.setAngularVelocity(agx.Vec3(0, 0, 0))
    
    los = LOSGuidance(
    waypoints=waypoints,
    params=LOSParams(
      Delta_min=LCFG.Delta_min, Delta_k=LCFG.Delta_k,
      switch_radius=LCFG.switch_radius, u_desired=LCFG.u_desired,
      u_approach=LCFG.u_approach, approach_dist=LCFG.approach_dist,
    ))
    
    ref = LOSReferenceFilter(
            head_params=HeadRefParams(
                omega=SCFG.ref_head_wn, zeta=SCFG.ref_head_zeta,
                rmax=SCFG.ref_head_rmax,
            ),
            speed_params=SpeedRefParams(
                omega=SCFG.ref_speed_wn, zeta=SCFG.ref_speed_zeta,
                umax=SCFG.ref_speed_umax,
            ),
        )

    x0, y0, psi0 = get_ship_state(body)
    ref.reset(psi_now=psi0)

    ctl = LOSPIDController(
            M_diag=M, D_diag=D,
            gains=LOSPIDGains(
                Kp_u=SCFG.Kp_u, Ki_u=SCFG.Ki_u,
                Kp_psi=SCFG.Kp_psi, Kd_psi=SCFG.Kd_psi, Ki_psi=SCFG.Ki_psi,
                Kd_v=SCFG.Kd_v,
                tau_surge_max=SCFG.tau_surge_max, tau_sway_max=SCFG.tau_sway_max,
                tau_yaw_max=SCFG.tau_yaw_max,
            ),
        )

    obs = SimpleObserver(ObsGains(
        L_eta=SCFG.obs_L_eta, L_nu_xy=SCFG.obs_L_nu_xy,
        L_nu_psi=SCFG.obs_L_nu_psi, filter_alpha=SCFG.obs_filter_alpha,
    ))
    obs.reset(x0, y0, psi0)

    last_tau = (0.0, 0.0, 0.0)
    t_ep = 0.0
    ep_start = time.time()
    ep_rows = 0

    max_steps = int(max_ep_time / dt)

    for step in range(max_steps):
        t_ep = step * dt

        # Update wave heightfield
        update_waves(t_ep)

        # Read ship state
        x, y, psi = get_ship_state(body)

        # Optional GNSS noise
        if getattr(NCFG, "disable_noise", False):
            x_m, y_m, psi_m = x, y, psi
        else:
            x_m = x + random.gauss(0.0, NCFG.sigma_pos)
            y_m = y + random.gauss(0.0, NCFG.sigma_pos)
            psi_m = wrap_pi(psi + random.gauss(0.0, NCFG.sigma_psi))

        # Observer
        (xh, yh, psih), (uh, vh, rh) = obs.step(
            dt, meas_x=x_m, meas_y=y_m, meas_psi=psi_m,
            tau_x=last_tau[0], tau_y=last_tau[1], tau_n=last_tau[2],
            M=M, D=D,
        )

        # LOS guidance
        g = los.step(xh, yh, u=uh)
        chi_los  = g["chi_los"]
        u_d      = g["u_d"]
        e_ct     = g["e_ct"]
        leg      = g["leg"]
        finished = g["finished"]

        # Heading-error speed governor
        heading_err = abs(wrap_pi(chi_los - psih))
        if heading_err > math.radians(15):
            blend = min(1.0, (heading_err - math.radians(15)) /
                        (math.radians(60) - math.radians(15)))
            u_d = u_d * (1.0 - blend) + LCFG.u_approach * blend

        # Reference filter
        u_r, psi_r, r_r = ref.step(dt, chi_los, u_d)

        # PID controller
        tau_x, tau_y, tau_psi = ctl.step(
            dt, u_r=u_r, psi_r=psi_r, r_r=r_r,
            u_hat=uh, v_hat=vh, r_hat=rh, psi_hat=psih,
        )

        # Thrust allocation and force application
        Fx1, Fy1, Fx2, Fy2 = alloc.allocate(tau_x, tau_y, tau_psi)
        apply_thruster_force(body, Fx1, Fy1, Fx2, Fy2, thr_port, thr_star)
        last_tau = (tau_x, tau_y, tau_psi)

        # Compute NN features
        dx = DOCK_CFG.dock_x - xh
        dy = DOCK_CFG.dock_y - yh
        dist_dock = math.hypot(dx, dy)
        e_psi = math.degrees(wrap_pi(psi_r - psih))

        # Write row
        writer.writerow([
            ep, seed,
            f"{t_ep:.4f}", f"{xh:.4f}", f"{yh:.4f}", f"{psih:.4f}",
            f"{uh:.4f}", f"{vh:.4f}", f"{rh:.4f}",
            f"{chi_los:.4f}", f"{psi_r:.4f}", f"{u_r:.4f}",
            f"{e_ct:.4f}", f"{e_psi:.4f}",
            f"{tau_x:.1f}", f"{tau_y:.1f}", f"{tau_psi:.1f}",
            f"{Fx1:.1f}", f"{Fy1:.1f}", f"{Fx2:.1f}", f"{Fy2:.1f}",
            leg, int(finished),
            f"{dx:.4f}", f"{dy:.4f}", f"{dist_dock:.4f}",
        ])
        ep_rows += 1


        sim.stepForward()

        # Check if episode is done
        if finished:
            break

    # Episode summary
    ep_time = time.time() - ep_start
    total_rows += ep_rows
    dist_final = math.hypot(DOCK_CFG.dock_x - x, DOCK_CFG.dock_y - y)
    status = "DOCKED" if finished else "TIMEOUT"

    print(f"  Episode {ep:3d}/{n_episodes-1}  seed={seed:4d}  "
          f"{status}  t={t_ep:.1f}s  dist={dist_final:.1f}m  "
          f"rows={ep_rows}  wall={ep_time:.1f}s")

    # Flush periodically so data is saved even if interrupted
    if ep % 10 == 0:
        csv_file.flush()

  csv_file.close()
  total_time = time.time() - total_start

  print(f"\n{'='*60}")
  print(f"Collection complete!")
  print(f"  Episodes:    {n_episodes}")
  print(f"  Total rows:  {total_rows}")
  print(f"  Output:      {output_path}")
  print(f"  Wall time:   {total_time:.1f}s ({total_time/n_episodes:.1f}s per episode)")


# Entry point 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless LOS docking data collection")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=300.0,
                        help="Max seconds per episode before timeout")
    parser.add_argument("--output", default="data/training_data.csv")
    args = parser.parse_args()

    print(f"AGX Headless Data Collection")
    print(f"  Episodes: {args.episodes}")
    print(f"  Seed:     {args.seed}")
    print(f"  Output:   {args.output}")
    print(f"  Max time: {args.max_time}s per episode")
    print()

    agx.init()
    try:
      collect_data(
          n_episodes=args.episodes,
          base_seed=args.seed,
          max_ep_time=args.max_time,
          output_path=args.output,
      )
    finally:
      agx.shutdown()