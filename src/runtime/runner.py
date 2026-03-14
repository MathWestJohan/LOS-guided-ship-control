import math
import random
import csv, os, atexit
import agx
from agxPythonModules.utils.environment import simulation, application
from agxPythonModules.utils.callbacks import StepEventCallback as Sec

from agx_wrap.world import create_ocean
from modeling.vessel import Ship
from control.los_guidance import LOSGuidance, LOSParams
from control.reference import LOSReferenceFilter, HeadRefParams, SpeedRefParams
from control.controller import LOSPIDController, LOSPIDGains
from control.observer import SimpleObserver, ObsGains
from control.allocation import TwoThrusterAllocator, Geometry2Thrusters
from runtime.config import vessel as VCFG, los as LCFG, scene as SCFG, route as RCFG, gnss as NCFG

# ── CSV logger ──────────────────────────────────────────────────────
log_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "los_log.csv")
_log_file = open(log_path, "w", newline="")
_log_writer = csv.writer(_log_file)
_log_writer.writerow([
    "t", "x", "y", "psi", "u", "v", "r",
    "chi_los", "psi_r", "u_r",
    "e_ct", "e_psi",
    "tau_x", "tau_y", "tau_psi",
    "Fx1", "Fy1", "Fx2", "Fy2",
    "leg", "finished"
])

def _wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def build_scene_and_start():
    application().getSceneDecorator().setEnableShadows(False)
    application().setEnableDebugRenderer(True)
    
    create_ocean(height=SCFG.wave_height)
    
    #for i, (wx, wy) in enumerate(RCFG.waypoints):
        #sphere = agxCollide.Geometry(agxCollide.Sphere(1.0))
        #sphere.setPosition(agx.Vec3(wx, wy, 3.0))
        #sphere.setSensor(True)  # no physics interaction
        #node = agxOSG.createVisual(sphere, root())
        #if i == len(RCFG.waypoints) - 1:
            #agxOSG.setDiffuseColor(node, agx.Vec4f(0, 1, 0, 1))    # green = goal
        #else:
            #agxOSG.setDiffuseColor(node, agx.Vec4f(1, 0.5, 0, 1))  # orange = waypoint
        #simulation().add(sphere)

    # Draw path lines between waypoints
    #for i in range(len(RCFG.waypoints) - 1):
        #x0, y0 = RCFG.waypoints[i]
        #x1, y1 = RCFG.waypoints[i + 1]
        #mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        #length = math.hypot(x1 - x0, y1 - y0)
        #angle = math.atan2(y1 - y0, x1 - x0)

        #box = agxCollide.Geometry(agxCollide.Box(length / 2, 0.1, 0.1))
        #box.setPosition(agx.Vec3(mx, my, 2.5))
        #box.setRotation(agx.EulerAngles(0, 0, angle))
        #box.setSensor(True)
        #node = agxOSG.createVisual(box, root())
        #agxOSG.setDiffuseColor(node, agx.Vec4f(1, 1, 0, 0.8))
        #simulation().add(box)

    # ── Ship ────────────────────────────────────────────────────────
    ship = Ship(
        mass_kg=VCFG.mass,
        half_length=VCFG.half_length,
        half_width=VCFG.half_width,
        half_height=VCFG.half_height,
        cm_shift_x=VCFG.cm_shift_x,
        thruster_z_offset=VCFG.thruster_z_offset,
        stern_x_offset=VCFG.stern_x_offset,
        thr_port_x=VCFG.thr_port_x,
        thr_port_y=VCFG.thr_port_y,
        thr_star_x=VCFG.thr_star_x,
        thr_star_y=VCFG.thr_star_y,
    )
    start = RCFG.waypoints[0]
    ship.setPosition(agx.Vec3(start[0], start[1], 2.0))
    simulation().add(ship)

    # ── LOS guidance ────────────────────────────────────────────────
    los = LOSGuidance(
        waypoints=RCFG.waypoints,
        params=LOSParams(
            Delta_min=LCFG.Delta_min,
            Delta_k=LCFG.Delta_k,
            switch_radius=LCFG.switch_radius,
            u_desired=LCFG.u_desired,
            u_approach=LCFG.u_approach,
            approach_dist=LCFG.approach_dist,
        ),
    )

    # ── Reference filter ────────────────────────────────────────────
    ref = LOSReferenceFilter(
        head_params=HeadRefParams(
            omega=SCFG.ref_head_wn,
            zeta=SCFG.ref_head_zeta,
            rmax=SCFG.ref_head_rmax,
        ),
        speed_params=SpeedRefParams(
            omega=SCFG.ref_speed_wn,
            zeta=SCFG.ref_speed_zeta,
            umax=SCFG.ref_speed_umax,
        ),
    )
    ref.reset(psi_now=ship.get_xy_psi()[2])

    # ── Observer ────────────────────────────────────────────────────
    obs = SimpleObserver(ObsGains(
        L_eta=SCFG.obs_L_eta,
        L_nu_xy=SCFG.obs_L_nu_xy,
        L_nu_psi=SCFG.obs_L_nu_psi,
    ))
    x0, y0, psi0 = ship.get_xy_psi()
    obs.reset(x0, y0, psi0)

    # ── Allocator ───────────────────────────────────────────────────
    lx1 = float(ship.thruster_port_local.x())
    ly1 = float(ship.thruster_port_local.y())
    lx2 = float(ship.thruster_star_local.x())
    ly2 = float(ship.thruster_star_local.y())
    alloc = TwoThrusterAllocator(
        Geometry2Thrusters(lx1=lx1, ly1=ly1, lx2=lx2, ly2=ly2,
                           biasFy=VCFG.alloc_bias_Fy),
        Tmax=VCFG.Tmax_thruster,
    )

    # ── Controller ──────────────────────────────────────────────────
    M = [VCFG.mass, VCFG.mass, VCFG.Iz]
    D = [VCFG.Xu,   VCFG.Yv,   VCFG.Nr]
    ctl = LOSPIDController(
        M_diag=M, D_diag=D,
        gains=LOSPIDGains(
            Kp_u=SCFG.Kp_u, Ki_u=SCFG.Ki_u,
            Kp_psi=SCFG.Kp_psi, Kd_psi=SCFG.Kd_psi, Ki_psi=SCFG.Ki_psi,
            Kd_v=SCFG.Kd_v,
            tau_surge_max=SCFG.tau_surge_max,
            tau_sway_max=SCFG.tau_sway_max,
            tau_yaw_max=SCFG.tau_yaw_max,
        ),
    )

    # ── HUD ─────────────────────────────────────────────────────────
    sd = application().getSceneDecorator()
    sd.setText(1, "LOS guidance active")
    sd.setText(2, "Thrusters [Fx1,Fy1,Fx2,Fy2] (kN)")
    sd.setText(3, "τ [X,Y,N] (kN)")

    last_tau = (0.0, 0.0, 0.0)
    t_sim = 0.0

    # ── Step callback ───────────────────────────────────────────────
    def los_step(_time: float):
        nonlocal last_tau, t_sim
        dt = simulation().getTimeStep()
        t_sim += dt

        # Raw measurement (+ optional noise)
        x, y, psi = ship.get_xy_psi()
        if getattr(NCFG, "disable_noise", False):
            x_m, y_m, psi_m = x, y, psi
        else:
            x_m   = x   + random.gauss(0.0, NCFG.sigma_pos)
            y_m   = y   + random.gauss(0.0, NCFG.sigma_pos)
            psi_m = _wrap_pi(psi + random.gauss(0.0, NCFG.sigma_psi))

        # Observer
        (xh, yh, psih), (uh, vh, rh) = obs.step(
            dt,
            meas_x=x_m, meas_y=y_m, meas_psi=psi_m,
            tau_x=last_tau[0], tau_y=last_tau[1], tau_n=last_tau[2],
            M=M, D=D,
        )

        # LOS guidance (uses observer estimates)
        g = los.step(xh, yh, u=uh)
        chi_los  = g["chi_los"]
        u_d      = g["u_d"]
        e_ct     = g["e_ct"]
        e_at     = g["e_at"]
        leg      = g["leg"]
        finished = g["finished"]

        # Reference filter
        u_r, psi_r, r_r = ref.step(dt, chi_los, u_d)

        # Controller
        tau_x, tau_y, tau_psi = ctl.step(
            dt,
            u_r=u_r, psi_r=psi_r, r_r=r_r,
            u_hat=uh, v_hat=vh, r_hat=rh, psi_hat=psih,
        )

        # Allocate & apply
        Fx1, Fy1, Fx2, Fy2 = alloc.allocate(tau_x, tau_y, tau_psi)
        ship.apply_thruster_forces(Fx1, Fy1, Fx2, Fy2)
        last_tau = (tau_x, tau_y, tau_psi)

        # HUD
        Delta_now = LCFG.Delta_min + LCFG.Delta_k * abs(uh)
        status = "FINISHED" if finished else f"Leg {leg}/{len(RCFG.waypoints)-2}"
        e_psi_deg = math.degrees(_wrap_pi(psi_r - psih))

        sd.setText(0, f"── LOS Guidance ── {status}  t={t_sim:.1f}s")
        sd.setText(1, f"Position:  x={xh:.1f} m   y={yh:.1f} m   ψ={math.degrees(psih):.1f}°")
        sd.setText(2, f"Velocity:  surge={uh:.2f} m/s   sway={vh:.2f} m/s   yaw rate={math.degrees(rh):.2f} °/s")
        sd.setText(3, f"Path:  cross-track={e_ct:+.1f} m   dist-to-WP={e_at:.1f} m   Δ={Delta_now:.1f} m")
        sd.setText(4, f"Reference:  u_ref={u_r:.2f} m/s   ψ_ref={math.degrees(psi_r):.1f}°   r_ref={math.degrees(r_r):.2f} °/s")
        sd.setText(5, f"Heading err: {e_psi_deg:+.1f}°   χ_LOS={math.degrees(chi_los):.1f}°   u_desired={u_d:.2f} m/s")
        sd.setText(6, f"── Control Forces ──")
        sd.setText(7, f"τ_surge={tau_x/1e3:+.1f} kN   τ_sway={tau_y/1e3:+.1f} kN   τ_yaw={tau_psi/1e3:+.1f} kNm")
        sd.setText(8, f"Port:  Fx={Fx1/1e3:+.1f} kN  Fy={Fy1/1e3:+.1f} kN   Star:  Fx={Fx2/1e3:+.1f} kN  Fy={Fy2/1e3:+.1f} kN")
        
        # Log
        _log_writer.writerow([
            f"{t_sim:.4f}", f"{xh:.4f}", f"{yh:.4f}", f"{psih:.4f}",
            f"{uh:.4f}", f"{vh:.4f}", f"{rh:.4f}",
            f"{chi_los:.4f}", f"{psi_r:.4f}", f"{u_r:.4f}",
            f"{e_ct:.4f}", f"{e_psi_deg:.4f}",
            f"{tau_x:.1f}", f"{tau_y:.1f}", f"{tau_psi:.1f}",
            f"{Fx1:.1f}", f"{Fy1:.1f}", f"{Fx2:.1f}", f"{Fy2:.1f}",
            leg, int(finished),
        ])

        if int(t_sim / dt) % 200 == 0:
            print(f"[{t_sim:6.1f}s] leg={leg} e_ct={e_ct:+.2f} u_r={u_r:.3f} chi={math.degrees(chi_los):+.1f}°")

    Sec.preCallback(lambda t: los_step(t))

    # ── Cleanup ─────────────────────────────────────────────────────
    def close_log():
        try:
            _log_file.close()
        except Exception:
            pass
    atexit.register(close_log)

    # ── Camera ──────────────────────────────────────────────────────
    cam = application().getCameraData()
    cam.eye    = agx.Vec3(start[0] - 30.0, start[1] - 80.0, 45.0)
    cam.center = agx.Vec3(start[0], start[1], 5.0)
    cam.up     = agx.Vec3(0.0, 0.0, 1.0)
    cam.nearClippingPlane = 0.1
    cam.farClippingPlane  = 5000.0