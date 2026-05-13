import math
import random
import csv, os, atexit
import agx
import agxCollide
import agxOSG
from agxPythonModules.utils.environment import simulation, application, root
from agxPythonModules.utils.callbacks import StepEventCallback as Sec

from agx_wrap.world import create_ocean
from modeling.vessel import Ship
from control.los_guidance import LOSGuidance, LOSParams
from control.reference import LOSReferenceFilter, HeadRefParams, SpeedRefParams
from control.controller import LOSPIDController, LOSPIDGains
from control.observer import SimpleObserver, ObsGains
from control.allocation import TwoThrusterAllocator, Geometry2Thrusters
from control.route_generator import generate_random_route, analyze_route
from model.model_controller import MLPController
from runtime.config import (
    vessel as VCFG,
    los as LCFG,
    scene as SCFG,
    dock as DCFG,
    init_area as ICFG,
    route_gen as RCFG,
    gnss as NCFG,
)

_seed_env = os.environ.get("ROUTE_SEED", None)
ROUTE_SEED = int(_seed_env) if _seed_env is not None else None

USE_MLP = True
CONTROLLER_NAME = "MLP" if USE_MLP else "PID"
controller = MLPController() if USE_MLP else None

waypoints, (start_x, start_y, start_psi) = generate_random_route(
    area=ICFG,
    dock=DCFG,
    params=RCFG,
    seed=ROUTE_SEED,
)
print(f"\n── Route generated (seed={ROUTE_SEED}) ──")
print(f"   Controller: {CONTROLLER_NAME}")
print(f"   Wave:       {SCFG.wave_height:.1f} m")
print(f"   Start:      ({start_x:.1f}, {start_y:.1f}, {math.degrees(start_psi):.1f}°)")
print(f"   Dock:       ({DCFG.dock_x:.1f}, {DCFG.dock_y:.1f})")
analyze_route(waypoints)

log_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "los_log.csv")
_header = [
    "run_id",
    "seed",
    "controller",
    "wave_height",
    "t",
    "x",
    "y",
    "psi",
    "u",
    "v",
    "r",
    "chi_los",
    "psi_r",
    "u_r",
    "e_ct",
    "e_psi",
    "tau_x",
    "tau_y",
    "tau_psi",
    "Fx1",
    "Fy1",
    "Fx2",
    "Fy2",
    "leg",
    "finished",
    "dist_dock",
]

_file_exists = os.path.isfile(log_path) and os.path.getsize(log_path) > 0
if _file_exists:
    with open(log_path, "r", newline="") as f:
        reader = csv.reader(f)
        header_row = next(reader, None)
        if header_row != _header:
            raise RuntimeError(
                "Existing data/los_log.csv uses an older header. "
                "Rename or delete it before logging PID vs MLP comparison runs."
            )
        rid_col = header_row.index("run_id")
        max_rid = max((int(row[rid_col]) for row in reader if row), default=-1)
        RUN_ID = max_rid + 1
else:
    RUN_ID = 0

_log_file = open(log_path, "a" if _file_exists else "w", newline="")
_log_writer = csv.writer(_log_file)
if not _file_exists:
    _log_writer.writerow(_header)

print(f"   Logging as run_id={RUN_ID} → {os.path.abspath(log_path)}")


def _wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def build_scene_and_start():
    application().getSceneDecorator().setEnableShadows(False)
    application().setEnableDebugRenderer(True)

    _, _, wwc = create_ocean(height=SCFG.wave_height)

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
    ship.setPosition(agx.Vec3(start_x, start_y, 2.0))
    ship.setRotation(agx.EulerAngles(0, 0, start_psi))
    simulation().add(ship)
    print(
        f"Ship initial position: x={start_x:.1f} m, y={start_y:.1f} m, "
        f"psi={math.degrees(start_psi):.1f} °"
    )
    print(
        f"   Spawned heading: {math.degrees(ship.get_xy_psi()[2]):.1f}° "
        f"(expected {math.degrees(start_psi):.1f}°)"
    )

    los = LOSGuidance(
        waypoints=waypoints,
        params=LOSParams(
            Delta_min=LCFG.Delta_min,
            Delta_k=LCFG.Delta_k,
            switch_radius=LCFG.switch_radius,
            final_dock_radius=LCFG.final_dock_radius,
            u_desired=LCFG.u_desired,
            u_approach=LCFG.u_approach,
            approach_dist=LCFG.approach_dist,
        ),
    )

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

    obs = SimpleObserver(
        ObsGains(
            L_eta=SCFG.obs_L_eta,
            L_nu_xy=SCFG.obs_L_nu_xy,
            L_nu_psi=SCFG.obs_L_nu_psi,
            filter_alpha=SCFG.obs_filter_alpha,
        )
    )
    x0, y0, psi0 = ship.get_xy_psi()
    obs.reset(x0, y0, psi0)

    lx1 = float(ship.thruster_port_local.x())
    ly1 = float(ship.thruster_port_local.y())
    lx2 = float(ship.thruster_star_local.x())
    ly2 = float(ship.thruster_star_local.y())
    alloc = TwoThrusterAllocator(
        Geometry2Thrusters(
            lx1=lx1,
            ly1=ly1,
            lx2=lx2,
            ly2=ly2,
            biasFy=VCFG.alloc_bias_Fy,
        ),
        Tmax=VCFG.Tmax_thruster,
    )

    M = [VCFG.mass, VCFG.mass, VCFG.Iz]
    D = [VCFG.Xu, VCFG.Yv, VCFG.Nr]
    ctl = LOSPIDController(
        M_diag=M,
        D_diag=D,
        gains=LOSPIDGains(
            Kp_u=SCFG.Kp_u,
            Ki_u=SCFG.Ki_u,
            Kp_psi=SCFG.Kp_psi,
            Kd_psi=SCFG.Kd_psi,
            Ki_psi=SCFG.Ki_psi,
            Kd_v=SCFG.Kd_v,
            tau_surge_max=SCFG.tau_surge_max,
            tau_sway_max=SCFG.tau_sway_max,
            tau_yaw_max=SCFG.tau_yaw_max,
        ),
    )

    sd = application().getSceneDecorator()
    sd.setText(1, "PID/MLP comparison logging active")
    sd.setText(2, "Thrusters [Fx1,Fy1,Fx2,Fy2] (kN)")
    sd.setText(3, "τ [X,Y,N] (kN)")

    last_tau = (0.0, 0.0, 0.0)
    t_sim = 0.0
    n_legs = len(waypoints) - 1
    run_finished = False

    def los_step(_time: float):
        nonlocal last_tau, t_sim, run_finished
        
        if run_finished:
            return
        
        dt = simulation().getTimeStep()
        t_sim += dt

        x, y, psi = ship.get_xy_psi()
        if getattr(NCFG, "disable_noise", False):
            x_m, y_m, psi_m = x, y, psi
        else:
            x_m = x + random.gauss(0.0, NCFG.sigma_pos)
            y_m = y + random.gauss(0.0, NCFG.sigma_pos)
            psi_m = _wrap_pi(psi + random.gauss(0.0, NCFG.sigma_psi))

        (xh, yh, psih), (uh, vh, rh) = obs.step(
            dt,
            meas_x=x_m,
            meas_y=y_m,
            meas_psi=psi_m,
            tau_x=last_tau[0],
            tau_y=last_tau[1],
            tau_n=last_tau[2],
            M=M,
            D=D,
        )

        g = los.step(xh, yh, u=uh)
        chi_los = g["chi_los"]
        u_d = g["u_d"]
        e_ct = g["e_ct"]
        e_at = g["e_at"]
        leg = g["leg"]
        finished = g["finished"]
        if finished:
            run_finished = True
            _log_file.flush()
            close_log()
            print(f"Docking finished at t={t_sim:.2f}s. Closing viewer.")
            raise SystemExit(0)
            
        heading_err = abs(_wrap_pi(chi_los - psih))
        heading_low = math.radians(15)
        heading_high = math.radians(60)
        if heading_err > heading_low:
            blend = min(1.0, (heading_err - heading_low) / (heading_high - heading_low))
            u_d = u_d * (1.0 - blend) + LCFG.u_approach * blend

        u_r, psi_r, r_r = ref.step(dt, chi_los, u_d)

        e_psi_deg = math.degrees(_wrap_pi(psi_r - psih))
        dx = DCFG.dock_x - xh
        dy = DCFG.dock_y - yh
        dist_dock = math.hypot(dx, dy)

        if USE_MLP:
            tau_x, tau_y, tau_psi = controller.predict(
                x=xh,
                y=yh,
                psi=psih,
                u=uh,
                v=vh,
                r=rh,
                u_r=u_r,
                e_ct=e_ct,
                e_psi=e_psi_deg,
                dist_dock=dist_dock,
            )
        else:
            tau_x, tau_y, tau_psi = ctl.step(
                dt,
                u_r=u_r,
                psi_r=psi_r,
                r_r=r_r,
                u_hat=uh,
                v_hat=vh,
                r_hat=rh,
                psi_hat=psih,
            )

        Fx1, Fy1, Fx2, Fy2 = alloc.allocate(tau_x, tau_y, tau_psi)
        ship.apply_thruster_forces(Fx1, Fy1, Fx2, Fy2)
        last_tau = (tau_x, tau_y, tau_psi)

        Delta_now = LCFG.Delta_min + LCFG.Delta_k * abs(uh)
        status = "FINISHED" if finished else f"Leg {leg}/{n_legs - 1}"

        sd.setText(
            0,
            f" LOS Docking {status}  ctrl={CONTROLLER_NAME}  wave={SCFG.wave_height:.1f}m  "
            f"t={t_sim:.1f}s  seed={ROUTE_SEED}"
        )
        sd.setText(1, f"Position:  x={xh:.1f} m   y={yh:.1f} m   ψ={math.degrees(psih):.1f}°")
        sd.setText(
            2,
            f"Velocity:  surge={uh:.2f} m/s   sway={vh:.2f} m/s   "
            f"yaw rate={math.degrees(rh):.2f} °/s"
        )
        sd.setText(
            3,
            f"Path:  cross-track={e_ct:+.1f} m   dist-to-WP={e_at:.1f} m   "
            f"dist-to-dock={dist_dock:.1f} m   Δ={Delta_now:.1f} m"
        )
        sd.setText(
            4,
            f"Reference:  u_ref={u_r:.2f} m/s   ψ_ref={math.degrees(psi_r):.1f}°   "
            f"r_ref={math.degrees(r_r):.2f} °/s"
        )
        sd.setText(
            5,
            f"Heading err: {e_psi_deg:+.1f}°   χ_LOS={math.degrees(chi_los):.1f}°   "
            f"u_desired={u_d:.2f} m/s"
        )
        sd.setText(6, "── Control Forces ──")
        sd.setText(
            7,
            f"τ_surge={tau_x / 1e3:+.1f} kN   τ_sway={tau_y / 1e3:+.1f} kN   "
            f"τ_yaw={tau_psi / 1e3:+.1f} kNm"
        )
        sd.setText(
            8,
            f"Port:  Fx={Fx1 / 1e3:+.1f} kN  Fy={Fy1 / 1e3:+.1f} kN   "
            f"Star:  Fx={Fx2 / 1e3:+.1f} kN  Fy={Fy2 / 1e3:+.1f} kN"
        )

        _log_writer.writerow(
            [
                RUN_ID,
                ROUTE_SEED if ROUTE_SEED is not None else "",
                CONTROLLER_NAME,
                f"{SCFG.wave_height:.3f}",
                f"{t_sim:.4f}",
                f"{xh:.4f}",
                f"{yh:.4f}",
                f"{psih:.4f}",
                f"{uh:.4f}",
                f"{vh:.4f}",
                f"{rh:.4f}",
                f"{chi_los:.4f}",
                f"{psi_r:.4f}",
                f"{u_r:.4f}",
                f"{e_ct:.4f}",
                f"{e_psi_deg:.4f}",
                f"{tau_x:.1f}",
                f"{tau_y:.1f}",
                f"{tau_psi:.1f}",
                f"{Fx1:.1f}",
                f"{Fy1:.1f}",
                f"{Fx2:.1f}",
                f"{Fy2:.1f}",
                leg,
                int(finished),
                f"{dist_dock:.4f}",
            ]
        )

        if int(t_sim / dt) % 200 == 0:
            print(
                f"[{t_sim:6.1f}s] ctrl={CONTROLLER_NAME} wave={SCFG.wave_height:.1f}m "
                f"leg={leg}/{n_legs - 1} e_ct={e_ct:+.2f} "
                f"dist_dock={dist_dock:.2f} u_r={u_r:.3f} "
                f"chi={math.degrees(chi_los):+.1f}°"
            )

    Sec.preCallback(lambda t: los_step(t))

    def close_log():
        try:
            _log_file.close()
        except Exception:
            pass

    atexit.register(close_log)

    cam = application().getCameraData()
    cam.eye = agx.Vec3(start_x - 30.0, start_y - 80.0, 45.0)
    cam.center = agx.Vec3(start_x, start_y, 5.0)
    cam.up = agx.Vec3(0.0, 0.0, 1.0)
    cam.nearClippingPlane = 0.1
    cam.farClippingPlane = 5000.0