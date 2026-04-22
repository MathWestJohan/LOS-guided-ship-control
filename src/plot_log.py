import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Rectangle
from runtime.config import dock as DCFG, init_area as ICFG
from pathlib import Path
from matplotlib import transforms


def _leg_transitions(df):
    changes = df.leg.diff().fillna(0).abs() > 0
    return df.index[changes].tolist(), df.t.iloc[df.index[changes]].tolist()

def plot_los_log(csv_path="data/los_log.csv", out_path="data/los_log_plot.png"):
    df = pd.read_csv(csv_path)

    if "run_id" not in df.columns:
        df["run_id"] = 0
    run_ids = sorted(df.run_id.unique())
    n_runs = len(run_ids)
    cmap = get_cmap("tab10") if n_runs <= 10 else get_cmap("turbo", n_runs)

    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.32, wspace=0.30)

    ax_traj  = fig.add_subplot(gs[0, 0])
    ax_head  = fig.add_subplot(gs[0, 1])
    ax_ct    = fig.add_subplot(gs[1, 0])
    ax_sp    = ax_ct.twinx()
    ax_epsi  = fig.add_subplot(gs[1, 1])
    ax_tau   = fig.add_subplot(gs[2, 0])
    ax_thr   = fig.add_subplot(gs[2, 1])

    for idx, rid in enumerate(run_ids):
        rd = df[df.run_id == rid].reset_index(drop=True)
        c = cmap(idx / max(n_runs - 1, 1))
        label = f"Run {rid}"
        alpha = max(0.3, 1.0 - 0.04 * n_runs)  # fade for many runs

        leg_idxs, leg_times = _leg_transitions(rd)
        wp_x = [rd.x.iloc[0]] + [rd.x.iloc[i] for i in leg_idxs] + [rd.x.iloc[-1]]
        wp_y = [rd.y.iloc[0]] + [rd.y.iloc[i] for i in leg_idxs] + [rd.y.iloc[-1]]

        # 2D trajectory
        ax_traj.plot(rd.x, rd.y, color=c, lw=1.0, alpha=alpha, label=label)
        ax_traj.plot(wp_x, wp_y, "o--", color=c, ms=3, lw=0.5, alpha=0.5)
        ax_traj.plot(rd.x.iloc[0], rd.y.iloc[0], "s", color=c, ms=6)
        ax_traj.plot(rd.x.iloc[-1], rd.y.iloc[-1], "^", color=c, ms=6)

        # Heading
        psi_u  = np.unwrap(rd.psi)
        chi_u  = np.unwrap(rd.chi_los)
        psir_u = np.unwrap(rd.psi_r)
        ax_head.plot(rd.t, np.degrees(psi_u), color=c, alpha=alpha, lw=0.8)
        ax_head.plot(rd.t, np.degrees(chi_u), "--", color=c, alpha=alpha*0.6, lw=0.7)

        # Cross-track & speed
        ax_ct.plot(rd.t, rd.e_ct, color=c, alpha=alpha, lw=0.8)
        ax_sp.plot(rd.t, rd.u, color=c, alpha=alpha*0.6, lw=0.7, ls="--")

        # Heading error
        ax_epsi.plot(rd.t, rd.e_psi, color=c, alpha=alpha, lw=0.8)

        # Control forces
        ax_tau.plot(rd.t, rd.tau_x / 1e3, color=c, alpha=alpha, lw=0.7)

        # Thruster forces
        ax_thr.plot(rd.t, rd.Fx1 / 1e3, color=c, alpha=alpha, lw=0.7)
        ax_thr.plot(rd.t, rd.Fx2 / 1e3, "--", color=c, alpha=alpha*0.6, lw=0.7)
    
    # Dock location
    dock_w = 20.0
    dock_h = 6.0

    dock_rect = Rectangle(
        (DCFG.dock_x - dock_w / 2, DCFG.dock_y - dock_h / 2),
        dock_w,
        dock_h,
        linewidth=2.0,
        edgecolor="red",
        facecolor="none",
        zorder=10,
        label="Dock",
    )

    dock_transform = transforms.Affine2D().rotate_around(
        DCFG.dock_x,
        DCFG.dock_y,
        DCFG.approach_heading,
    ) + ax_traj.transData

    dock_rect.set_transform(dock_transform)
    ax_traj.add_patch(dock_rect)

    ax_traj.annotate(
        "DOCK",
        (DCFG.dock_x, DCFG.dock_y),
        textcoords="offset points",
        xytext=(8, -12),
        fontsize=9,
        fontweight="bold",
        color="red",
    )

    # Initial start zone
    zone_w = ICFG.x_max - ICFG.x_min
    zone_h = ICFG.y_max - ICFG.y_min
    rect = Rectangle((ICFG.x_min, ICFG.y_min), zone_w, zone_h,
                      linewidth=1.5, edgecolor="green", facecolor="green",
                      alpha=0.12, linestyle="--", label="Start zone", zorder=1)
    ax_traj.add_patch(rect)

    # Axis labels & titles
    ax_traj.set_xlabel("X [m]"); ax_traj.set_ylabel("Y [m]")
    ax_traj.set_title(f"2D Trajectories ({n_runs} runs)")
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.grid(True, alpha=0.3)
    if n_runs <= 15:
        ax_traj.legend(fontsize=6, ncol=2)

    ax_head.set_ylabel("Heading [deg]"); ax_head.set_title("Heading")
    ax_head.grid(True, alpha=0.3)

    ax_ct.set_ylabel("Cross-track [m]", color="r")
    ax_ct.axhline(0, color="r", ls=":", lw=0.5)
    ax_sp.set_ylabel("Surge speed [m/s]", color="b")
    ax_ct.grid(True, alpha=0.3)

    ax_epsi.set_ylabel("Heading error [deg]")
    ax_epsi.set_title("Heading error (ψ_ref − ψ)")
    ax_epsi.axhline(0, color="gray", ls=":", lw=0.5)
    ax_epsi.grid(True, alpha=0.3)

    ax_tau.set_ylabel("τ_surge [kN]"); ax_tau.set_xlabel("Time [s]")
    ax_tau.grid(True, alpha=0.3)

    ax_thr.set_ylabel("Fx thruster [kN]"); ax_thr.set_xlabel("Time [s]")
    ax_thr.grid(True, alpha=0.3)

    fig.suptitle(f"LOS Guidance — {n_runs} run(s)", fontsize=14, y=0.99)
    plt.tight_layout()
    out = Path(out_path)
    plt.savefig(out, dpi=150)
    print(f"Saved {out.resolve()}")


if __name__ == "__main__":
    plot_los_log()