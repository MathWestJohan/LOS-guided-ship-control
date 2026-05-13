import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib import transforms

from runtime.config import dock as DCFG, init_area as ICFG, vessel as VCFG


PLOT_ACCEPTANCE_RADIUS_M = 5.0
DOCK_DRAW_LENGTH_M = 20.0
DOCK_DRAW_WIDTH_M = 6.0
BERTH_GAP_M = 1.5
BERTH_SIDE = -1.0
DOCKING_ZOOM_RADIUS_M = 45.0
SHIP_PLOT_SCALE_TRAJ = 0.50
SHIP_PLOT_SCALE_ZOOM = 0.65
SHIP_PLOT_SCALE_BERTH = 0.65

PID_COLOR = "#1f77b4"
MLP_COLOR = "#ff7f0e"


def _wrap_deg(rad_values):
    rad_values = np.asarray(rad_values, dtype=float)
    return np.degrees(np.arctan2(np.sin(rad_values), np.cos(rad_values)))


def _rigid_rect_polygon(cx, cy, psi, half_length, half_width):
    local = np.array(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=float,
    )

    c = float(np.cos(psi))
    s = float(np.sin(psi))
    rot = np.array([[c, -s], [s, c]], dtype=float)

    world = local @ rot.T
    world[:, 0] += cx
    world[:, 1] += cy
    return world


def _draw_ship_patch(
    ax,
    x,
    y,
    psi,
    edgecolor,
    facecolor="none",
    alpha=0.30,
    lw=1.0,
    zorder=12,
    scale=1.0,
):
    hull = _rigid_rect_polygon(
        x,
        y,
        psi,
        VCFG.half_length * scale,
        VCFG.half_width * scale,
    )
    patch = Polygon(
        hull,
        closed=True,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _berth_pose():
    psi = DCFG.approach_heading
    nx = -float(np.sin(psi))
    ny = float(np.cos(psi))

    offset = BERTH_SIDE * (DOCK_DRAW_WIDTH_M / 2.0 + VCFG.half_width + BERTH_GAP_M)
    bx = DCFG.dock_x + offset * nx
    by = DCFG.dock_y + offset * ny
    return bx, by, psi


def _decorate_traj_axis(ax):
    dock_rect = Rectangle(
        (DCFG.dock_x - DOCK_DRAW_LENGTH_M / 2, DCFG.dock_y - DOCK_DRAW_WIDTH_M / 2),
        DOCK_DRAW_LENGTH_M,
        DOCK_DRAW_WIDTH_M,
        linewidth=2.0,
        edgecolor="red",
        facecolor="none",
        zorder=10,
    )
    dock_transform = transforms.Affine2D().rotate_around(
        DCFG.dock_x,
        DCFG.dock_y,
        DCFG.approach_heading,
    ) + ax.transData
    dock_rect.set_transform(dock_transform)
    ax.add_patch(dock_rect)

    accept_circle = Circle(
        (DCFG.dock_x, DCFG.dock_y),
        radius=PLOT_ACCEPTANCE_RADIUS_M,
        edgecolor="red",
        facecolor="red",
        alpha=0.07,
        lw=1.5,
        linestyle="--",
        zorder=0,
    )
    ax.add_patch(accept_circle)

    ax.plot(
        DCFG.dock_x,
        DCFG.dock_y,
        marker="+",
        color="red",
        ms=9,
        mew=1.6,
        linestyle="None",
        zorder=12,
    )

    bx, by, bpsi = _berth_pose()
    berth_poly = _rigid_rect_polygon(
        bx,
        by,
        bpsi,
        VCFG.half_length * SHIP_PLOT_SCALE_BERTH,
        VCFG.half_width * SHIP_PLOT_SCALE_BERTH,
    )
    ax.add_patch(
        Polygon(
            berth_poly,
            closed=True,
            edgecolor="firebrick",
            facecolor="none",
            linestyle=":",
            linewidth=1.6,
            alpha=0.95,
            zorder=11,
        )
    )

    ax.annotate(
        "DOCK",
        (DCFG.dock_x, DCFG.dock_y),
        textcoords="offset points",
        xytext=(8, -12),
        fontsize=9,
        fontweight="bold",
        color="red",
    )

    ax.annotate(
        "BERTH",
        (bx, by),
        textcoords="offset points",
        xytext=(8, 6),
        fontsize=8,
        color="firebrick",
    )

    zone_w = ICFG.x_max - ICFG.x_min
    zone_h = ICFG.y_max - ICFG.y_min
    start_rect = Rectangle(
        (ICFG.x_min, ICFG.y_min),
        zone_w,
        zone_h,
        linewidth=1.5,
        edgecolor="green",
        facecolor="green",
        alpha=0.10,
        linestyle="--",
        zorder=1,
    )
    ax.add_patch(start_rect)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)


def _save_fig(fig, out_dir, filename):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / filename
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.resolve()}")


def _make_wave_axes(wave_levels, width=7.0, height=6.0):
    ncols = min(3, len(wave_levels))
    nrows = int(np.ceil(len(wave_levels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width * ncols, height * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for ax in flat_axes[len(wave_levels):]:
        ax.set_visible(False)
    return fig, flat_axes


def _controller_color(controller):
    controller = str(controller).upper()
    if controller == "PID":
        return PID_COLOR
    if controller == "MLP":
        return MLP_COLOR
    return "#6c757d"


def _load_runs(csv_path):
    df = pd.read_csv(csv_path)

    if "run_id" in df.columns:
        group_col = "run_id"
    elif "episode" in df.columns:
        group_col = "episode"
    else:
        raise ValueError(
            "Expected a 'run_id' or 'episode' column in the log. "
            "Use runtime logging or headless evaluation logs."
        )

    required_cols = [
        "controller",
        "wave_height",
        "x",
        "y",
        "psi",
        "t",
        "e_ct",
        "e_psi",
        "u",
        "tau_x",
        "finished",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "Comparison plotting requires the following log columns: "
            + ", ".join(missing)
            + ". Update runner logging and regenerate data/los_log.csv."
        )

    runs = []
    final_rows = []

    for rid in sorted(df[group_col].unique()):
        rd = df[df[group_col] == rid].reset_index(drop=True)
        if rd.empty:
            continue

        controller = str(rd["controller"].iloc[0]).upper()
        wave_height = float(rd["wave_height"].iloc[0])
        final_finished = bool(int(rd["finished"].iloc[-1]))

        if "dist_dock" in rd.columns:
            final_dist = float(rd["dist_dock"].iloc[-1])
        else:
            final_dist = float(
                np.hypot(
                    float(rd["x"].iloc[-1]) - DCFG.dock_x,
                    float(rd["y"].iloc[-1]) - DCFG.dock_y,
                )
            )

        runs.append(
            {
                "rid": rid,
                "controller": controller,
                "wave_height": wave_height,
                "df": rd,
                "color": _controller_color(controller),
                "final_finished": final_finished,
            }
        )

        final_rows.append(
            {
                "rid": rid,
                "controller": controller,
                "wave_height": wave_height,
                "finished_plot": int(final_finished),
                "dist_dock": final_dist,
                "x_end": float(rd["x"].iloc[-1]),
                "y_end": float(rd["y"].iloc[-1]),
            }
        )

    final_df = pd.DataFrame(final_rows).sort_values(
        ["wave_height", "controller", "rid"]
    ).reset_index(drop=True)

    if final_df.empty:
        raise ValueError("No runs found in the provided log.")

    wave_levels = sorted(final_df["wave_height"].unique())
    return runs, final_df, wave_levels


def _sample_runs(runs, max_runs):
    if len(runs) <= max_runs:
        return runs
    sample_idx = np.linspace(0, len(runs) - 1, max_runs, dtype=int)
    return [runs[i] for i in sample_idx]


def _final_subset(final_df, wave_height, controller=None):
    mask = np.isclose(final_df["wave_height"].to_numpy(dtype=float), wave_height)
    subset = final_df.loc[mask]
    if controller is not None:
        subset = subset.loc[subset["controller"] == controller]
    return subset


def _wave_summary_text(final_df, wave_height):
    parts = []
    for controller in ["PID", "MLP"]:
        subset = _final_subset(final_df, wave_height, controller)
        if subset.empty:
            continue
        docked = int(subset["finished_plot"].sum())
        parts.append(f"{controller} {docked}/{len(subset)} docked")
    return " | ".join(parts)


def _trajectory_legend_handles():
    return [
        Line2D([0], [0], color=PID_COLOR, lw=1.8, label="PID trajectories"),
        Line2D([0], [0], color=MLP_COLOR, lw=1.8, label="MLP trajectories"),
        Line2D([0], [0], marker="o", color="black", lw=0, label="Docked end"),
        Line2D([0], [0], marker="x", color="black", lw=0, label="Timeout end"),
    ]


def _series_legend_handles():
    return [
        Line2D([0], [0], color=PID_COLOR, lw=1.8, label="PID"),
        Line2D([0], [0], color=MLP_COLOR, lw=1.8, label="MLP"),
    ]


def _plot_success_rate(final_df, wave_levels, out_dir):
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)

    x = np.arange(len(wave_levels))
    width = 0.36

    for offset, controller in [(-width / 2, "PID"), (width / 2, "MLP")]:
        values = []
        counts = []
        for wave in wave_levels:
            subset = _final_subset(final_df, wave, controller)
            values.append(100.0 * float(subset["finished_plot"].mean()) if len(subset) else np.nan)
            counts.append(len(subset))

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=_controller_color(controller),
            alpha=0.88,
            label=controller,
        )

        for bar, value, count in zip(bars, values, counts):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.5,
                f"{value:.0f}%\n(n={count})",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{wave:.1f} m" for wave in wave_levels])
    ax.set_ylim(0, 110)
    ax.set_xlabel("Wave height")
    ax.set_ylabel("Docking success rate [%]")
    ax.set_title("PID vs MLP docking success rate by wave height")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save_fig(fig, out_dir, "success_rate_by_wave.png")


def _plot_final_distance_summary(final_df, wave_levels, out_dir):
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)

    x = np.arange(len(wave_levels))
    width = 0.36

    for offset, controller in [(-width / 2, "PID"), (width / 2, "MLP")]:
        values = []
        for wave in wave_levels:
            subset = _final_subset(final_df, wave, controller)
            values.append(float(subset["dist_dock"].median()) if len(subset) else np.nan)

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=_controller_color(controller),
            alpha=0.88,
            label=controller,
        )

        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.5,
                f"{value:.1f} m",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.axhline(PLOT_ACCEPTANCE_RADIUS_M, color="red", ls="--", lw=1.2, label="Acceptance radius")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{wave:.1f} m" for wave in wave_levels])
    ax.set_xlabel("Wave height")
    ax.set_ylabel("Median final distance to dock [m]")
    ax.set_title("PID vs MLP final dock distance by wave height")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save_fig(fig, out_dir, "final_distance_by_wave.png")


def _plot_trajectory_comparison(runs, final_df, wave_levels, out_dir, max_runs_per_controller):
    fig, axes = _make_wave_axes(wave_levels, width=7.2, height=6.8)

    for ax, wave in zip(axes, wave_levels):
        _decorate_traj_axis(ax)

        wave_runs = [run for run in runs if np.isclose(run["wave_height"], wave)]
        for controller in ["PID", "MLP"]:
            ctrl_runs = _sample_runs(
                [run for run in wave_runs if run["controller"] == controller],
                max_runs_per_controller,
            )
            color = _controller_color(controller)

            for run in ctrl_runs:
                rd = run["df"]
                ax.plot(rd["x"], rd["y"], color=color, lw=1.0, alpha=0.24)

                x_end = float(rd["x"].iloc[-1])
                y_end = float(rd["y"].iloc[-1])
                psi_end = float(rd["psi"].iloc[-1]) if "psi" in rd.columns else 0.0

                _draw_ship_patch(
                    ax,
                    x_end,
                    y_end,
                    psi_end,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.18 if run["final_finished"] else 0.08,
                    lw=1.0,
                    zorder=11,
                    scale=SHIP_PLOT_SCALE_TRAJ,
                )

                ax.plot(
                    x_end,
                    y_end,
                    marker="o" if run["final_finished"] else "x",
                    color=color,
                    ms=5,
                    alpha=0.95,
                    linestyle="None",
                    zorder=12,
                )

        ax.set_title(
            "PID vs MLP trajectories\n"
            f"Wave {wave:.1f} m | {_wave_summary_text(final_df, wave)}"
        )
        ax.legend(handles=_trajectory_legend_handles(), fontsize=8, loc="upper left")

    _save_fig(fig, out_dir, "trajectory_comparison.png")


def _plot_docking_zoom_comparison(runs, final_df, wave_levels, out_dir, max_runs_per_controller):
    fig, axes = _make_wave_axes(wave_levels, width=7.0, height=6.8)

    for ax, wave in zip(axes, wave_levels):
        _decorate_traj_axis(ax)
        ax.set_xlim(DCFG.dock_x - DOCKING_ZOOM_RADIUS_M, DCFG.dock_x + DOCKING_ZOOM_RADIUS_M)
        ax.set_ylim(DCFG.dock_y - DOCKING_ZOOM_RADIUS_M, DCFG.dock_y + DOCKING_ZOOM_RADIUS_M)

        wave_runs = [run for run in runs if np.isclose(run["wave_height"], wave)]
        for controller in ["PID", "MLP"]:
            ctrl_runs = _sample_runs(
                [run for run in wave_runs if run["controller"] == controller],
                max_runs_per_controller,
            )
            color = _controller_color(controller)

            for run in ctrl_runs:
                rd = run["df"]

                if "dist_dock" in rd.columns:
                    rd_plot = rd.loc[rd["dist_dock"] <= 60.0].copy()
                    if rd_plot.empty:
                        rd_plot = rd.tail(250).copy()
                else:
                    rd_plot = rd.tail(250).copy()

                ax.plot(rd_plot["x"], rd_plot["y"], color=color, lw=1.1, alpha=0.35)

                x_end = float(rd["x"].iloc[-1])
                y_end = float(rd["y"].iloc[-1])
                psi_end = float(rd["psi"].iloc[-1]) if "psi" in rd.columns else 0.0

                _draw_ship_patch(
                    ax,
                    x_end,
                    y_end,
                    psi_end,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.22 if run["final_finished"] else 0.10,
                    lw=1.0,
                    zorder=12,
                    scale=SHIP_PLOT_SCALE_ZOOM,
                )
                ax.plot(
                    x_end,
                    y_end,
                    marker="o" if run["final_finished"] else "x",
                    color=color,
                    ms=5,
                    alpha=0.95,
                    linestyle="None",
                    zorder=13,
                )

        ax.set_title(
            "PID vs MLP docking zoom\n"
            f"Wave {wave:.1f} m | {_wave_summary_text(final_df, wave)}"
        )
        ax.legend(handles=_trajectory_legend_handles(), fontsize=8, loc="upper left")

    _save_fig(fig, out_dir, "docking_zoom_comparison.png")


def _plot_series_comparison(
    runs,
    final_df,
    wave_levels,
    out_dir,
    value_getter,
    ylabel,
    title_prefix,
    filename,
    max_runs_per_controller,
    zero_line=False,
):
    fig, axes = _make_wave_axes(wave_levels, width=6.6, height=4.6)

    for ax, wave in zip(axes, wave_levels):
        wave_runs = [run for run in runs if np.isclose(run["wave_height"], wave)]

        for controller in ["PID", "MLP"]:
            ctrl_runs = _sample_runs(
                [run for run in wave_runs if run["controller"] == controller],
                max_runs_per_controller,
            )
            color = _controller_color(controller)

            for run in ctrl_runs:
                rd = run["df"]
                ax.plot(
                    rd["t"],
                    value_getter(rd),
                    color=color,
                    alpha=0.20,
                    lw=0.9,
                )

        if zero_line:
            ax.axhline(0.0, color="gray", ls=":", lw=0.7)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{title_prefix}\nWave {wave:.1f} m | {_wave_summary_text(final_df, wave)}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(handles=_series_legend_handles(), fontsize=8, loc="upper right")

    _save_fig(fig, out_dir, filename)


def plot_los_log(
    csv_path="data/los_log.csv",
    out_dir="data/plots/pid_vs_mlp",
    max_runs_per_controller=8,
):
    runs, final_df, wave_levels = _load_runs(csv_path)

    _plot_success_rate(final_df, wave_levels, out_dir)
    _plot_final_distance_summary(final_df, wave_levels, out_dir)
    _plot_trajectory_comparison(runs, final_df, wave_levels, out_dir, max_runs_per_controller)
    _plot_docking_zoom_comparison(runs, final_df, wave_levels, out_dir, max_runs_per_controller)

    _plot_series_comparison(
        runs,
        final_df,
        wave_levels,
        out_dir,
        value_getter=lambda rd: rd["e_ct"].to_numpy(dtype=float),
        ylabel="Cross-track error [m]",
        title_prefix="PID vs MLP cross-track error",
        filename="cross_track_comparison.png",
        max_runs_per_controller=max_runs_per_controller,
        zero_line=True,
    )

    _plot_series_comparison(
        runs,
        final_df,
        wave_levels,
        out_dir,
        value_getter=lambda rd: rd["e_psi"].to_numpy(dtype=float),
        ylabel="Heading error [deg]",
        title_prefix="PID vs MLP heading error",
        filename="heading_error_comparison.png",
        max_runs_per_controller=max_runs_per_controller,
        zero_line=True,
    )

    _plot_series_comparison(
        runs,
        final_df,
        wave_levels,
        out_dir,
        value_getter=lambda rd: rd["tau_x"].to_numpy(dtype=float) / 1e3,
        ylabel="Surge control force [kN]",
        title_prefix="PID vs MLP surge control effort",
        filename="tau_surge_comparison.png",
        max_runs_per_controller=max_runs_per_controller,
        zero_line=False,
    )

    waves_txt = ", ".join(f"{wave:.1f} m" for wave in wave_levels)
    print(f"Saved PID vs MLP comparison plots for wave heights: {waves_txt}")


if __name__ == "__main__":
    plot_los_log()