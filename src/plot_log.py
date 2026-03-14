import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_los_log(csv_path="data/los_log.csv", out_path="data/los_log_plot.png"):
    df = pd.read_csv(csv_path)

    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Position XY
    axs[0].plot(df.t, df.x, label="x")
    axs[0].plot(df.t, df.y, label="y")
    axs[0].set_ylabel("Position [m]")
    axs[0].legend()
    axs[0].set_title("LOS-guided trajectory")

    # Heading
    psi_u = np.unwrap(df.psi)
    chi_u = np.unwrap(df.chi_los)
    psir_u = np.unwrap(df.psi_r)
    axs[1].plot(df.t, np.degrees(psi_u), label="ψ (actual)")
    axs[1].plot(df.t, np.degrees(chi_u), "--", label="χ_LOS")
    axs[1].plot(df.t, np.degrees(psir_u), ":", label="ψ_ref")
    axs[1].set_ylabel("Heading [deg]")
    axs[1].legend()

    # Cross-track error & speed
    ax3a = axs[2]
    ax3b = ax3a.twinx()
    ax3a.plot(df.t, df.e_ct, "r", label="e_ct")
    ax3a.set_ylabel("Cross-track [m]", color="r")
    ax3a.axhline(0, color="r", ls=":", lw=0.5)
    ax3b.plot(df.t, df.u, "b", label="u (actual)")
    ax3b.plot(df.t, df.u_r, "b--", label="u_ref")
    ax3b.set_ylabel("Surge speed [m/s]", color="b")
    lines1, labs1 = ax3a.get_legend_handles_labels()
    lines2, labs2 = ax3b.get_legend_handles_labels()
    ax3a.legend(lines1 + lines2, labs1 + labs2, loc="upper right")

    # Control forces
    axs[3].plot(df.t, df.tau_x / 1e3, label="τ_x")
    axs[3].plot(df.t, df.tau_y / 1e3, label="τ_y")
    axs[3].plot(df.t, df.tau_psi / 1e3, label="τ_ψ")
    axs[3].set_ylabel("Control [kN / kNm]")
    axs[3].set_xlabel("Time [s]")
    axs[3].legend()

    plt.tight_layout()
    out = Path(out_path)
    plt.savefig(out, dpi=150)
    print(f"Saved {out.resolve()}")


if __name__ == "__main__":
    plot_los_log()