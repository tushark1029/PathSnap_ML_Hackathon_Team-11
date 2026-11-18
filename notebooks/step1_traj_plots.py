"""Step 1: GPS trajectory plots for representative and irregular training trips.

Run from repo root:  python notebooks/step1_traj_plots.py

All figures use ONLY training-partition trips and are labelled with the
partition. Coordinates are plotted in a local equirectangular projection
(project_local_xy, metres) so aspect ratio is truthful; points are coloured
by timestamp. Figures saved to results/figures/ at dpi=200.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loading import TRAIN_PARTITION, load_track, project_local_xy

OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Representative trips chosen from results/train_track_stats.csv:
REPRESENTATIVE = {
    "00000018": "short (379 obs, 451 s)",
    "00000057": "median duration (~1669 s)",
    "00000092": "long, regular sampling (7892 obs, 7891 s)",
    "00000004": "irregular sampling (46% non-1 s steps)",
    "00000022": "irregular sampling (47% non-1 s steps)",
    "00000039": "irregular sampling (42% non-1 s steps)",
    "00000081": "irregular + gaps (max gap 536 s)",
    "00000090": "extreme: 153064 s (~42 h) gap",
    "00000078": "extreme: 52810 s gap",
    "00000098": "moderately irregular (23% non-1 s steps)",
}


def plot_trip(trip_id: str, label: str) -> Path:
    df = load_track(TRAIN_PARTITION, trip_id)
    x, y = project_local_xy(df["lon"].to_numpy(), df["lat"].to_numpy())
    t = df["t"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(x, y, c=t, s=4, cmap="viridis", linewidths=0)
    ax.plot(x, y, color="0.6", lw=0.4, zorder=0)  # faint connecting line
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("easting (m, local equirectangular)")
    ax.set_ylabel("northing (m, local equirectangular)")
    ax.set_title(
        f"Trip {trip_id} — GPS trajectory ({label})\n"
        f"partition: training (80 trips) | {len(df)} observations"
    )
    cb = fig.colorbar(sc, ax=ax, label="timestamp t (s)")
    cb.ax.yaxis.label.set_size(9)

    # Mark large sampling gaps (>10 s) with red markers so they are visible.
    dt = np.diff(t)
    gap_idx = np.where(dt > 10)[0]
    if gap_idx.size:
        ax.scatter(
            x[gap_idx + 1],
            y[gap_idx + 1],
            facecolors="none",
            edgecolors="red",
            s=60,
            linewidths=1.2,
            zorder=3,
            label=f"sampling gap >10 s ({gap_idx.size})",
        )
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = OUT / f"traj_{trip_id}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    for tid, label in REPRESENTATIVE.items():
        path = plot_trip(tid, label)
        print("saved", path)


if __name__ == "__main__":
    main()
