"""Step 0 audit: per-trip sampling statistics and data-quality checks.

Run from the repo root:
    python notebooks/step0_timing_audit.py

Uses ONLY the training partition (80 trips). Prints a summary table of
per-trip timing stats and lists data-quality problems found by the
validators in src/data/loading.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loading import (
    TRAIN_PARTITION,
    list_trips,
    load_trip,
    load_track,
    validate_track,
)


def track_stats(partition: str, trip_id: str) -> dict:
    """Compute per-trip sampling statistics from the .track file."""
    df = load_track(partition, trip_id)
    t = df["t"].to_numpy()
    dt = np.diff(t)
    return {
        "trip_id": trip_id,
        "n_obs": len(df),
        "duration_s": float(t[-1] - t[0]),
        "t_start": float(t[0]),
        "t_end": float(t[-1]),
        "dt_mean": float(dt.mean()) if dt.size else np.nan,
        "dt_std": float(dt.std()) if dt.size else np.nan,
        "frac_dt_ne_1s": float(np.mean(dt != 1.0)) if dt.size else np.nan,
        "max_gap_s": float(dt.max()) if dt.size else np.nan,
        "min_dt_s": float(dt.min()) if dt.size else np.nan,
    }


def main() -> None:
    trips = list_trips(TRAIN_PARTITION)
    print(f"partition={TRAIN_PARTITION!r}, n_trips={len(trips)}")

    # ---- 1. Timing audit over all training tracks -------------------------
    rows = [track_stats(TRAIN_PARTITION, tid) for tid in trips]
    stats = pd.DataFrame(rows).set_index("trip_id")

    print("\n=== Overall timing summary (training, 80 trips) ===")
    print(
        stats[["n_obs", "duration_s", "dt_mean", "dt_std",
               "frac_dt_ne_1s", "max_gap_s"]].describe().round(4).to_string()
    )

    # Pooled fraction of 1-second steps across ALL observations.
    total_steps = int((stats["n_obs"] - 1).sum())
    bad_steps = stats["frac_dt_ne_1s"] * (stats["n_obs"] - 1)
    pooled_bad = float(bad_steps.sum() / total_steps)
    print(f"\npooled share of steps != 1 s: {pooled_bad:.4%} "
          f"({int(round(bad_steps.sum()))}/{total_steps} steps)")

    # ---- 2. Outlier trips --------------------------------------------------
    print("\n=== Top outliers ===")
    print("-- largest share of non-1s steps --")
    print(stats.nlargest(8, "frac_dt_ne_1s")[
        ["n_obs", "duration_s", "frac_dt_ne_1s", "max_gap_s", "dt_mean"]
    ].round(4).to_string())
    print("\n-- largest single gap (s) --")
    print(stats.nlargest(8, "max_gap_s")[
        ["n_obs", "duration_s", "frac_dt_ne_1s", "max_gap_s", "dt_mean"]
    ].round(4).to_string())
    print("\n-- shortest / longest trips by duration --")
    print("shortest:")
    print(stats.nsmallest(5, "duration_s")[["n_obs", "duration_s"]].round(1).to_string())
    print("longest:")
    print(stats.nlargest(5, "duration_s")[["n_obs", "duration_s"]].round(1).to_string())
    print("\n-- trips whose mean dt deviates most from 1.0 s --")
    print(stats.reindex((stats["dt_mean"] - 1.0).abs().sort_values(ascending=False).index).head(8)[
        ["n_obs", "dt_mean", "dt_std", "frac_dt_ne_1s", "max_gap_s"]
    ].round(4).to_string())

    # ---- 3. Validation of tracks, networks and routes ----------------------
    print("\n=== Validation (all 80 training trips: track+nodes+arcs+route) ===")
    all_problems: list[str] = []
    shapes = []
    for tid in trips:
        bundle = load_trip(TRAIN_PARTITION, tid)
        all_problems.extend(bundle["problems"])
        shapes.append(
            {
                "trip_id": tid,
                "n_track": len(bundle["track"]),
                "n_nodes": len(bundle["nodes"]),
                "n_arcs": len(bundle["arcs"]),
                "n_route": len(bundle["route"]),
            }
        )
    shapes_df = pd.DataFrame(shapes).set_index("trip_id")
    print(shapes_df.describe().round(1).to_string())
    if all_problems:
        print(f"\n{len(all_problems)} problem(s) found:")
        for p in all_problems:
            print(" -", p)
    else:
        print("no problems found by validators")

    # ---- 4. Timestamp origin check ----------------------------------------
    print("\n=== Timestamp origin ===")
    t0 = stats["t_start"]
    print(f"t_start min={t0.min():.3f} max={t0.max():.3f} "
          f"(unique values: {t0.nunique()})")
    print("first 10 t_start values:", np.sort(t0.unique())[:10])

    # ---- 5. Save the stats table for later steps --------------------------
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    stats.round(6).to_csv(out / "train_track_stats.csv")
    shapes_df.to_csv(out / "train_network_shapes.csv")
    print(f"\nsaved {out / 'train_track_stats.csv'} and train_network_shapes.csv")


if __name__ == "__main__":
    main()
