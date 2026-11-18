"""Step 3: generate interactive Folium maps for representative training trips.

Run from repo root:  python notebooks/step3_folium_maps.py

Trips chosen to cover: median duration (00000057), long trip exercising GPS
downsampling (00000092, 7892 obs), and an irregular-sampling trip (00000004).
No predicted_route is supplied yet -- the layer is only added when provided.
Maps are saved to results/maps/. Open the .html files in any browser
(internet needed for the OSM base tiles).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualization.folium_map import save_trip_map  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "maps"

TRIPS = [
    ("00000057", "median-duration trip (~1669 s, regular 1 Hz)"),
    ("00000092", "long trip (7892 obs) - exercises GPS downsampling"),
    ("00000004", "irregular sampling (46% non-1 s steps)"),
]


def main() -> None:
    for tid, desc in TRIPS:
        path = save_trip_map(
            OUT / f"trip_{tid}.html",
            tid,
            partition="train",
            max_gps_points=3000,
        )
        print(f"saved {path}  ({desc})  [{path.stat().st_size/1e6:.1f} MB]")


if __name__ == "__main__":
    main()
