"""Step 2: road-network visualisation for chosen training trips.

Run from repo root:  python notebooks/step2_network_plots.py

For each trip: clip the per-trip .nodes/.arcs to the GPS bounding box plus a
buffer (vectorised boolean masks -- no Python loop over arcs), draw arcs with
matplotlib.collections.LineCollection in a local metric projection, overlay
the reference route (from .route arc ids) and the GPS track. Also reports how
closely GPS points sit to the network (distance to nearest node, via cKDTree)
as a first mismatch check. Figures -> results/figures/ at dpi=200.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loading import (
    TRAIN_PARTITION,
    load_trip,
    project_local_xy,
)

OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TRIPS = ["00000057", "00000018", "00000004"]
BUFFER_M = 400.0  # clip buffer around the track+route extent


def clip_network(nodes_xy, arcs, mask_xy, buffer_m):
    """Vectorised clip: keep arcs with >=1 endpoint inside the buffered box.

    nodes_xy: (n,2) projected coords; arcs: (m,2) int endpoint indices;
    mask_xy: (k,2) reference points defining the box; buffer_m: metres.
    Returns (arc_segments (p,2,2), kept_arc_ids).
    """
    lo = mask_xy.min(axis=0) - buffer_m
    hi = mask_xy.max(axis=0) + buffer_m
    inside = (
        (nodes_xy[:, 0] >= lo[0])
        & (nodes_xy[:, 0] <= hi[0])
        & (nodes_xy[:, 1] >= lo[1])
        & (nodes_xy[:, 1] <= hi[1])
    )
    keep = inside[arcs[:, 0]] | inside[arcs[:, 1]]
    seg_ids = np.flatnonzero(keep)
    seg = nodes_xy[arcs[seg_ids]]  # (p, 2, 2)
    return seg, seg_ids


def route_segments(nodes_xy, arcs, route_ids):
    """Polyline segments for the reference route arc sequence."""
    rarc = arcs[route_ids]  # (r,2) node ids
    return nodes_xy[rarc]  # (r,2,2)


def plot_trip(trip_id: str) -> Path:
    bundle = load_trip(TRAIN_PARTITION, trip_id)
    nodes, arcs_df, route, track = (
        bundle["nodes"],
        bundle["arcs"],
        bundle["route"],
        bundle["track"],
    )

    # Project everything into a shared local metric frame (origin = track mean).
    lon0, lat0 = track["lon"].mean(), track["lat"].mean()
    nx, ny = project_local_xy(nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), lon0, lat0)
    nodes_xy = np.column_stack([nx, ny])
    gx, gy = project_local_xy(track["lon"].to_numpy(), track["lat"].to_numpy(), lon0, lat0)
    gps_xy = np.column_stack([gx, gy])

    arcs = arcs_df[["from_node", "to_node"]].to_numpy()
    # Reference extent = union of GPS points and route nodes.
    route_nodes = np.unique(arcs[route].ravel())
    ref_xy = np.vstack([gps_xy, nodes_xy[route_nodes]])

    seg, seg_ids = clip_network(nodes_xy, arcs, ref_xy, BUFFER_M)
    rseg = route_segments(nodes_xy, arcs, route)

    # --- mismatch check: GPS distance to nearest network node --------------
    tree = cKDTree(nodes_xy)
    d_node, _ = tree.query(gps_xy, k=1)
    # distance to nearest *route* node (route may be sparse; nodes of route arcs)
    rtree = cKDTree(nodes_xy[route_nodes])
    d_route_node, _ = rtree.query(gps_xy, k=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.add_collection(
        LineCollection(seg, colors="0.75", linewidths=0.5, zorder=1)
    )
    ax.add_collection(
        LineCollection(rseg, colors="#d62728", linewidths=1.4, zorder=3)
    )
    ax.scatter(gps_xy[:, 0], gps_xy[:, 1], c=track["t"], s=5, cmap="viridis", zorder=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    ax.set_title(
        f"Trip {trip_id} — clipped road network + reference route + GPS\n"
        f"partition: training (80 trips) | buffer {BUFFER_M:.0f} m | "
        f"{len(seg)} arcs drawn of {len(arcs)}, route {len(route)} arcs"
    )
    ax.plot(
        [], [], color="0.75", lw=1, label=f"road arcs ({len(seg)})"
    )
    ax.plot([], [], color="#d62728", lw=1.5, label="reference route")
    ax.scatter([], [], c="k", s=5, label="GPS observations")
    ax.legend(loc="best", fontsize=8)

    stats = (
        f"GPS->nearest node: median {np.median(d_node):.1f} m, "
        f"p95 {np.percentile(d_node, 95):.1f} m, max {d_node.max():.1f} m\n"
        f"GPS->nearest route-node: median {np.median(d_route_node):.1f} m, "
        f"p95 {np.percentile(d_route_node, 95):.1f} m"
    )
    ax.text(
        0.01,
        0.01,
        stats,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )
    fig.tight_layout()
    out = OUT / f"network_{trip_id}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print(
        f"{trip_id}: arcs drawn {len(seg)}/{len(arcs)}, "
        f"GPS->node median {np.median(d_node):.1f} m p95 {np.percentile(d_node, 95):.1f} m, "
        f"GPS->route-node median {np.median(d_route_node):.1f} m"
    )
    return out


def main() -> None:
    for tid in TRIPS:
        print("saved", plot_trip(tid))


if __name__ == "__main__":
    main()
