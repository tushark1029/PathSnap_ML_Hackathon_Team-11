"""Interactive Folium map for a single trip.

Layers (toggleable via LayerControl):
  * GPS observations   - CircleMarkers, coloured by timestamp
  * road network       - clipped arcs as PolyLines (grey)
  * reference route    - PolyLine through the .route arc sequence (red)
  * predicted route    - OPTIONAL, only added when `predicted_route` given

Coordinates: data files store (lon, lat); Folium wants (lat, lon) -- every
call site below passes them explicitly in that order.

Usage:
    from src.visualization.folium_map import build_trip_map
    m = build_trip_map("00000057", partition="train")
    m.save("results/maps/trip_00000057.html")
"""

from __future__ import annotations

import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.loading import load_trip  # noqa: E402

M_PER_DEG_LAT = 111_320.0


def _clip_arcs(nodes: pd.DataFrame, arcs: pd.DataFrame, ref_lonlat, buffer_m: float):
    """Vectorised clip of arcs to the bounding box of ref_lonlat + buffer.

    ref_lonlat: (k, 2) array of (lon, lat). Returns (kept arc segments as
    (p, 2, 2) lon/lat pairs, kept arc positional ids).
    """
    nlon = nodes["lon"].to_numpy()
    nlat = nodes["lat"].to_numpy()
    buf_lat = buffer_m / M_PER_DEG_LAT
    # metres-per-degree-lon varies with latitude; use mean reference latitude.
    lat0 = float(np.mean(ref_lonlat[:, 1]))
    buf_lon = buffer_m / (M_PER_DEG_LAT * np.cos(np.radians(lat0)))

    lo_lon = ref_lonlat[:, 0].min() - buf_lon
    hi_lon = ref_lonlat[:, 0].max() + buf_lon
    lo_lat = ref_lonlat[:, 1].min() - buf_lat
    hi_lat = ref_lonlat[:, 1].max() + buf_lat

    inside = (nlon >= lo_lon) & (nlon <= hi_lon) & (nlat >= lo_lat) & (nlat <= hi_lat)
    a = arcs[["from_node", "to_node"]].to_numpy()
    keep = inside[a[:, 0]] | inside[a[:, 1]]
    ids = np.flatnonzero(keep)
    seg = np.stack(
        [
            np.column_stack([nlon[a[ids, 0]], nlat[a[ids, 0]]]),
            np.column_stack([nlon[a[ids, 1]], nlat[a[ids, 1]]]),
        ],
        axis=1,
    )
    return seg, ids


def _route_polyline(nodes: pd.DataFrame, arcs: pd.DataFrame, route_ids) -> list:
    """Return list of (lat, lon) tuples tracing the route arc sequence.

    The route is an ordered arc-id sequence; each arc contributes its
    from->to node pair, so flattening gives a continuous path.
    """
    a = arcs[["from_node", "to_node"]].to_numpy()
    rarc = a[np.asarray(route_ids)]
    nlon = nodes["lon"].to_numpy()
    nlat = nodes["lat"].to_numpy()
    lon = nlon[rarc.ravel()]
    lat = nlat[rarc.ravel()]
    return list(zip(lat.tolist(), lon.tolist()))


def _downsample_indices(n: int, max_points: int) -> np.ndarray:
    """Evenly spaced indices keeping <= max_points observations."""
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).round().astype(int)


def build_trip_map(
    trip_id: str | int,
    partition: str = "train",
    predicted_route: np.ndarray | list[int] | None = None,
    max_gps_points: int = 3000,
    max_network_arcs: int = 15000,
    network_buffer_m: float = 400.0,
    tiles: str = "OpenStreetMap.Mapnik",
) -> folium.Map:
    """Build a Folium map for one trip.

    Parameters
    ----------
    trip_id : str | int
        8-digit trip identifier.
    partition : str
        'train' or 'student_test' (evaluator data must never be used).
    predicted_route : array-like of arc ids, optional
        If provided, a 'predicted route' layer is added. Arc ids refer to the
        trip's own .arcs file (row index = arc id).
    max_gps_points : int
        GPS CircleMarkers are evenly downsampled to at most this many.
    max_network_arcs : int
        If the clipped network has more arcs, every k-th arc is drawn so the
        HTML stays loadable.
    network_buffer_m : float
        Clip the road network to track+route extent plus this buffer.
    tiles : str
        Folium tile layer name.
    """
    tid = f"{int(trip_id):08d}"
    bundle = load_trip(partition, tid)
    track, nodes, arcs, route = (
        bundle["track"],
        bundle["nodes"],
        bundle["arcs"],
        bundle["route"],
    )

    # --- extent: union of GPS points and reference-route nodes --------------
    a = arcs[["from_node", "to_node"]].to_numpy()
    route_node_ids = np.unique(a[np.asarray(route)].ravel())
    ref_lon = np.concatenate(
        [track["lon"].to_numpy(), nodes["lon"].to_numpy()[route_node_ids]]
    )
    ref_lat = np.concatenate(
        [track["lat"].to_numpy(), nodes["lat"].to_numpy()[route_node_ids]]
    )
    ref_lonlat = np.column_stack([ref_lon, ref_lat])
    center = [float(ref_lat.mean()), float(ref_lon.mean())]  # (lat, lon)

    m = folium.Map(location=center, zoom_start=13, tiles=tiles, control_scale=True)

    # --- Layer 1: road network arcs (clipped) -------------------------------
    net_group = folium.FeatureGroup(name="Road network (clipped)", show=True)
    seg, _ = _clip_arcs(nodes, arcs, ref_lonlat, network_buffer_m)
    if len(seg) > max_network_arcs:
        stride = int(np.ceil(len(seg) / max_network_arcs))
        seg = seg[::stride]
    for s in seg:  # s: (2 nodes, 2 coords) -> [[lat, lon], [lat, lon]]
        net_group.add_child(
            folium.PolyLine(
                locations=[[float(pt[1]), float(pt[0])] for pt in s],
                color="#888888",
                weight=1,
                opacity=0.6,
            )
        )
    m.add_child(net_group)

    # --- Layer 2: reference route -------------------------------------------
    ref_group = folium.FeatureGroup(name="Reference route", show=True)
    ref_group.add_child(
        folium.PolyLine(
            locations=_route_polyline(nodes, arcs, route),
            color="#d62728",
            weight=4,
            opacity=0.9,
        )
    )
    m.add_child(ref_group)

    # --- Layer 3: GPS observations (downsampled CircleMarkers) --------------
    gps_group = folium.FeatureGroup(name="GPS observations", show=True)
    idx = _downsample_indices(len(track), max_gps_points)
    t_min, t_max = float(track["t"].min()), float(track["t"].max())
    span = max(t_max - t_min, 1e-9)
    for i in idx:  # blue->red ramp by normalised time
        row = track.iloc[int(i)]
        frac = (float(row["t"]) - t_min) / span
        r, g, b = int(255 * frac), int(80 * (1 - frac)), int(255 * (1 - frac))
        gps_group.add_child(
            folium.CircleMarker(
                location=[float(row["lat"]), float(row["lon"])],  # (lat, lon)
                radius=3,
                color=f"#{r:02x}{g:02x}{b:02x}",
                weight=1,
                fill=True,
                fillOpacity=0.8,
                popup=f"obs {int(i)} | t={float(row['t']):.0f}s",
            )
        )
    m.add_child(gps_group)

    # --- Layer 4 (optional): predicted route --------------------------------
    if predicted_route is not None:
        pred_group = folium.FeatureGroup(name="Predicted route", show=True)
        pred_group.add_child(
            folium.PolyLine(
                locations=_route_polyline(nodes, arcs, np.asarray(predicted_route)),
                color="#1f77b4",
                weight=4,
                opacity=0.9,
                dash_array="8",
            )
        )
        m.add_child(pred_group)

    m.fit_bounds(
        [[float(ref_lat.min()), float(ref_lon.min())],
         [float(ref_lat.max()), float(ref_lon.max())]]
    )
    folium.LayerControl(collapsed=False).add_to(m)
    title = (
        f"PathSnap trip {tid} | partition: {partition} | "
        f"GPS points shown: {len(idx)}/{len(track)}"
    )
    m.get_root().html.add_child(folium.Element(f"<h3 style='margin:0'>{title}</h3>"))
    return m


def save_trip_map(
    out_path: str | Path,
    trip_id: str | int,
    partition: str = "train",
    **kwargs,
) -> Path:
    """Convenience wrapper: build and save to an HTML file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = build_trip_map(trip_id, partition=partition, **kwargs)
    m.save(str(out_path))
    return out_path
