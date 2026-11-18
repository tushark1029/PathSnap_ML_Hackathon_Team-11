"""Data loaders for the PathSnap GPS trajectory / road-network dataset.

Layout on disk (relative to DATA_ROOT)::

    <partition>/<trip_id>/<trip_id>.track   lon, lat, timestamp  (whitespace-separated, no header)
    <partition>/<trip_id>/<trip_id>.nodes   lon, lat             (row index = node id)
    <partition>/<trip_id>/<trip_id>.arcs    from_node, to_node   (row index = arc id)
    <partition>/<trip_id>/<trip_id>.route   arc id sequence      (reference route)

IMPORTANT: each trip folder carries its OWN .nodes/.arcs file -- these are
per-trip (clipped) road networks, not one shared global network. The
"evaluator test" partition described in the brief is not present in this
workspace; only `train` (80 trips) and `student_test` (10 trips) exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Repo root is two levels up from this file (src/data/loading.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "Student folder" / "Student folder"

TRAIN_PARTITION = "train"
STUDENT_TEST_PARTITION = "student_test"
PARTITIONS = (TRAIN_PARTITION, STUDENT_TEST_PARTITION)

# Valid WGS84 ranges. NOTE: trips in this dataset are worldwide (not one
# region), so we only check that coordinates are finite and in-range --
# a regional bounding box would produce false positives.
LON_RANGE = (-180.0, 180.0)
LAT_RANGE = (-90.0, 90.0)


def trip_dir(partition: str, trip_id: str | int) -> Path:
    """Return the folder holding all four files for a trip."""
    trip_id = f"{int(trip_id):08d}"
    return DATA_ROOT / partition / trip_id


def list_trips(partition: str = TRAIN_PARTITION) -> list[str]:
    """Sorted list of trip ids available in a partition."""
    base = DATA_ROOT / partition
    if not base.is_dir():
        raise FileNotFoundError(f"partition folder not found: {base}")
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def load_track(partition: str, trip_id: str | int) -> pd.DataFrame:
    """Load a .track file -> DataFrame[lon, lat, t]."""
    path = trip_dir(partition, trip_id) / f"{int(trip_id):08d}.track"
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "t"],
        engine="python",
    )
    return df


def load_nodes(partition: str, trip_id: str | int) -> pd.DataFrame:
    """Load a .nodes file -> DataFrame[lon, lat]; index = node id."""
    path = trip_dir(partition, trip_id) / f"{int(trip_id):08d}.nodes"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["lon", "lat"], engine="python")
    df.index.name = "node_id"
    return df


def load_arcs(partition: str, trip_id: str | int) -> pd.DataFrame:
    """Load a .arcs file -> DataFrame[from_node, to_node]; index = arc id."""
    path = trip_dir(partition, trip_id) / f"{int(trip_id):08d}.arcs"
    df = pd.read_csv(
        path, sep=r"\s+", header=None, names=["from_node", "to_node"], engine="python"
    )
    df.index.name = "arc_id"
    return df


def load_route(partition: str, trip_id: str | int) -> np.ndarray:
    """Load a .route file -> 1-D int array of arc ids in chronological order."""
    path = trip_dir(partition, trip_id) / f"{int(trip_id):08d}.route"
    return np.loadtxt(path, dtype=np.int64)


def validate_track(df: pd.DataFrame, trip_id: str | int) -> list[str]:
    """Return a list of data-quality problems found in a track DataFrame."""
    problems: list[str] = []
    tid = f"{int(trip_id):08d}"
    if len(df) == 0:
        return [f"{tid}: empty track"]
    if df.isna().any().any():
        problems.append(f"{tid}: NaN values present ({int(df.isna().sum().sum())} cells)")
    if not df["t"].is_monotonic_increasing:
        problems.append(f"{tid}: timestamps not monotonically increasing")
    if (df["t"].diff().dropna() <= 0).any():
        problems.append(f"{tid}: non-positive time deltas present")
    if not np.isfinite(df[["lon", "lat", "t"]].to_numpy()).all():
        problems.append(f"{tid}: non-finite coordinate/timestamp values")
    lon_ok = df["lon"].between(*LON_RANGE).all()
    lat_ok = df["lat"].between(*LAT_RANGE).all()
    if not (lon_ok and lat_ok):
        problems.append(
            f"{tid}: coordinates outside sanity box lon{LON_RANGE} lat{LAT_RANGE} "
            f"(lon [{df['lon'].min():.4f},{df['lon'].max():.4f}], "
            f"lat [{df['lat'].min():.4f},{df['lat'].max():.4f}])"
        )
    return problems


def validate_network(
    nodes: pd.DataFrame, arcs: pd.DataFrame, trip_id: str | int
) -> list[str]:
    """Sanity-check a per-trip nodes/arcs pair."""
    problems: list[str] = []
    tid = f"{int(trip_id):08d}"
    n, a = len(nodes), len(arcs)
    if a == 0 or n == 0:
        return [f"{tid}: empty network (nodes={n}, arcs={a})"]
    bad_from = (arcs["from_node"] < 0) | (arcs["from_node"] >= n)
    bad_to = (arcs["to_node"] < 0) | (arcs["to_node"] >= n)
    if bad_from.any() or bad_to.any():
        problems.append(
            f"{tid}: arc endpoint out of node range "
            f"(from: {int(bad_from.sum())}, to: {int(bad_to.sum())})"
        )
    lon_ok = nodes["lon"].between(*LON_RANGE).all()
    lat_ok = nodes["lat"].between(*LAT_RANGE).all()
    if not (lon_ok and lat_ok):
        problems.append(f"{tid}: node coordinates outside sanity box")
    return problems


def validate_route(route: np.ndarray, n_arcs: int, trip_id: str | int) -> list[str]:
    """Check the reference route arc ids are in range."""
    problems: list[str] = []
    tid = f"{int(trip_id):08d}"
    if route.size == 0:
        return [f"{tid}: empty route"]
    if route.min() < 0 or route.max() >= n_arcs:
        problems.append(
            f"{tid}: route arc id out of range [0,{n_arcs}) "
            f"(min={route.min()}, max={route.max()})"
        )
    return problems


def load_trip(partition: str, trip_id: str | int) -> dict:
    """Load all four files for a trip and run validation.

    Returns a dict with keys 'track', 'nodes', 'arcs', 'route', 'problems'.
    """
    track = load_track(partition, trip_id)
    nodes = load_nodes(partition, trip_id)
    arcs = load_arcs(partition, trip_id)
    route = load_route(partition, trip_id)
    problems = (
        validate_track(track, trip_id)
        + validate_network(nodes, arcs, trip_id)
        + validate_route(route, len(arcs), trip_id)
    )
    return {
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
        "problems": problems,
    }


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in metres between two (lon, lat) point sets.

    Vectorised numpy implementation; used instead of Euclidean degrees
    because lon/lat are angular coordinates.
    """
    lon1, lat1, lon2, lat2 = map(np.asarray, (lon1, lat1, lon2, lat2))
    r = 6_371_000.0  # Earth mean radius, metres
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Initial bearing in degrees [0, 360) from point 1 to point 2."""
    lon1, lat1, lon2, lat2 = map(np.asarray, (lon1, lat1, lon2, lat2))
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlmb = np.radians(lon2 - lon1)
    y = np.sin(dlmb) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlmb)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def angular_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed smallest angular difference a - b wrapped to [-180, 180)."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def speed_kmh(dist_m: np.ndarray, dt_s: np.ndarray) -> np.ndarray:
    """Speed in km/h from distance (m) and elapsed time (s)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return dist_m / dt_s * 3.6


def project_local_xy(lon, lat, lon0=None, lat0=None):
    """Equirectangular projection to local metres around (lon0, lat0).

    Adequate for trip-scale extents (a few km); haversine is used for
    distances. lon0/lat0 default to the mean of the inputs.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon0 is None:
        lon0 = lon.mean()
    if lat0 is None:
        lat0 = lat.mean()
    x = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * 6_371_000.0
    y = np.radians(lat - lat0) * 6_371_000.0
    return x, y


def load_trip_track_only(partition: str, trip_id: str | int) -> pd.DataFrame:
    """Convenience: load only the .track file."""
    return load_track(partition, trip_id)


__all__ = [
    "DATA_ROOT",
    "REPO_ROOT",
    "TRAIN_PARTITION",
    "STUDENT_TEST_PARTITION",
    "PARTITIONS",
    "trip_dir",
    "list_trips",
    "load_track",
    "load_nodes",
    "load_arcs",
    "load_route",
    "load_trip",
    "load_trip_track_only",
    "validate_track",
    "validate_network",
    "validate_route",
    "haversine_m",
    "bearing_deg",
    "angular_diff_deg",
    "speed_kmh",
    "project_local_xy",
]
