import pandas as pd
import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two GPS coordinates in meters.
    """

    earth_radius = 6371000

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return earth_radius * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate compass bearing between two GPS points.
    """

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    delta_lon = np.radians(lon2 - lon1)

    y = np.sin(delta_lon) * np.cos(lat2)

    x = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1)
        * np.cos(lat2)
        * np.cos(delta_lon)
    )

    bearing = np.degrees(np.arctan2(y, x))

    return (bearing + 360) % 360


def process_gps_trajectory(df):
    """
    Validate and extract basic trajectory features.
    """

    required_columns = [
        "longitude",
        "latitude",
        "timestamp"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    df = df.copy()

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["longitude", "latitude", "timestamp"]
    )

    df = df.reset_index(drop=True)

    if len(df) < 2:
        raise ValueError(
            "Trajectory must contain at least two GPS observations."
        )

    if not df["longitude"].between(-180, 180).all():
        raise ValueError("Invalid longitude values detected.")

    if not df["latitude"].between(-90, 90).all():
        raise ValueError("Invalid latitude values detected.")

    df["time_diff_s"] = (
        df["timestamp"].diff().fillna(0)
    )

    df["dist_m"] = haversine_distance(
        df["latitude"].shift(1),
        df["longitude"].shift(1),
        df["latitude"],
        df["longitude"]
    ).fillna(0)

    df["speed_ms"] = np.where(
        df["time_diff_s"] > 0,
        df["dist_m"] / df["time_diff_s"],
        0
    )

    df["speed_kmh"] = df["speed_ms"] * 3.6

    df["heading_deg"] = calculate_bearing(
        df["latitude"].shift(1),
        df["longitude"].shift(1),
        df["latitude"],
        df["longitude"]
    )

    df["heading_deg"] = df["heading_deg"].fillna(0)

    df["flag_speed_anomaly"] = (
        df["speed_ms"] > 50
    )

    df["flag_gap_gt_5s"] = (
        df["time_diff_s"] > 5
    )

    df["flag_extreme_gap"] = (
        df["time_diff_s"] > 60
    )

    return df