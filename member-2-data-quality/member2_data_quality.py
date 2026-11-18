
from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium

warnings.filterwarnings("ignore")

# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "Student folder"

TRAIN_DIR = DATA_ROOT / "train"
STUDENT_TEST_DIR = DATA_ROOT / "student_test"

RESULTS_DIR = PROJECT_ROOT / "results"
GRAPHS_DIR = PROJECT_ROOT / "graphs"
MAPS_DIR = PROJECT_ROOT / "maps"

for directory in [RESULTS_DIR, GRAPHS_DIR, MAPS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================

# A time interval greater than this is considered a large gap.
LARGE_GAP_SECONDS = 5

# Speed above this is flagged as suspicious.
# This is a detection threshold, NOT a deletion rule.
SUSPICIOUS_SPEED_MPS = 50

# ============================================================
# DISTANCE CALCULATION: HAVERSINE
# ============================================================

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate geographical distance between two GPS points in meters.
    """

    earth_radius = 6371000

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return earth_radius * c


# ============================================================
# DATA LOADING
# ============================================================

def load_track(track_path):
    """
    Load a .track file.

    Expected format:
    longitude latitude timestamp

    No header is expected.
    """

    columns = ["longitude", "latitude", "timestamp"]

    try:
        df = pd.read_csv(
            track_path,
            sep=r"\s+",
            header=None,
            names=columns,
            engine="python"
        )

        return df

    except Exception as error:
        print(f"ERROR loading {track_path}: {error}")
        return None


# ============================================================
# DATASET INVENTORY
# ============================================================

def inspect_partition(partition_dir, partition_name):
    """
    Inspect trips in a dataset partition.

    This overlaps with Member 1's dataset inventory work.
    """

    inventory = []

    if not partition_dir.exists():
        print(f"Partition does not exist: {partition_dir}")
        return inventory

    trip_dirs = sorted(
        [p for p in partition_dir.iterdir() if p.is_dir()]
    )

    print(f"\n{'=' * 60}")
    print(f"PARTITION: {partition_name}")
    print(f"TRIPS FOUND: {len(trip_dirs)}")
    print(f"{'=' * 60}")

    for trip_dir in trip_dirs:

        trip_id = trip_dir.name

        track_file = trip_dir / f"{trip_id}.track"
        nodes_file = trip_dir / f"{trip_id}.nodes"
        arcs_file = trip_dir / f"{trip_id}.arcs"
        route_file = trip_dir / f"{trip_id}.route"

        track = load_track(track_file) if track_file.exists() else None

        inventory.append({
            "partition": partition_name,
            "trip_id": trip_id,
            "track_exists": track_file.exists(),
            "nodes_exists": nodes_file.exists(),
            "arcs_exists": arcs_file.exists(),
            "route_exists": route_file.exists(),
            "track_size_bytes": (
                track_file.stat().st_size
                if track_file.exists()
                else 0
            ),
            "nodes_size_bytes": (
                nodes_file.stat().st_size
                if nodes_file.exists()
                else 0
            ),
            "arcs_size_bytes": (
                arcs_file.stat().st_size
                if arcs_file.exists()
                else 0
            ),
            "route_size_bytes": (
                route_file.stat().st_size
                if route_file.exists()
                else 0
            ),
            "gps_observations": (
                len(track)
                if track is not None
                else 0
            )
        })

    return inventory


# ============================================================
# DATA QUALITY ANALYSIS
# ============================================================

def analyze_trip(track_path, partition_name):

    trip_id = track_path.parent.name

    df = load_track(track_path)

    if df is None or len(df) == 0:

        return None, None

    original_count = len(df)

    # --------------------------------------------------------
    # NUMERIC VALIDATION
    # --------------------------------------------------------

    for column in ["longitude", "latitude", "timestamp"]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    missing_longitude = df["longitude"].isna().sum()
    missing_latitude = df["latitude"].isna().sum()
    missing_timestamp = df["timestamp"].isna().sum()

    missing_any = df[
        ["longitude", "latitude", "timestamp"]
    ].isna().any(axis=1)

    # --------------------------------------------------------
    # COORDINATE VALIDATION
    # --------------------------------------------------------

    invalid_latitude = (
        (df["latitude"] < -90)
        | (df["latitude"] > 90)
    )

    invalid_longitude = (
        (df["longitude"] < -180)
        | (df["longitude"] > 180)
    )

    invalid_coordinates = (
        invalid_latitude
        | invalid_longitude
        | missing_longitude
        | missing_latitude
    )

    # --------------------------------------------------------
    # CHRONOLOGICAL ANALYSIS
    # --------------------------------------------------------

    df["time_difference_seconds"] = (
        df["timestamp"].diff()
    )

    negative_time = (
        df["time_difference_seconds"] < 0
    )

    zero_time = (
        df["time_difference_seconds"] == 0
    )

    large_gap = (
        df["time_difference_seconds"]
        > LARGE_GAP_SECONDS
    )

    # --------------------------------------------------------
    # DISTANCE BETWEEN GPS OBSERVATIONS
    # --------------------------------------------------------

    df["distance_meters"] = haversine_distance(
        df["longitude"].shift(1),
        df["latitude"].shift(1),
        df["longitude"],
        df["latitude"]
    )

    # First observation has no previous point.
    df.loc[df.index[0], "distance_meters"] = np.nan

    # --------------------------------------------------------
    # SPEED CALCULATION
    # --------------------------------------------------------

    df["speed_mps"] = (
        df["distance_meters"]
        / df["time_difference_seconds"]
    )

    df["speed_kmh"] = df["speed_mps"] * 3.6

    invalid_speed = (
        (df["speed_mps"] < 0)
        | (df["time_difference_seconds"] <= 0)
    )

    suspicious_speed = (
        df["speed_mps"]
        > SUSPICIOUS_SPEED_MPS
    )

    # --------------------------------------------------------
    # GPS JUMP DETECTION
    # --------------------------------------------------------

    suspicious_jump = (
        suspicious_speed
        | large_gap
        | negative_time
    )

    # --------------------------------------------------------
    # FLAG OBSERVATIONS
    # --------------------------------------------------------

    df["missing_data"] = missing_any

    df["invalid_coordinates"] = (
        invalid_coordinates
    )

    df["negative_time"] = negative_time

    df["zero_time_interval"] = zero_time

    df["large_time_gap"] = large_gap

    df["invalid_speed"] = invalid_speed

    df["suspicious_speed"] = suspicious_speed

    df["suspicious_movement"] = suspicious_jump

    df["trip_id"] = trip_id

    df["partition"] = partition_name

    # --------------------------------------------------------
    # TRIP SUMMARY
    # --------------------------------------------------------

    valid_time = df[
        df["time_difference_seconds"] > 0
    ]

    valid_speed = df[
        df["speed_mps"].notna()
        & (df["speed_mps"] >= 0)
    ]

    summary = {

        "partition": partition_name,

        "trip_id": trip_id,

        "total_observations": original_count,

        "missing_longitude": missing_longitude,

        "missing_latitude": missing_latitude,

        "missing_timestamp": missing_timestamp,

        "observations_with_missing_data": (
            missing_any.sum()
        ),

        "invalid_coordinates": (
            invalid_coordinates.sum()
        ),

        "negative_time_intervals": (
            negative_time.sum()
        ),

        "zero_time_intervals": (
            zero_time.sum()
        ),

        "large_time_gaps": (
            large_gap.sum()
        ),

        "mean_time_interval_seconds": (
            df["time_difference_seconds"].mean()
        ),

        "median_time_interval_seconds": (
            df["time_difference_seconds"].median()
        ),

        "maximum_time_interval_seconds": (
            df["time_difference_seconds"].max()
        ),

        "mean_distance_meters": (
            df["distance_meters"].mean()
        ),

        "median_distance_meters": (
            df["distance_meters"].median()
        ),

        "maximum_distance_meters": (
            df["distance_meters"].max()
        ),

        "mean_speed_mps": (
            valid_speed["speed_mps"].mean()
        ),

        "median_speed_mps": (
            valid_speed["speed_mps"].median()
        ),

        "maximum_speed_mps": (
            valid_speed["speed_mps"].max()
        ),

        "maximum_speed_kmh": (
            valid_speed["speed_kmh"].max()
        ),

        "suspicious_speed_observations": (
            suspicious_speed.sum()
        ),

        "suspicious_movement_observations": (
            suspicious_jump.sum()
        ),

        "latitude_min": (
            df["latitude"].min()
        ),

        "latitude_max": (
            df["latitude"].max()
        ),

        "longitude_min": (
            df["longitude"].min()
        ),

        "longitude_max": (
            df["longitude"].max()
        )
    }

    return summary, df


# ============================================================
# PROCESS ALL TRIPS
# ============================================================

def process_partition(partition_dir, partition_name):

    summaries = []

    detailed_data = []

    trip_dirs = sorted(
        [p for p in partition_dir.iterdir() if p.is_dir()]
    )

    print(f"\nProcessing {partition_name}")
    print(f"Number of trips: {len(trip_dirs)}")

    for index, trip_dir in enumerate(trip_dirs, start=1):

        trip_id = trip_dir.name

        track_path = (
            trip_dir / f"{trip_id}.track"
        )

        if not track_path.exists():

            print(
                f"[WARNING] Missing track file: {trip_id}"
            )

            continue

        summary, detailed_df = analyze_trip(
            track_path,
            partition_name
        )

        if summary is not None:

            summaries.append(summary)

            detailed_data.append(detailed_df)

        print(
            f"Processed {index}/{len(trip_dirs)}: "
            f"{trip_id}"
        )

    return summaries, detailed_data


# ============================================================
# GRAPHS
# ============================================================

def generate_graphs(summary_df, detailed_df):

    # --------------------------------------------------------
    # 1. TIMESTAMP INTERVAL DISTRIBUTION
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    intervals = detailed_df[
        detailed_df["time_difference_seconds"].notna()
    ]["time_difference_seconds"]

    intervals = intervals[
        (intervals >= 0)
        & (intervals <= 60)
    ]

    plt.hist(
        intervals,
        bins=50
    )

    plt.xlabel("Time Interval (seconds)")
    plt.ylabel("Frequency")
    plt.title("GPS Timestamp Interval Distribution")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        GRAPHS_DIR / "timestamp_interval_distribution.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # 2. SPEED DISTRIBUTION
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    speeds = detailed_df[
        "speed_kmh"
    ].dropna()

    speeds = speeds[
        (speeds >= 0)
        & (speeds <= 250)
    ]

    plt.hist(
        speeds,
        bins=50
    )

    plt.xlabel("Speed (km/h)")
    plt.ylabel("Frequency")
    plt.title("GPS Speed Distribution")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        GRAPHS_DIR / "speed_distribution.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # 3. DISTANCE BETWEEN OBSERVATIONS
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    distances = detailed_df[
        "distance_meters"
    ].dropna()

    distances = distances[
        distances <= 1000
    ]

    plt.hist(
        distances,
        bins=50
    )

    plt.xlabel("Distance Between GPS Points (meters)")
    plt.ylabel("Frequency")
    plt.title("GPS Movement Distance Distribution")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        GRAPHS_DIR / "gps_distance_distribution.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # 4. LARGE TIME GAPS PER TRIP
    # --------------------------------------------------------

    plt.figure(figsize=(14, 6))

    plt.bar(
        summary_df["trip_id"],
        summary_df["large_time_gaps"]
    )

    plt.xlabel("Trip ID")
    plt.ylabel("Number of Large Time Gaps")
    plt.title(
        f"Large Timestamp Gaps per Trip "
        f"(>{LARGE_GAP_SECONDS} seconds)"
    )

    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.savefig(
        GRAPHS_DIR / "large_time_gaps_per_trip.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # 5. SUSPICIOUS MOVEMENTS PER TRIP
    # --------------------------------------------------------

    plt.figure(figsize=(14, 6))

    plt.bar(
        summary_df["trip_id"],
        summary_df[
            "suspicious_movement_observations"
        ]
    )

    plt.xlabel("Trip ID")
    plt.ylabel("Suspicious Observations")
    plt.title("Suspicious GPS Movements per Trip")

    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.savefig(
        GRAPHS_DIR / "suspicious_movements_per_trip.png",
        dpi=300
    )

    plt.close()

    print("\nGraphs generated successfully.")


# ============================================================
# FOLIUM MAP
# ============================================================

def generate_map(df, trip_id):

    valid = df[
        df["longitude"].notna()
        & df["latitude"].notna()
        & (df["latitude"].between(-90, 90))
        & (df["longitude"].between(-180, 180))
    ].copy()

    if len(valid) == 0:

        print("No valid GPS points for map.")

        return

    center_lat = valid["latitude"].mean()

    center_lon = valid["longitude"].mean()

    gps_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14
    )

    # Normal GPS observations.
    for _, row in valid.iterrows():

        if row["suspicious_movement"]:

            folium.CircleMarker(
                location=[
                    row["latitude"],
                    row["longitude"]
                ],
                radius=5,
                color="red",
                fill=True,
                fill_color="red",
                popup=(
                    f"Trip: {trip_id}<br>"
                    f"Timestamp: {row['timestamp']}<br>"
                    f"Speed: {row['speed_kmh']:.2f} km/h<br>"
                    f"Distance: {row['distance_meters']:.2f} m"
                )
            ).add_to(gps_map)

        else:

            folium.CircleMarker(
                location=[
                    row["latitude"],
                    row["longitude"]
                ],
                radius=2,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.5
            ).add_to(gps_map)

    # Draw the trajectory.
    coordinates = list(
        zip(
            valid["latitude"],
            valid["longitude"]
        )
    )

    folium.PolyLine(
        coordinates,
        color="blue",
        weight=3,
        opacity=0.7,
        tooltip="GPS Trajectory"
    ).add_to(gps_map)

    folium.LayerControl().add_to(gps_map)

    output_path = (
        MAPS_DIR
        / f"trip_{trip_id}_noise_analysis.html"
    )

    gps_map.save(output_path)

    print(f"Map saved: {output_path}")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("=" * 70)

    print("PATHSNAP - MEMBER 2 DATA QUALITY ANALYSIS")

    print("=" * 70)

    # --------------------------------------------------------
    # DATASET INVENTORY
    # --------------------------------------------------------

    train_inventory = inspect_partition(
        TRAIN_DIR,
        "train"
    )

    test_inventory = inspect_partition(
        STUDENT_TEST_DIR,
        "student_test"
    )

    inventory_df = pd.DataFrame(
        train_inventory + test_inventory
    )

    inventory_df.to_csv(
        RESULTS_DIR / "dataset_inventory.csv",
        index=False
    )

    # --------------------------------------------------------
    # TRAINING DATA ANALYSIS
    # --------------------------------------------------------

    train_summaries, train_details = process_partition(
        TRAIN_DIR,
        "train"
    )

    train_summary_df = pd.DataFrame(
        train_summaries
    )

    train_detail_df = pd.concat(
        train_details,
        ignore_index=True
    )

    train_summary_df.to_csv(
        RESULTS_DIR / "train_data_quality_summary.csv",
        index=False
    )

    train_detail_df.to_csv(
        RESULTS_DIR / "train_gps_detailed_analysis.csv",
        index=False
    )

    # --------------------------------------------------------
    # STUDENT TEST DATA ANALYSIS
    # --------------------------------------------------------

    test_summaries, test_details = process_partition(
        STUDENT_TEST_DIR,
        "student_test"
    )

    test_summary_df = pd.DataFrame(
        test_summaries
    )

    test_detail_df = pd.concat(
        test_details,
        ignore_index=True
    )

    test_summary_df.to_csv(
        RESULTS_DIR / "student_test_data_quality_summary.csv",
        index=False
    )

    test_detail_df.to_csv(
        RESULTS_DIR / "student_test_gps_detailed_analysis.csv",
        index=False
    )

    # --------------------------------------------------------
    # GRAPH GENERATION
    # --------------------------------------------------------

    generate_graphs(
        train_summary_df,
        train_detail_df
    )

    # --------------------------------------------------------
    # MAP GENERATION
    # --------------------------------------------------------

    # Generate a map for Trip 00000001.
    selected_trip = "00000001"

    selected_trip_df = train_detail_df[
        train_detail_df["trip_id"] == selected_trip
    ]

    if len(selected_trip_df) > 0:

        generate_map(
            selected_trip_df,
            selected_trip
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("ANALYSIS COMPLETED")

    print("=" * 70)

    print("\nTRAINING DATA SUMMARY")

    print(f"Trips analysed: {len(train_summary_df)}")

    print(
        "Total GPS observations:",
        train_summary_df[
            "total_observations"
        ].sum()
    )

    print(
        "Total missing records:",
        train_summary_df[
            "observations_with_missing_data"
        ].sum()
    )

    print(
        "Total invalid coordinates:",
        train_summary_df[
            "invalid_coordinates"
        ].sum()
    )

    print(
        "Total large time gaps:",
        train_summary_df[
            "large_time_gaps"
        ].sum()
    )

    print(
        "Total suspicious movements:",
        train_summary_df[
            "suspicious_movement_observations"
        ].sum()
    )

    print("\nFILES SAVED IN:")

    print(RESULTS_DIR)

    print(GRAPHS_DIR)

    print(MAPS_DIR)


if __name__ == "__main__":

    main()
