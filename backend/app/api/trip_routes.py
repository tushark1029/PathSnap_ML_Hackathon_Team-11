from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import io

from app.services.gps_service import process_gps_trajectory

router = APIRouter(
    prefix="/api/trips",
    tags=["Trips"]
)


@router.post("/analyze")
async def analyze_trip(file: UploadFile = File(...)):
    """
    Upload a GPS trajectory and perform GPS analysis.
    """

    try:
        contents = await file.read()

        filename = file.filename.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(
                io.BytesIO(contents)
            )

        elif filename.endswith(".track"):
            df = pd.read_csv(
                io.BytesIO(contents),
                sep=r"\s+",
                header=None,
                names=[
                    "longitude",
                    "latitude",
                    "timestamp"
                ]
            )

        else:
            raise HTTPException(
                status_code=400,
                detail="Upload a CSV or TRACK file."
            )

        processed_df = process_gps_trajectory(df)

        total_distance = float(
            processed_df["dist_m"].sum()
        )

        total_duration = float(
            processed_df["timestamp"].iloc[-1]
            - processed_df["timestamp"].iloc[0]
        )

        average_speed = float(
            processed_df["speed_kmh"].mean()
        )

        maximum_speed = float(
            processed_df["speed_kmh"].max()
        )

        anomaly_count = int(
            processed_df["flag_speed_anomaly"].sum()
        )

        gap_count = int(
            processed_df["flag_gap_gt_5s"].sum()
        )

        trajectory = processed_df[
            ["longitude", "latitude", "timestamp"]
        ].to_dict(orient="records")

        return {
            "filename": file.filename,
            "status": "success",
            "analysis": {
                "observations": len(processed_df),
                "total_distance_m": round(
                    total_distance, 2
                ),
                "total_distance_km": round(
                    total_distance / 1000, 3
                ),
                "duration_seconds": round(
                    total_duration, 2
                ),
                "average_speed_kmh": round(
                    average_speed, 2
                ),
                "maximum_speed_kmh": round(
                    maximum_speed, 2
                ),
                "speed_anomalies": anomaly_count,
                "large_time_gaps": gap_count
            },
            "trajectory": trajectory
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )