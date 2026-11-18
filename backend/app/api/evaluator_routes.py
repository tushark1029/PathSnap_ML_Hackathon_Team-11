"""Evaluator trip endpoints.

Exposes the ten reserved evaluator trips to the frontend:

- GET  /api/evaluator/trips                  -> list trip IDs + obs counts
- POST /api/evaluator/trips/{trip_id}/predict -> predicted route + metrics

Ground-truth labels (.route) are used server-side only for the accuracy
metric and are never included in any response.
"""

import re

from fastapi import APIRouter, HTTPException

from app.services import matching_service

router = APIRouter(prefix="/api/evaluator", tags=["Evaluator"])

_TRIP_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _valid_trip_id(trip_id: str) -> str:
    if not _TRIP_ID_RE.match(trip_id):
        raise HTTPException(status_code=400, detail="Invalid trip_id format.")
    return trip_id


@router.get("/trips")
def list_trips():
    """List the reserved evaluator trips (IDs and observation counts only)."""
    trips = matching_service.list_evaluator_trips()
    return {"trips": trips, "count": len(trips)}


@router.post("/trips/{trip_id}/predict")
def predict(trip_id: str):
    """Run map matching for one evaluator trip.

    Returns the predicted route as [lon, lat] coordinates, the GPS
    observations, evaluation metrics (computed server-side against the
    reference route — labels never leave the server), and timings.
    """
    _valid_trip_id(trip_id)
    known = {t["trip_id"] for t in matching_service.list_evaluator_trips()}
    if trip_id not in known:
        raise HTTPException(status_code=404,
                            detail=f"Unknown evaluator trip: {trip_id}")
    try:
        return matching_service.predict_trip(trip_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # keep the API responsive; log detail server-side
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
