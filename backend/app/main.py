from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.trip_routes import router as trip_router


app = FastAPI(
    title="PathSnap API",
    description=(
        "GPS Trajectory Analysis and "
        "Machine Learning Map Matching API"
    ),
    version="1.0.0"
)


# Frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register API routes
app.include_router(trip_router)


@app.get("/")
def home():
    return {
        "project": "PathSnap",
        "message": "PathSnap Backend is running",
        "status": "online"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PathSnap API"
    }


@app.get("/api/models")
def get_models():
    return {
        "models": [
            {
                "name": "Random Forest",
                "status": "trained",
                "accuracy": 92.7,
                "selected": True
            },
            {
                "name": "KNN",
                "status": "trained",
                "accuracy": 87.0,
                "selected": False
            }
        ]
    }