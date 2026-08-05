from fastapi import FastAPI
from app.database.connection import engine, Base

# Import models so SQLAlchemy knows them
from app.models.water_models import (
    User,
    WaterQuality,
    Treatment,
    SHAPResult,
    ChatHistory
)


# Create database tables
Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="Aqua_XAI API",
    description="Explainable AI Based Water Quality Monitoring and Treatment Recommendation System",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "project": "Aqua_XAI",
        "message": "Backend running successfully"
    }