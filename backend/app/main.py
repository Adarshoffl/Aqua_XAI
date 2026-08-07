from fastapi import FastAPI

from backend.app.api.routes import router


app = FastAPI(
    title="Aqua_XAI API",
    description="Explainable AI Based Water Quality Prediction System",
    version="1.0"
)


app.include_router(
    router,
    prefix="/api"
)


@app.get("/")
def home():

    return {
        "message": "Aqua_XAI API is running"
    }