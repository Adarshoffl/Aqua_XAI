from fastapi import APIRouter, HTTPException
from app.schemas.water import WaterQualityInput, WaterQualityResponse
from app.services.prediction import predict_water_quality

router = APIRouter()

@router.post("/predict", response_model=WaterQualityResponse)
def predict(payload: WaterQualityInput):
    try:
        result = predict_water_quality(payload.model_dump()) # Use .dict() if using older Pydantic
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))