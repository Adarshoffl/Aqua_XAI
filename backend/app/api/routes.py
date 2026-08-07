from fastapi import APIRouter

from backend.app.schemas.water import (
    WaterQualityInput,
    WaterQualityResponse
)

from backend.app.services.prediction import (
    predict_water_quality
)


router = APIRouter()


@router.post(
    "/predict",
    response_model=WaterQualityResponse
)
def predict(
    water_data: WaterQualityInput
):

    data = {

        "Country": water_data.Country,

        "Waterbody Type": water_data.Waterbody_Type,

        "Ammonia (mg/l)": water_data.Ammonia,

        "Biochemical Oxygen Demand (mg/l)": 
            water_data.Biochemical_Oxygen_Demand,

        "Dissolved Oxygen (mg/l)": 
            water_data.Dissolved_Oxygen,

        "Orthophosphate (mg/l)": 
            water_data.Orthophosphate,

        "pH (ph units)": water_data.pH,

        "Temperature (cel)": 
            water_data.Temperature,

        "Nitrogen (mg/l)": 
            water_data.Nitrogen,

        "Nitrate (mg/l)": 
            water_data.Nitrate,

        "Year": water_data.Year,

        "Month": water_data.Month
    }


    try:

        result = predict_water_quality(data)

        return result


    except Exception as e:

        print("API ERROR:", e)

        raise e