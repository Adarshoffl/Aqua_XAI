from pydantic import BaseModel


class WaterQualityInput(BaseModel):

    Country: str
    Waterbody_Type: str

    Ammonia: float
    Biochemical_Oxygen_Demand: float
    Dissolved_Oxygen: float
    Orthophosphate: float
    pH: float
    Temperature: float
    Nitrogen: float
    Nitrate: float

    Year: int
    Month: int


class WaterQualityResponse(BaseModel):

    water_quality: str
    confidence: float