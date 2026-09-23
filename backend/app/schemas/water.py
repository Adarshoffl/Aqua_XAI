from pydantic import BaseModel
from typing import List, Optional

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

# Schema for SHAP explanation
class FeatureImpact(BaseModel):
    feature: str
    importance: float

# Highly detailed schema for the treatment/maintenance protocols
class ActionRecommendation(BaseModel):
    action_type: str
    parameter: str
    treatment_name: str
    description: str
    working_principle: str
    advantages: str
    limitations: str
    maintenance: str
    estimated_cost: str
    precautions: str

# Final Response Schema including the new AI AI field
class WaterQualityResponse(BaseModel):
    water_quality: str
    confidence: float
    explanation: List[FeatureImpact]
    recommended_actions: List[ActionRecommendation] = [] 
    ai_treatment_plan: Optional[str] = None