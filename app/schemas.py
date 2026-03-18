from pydantic import BaseModel, Field, validator
from typing import Optional

class TitanicFeatures(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Ticket class: 1st, 2nd, 3rd")
    Sex: str = Field(..., description="male or female")
    Age: Optional[float] = Field(None, ge=0, le=100, description="Age in years")
    SibSp: int = Field(0, ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(0, ge=0, description="Number of parents/children aboard")
    Fare: float = Field(..., ge=0, description="Passenger fare")
    Embarked: str = Field("S", description="Port of embarkation: C, Q, S")

    @validator('Sex')
    def validate_sex(cls, v):
        if v.lower() not in ['male', 'female']:
            raise ValueError('Sex must be male or female')
        return v.lower()

    @validator('Embarked')
    def validate_embarked(cls, v):
        if v.upper() not in ['C', 'Q', 'S']:
            return 'S'
        return v.upper()

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability: float

class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_type: str
