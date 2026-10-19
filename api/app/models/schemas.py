from pydantic import BaseModel, Field
from typing import Optional

# this matches our dataset
class LoanApplication(BaseModel):
    year: int
    Gender: str
    approv_in_adv: str
    loan_type: str
    loan_purpose: str
    Credit_Worthiness: str
    open_credit: str
    business_or_commercial: str
    loan_amount: int
    term: float
    property_value: float
    construction_type: str
    occupancy_type: str
    Secured_by: str
    total_units: str
    income: float
    credit_type: str
    Credit_Score: int
    co_applicant_credit_type: str = Field(alias="co-applicant_credit_type")
    age: str
    submission_of_application: str
    Region: str
    Security_Type: str
    dtir1: float

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_label: str
    threshold_used: float


class BatchPredictionRequest(BaseModel):
    applications: list[LoanApplication]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    high_risk_count: int
    low_risk_count: int