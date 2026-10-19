from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    LoanApplication,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)
from app.services.prediction_service import prediction_service

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    try:
        data = application.model_dump(by_alias=True)
        result = prediction_service.predict(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    try:
        applications = [
            app.model_dump(by_alias=True) for app in request.applications
        ]
        
        results = prediction_service.predict_batch(applications)
        high_risk = sum(1 for r in results if r["prediction"] == 1)
        
        return {
                "predictions": results,
                "total": len(results),
                "high_risk_count": high_risk,
                "low_risk_count": len(results) - high_risk
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
