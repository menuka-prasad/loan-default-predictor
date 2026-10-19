from fastapi import APIRouter
from app.services.prediction_service import prediction_service

router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": prediction_service._loaded,
        "version": prediction_service.model_info["version"] if prediction_service._loaded else None
    }
    
    
@router.get("/model-info")
def model_info():
    return prediction_service.model_info