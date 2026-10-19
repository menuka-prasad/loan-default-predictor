import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from app.services.prediction_service import PredictionService


@pytest.fixture
def service():
    """Create a PredictionService with mocked artifacts."""
    
    svc = PredictionService()
    
    # Mock preprocessor - transform() returns a fake array
    svc.preprocessor = MagicMock()
    svc.preprocessor.transform.return_value = np.zeros((1, 35))
    
    # Mock Model - predict_proba() returns probability we control
    svc.model = MagicMock()
    svc.model.predict_proba.return_value = np.array([[0.15, 0.85]])
    
    # Set threshold directly - no file needed
    svc.threshold = 0.40
    svc._loaded = True
    
    return svc

def test_high_risk_prediction(service):
    """probability 0.85 > threshold 0.40 should return HIGH RISK"""
    application = {
        "loan_amount": 450000,
        "property_value": 390000.0,
        "year": 2019,
        "Gender": "Male",
        # ... fill in all required fields
    }
    result = service.predict(application)
    
    # Write your assertions here
    assert result["prediction"] == 1
    assert result["risk_label"] == "HIGH RISK"
    assert result["threshold_used"] == 0.40


def test_low_risk_prediction(service):
    """Change mock to return 0.15 probability — should return LOW RISK"""
    service.model.predict_proba.return_value = np.array([[0.85, 0.15]])
    
    application = {
        "loan_amount": 158000,
        "property_value": 210000.0,
        # ... fill in required fields
    }
    result = service.predict(application)
    
    # Write your assertions here
    assert result["prediction"] == 0
    assert result["risk_label"] == "LOW RISK"


def test_probability_is_rounded_to_4_decimals(service):
    """Check that probability is rounded correctly"""
    service.model.predict_proba.return_value = np.array([[0.15, 0.123456789]])
    
    application = {"loan_amount": 158000, "property_value": 210000.0}
    result = service.predict(application)
    
    # What should this assert?
    assert result["probability"] == 0.1235