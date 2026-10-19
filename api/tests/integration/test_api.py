import pytest
from fastapi.testclient import TestClient
from app.main import app

# TestClient wraps your FastAPI app and lets you send HTTP requests
# without starting a real server — it runs in-process, fast
@pytest.fixture(scope="module")
def client():
    """
    scope="module" means this fixture runs once for the entire test file,
    not once per test. The model loads once, all tests share it.
    
    Using TestClient as context manager triggers lifespan events —
    startup (load_artifacts) runs on __enter__, shutdown runs on __exit__
    """
    with TestClient(app) as c:
        yield c

# Sample application data — reuse your working HIGH RISK sample
HIGH_RISK_APPLICATION = {
    "year": 2019,
    "Gender": "Male",
    "approv_in_adv": "nopre",
    "loan_type": "type3",
    "loan_purpose": "p4",
    "Credit_Worthiness": "l2",
    "open_credit": "opc",
    "business_or_commercial": "b/c",
    "loan_amount": 450000,
    "term": 180.0,
    "property_value": 390000.0,
    "construction_type": "mh",
    "occupancy_type": "ir",
    "Secured_by": "land",
    "total_units": "4U",
    "income": 1200.0,
    "credit_type": "CIB",
    "Credit_Score": 548,
    "co-applicant_credit_type": "EXP",
    "age": ">74",
    "submission_of_application": "not_inst",
    "Region": "south",
    "Security_Type": "Indirect",
    "dtir1": 67.2
}



def test_health_endpoint_returns_healthy(client):
    # hit GET /health, assert status 200 and status == "healthy"
    # Send GET request to /health
    # client.get() is equivalent to hitting the endpoint in Swagger
    response = client.get("/health")
    
    # response.status_code — the HTTP status number (200, 404, 500 etc)
    assert response.status_code == 200
    
    # response.json() — parses the JSON response body into a dict
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True


def test_model_info_endpoint(client):
    # hit GET /model-info, assert status 200 and model_type == "XGBoost"
    response = client.get("/model-info")
    
    assert response.status_code == 200
    
    data = response.json()
    # Verify the model info matches what we set in config/service
    assert data["model_type"] == "XGBoost"
    assert data["threshold"] == 0.40
    assert data["features"] == 35

def test_predict_endpoint_high_risk(client):
    # hit POST /predict with HIGH_RISK_APPLICATION
    # assert status 200, prediction == 1, risk_label == "HIGH RISK"

    response = client.post("/predict", json=HIGH_RISK_APPLICATION)
    assert response.status_code == 200
    
    data = response.json()
    
    # Verify response shape matches PredictionResponse Schema
    assert "prediction" in data
    assert "probability" in data
    assert "risk_label" in data
    assert "threshold_used" in data
    
    # verify actual prediction values
    assert data["prediction"] == 1
    assert data["risk_label"] == "HIGH RISK"
    assert data["threshold_used"] == 0.40
    # Probability should be between 0 and 1
    assert 0 <= data["probability"] <= 1
    
def test_predict_invalid_input(client):
    # hit POST /predict with an empty dict {}
    # what status code should FastAPI return for invalid input?
    # assert that status code
    
    # Send completely empty JSON — missing all required fields
    # Pydantic validation will fail before even reaching our service
    response = client.post("/predict", json={})
    
    # FastAPI returns 422 Unprocessable Entity when Pydantic validation fails
    # This is different from 400 Bad Request — 422 specifically means
    # "I understood the request but the data is semantically invalid"
    assert response.status_code == 422
    
    # FastAPI also returns detail about which fields failed validation
    data = response.json()
    assert "detail" in data
    

def test_batch_predict_endpoint(client):
    # Wrap two applications in the BatchPredictionRequest structure
    batch_payload = {
        "applications" : [
            HIGH_RISK_APPLICATION,
            HIGH_RISK_APPLICATION
        ]
    }
    
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    
    data = response.json()
    
    # verify batch response structure
    assert data["total"] == 2
    assert len(data["predictions"]) == 2
    
    # high_risk_count + low_risk_count should always equal total
    assert data["high_risk_count"] + data["low_risk_count"] == data["total"]