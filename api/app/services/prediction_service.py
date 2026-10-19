import json
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from app.core.config import settings

# now we use hugging face to get model artifacts instead of from local folder
HF_REPO_ID = "menuka17/loan-default-predictor"

def download_artifacts_if_missing():
    """Download artifacts from huggingface hub if not present locally."""
    from huggingface_hub import hf_hub_download
    
    os.makedirs(settings.ARTIFACTS_DIR, exist_ok=True)
    
    files = ["model.pkl", "preprocessor.pkl", "threshold.json"]
    
    for filename in files:
        local_path = settings.ARTIFACTS_DIR / filename
        if not local_path.exists():
            print(f"Downloading {filename} from HF hub...")
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                local_dir=settings.ARTIFACTS_DIR,
                repo_type="model"
            )
            print(f"Downloaded: {filename}")
            
        else:
            print(f"Found locally: {filename}")

class PredictionService:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.threshold = None
        self.model_info = None
        self._loaded = False

    def load_artifacts(self):
        """Load all artifacts once at startup."""
        download_artifacts_if_missing()
        
        self.preprocessor = joblib.load(settings.PREPROCESSOR_PATH)
        self.model = joblib.load(settings.MODEL_PATH)
        
        with open(settings.THRESHOLD_PATH) as f:
            threshold_data = json.load(f)
        
        self.threshold = threshold_data["threshold"]
        self.model_info = {
            "model_type": "XGBoost",
            "version": settings.VERSION,
            "threshold": self.threshold,
            "features": 35,
            "description": settings.DESCRIPTION
        }
        self._loaded = True

    def predict(self, application_dict: dict) -> dict:
        """Run single prediction."""
        
        # Compute LTV before creating DataFrame
        application_dict["LTV"] = (
            application_dict["loan_amount"] / application_dict["property_value"]
            )
        # Convert to DataFrame — preprocessor expects a DataFrame, not a dict
        df = pd.DataFrame([application_dict])
        
        # Preprocess
        X = self.preprocessor.transform(df)
        
        # Get probability
        probability = float(self.model.predict_proba(X)[0][1])
        
        # Apply threshold
        prediction = int(probability >= self.threshold)
        risk_label = "HIGH RISK" if prediction == 1 else "LOW RISK"
        
        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk_label": risk_label,
            "threshold_used": self.threshold
        }

    def predict_batch(self, applications: list[dict]) -> list[dict]:
        """Run batch prediction."""
        
        # Compute LTV for each application
        for app in applications:
            app["LTV"] = app["loan_amount"] / app["property_value"]
            
            
        df = pd.DataFrame(applications)
        X = self.preprocessor.transform(df)
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)
        
        return [
            {
                "prediction": int(pred),
                "probability": round(float(prob), 4),
                "risk_label": "HIGH RISK" if pred == 1 else "LOW RISK",
                "threshold_used": self.threshold
            }
            for pred, prob in zip(predictions, probabilities)
        ]

# Singleton instance
prediction_service = PredictionService()