from pathlib import Path

# Goto Project Root
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

class Settings:
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
    ONNX_MODEL_PATH = ARTIFACTS_DIR / "model.onnx"
    PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
    THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"
    
    
    APP_NAME = "Loan Default Predictor API"
    VERSION = "1.0.0"
    DESCRIPTION = "Predicts loan default risk from application data"
    
    
settings = Settings()