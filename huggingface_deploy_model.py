from huggingface_hub import HfApi
import os

api = HfApi()

# Create a model repository on HF Hub
# Change "your-username" to your actual HF username
repo_id = "menuka17/loan-default-predictor"

api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True,
    private=False
)

# Upload all artifacts
artifact_files = [
    "artifacts/model.pkl",
    "artifacts/model.onnx", 
    "artifacts/preprocessor.pkl",
    "artifacts/threshold.json"
]

for file_path in artifact_files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id=repo_id,
        repo_type="model"
    )
    print(f"Uploaded: {file_path}")

print(f"Done. Repo: https://huggingface.co/{repo_id}")