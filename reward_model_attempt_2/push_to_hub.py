from huggingface_hub import HfApi

MODEL_DIRECTORY = "./exported_model_hf"
REPOSITORY_ID = "philomath-1209/gpt2-reward_model_hh-rlhf"

api = HfApi()

api.create_repo(
    repo_id=REPOSITORY_ID,
    repo_type="model",
    exist_ok=True,
)

api.upload_folder(
    folder_path=MODEL_DIRECTORY,
    repo_id=REPOSITORY_ID,
    repo_type="model",
)

print(f"Model uploaded successfully: {REPOSITORY_ID}")