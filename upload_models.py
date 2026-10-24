import os
import sys
from huggingface_hub import HfApi
from download_models import MODELS, repo_id

def upload(name, token):
    api = HfApi(token=token)
    rid = repo_id(name)
    api.create_repo(rid, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=MODELS[name],
        repo_id=rid,
        repo_type="model",
        ignore_patterns=["checkpoint-*", "runs/*", "optimizer.pt",
                         "scheduler.pt", "rng_state.pth"],
    )
    print("Đã upload:", f"https://huggingface.co/{rid}")

if __name__ == "__main__":
    token = os.environ["HF_TOKEN"]
    for n in (sys.argv[1:] or MODELS):
        upload(n, token)