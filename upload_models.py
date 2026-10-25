import os
import sys
from huggingface_hub import HfApi
from download_models import MODELS, repo_id

IGNORE = ["checkpoint-*", "runs/*", "optimizer.pt",
          "scheduler.pt", "rng_state.pth"]

def upload(name, token):
    api = HfApi(token=token)
    rid = repo_id(name)
    path = MODELS[name]
    api.create_repo(rid, repo_type="model", exist_ok=True)

    if os.path.isdir(path):
        api.upload_folder(
            folder_path=path,
            repo_id=rid,
            repo_type="model",
            ignore_patterns=IGNORE,
        )
    else:  
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=os.path.basename(path),
            repo_id=rid,
            repo_type="model",
        )
    print("Đã upload:", f"https://huggingface.co/{rid}")

if __name__ == "__main__":
    token = os.environ["HF_TOKEN"]
    for n in (sys.argv[1:] or MODELS):
        upload(n, token)
