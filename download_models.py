import os
import sys
from huggingface_hub import snapshot_download

HF_USER = "Uyen1"          
REPO_PREFIX = "nlp-toxic-vi-"
MODELS = {                             
    "baseline": "saved_models/baseline.joblib",
    "phobert":  "saved_models/phobert",
    "visobert": "saved_models/visobert",
}

def repo_id(name):
    return f"{HF_USER}/{REPO_PREFIX}{name}"

def download_all(force=False):
    for name, local_dir in MODELS.items():
        if os.path.isdir(local_dir) and os.listdir(local_dir) and not force:
            print(f"[bỏ qua] {name}: đã có trong {local_dir}")
            continue
        print(f"[tải] {repo_id(name)} -> {local_dir}")
        snapshot_download(repo_id=repo_id(name), local_dir=local_dir)

if __name__ == "__main__":
    download_all(force="--force" in sys.argv)