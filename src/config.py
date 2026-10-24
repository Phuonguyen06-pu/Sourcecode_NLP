# Cấu hình dùng chung: đường dẫn, nhãn, seed, preset tiền xử lý, tham số baseline

import os
from pathlib import Path

ROOT = Path(os.environ.get("NLP_ROOT", Path(__file__).resolve().parents[1]))

RAW_PATH = ROOT / "data" / "raw" / "dataset.xlsx"
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "saved_models"
LEXICON_DIR = ROOT / "lexicon"  #normalization.json (teencode), emoji_vi.json (emoji -> chữ)

RESULTS_METRICS = ROOT / "results" / "metrics"
RESULTS_FIGURES = ROOT / "results" / "figures"
RESULTS_PREDICTIONS = ROOT / "results" / "predictions"


# dữ liệu
LABELS = {0: "BÌNH THƯỜNG", 1: "XÚC PHẠM", 2: "THÙ GHÉT"}
N_CLASSES = len(LABELS)
TEXT_COL = "text"   # cột text đã xử lý theo từng mô hình
LABEL_COL = "label_id"
RANDOM_SEED = 42


# preset tiền xử lý theo từng mô hình 
'''
lower: hạ chữ thường
repeat: gộp chữ/dấu câu lặp
teencode: đổi teencode 
emoji: "token" -> tách riêng thành token | "text" -> đổi sang chữ, lạ bị loại bỏ | "keep" -> giữ nguyên
segment: tách từ bằng underthesea (chỉ dùng cho PhoBERT)
'''
PREPROCESS_PRESETS = {
    "baseline": {"lower": True,  "repeat": True, "teencode": True,  "emoji": "token", "segment": False},
    "phobert":  {"lower": False, "repeat": True, "teencode": True,  "emoji": "text",  "segment": True},
    "visobert": {"lower": False, "repeat": True, "teencode": False, "emoji": "keep",  "segment": False},
}

#tên cột văn bản đã xử lý của từng mô hình
def text_col(model):
    return f"text_{model}"

#cấu hình baseline: TF-IDF char + word ->SVM
BASELINE = {
    "char": {"analyzer": "char_wb", "ngram_range": (2, 5), "min_df": 2, "max_features": 300_000},
    "word": {"analyzer": "word", "ngram_range": (1, 3), "min_df": 2, "max_features": 300_000},
    "c_grid": [0.1, 0.3, 1.0, 2.0],
}

# cấu hình fine-tune PhoBERT / VisoBERT
TRANSFORMER_COMMON ={
    "max_len": 128,        
    "batch_size": 32,       # đưa 32 câu xử lý cùng lúc trong mỗi bước train
    "epochs": 5,            # học đi học lại dữ liệu 5 lần
    "lr": 2e-5,             
    "weight_decay": 0.01,   
    "warmup_ratio": 0.1,    
    "max_grad_norm": 1.0, 
    "patience": 2,          # dừng sớm nếu val macro-f1 không tăng
    "class_weight": True    # loss có trọng số lớp -> xử lý dữ liệu mất cân bằng
}
TRANSFORMERS = {
    "phobert": {"hf_name": "vinai/phobert-base-v2"},
    "visobert": {"hf_name": "uitnlp/visobert"}
}


def ensure_dirs():
    for d in (DATA_DIR, MODELS_DIR, RESULTS_METRICS, RESULTS_FIGURES, RESULTS_PREDICTIONS):
        d.mkdir(parents=True, exist_ok=True)