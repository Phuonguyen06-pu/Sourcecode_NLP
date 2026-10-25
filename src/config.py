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

": 0.1,
        "max_grad_norm": 1.0,
        "patience": 2,
        "class_weight": True,
        "class_weight_power": 1.0,
    },
}

_TRANSFORMER_KEYS = ("hf_name", "max_len", "batch_size", "epochs", "lr", "weight_decay", "warmup_ratio",
                     "max_grad_norm", "patience", "class_weight", "class_weight_power")

def get_transformer_cfg(name):
    '''trả về bản sao cấu hình riêng của mô hình'''
    if name not in TRANSFORMERS:
        raise ValueError(f"Mô hình '{name}' không có. Các mô hình hiện có: {list(TRANSFORMERS)}")
    cfg = dict(TRANSFORMERS[name])
    missing = [k for k in _TRANSFORMER_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Cấu hình '{name}' thiếu tham số: {missing}")
    return cfg

def ensure_dirs():
    for d in (DATA_DIR, MODELS_DIR, RESULTS_METRICS, RESULTS_FIGURES, RESULTS_PREDICTIONS):
        d.mkdir(parents=True, exist_ok=True)
