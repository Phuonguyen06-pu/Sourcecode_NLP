# đọc 3 tập train/val/test, lấy đúng cột văn bản đã xử lý của mô hình


import pandas as pd

from src.config import DATA_DIR, LABEL_COL, PREPROCESS_PRESETS, TEXT_COL, text_col

def load_splits(model="baseline", limit=None):
    if model not in PREPROCESS_PRESETS:
        raise ValueError(f"Các mô hình hiện có: {list(PREPROCESS_PRESETS)}")
    col = text_col(model)
    out = {}
    
    for split in ("train", "val", "test"):
        path = DATA_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy {path}. Chạy src/prepare_data.py")
        
        df = pd.read_csv(path)
        if col not in df.columns:
            raise KeyError(f"{path.name} chưa có cột {col}. Chạy lại src/prepare_data.py --models {model}")
        df[TEXT_COL] = df[col].fillna("").astype(str)
        df[LABEL_COL] = df[LABEL_COL].astype(int)
        
        if limit:
            df = df.head(limit)
        out[split] = df.reset_index(drop=True)
    
    return out
       