#đọc dữ liệu, loại bỏ các dòng thiếu và câu trùng lặp, chia tập train/val/test, tiền xử lý, sinh cột text_<preset> cho từng mô hình

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_DIR, LABELS, PREPROCESS_PRESETS, RANDOM_SEED, RAW_PATH
from preprocess import clean_text, base_clean

def load_raw():
    '''đọc và gộp các sheet, bỏ dòng thiếu text/nhãn'''
    sheets = pd.read_excel(RAW_PATH, sheet_name=None)
    for name, part in sheets.items():
        print(f"Sheet {name}: {len(part)}")
    df = pd.concat(sheets.values(), ignore_index=True)
    df["id"] = df.index
    print("Gộp các sheet:", len(df))
    
    df = df.dropna(subset=["free_text", "label_id"]).copy()
    df["label_id"] = df["label_id"].astype(int)
    print("Sau khi bỏ dòng thiếu text/nhãn:", len(df))
    return df

def deduplicate(df):
    '''loại các câu trùng lặp
    câu trùng khác nhãn bị loại, câu trùng cùng nhãn giữ 1'''
    df = df.assign(key=df["free_text"].map(base_clean).str.casefold())
    df = df[df["key"].str.len() > 0]
    
    n_labels = df.groupby("key")["label_id"].nunique()
    conflict = n_labels[n_labels > 1].index
    df = df[~df["key"].isin(conflict)].drop_duplicates("key")
    print("Sau khi khử trùng lặp:", len(df))
    return df

def split(df):
    '''chia 80/10/10'''
    train, rest = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=RANDOM_SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label_id"], random_state=RANDOM_SEED)
    return {"train": train, "val": val, "test": test}


def main(models):
    df = deduplicate(load_raw())  #loại bỏ trùng lặp cơ bản trên data gốc
    parts = split(df)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    final = {}
    
    # 1. Sinh cột text_<preset> cho từng tập 
    for name, part in parts.items():  
        part = part.copy()
        for m in models:   
            part[f"text_{m}"] = [clean_text(t, m) for t in part["free_text"]]
        parts[name] = part

    # 2. Lọc bỏ các dòng ở train/val/test bị trùng với nhau sau khi đã qua preset
    for m in models:
        text_col = f"text_{m}"
        train_texts = set(parts["train"][text_col])
        parts["val"] = parts["val"][~parts["val"][text_col].isin(train_texts)]
        seen_texts = train_texts | set(parts["val"][text_col])  
        parts["test"] = parts["test"][~parts["test"][text_col].isin(seen_texts)]
        parts["val"] = parts["val"].drop_duplicates(subset=[text_col])

    # 3. Lưu file và kiểm tra phân phối
    for name, part in parts.items():
        text_cols = [f"text_{m}" for m in models]
        empty = (part[text_cols] == "").any(axis=1)
        if empty.any():
            part = part[~empty]
            
        columns = ["id", "free_text", "label_id"] + text_cols
        part[columns].to_csv(DATA_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
 
        dist = part["label_id"].value_counts().sort_index().to_dict()
        print(name, len(part), {LABELS[k]: f"{v}" for k, v in dist.items()})
 
    # Kiểm tra lại sau khi lọc
    keys = [set(p[f"text_{models[0]}"]) for p in parts.values()]
    print("Sau khi lọc - Trùng train-val:", len(keys[0] & keys[1]), "| train-test:", len(keys[0] & keys[2]),
          "| val-test:", len(keys[1] & keys[2]))
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(PREPROCESS_PRESETS), choices=list(PREPROCESS_PRESETS),
                    help="các preset cần sinh cột văn bản (mặc định: tất cả)")
    main(ap.parse_args().models)
 