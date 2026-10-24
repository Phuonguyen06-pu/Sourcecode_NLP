#Dự đoán cho câu mới: nhãn, xác suất từng lớp và mức xử lý (Cho phép / Cần kiểm tra / Chặn)

import argparse
import sys
from functools import lru_cache
from pathlib import Path
 
import numpy as np
 
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
 
import src.config as cfg
from src.config import LABELS, MODELS_DIR, N_CLASSES, PREPROCESS_PRESETS
from src.preprocess import clean_text

 
# ngưỡng xử lý: lấy từ config.py nếu có, không thì dùng giá trị mặc định dưới đây
# mặc định chọn theo tập val của baseline: T_LOW bắt được >= 85% câu độc hại, T_HIGH để câu bị chặn đúng >= ~90%
# PhoBERT/ViSoBERT phân bố xác suất khác baseline, nên chọn lại ngưỡng sau khi train xong rồi đặt vào config.py
T_LOW = getattr(cfg, "T_LOW", 0.20)     # điểm độc hại >= T_LOW  -> Cần kiểm tra
T_HIGH = getattr(cfg, "T_HIGH", 0.80)   # điểm độc hại >= T_HIGH -> Chặn
CLEAN_ID = getattr(cfg, "CLEAN_ID", 0)  # id của nhãn "BÌNH THƯỜNG"
 
ALLOW, REVIEW, BLOCK = "Cho phép", "Cần kiểm tra", "Chặn"
 
 
@lru_cache(maxsize=None)
def load_model(name):
    if name not in PREPROCESS_PRESETS:
        raise ValueError(f"Mô hình '{name}' không có. Các mô hình hiện có: {list(PREPROCESS_PRESETS)}")
    if name == "baseline":
        import joblib
        path = MODELS_DIR / "baseline.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Chưa có {path}. Chạy: python -m src.models.baseline")
        return joblib.load(path)
    from src.models.transformer_model import TransformerClassifier
    return TransformerClassifier(name)
 
 
def predict_proba_clean(name, clean_texts):
    '''xác suất (n x N_CLASSES) cho các câu đã tiền xử lý theo đúng preset của mô hình'''
    model = load_model(name)
    probs = np.asarray(model.predict_proba(list(clean_texts)))
    if probs.shape[1] != N_CLASSES:    
        full = np.zeros((len(probs), N_CLASSES))
        full[:, np.asarray(model.classes_, dtype=int)] = probs
        probs = full
    return probs
 
 
def toxic_score(probs):
    '''điểm độc hại = 1 - xác suất lớp BÌNH THƯỜNG'''
    return 1.0 - np.asarray(probs)[..., CLEAN_ID]
 
 
def decide(score, t_low=None, t_high=None):
    '''điểm độc hại -> mức xử lý'''
    t_low = T_LOW if t_low is None else t_low
    t_high = T_HIGH if t_high is None else t_high
    if not 0 <= t_low < t_high <= 1:
        raise ValueError(f"Cần 0 <= T_low < T_high <= 1 (đang là {t_low}, {t_high})")
    if score >= t_high:
        return BLOCK
    if score >= t_low:
        return REVIEW
    return ALLOW
 
 
def predict_texts(texts, model="baseline", t_low=None, t_high=None):
    '''dự đoán cho danh sách câu thô'''
    if isinstance(texts, str):
        texts = [texts]
    cleaned = [clean_text(t, model) for t in texts]
    probs = predict_proba_clean(model, cleaned)
    out = []
    for raw, clean, p in zip(texts, cleaned, probs):
        label_id = int(p.argmax())
        score = float(toxic_score(p))
        out.append({
            "text": raw,
            "clean_text": clean,
            "label_id": label_id,
            "label": LABELS[label_id],
            "probs": {LABELS[i]: float(p[i]) for i in sorted(LABELS)},
            "toxic_score": score,
            "action": decide(score, t_low, t_high),
        })
    return out
 
 
def predict_one(text, model="baseline", t_low=None, t_high=None):
    return predict_texts([text], model, t_low, t_high)[0]
 
 
def main():
    ap = argparse.ArgumentParser(description="Dự đoán độc hại cho câu tiếng Việt (chỉ in ra màn hình, không lưu)")
    ap.add_argument("texts", nargs="+", help="một hoặc nhiều câu cần dự đoán")
    ap.add_argument("--model", default="baseline", choices=list(PREPROCESS_PRESETS))
    ap.add_argument("--t-low", type=float, default=None)
    ap.add_argument("--t-high", type=float, default=None)
    args = ap.parse_args()
    for r in predict_texts(args.texts, args.model, args.t_low, args.t_high):
        print(f"\n{r['text']}")
        print(f"  -> {r['label']} | điểm độc hại = {r['toxic_score']:.3f} | mức xử lý: {r['action']}")
        print("     " + " | ".join(f"{k}: {v:.3f}" for k, v in r["probs"].items()))
 
 
if __name__ == "__main__":
    main()