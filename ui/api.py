'''Chạy từ thư mục gốc dự án:   python ui/api.py
rồi mở http://127.0.0.1:8000'''
 
import io
import json
import re
import sys
from pathlib import Path
 
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
 
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
 
from src.config import LABELS, MODELS_DIR, PREPROCESS_PRESETS, RESULTS_METRICS
from src.explain import explain, top_words
from src.predict import ALLOW, BLOCK, REVIEW, T_HIGH, T_LOW, predict_texts
 
MODELS = list(PREPROCESS_PRESETS)    # baseline, phobert, visobert
MODEL_TITLES = {"baseline": "Baseline", "phobert": "PhoBERT", "visobert": "ViSoBERT"}
TEXT_COLUMN_HINTS = ("text", "free_text", "comment", "content", "văn bản", "van_ban", "bình luận", "câu", "noi_dung")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_TEXTS = 2000
 
app = FastAPI(title="Phát hiện ngôn từ độc hại tiếng Việt")
 
 
def title(model):
    return MODEL_TITLES.get(model, model)
 
 
def model_available(model):
    '''mô hình đã train và lưu chưa'''
    path = MODELS_DIR / "baseline.joblib" if model == "baseline" else MODELS_DIR / model
    return path.exists()
 
 
def check_model(model):
    if model not in MODELS:
        raise HTTPException(400, f"Không có mô hình '{model}'.")
    if not model_available(model):
        raise HTTPException(409, f"{title(model)} chưa được train.")
 
 
#đọc file
 
def read_table(name, raw):
    '''đọc .csv hoặc .xlsx'''
    name = name.lower()
    if name.endswith(".csv"):
        for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except UnicodeDecodeError:
                continue
            except pd.errors.EmptyDataError:
                raise ValueError("File rỗng.")
    elif name.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(raw))
    else:
        raise ValueError("Chỉ hỗ trợ file .csv và .xlsx.")
    df.columns = [str(c) for c in df.columns]
    return df
 
 
def find_text_column(df):
    '''lấy cột văn bản'''
    lower = {c.lower().strip(): c for c in df.columns}
    for hint in TEXT_COLUMN_HINTS:
        if hint in lower:
            return lower[hint]
    best, best_len = None, 0.0
    for c in df.columns:
        if df[c].dtype == object:
            mean_len = df[c].dropna().astype(str).str.len().mean()
            if mean_len == mean_len and mean_len > best_len:      # mean_len == mean_len: loại NaN
                best, best_len = c, mean_len
    return best
 
 
def load_metrics():
    rows = []
    for model in MODELS:
        path = RESULTS_METRICS / f"{model}.json"
        row = {"Mô hình": title(model)}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                r = json.load(f)
            t = r["test"]
            row.update({
                "Accuracy": t["accuracy"],
                "Macro-F1": t["macro_f1"],
                "Weighted-F1": t["weighted_f1"],
                "Macro-F1 (val)": r["val"]["macro_f1"],
            })
            for cls, m in t["per_class"].items():
                row[f"F1 {cls}"] = m["f1"]
        rows.append(row)
    columns = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    return columns, [{c: row.get(c) for c in columns} for row in rows]
 
 
# API 
 
class PredictIn(BaseModel):
    model: str
    texts: list[str]
 
 
class ExplainIn(BaseModel):
    model: str
    text: str
 
 
@app.get("/api/status")
def api_status():
    return {
        "models": [{"id": m, "title": title(m), "available": model_available(m)} for m in MODELS],
        "labels": [LABELS[i] for i in sorted(LABELS)],
        "actions": [ALLOW, REVIEW, BLOCK],
        "thresholds": {"low": T_LOW, "high": T_HIGH},
    }
 
 
@app.get("/api/metrics")
def api_metrics():
    columns, rows = load_metrics()
    return {"columns": columns, "rows": rows}
 
 
@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File quá lớn (tối đa 5 MB).")
    try:
        df = read_table(file.filename or "", raw)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Không đọc được file: {e}")
    if not len(df.columns):
        raise HTTPException(400, "File không có cột nào.")
    guess = find_text_column(df)
    rows = df.fillna("").astype(str).values.tolist()
    return {"name": file.filename, "columns": list(df.columns), "guess": guess, "rows": rows}
 
 
@app.post("/api/predict")
def api_predict(body: PredictIn):
    check_model(body.model)
    if not body.texts:
        raise HTTPException(400, "Chưa có văn bản nào.")
    if len(body.texts) > MAX_TEXTS:
        raise HTTPException(413, f"Tối đa {MAX_TEXTS} câu mỗi lần.")
    if any(not re.search(r"\w", t) for t in body.texts):
        raise HTTPException(400, "Có câu không chứa chữ nào.")
    out = predict_texts(body.texts, body.model)
    for r in out:
        del r["text"]
    return {"model": body.model, "results": out}
 
 
@app.post("/api/explain")
def api_explain(body: ExplainIn):
    check_model(body.model)
    if not re.search(r"\w", body.text):
        raise HTTPException(400, "Câu không chứa chữ nào.")
    exp = explain(body.text, body.model)
    return {
        "tokens": exp["display"],
        "scores": exp["scores"],
        "top": [{"word": w, "score": s} for w, s in top_words(exp, 5)],
    }
 
 
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
 
 
if __name__ == "__main__":
    import uvicorn
    print("Mở trình duyệt tại http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
 