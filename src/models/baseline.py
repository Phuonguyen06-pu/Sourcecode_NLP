'''Baseline: TF-IDF char ngram + word ngram + SVM tuyến tính (sklearn LinearSVC, loss="hinge")
văn bản theo preset baseline (hạ chữ thường, teencode, emoji tách token, không tách từ)
chạy huấn luyện: python -m src.models.baseline'''

import argparse
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.config import BASELINE, LABEL_COL, LABELS, MODELS_DIR, RANDOM_SEED, TEXT_COL, ensure_dirs, N_CLASSES
from src.data import load_splits
from src.evaluate import compute_metrics, expand_probs, save_predictions, save_report
from src.models.baseline_model import TextSVM

NAME = "baseline"

def _tfidf(kind):
    '''vector hóa văn bản theo char hoặc word ngram với cấu hình trong config.py'''
    
    p = BASELINE[kind]
    kw = dict(
        analyzer=p["analyzer"], 
        ngram_range=p["ngram_range"], 
        min_df=p["min_df"], 
        max_features=p["max_features"], 
        sublinear_tf=True, 
        lowercase=True
    )
    
    if p["analyzer"] == "word":
        kw["token_pattern"] = r"(?u)\b\w+\b"
        
    return TfidfVectorizer(**kw)


def make_vectorizer():
    '''trả về bộ vector hóa văn bản
    char_word: ghép đặc trưng char và word thành một ma trận chung'''
    return FeatureUnion(
        [("char", _tfidf("char")), 
         ("word", _tfidf("word"))]
    )


def svm(c):
    '''tạo mô hình SVM với tham số C (kernel lấy từ config.py): 
        - C nhỏ: chấp nhận nhiều lỗi
        - C lớn: phạt lỗi mạnh hơn
    class_weight='balanced: tự điều chỉnh trọng số lớp dựa trên tần suất mẫu của từng lớp'''
    
    return LinearSVC(
        C=c,
        loss="hinge",
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_SEED
    )
    
    
def run(data, c=None):
    '''huấn luyện và đánh giá baseline
    các bước: vector hóa -> chọn C tốt nhất theo val -> hiệu chỉnh xác suất -> đo val/test -> lưu kết quả
    trả về (tên kết quả, dict kết quả val/test)'''
    
    t0 = time.time()
    train, val, test = data["train"], data["val"], data["test"]
    ytr = train[LABEL_COL]  #nhãn thật tập train
    
    #b1: fit TF-IDF trên train, rồi áp dụng cho val/test (tránh rò rỉ dư liệu)
    vector = make_vectorizer()
    Xtrain = vector.fit_transform(train[TEXT_COL])
    Xval, Xtest = vector.transform(val[TEXT_COL]), vector.transform(test[TEXT_COL])
    print(f"[{NAME}] số đặc trưng: {Xtrain.shape[1]:,}")
    
    #b2: thử từng C trong "c_grid", chọn C có macro-f1 trên val cao nhất
    c_values = [c] if c else BASELINE["c_grid"]
    best_c, best_f1, best_svm = None, -1, None
    
    for c_value in c_values:
        t = time.time()
        model = svm(c_value).fit(Xtrain, ytr)
        f1 = f1_score(val[LABEL_COL], model.predict(Xval), average="macro")
        print(f"    C={c_value:<4} val macro-f1 = {f1:.4f} ({time.time() - t:.0f}s)")
        
        if f1 > best_f1:
            best_c, best_f1, best_svm = c_value, f1, model
    print(f"    -> Chọn C={best_c}")
    
    #b3: hiệu chỉnh xs để có "độ tin cậy", số fold ko vượt quá số mẫu của lớp nhỏ nhất
    calibrator = LogisticRegression(max_iter=1000).fit(best_svm.decision_function(Xval), val[LABEL_COL])
    
    #b4: đo trên val và test, lưu dự đoán từng câu
    results = {}
    
    for split, df, X in (("val", val, Xval), ("test", test, Xtest)):
        pred = best_svm.predict(X)
        probs = expand_probs(calibrator.predict_proba(best_svm.decision_function(X)), calibrator.classes_, N_CLASSES)
        results[split] = compute_metrics(df[LABEL_COL], pred, LABELS)
        save_predictions(NAME, split, df, probs, LABELS, pred_ids=pred)
        
    print(f"  val  macro-F1 = {results['val']['macro_f1']:.4f} | acc = {results['val']['accuracy']:.4f}")
    print(f"  test macro-F1 = {results['test']['macro_f1']:.4f} | acc = {results['test']['accuracy']:.4f}")
    
    #b5: lưu model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(TextSVM(vector, best_svm, calibrator), MODELS_DIR / f"{NAME}.joblib", compress=3)
    save_report(NAME, results, LABELS, extra={"best_C": best_c, "model": "LinearSVC (loss=hinge)", 
                    "n_features": int(Xtrain.shape[1]), "seconds": round(time.time() - t0, 1)})
    print(f"    xong sau {time.time() - t0:.0f}s")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, default=None, help="dùng luôn giá trị C này, bỏ qua bước thử C (nhanh hơn)")
    ap.add_argument("--limit", type=int, default=None, help="chỉ dùng N dòng mỗi tập (chạy thử)")
    args = ap.parse_args()
 
    ensure_dirs()
    data = load_splits(NAME, args.limit)
    print("Dữ liệu:", {k: len(v) for k, v in data.items()})
    run(data, args.c)
 
 
if __name__ == "__main__":
    main()
 