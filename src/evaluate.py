#tính chỉ số, vẽ confusion matrix, lưu kết quả

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from src.config import (LABEL_COL, RESULTS_FIGURES, RESULTS_METRICS, RESULTS_PREDICTIONS, TEXT_COL,
                    ensure_dirs)


def compute_metrics(y_true, y_pred, labels):
    '''tính toán các chỉ số của cả bộ dữ liệu và chi tiết từng lớp'''
    
    ids = sorted(labels)
    names = [labels[i] for i in ids]
    
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=ids, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=ids)
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=ids, average="weighted", zero_division=0)),
        "per_class": {
            n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, n in enumerate(names)
        },
        "confusion_matrix": cm.tolist()
    }
    
    
def expand_probs(probs, classes, n_classes):
    '''đưa ma trận xác suất về đủ n_classes cột (nếu tập train thiếu 1 lớp)'''
    
    full = np.zeros((len(probs), n_classes))
    full[:, np.asanyarray(classes, dtype=int)] = probs
    return full

def plot_confusion(cm, names, title, path):
    '''vẽ confusion matrix'''
    cm = np.array(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n = len(names)
    
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * n + 1.5), max(4.2, 1.1 * n + 1.2)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), names, rotation=30, ha="right")
    ax.set_yticks(range(n), names)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    
    
def save_report(name, results, labels, extra=None):
    '''results: {'val': metrics, 'test': metrics}. Lưu json + hình confusion matrix'''
    
    ensure_dirs()
    names = [labels[i] for i in sorted(labels)]
    
    for split, m in results.items():
        plot_confusion(m["confusion_matrix"], names, f"{name} ({split})", RESULTS_FIGURES / f"cm_{name}_{split}.png")
        
    payload = {"model": name, **results}
    if extra:
        payload["extra"] = extra
    with open(RESULTS_METRICS / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        
        
def save_predictions(name, split, df, probs, labels, pred_ids=None):
    '''lưu dự đoán từng bình luận để phân tích lỗi
    - probs: ma trận xác suất (n_mẫu x n_lớp), cùng thứ tự với df
    - pred_ids: nhãn dự đoán; nếu không truyền thì lấy lớp có xác suất cao nhất
    - confidence: xác suất của nhãn được dự đoán'''
    
    ensure_dirs()
    ids = sorted(labels)
    if pred_ids is None:
        pred = np.asanyarray(ids)[probs.argmax(axis=1)]
    else:
        pred = np.asarray(pred_ids)
    out = pd.DataFrame({
        "text": df[TEXT_COL].values,
        "true_id": df[LABEL_COL].values,
        "pred_id": pred,
        "true_label": [labels[i] for i in df[LABEL_COL].values],
        "pred_label": [labels[i] for i in pred],
        "confidence": probs[np.arange(len(pred)), pred].round(4),
        "correct": df[LABEL_COL].values == pred
    })
    for j, i in enumerate(ids):
        out[f"p_{labels[i]}"] = probs[:, j].round(4)
        
    front = [c for c in ("id", "free_text") if c in df.columns]
    for k, c in enumerate(front):
        out.insert(k, c, df[c].values)
    out.to_csv(RESULTS_PREDICTIONS / f"{name}_{split}.csv", index=False, encoding="utf-8-sig")

def show_results(model):
    '''đọc kết quả đã lưu ra notebook'''
    
    from IPython.display import Image, display
    name = model
    path = RESULTS_METRICS / f"{name}.json"
    
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path}. phải chạy mô hình trước")
    
    r = json.load(open(path, encoding="utf-8"))
    print(f"{name}: val macro-F1 = {r['val']['macro_f1']:.4f} | "
          f"test macro-F1 = {r['test']['macro_f1']:.4f} | test acc = {r['test']['accuracy']:.4f}")
    
    if "extra" in r:
        print({k: v for k, v in r["extra"].items() if k in ("best_C", "n_features", "best_epoch", "n_params", "seconds")})
    display(pd.DataFrame(r["test"]["per_class"]).T.round(4))
    display(Image(filename=str(RESULTS_FIGURES / f"cm_{name}_test.png")))