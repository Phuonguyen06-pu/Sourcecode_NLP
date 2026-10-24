#gom kết quả mọi mô hình trong results/metrics thành bảng so sánh

import json
import pandas as pd
from src.config import RESULTS_METRICS, ROOT


def build_summary():
    '''đọc các file json kết quả của 3 mô hình, gộp thành 1 bảng
    lưu bảng ra results/summary.csv và trả về dataframe'''
    
    rows = []
    
    for f in sorted(RESULTS_METRICS.glob("*.json")):
        r = json.load(open(f, encoding="utf-8"))
        row = {
            "model": r["model"],
            "val_macro_f1": r["val"]["macro_f1"],
            "test_macro_f1": r["test"]["macro_f1"],
            "test_weight_f1": r["test"]["weighted_f1"],
            "test_acc": r["test"]["accuracy"] 
        }
        
        for cls, m in r["test"]["per_class"].items():
            row[f"test_f1_{cls}"] = m["f1"]
        rows.append(row)
        
    if not rows:
        raise FileNotFoundError(f"Chưa có kết quả nào trong {RESULTS_METRICS}")
    df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).round(4)
    df.to_csv(ROOT / "results" / "summary.csv", index=False)
    return df

if __name__ == "__main__":
    print(build_summary().to_string(index=False))