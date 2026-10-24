# highlight các từ độc hại

import argparse
import html as _html
 
import numpy as np
 
from src.predict import (LABELS, PREPROCESS_PRESETS, clean_text, decide, predict_proba_clean, toxic_score)

def _logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))

def explain(text, model="baseline", max_tokens=100, t_low=None, t_high=None):
    '''trả về dict: tokens (đã tiền xử lý), scores (đóng góp của từng token), toxic_score, label, action
    scores (log-odds) > 0: token đẩy về phía độc hại
    scores < 0: token kéo về phía bình thường'''
    
    clean = clean_text(text, model)
    tokens = clean.split()
    if not tokens:
        return {
            "text": text, 
            "clean_text": clean, 
            "tokens": [], 
            "display": [], 
            "scores": [],
            "toxic_score": 0.0, 
            "label": LABELS[0], 
            "action": decide(0.0, t_low, t_high)
        }
 
    n = min(len(tokens), max_tokens)
    variants = [clean] + [" ".join(tokens[:i] + tokens[i + 1:]) for i in range(n)]
    probs = predict_proba_clean(model, variants)          
    scores_all = toxic_score(probs)
    base = float(scores_all[0])
 
    lg = _logit(scores_all)
    scores = [float(lg[0] - v) for v in lg[1:]] + [0.0] * (len(tokens) - n)   # đơn vị: log-odds
    label_id = int(probs[0].argmax())
    return {
        "text": text,
        "clean_text": clean,
        "tokens": tokens,
        "display": [t.replace("_", " ") for t in tokens],    
        "scores": scores,
        "toxic_score": base,
        "label": LABELS[label_id],
        "action": decide(base, t_low, t_high),
    }
    
def top_words(result, k=5):
    '''k từ đóng góp nhiều nhất vào việc câu bị coi là độc hại (chỉ lấy điểm > 0)'''
    combined_pairs = zip(result["display"], result["scores"])

    pairs = [
        (word, score) 
        for word, score in combined_pairs 
        if score > 0
    ]   
    return sorted(pairs, key=lambda x: -x[1])[:k]
 
 
def highlight_html(result):
    '''HTML tô nền đỏ cho từ đẩy về phía độc hại (đậm theo mức đóng góp)'''
    scores = np.array(result["scores"]) if result["scores"] else np.array([0.0])
    top = max(float(scores.max()), 1e-9)
    parts = []
    for word, s in zip(result["display"], result["scores"]):
        w = _html.escape(word)
        if s > 0.05:
            alpha = 0.15 + 0.65 * min(s / top, 1.0)
            parts.append(f'<span title="+{s:.2f}" style="background:rgba(220,38,38,{alpha:.2f});'
                         f'border-radius:3px;padding:0 2px">{w}</span>')
        else:
            parts.append(w)
    return " ".join(parts)
 
 
def main():
    ap = argparse.ArgumentParser(description="Giải thích dự đoán cho một câu")
    ap.add_argument("text")
    ap.add_argument("--model", default="baseline", choices=list(PREPROCESS_PRESETS))
    args = ap.parse_args()
    r = explain(args.text, args.model)
    print(f"{r['label']} | điểm độc hại = {r['toxic_score']:.3f} | mức xử lý: {r['action']}")
    for w, s in sorted(zip(r["display"], r["scores"]), key=lambda x: -x[1]):
        print(f"  {s:+.2f}  {w}")
 
 
if __name__ == "__main__":
    main()