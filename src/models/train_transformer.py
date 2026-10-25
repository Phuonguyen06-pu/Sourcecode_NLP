'''Mô hình cải tiến: PhoBERT và VisoBERT
văn bản lấy từ cột text_<mô hình> (tiền xử lý riêng từng mô hình)
chạy huấn luyện:
    python -m src.models.train_transformer --model phobert
    python -m src.models.train_transformer --model visobert'''

import argparse
import copy
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.config import (LABEL_COL, LABELS, MODELS_DIR, N_CLASSES, RANDOM_SEED, TEXT_COL, TRANSFORMERS, ensure_dirs, get_transformer_cfg)
from src.data import load_splits
from src.evaluate import compute_metrics, save_predictions, save_report
from src.models.transformer_model import collate_ids, encode_texts, pick_device, predict_probs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def class_weights(y, n_classes, power=1.0):
    '''trọng số lớp = n / (n_lớp * số mẫu của lớp)'''
    counts = np.bincount(y, minlength=n_classes).clip(min=1)
    return len(y) / (n_classes * counts) ** power


def make_optimizer(model, lr, weight_decay):
    
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
            "weight_decay": weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
            "weight_decay": 0.0
        }
    ]
    return torch.optim.AdamW(groups, lr=lr)


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, device, max_grad_norm):
    model.train()
    total, n = 0.0, 0
    use_amp = scaler is not None
    
    for input_ids, mask, labels in loader:
        input_ids, mask, labels = input_ids.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=mask).logits
        loss = loss_fn(logits.float(), labels)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        scheduler.step()
        
        total += loss.item() * len(labels)
        n += len(labels)
        
    return total / max(n,1)


def run(name, args):
    cfg = get_transformer_cfg(name)
    for key in ("epochs", "batch_size", "lr", "max_len", "patience", "class_weight_power"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
            
    t0 = time.time()
    set_seed(RANDOM_SEED)
    ensure_dirs()
    device = pick_device()
    use_amp = device.type == "cuda"
    print(f"[{name}] {cfg['hf_name']} | thiết bị: {device} | cấu hình: "
          f"{ {k: cfg[k] for k in ('max_len', 'batch_size', 'epochs', 'lr')} }")
    assert sorted(LABELS) == list(range(N_CLASSES)), "id nhãn phải liên tục từ 0"
    
    data = load_splits(name, args.limit)
    print("Dữ liệu:", {k: len(v) for k, v in data.items()})
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])
    pad_id = tokenizer.pad_token_id
    ids = {s: encode_texts(tokenizer, df[TEXT_COL], cfg["max_len"]) for s, df in data.items()}
    y = {s: df[LABEL_COL].to_numpy() for s, df in data.items()}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["hf_name"], num_labels=N_CLASSES,
        id2label={i: LABELS[i] for i in LABELS}, 
        label2id={LABELS[i]: i for i in LABELS}
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    
    #dữ liệu huấn luyện:
    train_set = list(zip(ids["train"], y["train"].tolist()))
    def collate(batch):
        input_ids, mask = collate_ids([b[0] for b in batch], pad_id)
        return input_ids, mask, torch.tensor([b[1] for b in batch], dtype=torch.long)

    gen = torch.Generator().manual_seed(RANDOM_SEED)
    loader = DataLoader(
        train_set, 
        batch_size=cfg["batch_size"], 
        shuffle=True, 
        collate_fn=collate, 
        generator=gen
    )
    
    weight = torch.tensor(class_weights(y["train"], N_CLASSES, cfg["class_weight_power"]), dtype=torch.float, device=device) \
        if cfg["class_weight"] else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = make_optimizer(model, cfg["lr"], cfg["weight_decay"])
    total_steps = len(loader) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg["warmup_ratio"] * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # huấn luyện: mỗi epoch đo macro-F1 trên val, giữ trọng số tốt nhất, dừng sớm khi không tiến bộ
    best_f1, best_epoch, best_state, bad, history = -1.0, 0, None, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        t = time.time()
        loss = train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, device, cfg["max_grad_norm"])
        val_probs = predict_probs(model, ids["val"], pad_id, device)
        val_f1 = f1_score(y["val"], val_probs.argmax(axis=1), average="macro", labels=list(range(N_CLASSES)),
                          zero_division=0)
        history.append({"epoch": epoch, "train_loss": round(loss, 4), "val_macro_f1": round(float(val_f1), 4)})
        print(f"  epoch {epoch}/{cfg['epochs']} | loss = {loss:.4f} | val macro-F1 = {val_f1:.4f} ({time.time() - t:.0f}s)")

        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"  dừng sớm (val không tăng sau {cfg['patience']} epoch)")
                break

    model.load_state_dict(best_state)
    print(f"  -> chọn epoch {best_epoch} (val macro-F1 = {best_f1:.4f})")

    # đo trên val và test bằng trọng số tốt nhất, lưu dự đoán từng câu
    results = {}
    for split in ("val", "test"):
        probs = predict_probs(model, ids[split], pad_id, device)
        pred = probs.argmax(axis=1)
        results[split] = compute_metrics(y[split], pred, LABELS)
        save_predictions(name, split, data[split], probs, LABELS, pred_ids=pred)
    print(f"  val  macro-F1 = {results['val']['macro_f1']:.4f} | acc = {results['val']['accuracy']:.4f}")
    print(f"  test macro-F1 = {results['test']['macro_f1']:.4f} | acc = {results['test']['accuracy']:.4f}")

    # lưu model + tokenizer (để TransformerClassifier nạp lại) và báo cáo
    out_dir = MODELS_DIR / name
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    save_report(name, results, LABELS, extra={
        "model": cfg["hf_name"], "best_epoch": best_epoch, "n_params": n_params,
        "seconds": round(time.time() - t0, 1), "history": history,
        "hyperparams": {k: cfg[k] for k in ("max_len", "batch_size", "epochs", "lr", "weight_decay",
                                            "warmup_ratio", "patience", "class_weight", "class_weight_power")},
    })
    print(f"  xong sau {time.time() - t0:.0f}s, lưu tại {out_dir}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(TRANSFORMERS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-len", dest="max_len", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None, help="số epoch chờ trước khi dừng sớm")
    ap.add_argument("--class-weight-power", dest="class_weight_power", type=float, default=None,
                    help="1.0 = trọng số nghịch đảo tần suất, 0.5 = giảm nhẹ, 0 = không dùng trọng số")
    ap.add_argument("--limit", type=int, default=None, help="chỉ dùng N dòng mỗi tập (chạy thử)")
    args = ap.parse_args()
    run(args.model, args)


if __name__ == "__main__":
    main()
