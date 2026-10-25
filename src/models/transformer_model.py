'''Nạp mô hình đã fine-tune và tokenizer đã lưu để dự đoán nhãn, tính xác suất cho các câu mới'''

import numpy as np
import torch

def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_texts(tokenizer, texts, max_len):
    '''văn bản -> danh sách id token (cắt ở max_len, chưa padding)'''
    return tokenizer(list(texts), truncation=True, max_length=max_len)["input_ids"]


def collate_ids(ids_list, pad_id):
    '''gom các chuỗi id có độ dài khác nhau thành tensor (padding theo câu dài nhất trong batch)'''
    longest = max(len(x) for x in ids_list)
    input_ids = torch.full((len(ids_list), longest), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids_list), longest), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[i, :len(ids)] = 1
    return input_ids, mask


@torch.no_grad()
def predict_probs(model, ids_list, pad_id, device, batch_size=64):
    '''trả về ma trận xác suất softmax (n_mẫu x n_lớp), đúng thứ tự đầu vào
    sắp theo độ dài để batch ít padding hơn, chạy nhanh hơn'''
    model.eval()
    order = np.argsort([len(x) for x in ids_list])
    out = [None] * len(ids_list)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        input_ids, mask = collate_ids([ids_list[i] for i in idx], pad_id)
        logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        for j, i in enumerate(idx):
            out[i] = probs[j]
    return np.stack(out) if out else np.zeros((0, model.config.num_labels))


class TransformerClassifier:
    '''Nạp mô hình đã fine-tune từ models/<tên>/ để dự đoán thẳng từ chuỗi văn bản'''

    def __init__(self, name, device=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from src.config import MODELS_DIR, TRANSFORMER_COMMON, TRANSFORMERS

        path = MODELS_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Chưa có {path}. Chạy train_transformer.py --model {name} trước")
        self.name = name
        self.max_len = get_transformer_cfg(name)["max_len"]
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        self.classes_ = np.arange(self.model.config.num_labels)

    def predict_proba(self, texts, batch_size=64):
        ids = encode_texts(self.tokenizer, texts, self.max_len)
        return predict_probs(self.model, ids, self.tokenizer.pad_token_id, self.device, batch_size)

    def predict(self, texts, batch_size=64):
        return self.predict_proba(texts, batch_size).argmax(axis=1)
