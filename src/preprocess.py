# làm sạch bảng dữ liệu và chuẩn hóa văn bản 

import json
import os
import re
import unicodedata
from pathlib import Path
import pandas as pd
 
from config import PREPROCESS_PRESETS, LEXICON_DIR


# bước nền: 
BASE_SUBS = [
    (re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\ufe0e\ufe0f]"), ""),            # ký tự ẩn
    (re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"), " "),                       # ký tự điều khiển
    (re.compile(r"https?://\S+|www\.\S+"), " "),                                  # URL
    (re.compile(r"[\w.\-]+@[\w\-]+(?:\.[\w\-]+)+"), " "),                         # email
    (re.compile(r"(?<!\d)(?:\+84|0)(?:[\s.\-]?\d){9}(?!\d)"), " "),               # số điện thoại
    (re.compile(r"@\w+"), " <user> "),                                            # mention
    (re.compile(r"#(\w+)"), r" \1 "),                                             # hashtag: giữ chữ, bỏ dấu #
]
SPACE_RE = re.compile(r"\s+")
LETTER_REPEAT_RE = re.compile(r"([^\W\d_])\1{2,}")        # chữ cái lặp >=3 lần -> 2 ("dmmmm" -> "dmm")
PUNCT_REPEAT_RE = re.compile(r"([!?.,])\1+")              # "!!!" -> "!"
EMOJI_RE = re.compile("[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\u2B50\u2B55]")
EDGE_PUNCT = ".,!?;:\"'()[]{}…-~*"                         # dấu câu dính đầu/cuối token khi tra teencode


#chuẩn hóa vị trí dấu thanh
_TONE = {
    "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
    "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
    "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
}
TONE_MAP = {f(k): f(v) for k, v in _TONE.items() for f in (str, str.capitalize, str.upper)}
TONE_RE = re.compile("|".join(TONE_MAP))

# ký tự trông giống chữ Latinh (Kirin/Hy Lạp) hay bị chèn vào để né bộ lọc: đổi về chữ Latinh
HOMOGLYPH = str.maketrans(
    "аеорсхуіјѕһԁɡАВЕКМНОРСТХУІЈЅΑΒΕΖΗΙΚΜΝΟΡΤΥΧοιν",
    "aeopcxyijshdgABEKMHOPCTXYIJSABEZHIKMNOPTYXoiv",
)



def _load_json(name):
    '''Đọc từ điển trong thư mục lexicon/ (báo lỗi rõ nếu thiếu file)'''
    path = LEXICON_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Thiếu {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
 
TEENCODE = {k.lower(): v for k, v in _load_json("normalization.json").items()}   # "ko" -> "không"
EMOJI_TO_TEXT = _load_json("emoji_vi.json")      



def base_clean(text):
    '''dùng chung mọi mô hình, không hạ chữ thường'''
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return ""
    text = unicodedata.normalize("NFKC", str(text)).translate(HOMOGLYPH)
    for pattern, repl in BASE_SUBS:
        text = pattern.sub(repl, text)
    text = TONE_RE.sub(lambda m: TONE_MAP[m.group(0)], text)
    return SPACE_RE.sub(" ", text).strip()

def merge_repeats(text):
    '''Gộp chữ cái lặp (>=3 lần -> 2) và dấu câu lặp (-> 1)'''
    return PUNCT_REPEAT_RE.sub(r"\1", LETTER_REPEAT_RE.sub(r"\1\1", text))

def normalize_teencode(text):
    def fix(token):
        core = token.strip(EDGE_PUNCT)
        if core and core.lower() in TEENCODE:
            i = token.index(core)
            return token[:i] + TEENCODE[core.lower()] + token[i + len(core):]
        return token
    return " ".join(fix(t) for t in text.split(" "))

def segment_words(text):
    '''Tách từ tiếng việt (cho phobert)
    token <user> được tách ra khỏi câu trước khi tách từ (underthesea sẽ cắt "<user>" thành "< user >"
    và dính vào từ kế bên), rồi đưa vào lại dưới dạng chữ thường "user"'''
    from underthesea import word_tokenize         
    out = []
    for i, part in enumerate(text.split("<user>")):
        if i > 0:
            out.append("user")
        part = part.strip()
        if part:
            out.append(word_tokenize(part, format="text"))
    return " ".join(out)

def convert_emoji(text, mode):
    '''Xử lý emoji theo preset: 
    - token = tách riêng bằng khoảng trắng
    - mtext = đổi sang chữ (lạ thì bỏ)
    - keep = giữ nguyên'''
    if mode == "token":
        return EMOJI_RE.sub(lambda m: f" {m.group(0)} ", text)
    if mode == "text":
        return EMOJI_RE.sub(lambda m: f" {EMOJI_TO_TEXT.get(m.group(0), '')} ", text)
    if mode == "keep":
        return text
    raise ValueError(f"emoji mode không hợp lệ: {mode}")
 
 
 

def clean_text(text, model="baseline"):
    '''làm sạch dữ liệu theo mô hình baseline / phobert/visobert'''
    if model not in PREPROCESS_PRESETS:
        raise ValueError(f"Không có preset '{model}'. Các preset hiện có: {list(PREPROCESS_PRESETS)}")
    cfg = PREPROCESS_PRESETS[model]
 
    text = base_clean(text)
    if cfg["lower"]:
        text = text.lower()
    if cfg["repeat"]:
        text = merge_repeats(text)
    if cfg["teencode"]:
        text = normalize_teencode(text)
    text = SPACE_RE.sub(" ", convert_emoji(text, cfg["emoji"])).strip()
    if cfg["segment"] and text:
        text = segment_words(text)
    return text