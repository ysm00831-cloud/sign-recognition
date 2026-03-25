# inference.py
# 웹앱용 추론 모듈 — sign_lstm.py의 모델/특징추출 로직을 재사용

# -*- coding: utf-8 -*-

import os
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "models")
MODEL_JAMO   = os.path.join(MODEL_DIR, "mlp_jamo.pth")
MODEL_SEQ    = os.path.join(MODEL_DIR, "lstm_seq.pth")

SEQ_LEN      = 20
CONF_THRESH_JAMO = 0.20
CONF_THRESH_SEQ  = 0.50

# =============================================================================
# 자모 목록 (sign_lstm.py 와 동일)
# =============================================================================
CONSONANTS  = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
VOWELS      = list("ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ")
DIPHTHONGS  = ["ㅐ","ㅔ","ㅒ","ㅖ","ㅘ","ㅙ","ㅚ","ㅝ","ㅞ","ㅟ","ㅢ"]
SPECIALS    = ["(띄어쓰기)","(삭제)","(줄바꿈)"]
GESTURES_JAMO = CONSONANTS + VOWELS + DIPHTHONGS + SPECIALS

# 음절 조합 테이블
CHO  = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = ["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ",
        "ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"]
JONG = ["","ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ","ㄻ","ㄼ","ㄽ","ㄾ",
        "ㄿ","ㅀ","ㅁ","ㅂ","ㅄ","ㅅ","ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
CHO_MAP  = {c:i for i,c in enumerate(CHO)}
JUNG_MAP = {c:i for i,c in enumerate(JUNG)}
JONG_MAP = {c:i for i,c in enumerate(JONG)}

# =============================================================================
# 모델 클래스
# =============================================================================
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=0.3 if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))

# =============================================================================
# 모델 로드
# =============================================================================
def load_jamo_model():
    if not os.path.exists(MODEL_JAMO):
        print(f"[경고] 자모 모델 없음: {MODEL_JAMO}")
        return None, []
    ckpt = torch.load(MODEL_JAMO, map_location="cpu")
    label_list = ckpt.get("label_list", GESTURES_JAMO)
    n_cls = ckpt.get("num_classes", len(label_list))
    model = MLP(ckpt.get("input_dim", 12), n_cls)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[자모 모델 로드] {n_cls}클래스")
    return model, label_list

def load_seq_model():
    if not os.path.exists(MODEL_SEQ):
        print(f"[경고] 문장 모델 없음: {MODEL_SEQ}")
        return None, [], SEQ_LEN
    ckpt = torch.load(MODEL_SEQ, map_location="cpu")
    label_list = ckpt.get("label_list", [])
    n_cls = ckpt.get("num_classes", len(label_list))
    seq_len = ckpt.get("seq_len", SEQ_LEN)
    if ckpt.get("model_type") == "lstm":
        model = LSTMClassifier(
            input_size  = ckpt.get("input_size", 84),
            hidden_size = ckpt.get("hidden_size", 128),
            num_layers  = ckpt.get("num_layers", 2),
            num_classes = n_cls,
        )
    else:
        model = MLP(ckpt.get("input_dim", seq_len * 84), n_cls)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[문장 모델 로드] {n_cls}클래스  seq_len={seq_len}")
    return model, label_list, seq_len

# =============================================================================
# 특징 추출 (브라우저 raw 랜드마크 사용)
# raw_lm: [{x, y, z}, ...] 21개 dict 리스트
# =============================================================================
def extract_jamo_feat(raw_lm: list) -> Optional[np.ndarray]:
    """자모용: 손목+5손가락 끝 6점 → 12차원"""
    if not raw_lm or len(raw_lm) < 21:
        return None
    idx = [0, 4, 8, 12, 16, 20]
    pts = np.array([[raw_lm[i]['x'], raw_lm[i]['y']] for i in idx], dtype=np.float32)
    pts -= pts[0]
    scale = float(np.linalg.norm(pts[1])) + 1e-6
    pts /= scale
    return pts.flatten()  # (12,)

def extract_seq_feat(raw_lm_list: list, handedness_list: list) -> Optional[np.ndarray]:
    """문장용: 양손 21점 → 84차원"""
    if not raw_lm_list:
        return None

    left_lm = right_lm = None
    for i, lm in enumerate(raw_lm_list):
        label = handedness_list[i] if i < len(handedness_list) else None
        if label == "Left":
            left_lm = lm
        elif label == "Right":
            right_lm = lm
        else:
            if left_lm is None:
                left_lm = lm
            elif right_lm is None:
                right_lm = lm

    anchor_src = left_lm or right_lm
    if anchor_src is None:
        return None

    anchor = np.array([anchor_src[0]['x'], anchor_src[0]['y']], dtype=np.float32)
    scale = np.linalg.norm(
        np.array([anchor_src[8]['x'], anchor_src[8]['y']], dtype=np.float32) - anchor
    ) + 1e-6

    def hand_feat(lm):
        if lm is None:
            return np.zeros(42, dtype=np.float32)
        pts = np.array([[p['x'], p['y']] for p in lm], dtype=np.float32)
        pts -= anchor
        pts /= scale
        return pts.flatten()

    return np.concatenate([hand_feat(left_lm), hand_feat(right_lm)])  # (84,)

# =============================================================================
# 추론
# =============================================================================
def predict_jamo(model, label_list: list, feat: np.ndarray) -> Tuple[str, float]:
    x = torch.from_numpy(feat).float().unsqueeze(0)
    with torch.no_grad():
        prob = F.softmax(model(x), dim=1).numpy()[0]
    idx = int(prob.argmax())
    conf = float(prob[idx])
    label = label_list[idx] if idx < len(label_list) else "UNKNOWN"
    if conf < CONF_THRESH_JAMO:
        return "UNKNOWN", conf
    return label, conf

def predict_seq(model, label_list: list, seq_frames: List[np.ndarray], seq_len: int) -> Tuple[str, float]:
    if len(seq_frames) < 4:
        return "UNKNOWN", 0.0
    arr = np.array(seq_frames, dtype=np.float32)  # (T, 84)
    idx = np.linspace(0, len(arr) - 1, seq_len).astype(int)
    x = torch.from_numpy(arr[idx]).float().unsqueeze(0)  # (1, seq_len, 84)
    with torch.no_grad():
        if isinstance(model, LSTMClassifier):
            prob = F.softmax(model(x), dim=1).numpy()[0]
        else:
            prob = F.softmax(model(x.view(1, -1)), dim=1).numpy()[0]
    best = int(prob.argmax())
    conf = float(prob[best])
    label = label_list[best] if best < len(label_list) else "UNKNOWN"
    if conf < CONF_THRESH_SEQ:
        return "UNKNOWN", conf
    return label, conf

# =============================================================================
# 자모 → 음절 조합
# =============================================================================
def compose_syllables(jamo_stream: List[str]) -> str:
    out = []
    i = 0
    while i < len(jamo_stream):
        ch = jamo_stream[i]
        if ch in CHO_MAP and i+1 < len(jamo_stream) and jamo_stream[i+1] in JUNG_MAP:
            cho  = CHO_MAP[ch]
            jung = JUNG_MAP[jamo_stream[i+1]]
            i += 2
            jong = 0
            if i < len(jamo_stream) and jamo_stream[i] in JONG_MAP:
                jong = JONG_MAP[jamo_stream[i]]
                i += 1
            out.append(chr(0xAC00 + (cho*21 + jung)*28 + jong))
        else:
            if ch not in CHO_MAP and ch not in JUNG_MAP and ch not in JONG_MAP:
                i += 1
                continue
            out.append(ch)
            i += 1
    return "".join(out)
