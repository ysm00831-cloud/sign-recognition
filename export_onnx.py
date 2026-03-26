# export_onnx.py
# PyTorch 모델을 ONNX 형식으로 변환

# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
WEB_DIR   = MODEL_DIR  # ONNX 파일을 models/ 에 바로 저장
os.makedirs(WEB_DIR, exist_ok=True)

# =============================================================================
# 모델 클래스 (sign_mlp.py 와 동일)
# =============================================================================
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_classes)
    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

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
# 자모 MLP 변환
# =============================================================================
def export_jamo():
    path = os.path.join(MODEL_DIR, "mlp_jamo.pth")
    if not os.path.exists(path):
        print(f"[건너뜀] 자모 모델 없음: {path}")
        return None

    ckpt = torch.load(path, map_location="cpu")
    label_list = ckpt.get("label_list") or ckpt.get("classes", [])
    n_cls      = ckpt.get("num_classes", len(label_list))
    if n_cls <= 0:
        n_cls = len(label_list)
    input_dim  = ckpt.get("input_dim", 12)

    model = MLP(input_dim, n_cls)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.zeros(1, input_dim)
    out_path = os.path.join(WEB_DIR, "jamo.onnx")
    torch.onnx.export(model, dummy, out_path,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}},
                      opset_version=11,
                      dynamo=False)
    print(f"[완료] 자모 모델 → {out_path}  ({n_cls}클래스)")
    return label_list

# =============================================================================
# 문장 LSTM 변환
# =============================================================================
def export_seq():
    path = os.path.join(MODEL_DIR, "lstm_seq.pth")
    if not os.path.exists(path):
        print(f"[건너뜀] 문장 모델 없음: {path}")
        return None, 20

    ckpt = torch.load(path, map_location="cpu")
    label_list  = ckpt.get("label_list", [])
    n_cls       = ckpt.get("num_classes", len(label_list))
    seq_len     = ckpt.get("seq_len", 20)

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

    input_size = ckpt.get("input_size", 112)
    dummy    = torch.zeros(1, seq_len, input_size)
    out_path = os.path.join(WEB_DIR, "seq.onnx")
    torch.onnx.export(model, dummy, out_path,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}},
                      opset_version=11,
                      dynamo=False)
    print(f"[완료] 문장 모델 → {out_path}  ({n_cls}클래스, seq_len={seq_len})")
    return label_list, seq_len

# =============================================================================
# 라벨 JSON 저장
# =============================================================================
import json

def save_labels(jamo_labels, seq_labels, seq_len):
    data = {
        "jamo": jamo_labels or [],
        "seq":  seq_labels  or [],
        "seq_len": seq_len,
    }
    out_path = os.path.join(WEB_DIR, "labels.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[완료] 라벨 저장 → {out_path}")

# =============================================================================
# 실행
# =============================================================================
if __name__ == "__main__":
    print("=== ONNX 변환 시작 ===")
    jamo_labels          = export_jamo()
    seq_labels, seq_len  = export_seq()
    save_labels(jamo_labels, seq_labels, seq_len)
    print("\n=== 완료 ===")
    print(f"변환된 파일 위치: {WEB_DIR}")
