# train_seq_lstm.py
# 움직이는 수어(단어/문장)용 시퀀스 LSTM 학습 스크립트

# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# =============================================================================
# 경로 설정
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE_SEQ = os.path.join(BASE_DIR, "sign_sequences_dualhand.npy")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
MODEL_SEQ     = os.path.join(MODEL_DIR, "lstm_seq.pth")

os.makedirs(MODEL_DIR, exist_ok=True)

# 시퀀스 고정 길이 (프레임 수)
SEQ_LEN      = 20
INPUT_SIZE   = 112   # 손(84) + 포즈 상체(18) + 얼굴 핵심(10)
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2

# =============================================================================
# 시퀀스 → 고정 길이 텐서 (T, 112)
# =============================================================================
def seq_to_tensor(seq: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    seq: (T, 112)  →  return: (seq_len, 112)
    T 프레임에서 seq_len 개를 균등 샘플링
    """
    if seq.ndim != 2 or seq.shape[1] != INPUT_SIZE:
        raise ValueError(f"시퀀스 shape 이상: {seq.shape}, (T,{INPUT_SIZE}) 이어야 함.")
    T = seq.shape[0]
    if T == 0:
        return np.zeros((seq_len, INPUT_SIZE), dtype=np.float32)
    idx = np.linspace(0, T - 1, seq_len).astype(int)
    return seq[idx].astype(np.float32)   # (seq_len, 112)

# =============================================================================
# 데이터 로드
# =============================================================================
def load_data_seq(path: str) -> Dict[str, List[np.ndarray]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"시퀀스 데이터 파일을 찾을 수 없음: {path}")

    obj = np.load(path, allow_pickle=True).item()
    data: Dict[str, List[np.ndarray]] = {}
    for k, v in obj.items():
        seq_list = []
        for arr in v:
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == INPUT_SIZE:
                seq_list.append(arr)
            else:
                print(f"[경고] 라벨 '{k}'의 시퀀스 shape 이상: {arr.shape}, 건너뜀")
        data[k] = seq_list
    return data

# =============================================================================
# Dataset 정의
# =============================================================================
class SeqDataset(Dataset):
    def __init__(self,
                 data_dict: Dict[str, List[np.ndarray]],
                 label_to_idx: Dict[str, int],
                 seq_len: int = SEQ_LEN):
        super().__init__()
        X_list: List[np.ndarray] = []
        y_list: List[int] = []

        for label, idx in label_to_idx.items():
            for seq in data_dict.get(label, []):
                X_list.append(seq_to_tensor(seq, seq_len))   # (seq_len, 84)
                y_list.append(idx)

        if len(X_list) == 0:
            raise RuntimeError("학습에 사용할 시퀀스가 없습니다. 먼저 sign_lstm.py에서 시퀀스를 충분히 수집하세요.")

        self.X = torch.tensor(np.stack(X_list), dtype=torch.float32)  # (N, seq_len, 84)
        self.y = torch.tensor(y_list, dtype=torch.long)

        print(f"[Dataset] 샘플 수: {len(self.X)}, 시퀀스: {self.X.shape[1]}×{self.X.shape[2]}, 클래스: {len(label_to_idx)}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

# =============================================================================
# LSTM 모델
# =============================================================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])   # 마지막 타임스텝
        return self.fc(out)

# =============================================================================
# 학습 루프
# =============================================================================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / total if total > 0 else 0.0

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                val_loss += criterion(logits, y).item() * X.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        scheduler.step()

        print(f"[Epoch {epoch:02d}/{epochs}] "
              f"train_loss={total_loss/total:.4f}, train_acc={train_acc*100:5.1f}% | "
              f"val_acc={val_acc*100:5.1f}%")

# =============================================================================
# 메인
# =============================================================================
def main():
    print(f"[INFO] 시퀀스 데이터 로드: {DATA_FILE_SEQ}")
    data_seq = load_data_seq(DATA_FILE_SEQ)

    non_empty_labels = [k for k, v in data_seq.items() if len(v) > 0]
    if len(non_empty_labels) == 0:
        raise RuntimeError("시퀀스가 저장된 라벨이 없습니다. 먼저 sign_lstm.py에서 동작(시퀀스)을 저장하세요.")
    if len(non_empty_labels) == 1:
        print(f"[경고] 라벨이 1개뿐입니다: {non_empty_labels[0]}")

    label_names = sorted(non_empty_labels)
    label_to_idx = {lb: i for i, lb in enumerate(label_names)}
    num_classes = len(label_names)
    print(f"[라벨] {num_classes}개: {label_names}")

    full_dataset = SeqDataset(data_seq, label_to_idx, seq_len=SEQ_LEN)

    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * 0.2))
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(0)
    )
    print(f"[Split] train={len(train_dataset)} / val={len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

    model = LSTMClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    train_model(model=model, train_loader=train_loader, val_loader=val_loader,
                device=device, epochs=50, lr=1e-3)

    ckpt = {
        "model_type":  "lstm",
        "input_size":  INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers":  NUM_LAYERS,
        "num_classes": num_classes,
        "model_state": model.state_dict(),
        "label_list":  label_names,
        "seq_len":     SEQ_LEN,
    }
    torch.save(ckpt, MODEL_SEQ)
    print(f"[저장 완료] {MODEL_SEQ}")
    print("[DONE] 이제 sign_lstm.py에서 mlp_seq.pth를 로드해서 동작 인식에 사용할 수 있습니다.")

if __name__ == "__main__":
    main()
