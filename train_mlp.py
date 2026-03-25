# train_mlp.py
# sign_coords.npy 를 이용해서 자모 MLP 학습 -> models/mlp_jamo.pth 저장

import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# sign_lstm.py 에서 설정/모델을 가져온다
from sign_lstm import (
    BASE_DIR,
    DATA_FILE_JAMO,
    MODEL_JAMO,
    GESTURES_JAMO,
    MLP,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 12          # extract_jamo_keypoints 출력 차원
BATCH_SIZE = 32
EPOCHS = 60
LR = 1e-3


def load_jamo_dataset(path: str):
    """sign_coords.npy 를 읽어서 (X, y) 텐서로 변환"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 를 찾을 수 없습니다. 먼저 sign_lstm.py에서 자모 데이터를 저장하세요.")

    obj: Dict[str, List[np.ndarray]] = np.load(path, allow_pickle=True).item()

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for idx, label in enumerate(GESTURES_JAMO):
        samples = obj.get(label, [])
        for feat in samples:
            feat = np.asarray(feat, dtype=np.float32)
            if feat.size != INPUT_DIM:
                # 잘못된 크기의 데이터는 버린다.
                continue
            X_list.append(feat)
            y_list.append(idx)

    if not X_list:
        raise RuntimeError("학습에 사용할 자모 샘플이 없습니다. sign_lstm.py에서 데이터를 먼저 수집하세요.")

    X = np.stack(X_list).astype(np.float32)     # [N, 12]
    y = np.array(y_list, dtype=np.int64)        # [N]

    print(f"[데이터] 샘플 수: {len(X_list)}, 입력 차원: {INPUT_DIM}, 클래스 수: {len(GESTURES_JAMO)}")
    return torch.from_numpy(X), torch.from_numpy(y)


def train():
    X, y = load_jamo_dataset(DATA_FILE_JAMO)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True)

    model = MLP(INPUT_DIM, len(GESTURES_JAMO)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            _, pred = logits.max(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

        epoch_loss = running_loss / total
        acc = correct / total
        print(f"[{epoch:03d}/{EPOCHS}] loss={epoch_loss:.4f}, acc={acc:.4f}")

    # 체크포인트 저장
    ckpt = {
        "input_dim": INPUT_DIM,
        "classes": GESTURES_JAMO,
        "model_state": model.state_dict(),
    }
    os.makedirs(os.path.dirname(MODEL_JAMO), exist_ok=True)
    torch.save(ckpt, MODEL_JAMO)
    print(f"[저장 완료] {MODEL_JAMO}")


if __name__ == "__main__":
    print(f"[디바이스] {DEVICE}")
    train()
