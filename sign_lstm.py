# sign_lstm.py
# 통합 수화 인식기 (자모 MLP + 문장 LSTM, Holistic 버전)

# -*- coding: utf-8 -*-

import os, warnings, logging

# ── 경고 억제 (가장 먼저 실행) ──────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "3"   # TensorFlow C++ 로그 끔
os.environ["TF_ENABLE_ONEDNN_OPTS"]     = "0"   # oneDNN 알림 끔
os.environ["ABSL_MIN_LOG_LEVEL"]        = "3"   # absl 로그 끔
os.environ["GLOG_minloglevel"]          = "3"   # glog 끔
os.environ["MEDIAPIPE_DISABLE_GPU"]     = "1"   # GPU 관련 경고 억제
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

warnings.filterwarnings("ignore")              # Python warnings 전체 끔

logging.disable(logging.CRITICAL)             # Python logging 전체 끔

import time, tempfile, threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import pyttsx3, winsound
try:
    import mediapipe.framework.formats.landmark_pb2 as landmark_pb2
except (ModuleNotFoundError, ImportError):
    from typing import Any as _Any
    class _LandmarkPb2Stub:
        NormalizedLandmarkList = _Any
    landmark_pb2 = _LandmarkPb2Stub()

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 환경 & 설정
# =============================================================================
CAM_INDEX = 0  # 필요하면 0/1 바꿔서 사용

KOREAN_FONT = r"C:\Windows\Fonts\malgun.ttf"
WINDOW_TITLE = "통합 수화 인식기 (MLP 버전)"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE_JAMO = os.path.join(BASE_DIR, "sign_coords.npy")             # 자모
DATA_FILE_SEQ  = os.path.join(BASE_DIR, "sign_sequences_dualhand.npy") # 동작 시퀀스

MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_JAMO = os.path.join(MODEL_DIR, "mlp_jamo.pth")
MODEL_SEQ  = os.path.join(MODEL_DIR, "lstm_seq.pth")

os.makedirs(MODEL_DIR, exist_ok=True)

# ----- 라벨 정의 -----
CONSONANTS = ["ㄱ","ㄴ","ㄷ","ㄹ","ㅁ","ㅂ","ㅅ","ㅇ",
              "ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
VOWELS     = ["ㅏ","ㅑ","ㅓ","ㅕ","ㅗ","ㅛ","ㅜ","ㅠ","ㅡ","ㅣ"]
DIPHTHONGS = ["ㅐ","ㅔ","ㅒ","ㅖ","ㅘ","ㅙ","ㅚ","ㅝ","ㅞ","ㅟ","ㅢ"]

SPECIALS   = ["(띄어쓰기)", "(삭제)", "(줄바꿈)"]

# 자모(정지 포즈)
GESTURES_JAMO = CONSONANTS + VOWELS + DIPHTHONGS + SPECIALS

# 움직이는 동작(문장/단어) 라벨
SENTENCES = ["안녕하세요", "감사합니다", "반갑습니다", "만나서", "제", "이름은","입니다"]
GESTURES_SEQ = SENTENCES + SPECIALS

# UI에서 사용할 전체 라벨
ALL_LABELS = list(dict.fromkeys(GESTURES_JAMO + GESTURES_SEQ))

# ----- 임계값 / dwell / 시퀀스 설정 -----
CONF_THRESH_JAMO = 0.20   # 자모 softmax 확률 임계
CONF_THRESH_SEQ  = 0.50   # 동작 softmax 확률 임계

DWELL_SEC    = 0.6        # 같은 자모 라벨을 이 시간 이상 유지하면 커밋
COOLDOWN_SEC = 0.7        # 한 번 커밋 후 같은 라벨 재커밋까지 대기

MIN_SEQ_FRAMES = 12       # 동작 시퀀스로 인정할 최소 프레임 수
SEQ_LEN        = 20       # ★ train_seq_mlp.py의 SEQ_LEN과 반드시 동일해야 함 ★

RESULT_Y_OFFSET = -50

SCALE_MIN, SCALE_MAX, SCALE_STEP = 0.50, 2.00, 0.10
SCALE_DEFAULT = 1.00
RES_PRESETS: List[Tuple[int,int]] = [(640,480), (1280,720), (1920,1080)]

# ====== TTS 관련 ======
TTS_ENABLED   = False
SPEAK_UNKNOWN = False
REPEAT_PERIOD = 1.0
VOICE_RATE, VOICE_VOLUME = 180, 1.0

READABLE = {
    "ㄱ":"기역","ㄴ":"니은","ㄷ":"디귿","ㄹ":"리을","ㅁ":"미음","ㅂ":"비읍","ㅅ":"시옷",
    "ㅇ":"이응","ㅈ":"지읒","ㅊ":"치읓","ㅋ":"키읔","ㅌ":"티읕","ㅍ":"피읖","ㅎ":"히읗",
    "ㅏ":"아","ㅑ":"야","ㅓ":"어","ㅕ":"여","ㅗ":"오","ㅛ":"요","ㅜ":"우","ㅠ":"유","ㅡ":"으","ㅣ":"이",
    "ㅐ":"애","ㅔ":"에","ㅒ":"얘","ㅖ":"예","ㅘ":"와","ㅙ":"왜","ㅚ":"외","ㅝ":"워","ㅞ":"웨","ㅟ":"위","ㅢ":"의",
    "(띄어쓰기)":"띄어쓰기","(삭제)":"삭제","(줄바꿈)":"줄바꿈",
    "UNKNOWN":"알 수 없음"
}
for s in SENTENCES:
    READABLE[s] = s

SHOW_HELP = False

# ====== WAV 캐시 & TTS 함수 ======
CACHE_DIR = os.path.join(tempfile.gettempdir(), "sign_tts_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def safe_name(text):
    return "tts_" + "_".join(f"{ord(ch):04x}" for ch in text) + ".wav"

def ensure_wav_for(text: str) -> str:
    wav = os.path.join(CACHE_DIR, safe_name(text))
    if os.path.isfile(wav):
        return wav
    try:
        engine = pyttsx3.init(driverName='sapi5')
        for v in engine.getProperty('voices'):
            if "korean" in (v.name or "").lower() or "ko_" in (v.id or "").lower():
                engine.setProperty('voice', v.id); break
        engine.setProperty('rate', VOICE_RATE)
        engine.setProperty('volume', VOICE_VOLUME)
        engine.save_to_file(text, wav)
        engine.runAndWait()
    except Exception as e:
        print(f"[TTS] WAV 생성 실패({text}): {e}")
    return wav

def speak_label(label: Optional[str]):
    if not TTS_ENABLED or label is None:
        return
    if label == "UNKNOWN" and not SPEAK_UNKNOWN:
        return
    text = READABLE.get(label, label)
    wav = ensure_wav_for(text)
    try:
        winsound.PlaySound(None, 0)
        winsound.PlaySound(wav, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception as e:
        print(f"[TTS] 재생 실패: {e}")

def speak_last_word(committed: List[str]):
    """
    committed 리스트에서 마지막 '단어'를 찾아 읽어준다.
    - 공백/줄바꿈 토큰(" ", "\\n")은 건너뜀
    - 문장 동작으로 이미 읽어준 경우라도, 여기서는 그냥 문자열 그대로 읽는다.
      호출하는 쪽에서 자모로 만들어진 단어일 때만 부르도록 제어한다.
    """
    if not TTS_ENABLED:
        return
    if not committed:
        return

    # 뒤에서부터 공백 아닌 토큰 찾기
    i = len(committed) - 1
    while i >= 0 and committed[i] in (" ", "\n"):
        i -= 1
    if i < 0:
        return

    text = committed[i]
    if not text or not text.strip():
        return

    wav = ensure_wav_for(text)
    try:
        winsound.PlaySound(None, 0)
        winsound.PlaySound(wav, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception as e:
        print(f"[TTS] 단어 재생 실패: {e}")

# ====== MediaPipe ======
mp_hands    = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_pose     = mp.solutions.pose
mp_draw     = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# =============================================================================
# 그리기 함수
# =============================================================================
def draw_korean_text(frame, text, xy, font_size=20, color=(0,0,0)):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(KOREAN_FONT, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(xy, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def draw_korean_text_bottom_right(frame, text,
                                  margin=(40,30), font_size=80,
                                  color=(0,0,0), y_offset=0):
    h,w = frame.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(KOREAN_FONT, font_size)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0,0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x = w - margin[0] - tw
    y = h - margin[1] - th + y_offset
    draw.text((x,y), text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def draw_bottom_left_panel(frame, label_text, sentence_text,
                           font_size_label=72, font_size_sentence=26,
                           margin=16, pad=12, alpha=0.70):
    """왼쪽 하단에 어두운 반투명 배경 박스 + 흰 글씨로 라벨·문장 표시."""
    h, w = frame.shape[:2]
    try:
        font_lbl = ImageFont.truetype(KOREAN_FONT, font_size_label)
        font_sen = ImageFont.truetype(KOREAN_FONT, font_size_sentence)
    except:
        font_lbl = ImageFont.load_default()
        font_sen = font_lbl

    dummy = Image.new("RGB", (1, 1))
    dd = ImageDraw.Draw(dummy)

    lbl_bbox = dd.textbbox((0, 0), label_text, font=font_lbl) if label_text else (0,0,0,0)
    sen_bbox = dd.textbbox((0, 0), sentence_text, font=font_sen) if sentence_text else (0,0,0,0)

    lbl_w = lbl_bbox[2] - lbl_bbox[0]
    lbl_h = lbl_bbox[3] - lbl_bbox[1]
    sen_w = sen_bbox[2] - sen_bbox[0]
    sen_h = sen_bbox[3] - sen_bbox[1]

    box_w = max(lbl_w, sen_w, 100) + pad * 2
    box_h = (lbl_h if label_text else 0) + (sen_h + pad if sentence_text else 0) + pad * 2
    if box_h <= pad * 2:
        return frame

    x1 = margin
    y2 = h - margin
    y1 = y2 - box_h
    x2 = x1 + box_w

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    y = y1 + pad
    if label_text:
        draw.text((x1 + pad, y), label_text, font=font_lbl, fill=(255, 255, 255))
        y += lbl_h + pad
    if sentence_text:
        draw.text((x1 + pad, y), sentence_text, font=font_sen, fill=(220, 220, 220))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def draw_panel(frame, lines,
               font_size=16,
               color=(180,255,180),
               bg_color=(0,0,0),
               alpha=0.40,
               margin=10,
               line_gap=20):
    if not lines:
        return frame
    h, w = frame.shape[:2]
    try:
        font = ImageFont.truetype(KOREAN_FONT, font_size)
    except:
        font = ImageFont.load_default()

    # 박스 크기 계산 (더미 이미지로 측정)
    dummy = Image.new("RGB", (1, 1))
    ddraw = ImageDraw.Draw(dummy)
    max_line_w = max((ddraw.textbbox((0,0), t, font=font)[2] for t in lines), default=0)
    box_w = min(max_line_w + margin * 2, w - margin * 2)
    box_h = margin * 2 + line_gap * len(lines)
    x2 = w - margin
    x1 = x2 - box_w
    y1 = margin
    y2 = y1 + box_h

    # 배경 반투명 처리
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # PIL 변환 1회만 수행하여 모든 텍스트 그리기
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    fill = (color[2], color[1], color[0])
    y = y1 + margin
    for text in lines:
        draw.text((x1 + margin, y), text, font=font, fill=fill)
        y += line_gap
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# =============================================================================
# 샘플 수 요약
# =============================================================================
def _group_counts_str(data:Dict[str,List[np.ndarray]], labels:List[str], line_break:int=10)->List[str]:
    parts = [f"{lb}({len(data.get(lb, []))})" for lb in labels]
    lines = []
    for i in range(0, len(parts), line_break):
        lines.append(" ".join(parts[i:i+line_break]))
    return lines

def counts_summary_lines(data_jamo, data_seq)->List[str]:
    total_jamo = sum(len(v) for v in data_jamo.values())
    used_jamo  = sum(1 for v in data_jamo.values() if len(v) > 0)
    total_seq  = sum(len(v) for v in data_seq.values())
    used_seq   = sum(1 for v in data_seq.values() if len(v) > 0)
    lines = [
        f"[자모 샘플] 총 {total_jamo}개 | 라벨 사용 {used_jamo}/{len(data_jamo)}",
        "— 자음 —"
    ]
    lines += _group_counts_str(data_jamo, CONSONANTS)
    lines += ["— 모음 —"] + _group_counts_str(data_jamo, VOWELS)
    lines += ["— 이중모음 —"] + _group_counts_str(data_jamo, DIPHTHONGS)
    lines += ["— 자모 편집(제스처) —"] + _group_counts_str(data_jamo, SPECIALS, 10)

    all_seq_labels = sorted(k for k in data_seq if k not in SPECIALS and len(data_seq[k]) > 0)
    lines += [
        f"[동작 샘플] 총 {total_seq}개 | 라벨 사용 {used_seq}/{len(data_seq)}",
        "— 문장/단어 동작 —"
    ]
    lines += _group_counts_str(data_seq, all_seq_labels, 6)
    lines += ["— 동작 편집(제스처) —"] + _group_counts_str(data_seq, SPECIALS, 10)
    return lines

# =============================================================================
# 데이터 IO
# =============================================================================
def load_data_jamo(path:str)->Dict[str,List[np.ndarray]]:
    store = {g:[] for g in GESTURES_JAMO}
    if os.path.exists(path):
        try:
            obj = np.load(path, allow_pickle=True).item()
            for k,v in obj.items():
                if k in store:
                    store[k] = [np.array(a, dtype=np.float32) for a in v]
            print(f"[자모 로드] {path}")
        except Exception as e:
            print(f"[자모 로드 실패] {e}")
    return store

def save_data_jamo(path:str, data:Dict[str,List[np.ndarray]]):
    np.save(path, data)
    print(f"[자모 저장] {path}")

def load_data_seq(path:str)->Dict[str,List[np.ndarray]]:
    store = {g:[] for g in GESTURES_SEQ}
    if os.path.exists(path):
        try:
            obj = np.load(path, allow_pickle=True).item()
            for k,v in obj.items():
                store[k] = [np.array(a, dtype=np.float32) for a in v]
            print(f"[시퀀스 로드] {path}")
        except Exception as e:
            print(f"[시퀀스 로드 실패] {e}")
    return store

def save_data_seq(path:str, data:Dict[str,List[np.ndarray]]):
    np.save(path, data)
    print(f"[시퀀스 저장] {path}")

# =============================================================================
# 특징 & MLP 입력
# =============================================================================
def extract_jamo_keypoints(landmarks)->np.ndarray:
    """정지 자모용: 손목+5손가락 끝 6점 → 12차원"""
    idx = [0,4,8,12,16,20]
    pts = np.array([[landmarks.landmark[i].x, landmarks.landmark[i].y] for i in idx],
                   dtype=np.float32)
    pts -= pts[0]
    scale = float(np.linalg.norm(pts[1])) + 1e-6
    pts /= scale
    return pts.flatten()  # (12,)

def extract_seq_keypoints(results,
                          hand_landmarks_list: List
                          )->Optional[np.ndarray]:
    """양손 동작용: 21점×2 손 → 84차원"""
    if not hand_landmarks_list:
        return None

    left_hand_lm = None
    right_hand_lm = None

    for i, hand_lm in enumerate(hand_landmarks_list):
        handedness = None
        if results.multi_handedness and i < len(results.multi_handedness):
            handedness = results.multi_handedness[i].classification[0].label
        if handedness == "Left":
            left_hand_lm = hand_lm
        elif handedness == "Right":
            right_hand_lm = hand_lm
        else:
            if left_hand_lm is None:
                left_hand_lm = hand_lm
            elif right_hand_lm is None:
                right_hand_lm = hand_lm

    lm_source = left_hand_lm or right_hand_lm
    if lm_source is None:
        return None

    anchor = np.array([lm_source.landmark[0].x, lm_source.landmark[0].y], dtype=np.float32)
    scale = np.linalg.norm(
        np.array([lm_source.landmark[8].x, lm_source.landmark[8].y], dtype=np.float32) - anchor
    ) + 1e-6

    def hand_feat(hand_lm):
        num = 21
        if hand_lm is None:
            return np.zeros(num*2, dtype=np.float32)
        pts = np.array([[hand_lm.landmark[i].x, hand_lm.landmark[i].y]
                        for i in range(num)], dtype=np.float32)
        pts -= anchor
        pts /= scale
        return pts.flatten()

    left_feat  = hand_feat(left_hand_lm)
    right_feat = hand_feat(right_hand_lm)
    return np.concatenate([left_feat, right_feat])  # (84,)

def seq_to_fixed(seq: np.ndarray, seq_len:int = SEQ_LEN) -> np.ndarray:
    """
    가변 길이 시퀀스(프레임×84)를 seq_len 프레임으로 리샘플/패딩하고 1D로 펼침.
    train_seq_mlp.py와 동일한 방식이어야 함.
    """
    T, D = seq.shape
    if T >= seq_len:
        idx = np.linspace(0, T-1, seq_len).astype(int)
        fixed = seq[idx]
    else:
        pad = np.repeat(seq[-1:], seq_len-T, axis=0)
        fixed = np.concatenate([seq, pad], axis=0)
    return fixed.reshape(-1).astype(np.float32)  # (seq_len*D,)

def extract_seq_keypoints_holistic(result) -> Optional[np.ndarray]:
    """
    Holistic result → 112차원 특징 벡터
      - 손(양): 21점×2손×2좌표 = 84
      - 포즈 상체: 9점×2좌표    = 18  (코/어깨/팔꿈치/손목/엉덩이)
      - 얼굴 핵심: 5점×2좌표    = 10  (코끝/눈2/입꼬리2)
    정규화: 어깨 중점 기준, 어깨 너비로 스케일
    """
    # 손이 하나도 없으면 의미 없는 프레임
    left_lm  = result.left_hand_landmarks
    right_lm = result.right_hand_landmarks
    if left_lm is None and right_lm is None:
        return None

    pose_lm = result.pose_landmarks
    face_lm = result.face_landmarks

    # ── 정규화 기준: 포즈 어깨 중점 + 어깨 너비 ──
    if pose_lm is not None:
        ls = np.array([pose_lm.landmark[11].x, pose_lm.landmark[11].y], dtype=np.float32)
        rs = np.array([pose_lm.landmark[12].x, pose_lm.landmark[12].y], dtype=np.float32)
        anchor = (ls + rs) * 0.5
        scale  = float(np.linalg.norm(ls - rs)) + 1e-6
    else:
        src = left_lm or right_lm
        anchor = np.array([src.landmark[0].x, src.landmark[0].y], dtype=np.float32)
        scale  = float(np.linalg.norm(
            np.array([src.landmark[8].x, src.landmark[8].y], dtype=np.float32) - anchor)) + 1e-6

    def norm_pts(pts: np.ndarray) -> np.ndarray:
        return ((pts - anchor) / scale).flatten()

    # ── 손 특징 (84) ──
    def hand_feat(lm):
        if lm is None:
            return np.zeros(42, dtype=np.float32)
        pts = np.array([[lm.landmark[i].x, lm.landmark[i].y] for i in range(21)], dtype=np.float32)
        return norm_pts(pts)

    hand_vec = np.concatenate([hand_feat(left_lm), hand_feat(right_lm)])  # (84,)

    # ── 포즈 상체 특징 (18): 코0, 왼어깨11, 오른어깨12, 왼팔꿈치13, 오른팔꿈치14,
    #                         왼손목15, 오른손목16, 왼엉덩이23, 오른엉덩이24
    POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]
    if pose_lm is not None:
        pts = np.array([[pose_lm.landmark[i].x, pose_lm.landmark[i].y]
                        for i in POSE_IDX], dtype=np.float32)
        pose_vec = norm_pts(pts)  # (18,)
    else:
        pose_vec = np.zeros(18, dtype=np.float32)

    # ── 얼굴 핵심 특징 (10): 코끝1, 오른눈안쪽133, 왼눈안쪽362, 오른입꼬리61, 왼입꼬리291
    FACE_IDX = [1, 133, 362, 61, 291]
    if face_lm is not None:
        pts = np.array([[face_lm.landmark[i].x, face_lm.landmark[i].y]
                        for i in FACE_IDX], dtype=np.float32)
        face_vec = norm_pts(pts)  # (10,)
    else:
        face_vec = np.zeros(10, dtype=np.float32)

    return np.concatenate([hand_vec, pose_vec, face_vec])  # (112,)

def apply_resolution(cap:cv2.VideoCapture, wh:Tuple[int,int])->Tuple[int,int]:
    w, h = wh
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (aw,ah)!=(w,h):
        print(f"[경고] 요청 {w}x{h} → 실제 {aw}x{ah}")
    else:
        print(f"[해상도 적용] {aw}x{ah}")
    return aw, ah

# =============================================================================
# 자모 → 음절 합성
# =============================================================================
CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = ["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ",
        "ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"]
JONG = ["","ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ","ㄻ","ㄼ","ㄽ","ㄾ",
        "ㄿ","ㅀ","ㅁ","ㅂ","ㅄ","ㅅ","ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]

CHO_MAP  = {c:i for i,c in enumerate(CHO)}
JUNG_MAP = {c:i for i,c in enumerate(JUNG)}
JONG_MAP = {c:i for i,c in enumerate(JONG)}

def compose_syllables(jamo_stream: List[str]) -> str:
    """
    자모 스트림을 한국어 IME 방식으로 실시간 조합.
      ["ㄱ","ㅏ","ㄴ","ㅏ"] → "가나"
      ["ㄱ","ㅏ","ㄴ"]     → "간"
      ["ㄱ","ㅏ"]          → "가"
      ["ㄱ"]               → "ㄱ"
    """
    result: List[str] = []
    cho_i:  Optional[int] = None   # CHO 인덱스
    jung_i: Optional[int] = None   # JUNG 인덱스
    jong_i: Optional[int] = None   # JONG 인덱스 (1이상)

    def _flush():
        nonlocal cho_i, jung_i, jong_i
        if cho_i is None:
            pass
        elif jung_i is None:
            result.append(CHO[cho_i])
        else:
            result.append(chr(0xAC00 + (cho_i * 21 + jung_i) * 28 + (jong_i or 0)))
        cho_i = jung_i = jong_i = None

    for jamo in jamo_stream:
        if jamo in JUNG_MAP:            # ── 모음 입력 ──
            j = JUNG_MAP[jamo]
            if cho_i is None:
                # 초성 없이 모음 단독
                result.append(jamo)
            elif jung_i is None:
                # 초성 + 모음 결합
                jung_i = j
            elif jong_i is None:
                # (초성+중성) + 모음 → 현재 글자 commit 후 모음 단독
                _flush()
                result.append(jamo)
            else:
                # (초성+중성+종성) + 모음 → 종성을 다음 초성으로 분리
                prev_jong_char = JONG[jong_i]
                jong_i = None
                _flush()                       # 종성 없이 commit
                if prev_jong_char in CHO_MAP:
                    cho_i  = CHO_MAP[prev_jong_char]
                    jung_i = j
                else:
                    # 복합 종성(ㄳ 등) — 그대로 표시 후 모음 단독
                    result.append(prev_jong_char)
                    result.append(jamo)

        elif jamo in CHO_MAP:           # ── 자음 입력 ──
            c = CHO_MAP[jamo]
            if cho_i is None:
                cho_i = c
            elif jung_i is None:
                # 초성만 있는데 새 자음 → 이전 초성 단독 commit
                _flush()
                cho_i = c
            elif jong_i is None:
                # (초성+중성) + 자음 → 종성 후보
                if jamo in JONG_MAP:
                    jong_i = JONG_MAP[jamo]
                else:
                    _flush()
                    cho_i = c
            else:
                # (초성+중성+종성) + 자음 → 현재 글자 commit, 새 초성
                _flush()
                cho_i = c

    _flush()
    return "".join(result)

# =============================================================================
# 편집 제스처
# =============================================================================
def apply_edit_gesture(label:str, committed:List[str], jamo_buf:List[str]):
    if label == "(띄어쓰기)":
        if jamo_buf:
            committed.append(compose_syllables(jamo_buf))
            jamo_buf.clear()
        committed.append(" ")
    elif label == "(삭제)":
        if jamo_buf:
            jamo_buf.pop()
        elif committed:
            last = committed[-1]
            if isinstance(last,str) and len(last)>1:
                committed[-1] = last[:-1]
                if committed[-1] == "":
                    committed.pop()
            else:
                committed.pop()
    elif label == "(줄바꿈)":
        if jamo_buf:
            committed.append(compose_syllables(jamo_buf))
            jamo_buf.clear()
        committed.append("\n")

# =============================================================================
# MLP 모델 정의 & 로드
# =============================================================================
class MLP(nn.Module):
    def __init__(self, input_dim:int, num_classes:int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LSTMClassifier(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int, num_classes:int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=0.3 if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

def _try_load_state(model, ckpt, model_kind:str, num_classes:int):
    """state_dict 구조가 안 맞으면 예외 대신 경고만 찍고 None 반환용"""
    try:
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"[MLP {model_kind} 로드 완료] 입력 {ckpt.get('input_dim')}  클래스 {num_classes}")
        return model
    except Exception as e:
        print(f"[경고] {model_kind} MLP 모델 파라미터 로드 실패 (구조 불일치 가능)")
        print(f"       상세: {e}")
        print("       → models 폴더의 해당 .pth 파일을 삭제하고 다시 학습(train_mlp.py / train_seq_mlp.py)하세요.")
        return None

def load_mlp_jamo(model_path:str, num_classes:int)->Optional[MLP]:
    if not os.path.exists(model_path):
        print(f"[경고] 자모 MLP 모델 파일을 찾을 수 없습니다: {model_path}")
        print("       먼저 train_mlp.py 를 실행해서 자모 모델을 학습/저장하세요.")
        return None
    ckpt = torch.load(model_path, map_location="cpu")
    input_dim = ckpt.get("input_dim", 12)
    n_cls = ckpt.get("num_classes", num_classes)
    if n_cls <= 0:
        n_cls = num_classes
    model = MLP(input_dim, n_cls)
    return _try_load_state(model, ckpt, "자모", n_cls)

def load_mlp_seq(model_path:str):
    """
    동작용 MLP 모델과 label_list, seq_len을 함께 로드
    반환: (model or None, label_list(list[str]), seq_len(int))
    """
    if not os.path.exists(model_path):
        print(f"[경고] 동작 MLP 모델 파일을 찾을 수 없습니다: {model_path}")
        print("       먼저 train_seq_mlp.py 를 실행해서 동작 모델을 학습/저장하세요.")
        return None, [], SEQ_LEN

    ckpt = torch.load(model_path, map_location="cpu")
    label_list = ckpt.get("label_list", [])
    n_cls = ckpt.get("num_classes", len(label_list))
    if n_cls <= 0 and len(label_list) > 0:
        n_cls = len(label_list)

    if ckpt.get("model_type") == "lstm":
        model = LSTMClassifier(
            input_size  = ckpt.get("input_size",  112),
            hidden_size = ckpt.get("hidden_size", 128),
            num_layers  = ckpt.get("num_layers",  2),
            num_classes = n_cls,
        )
    else:
        input_dim = ckpt.get("input_dim", SEQ_LEN*84)
        model = MLP(input_dim, n_cls)
    model = _try_load_state(model, ckpt, "동작", n_cls)

    seq_len_from_model = ckpt.get("seq_len", SEQ_LEN)
    if seq_len_from_model != SEQ_LEN:
        print(f"[참고] 모델이 학습된 SEQ_LEN={seq_len_from_model}, 현재 코드 SEQ_LEN={SEQ_LEN}")
        print("       → train_seq_mlp.py와 sign_mlp.py의 SEQ_LEN을 맞추는 게 좋습니다.")

    if not label_list:
        print("[경고] ckpt에 label_list가 없어, 코드에 정의된 SENTENCES 순서를 그대로 사용합니다.")
        label_list = SENTENCES.copy()

    return model, label_list, seq_len_from_model

# =============================================================================
# 스레드 기반 카메라 캡처 (끊김 방지)
# =============================================================================
class ThreadedCapture:
    """별도 스레드에서 프레임을 계속 읽어 최신 프레임을 유지합니다."""
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self._frame = None
        self._ok = False
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while self._running:
            ok, frame = self.cap.read()
            with self._lock:
                self._ok = ok
                self._frame = frame

    def read(self):
        with self._lock:
            return self._ok, (self._frame.copy() if self._frame is not None else None)

    def stop(self):
        self._running = False

# =============================================================================
# 메인 루프
# =============================================================================
def main():
    global SHOW_HELP, TTS_ENABLED, SPEAK_UNKNOWN

    HELP_TITLE = "수화 인식기 — 도움말/상태"
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1280, 720)
    cv2.namedWindow(HELP_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(HELP_TITLE, 420, 600)
    cv2.moveWindow(HELP_TITLE, 1300, 0)

    pip_swapped = [False]  # False: 메인=관절, PiP=클린 / True: 반대
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pip_swapped[0] = not pip_swapped[0]
    cv2.setMouseCallback(WINDOW_TITLE, on_mouse)

    fullscreen = False
    scale = SCALE_DEFAULT
    show_counts = False

    # 인식 모드: both / jamo / seq
    recog_modes = ["both", "jamo", "seq"]
    recog_mode_idx = 0
    def recog_mode_str():
        return recog_modes[recog_mode_idx]

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ 웹캠 열기 실패")
        return

    data_jamo = load_data_jamo(DATA_FILE_JAMO)
    data_seq  = load_data_seq(DATA_FILE_SEQ)

    # 시작 시 저장된 단어 목록 출력
    print("\n" + "="*55)
    print("  저장된 동작(단어) 샘플 목록")
    print("="*55)
    seq_with_data = {k:v for k,v in data_seq.items() if len(v)>0}
    if seq_with_data:
        for i, (lbl, seqs) in enumerate(sorted(seq_with_data.items())):
            print(f"  {lbl:<14} {len(seqs):>3}개", end="   ")
            if (i+1) % 3 == 0:
                print()
        print()
    else:
        print("  (저장된 동작 샘플 없음)")
    print(f"  총 {len(seq_with_data)}개 단어 / {sum(len(v) for v in seq_with_data.values())}개 샘플")
    print("="*55 + "\n")

    mlp_jamo = load_mlp_jamo(MODEL_JAMO, len(GESTURES_JAMO))
    mlp_seq, seq_label_list, seq_len_model = load_mlp_seq(MODEL_SEQ)

    mode = "인식"   # "학습" / "인식"
    cur_idx = 0

    jamo_buf:  List[str] = []
    committed: List[str] = []

    # 자모 dwell 상태
    last_jamo_label = None
    last_change_ts = 0.0
    cooldown_until = 0.0

    # 동작 학습 상태
    is_capturing_seq = False
    capture_seq: List[np.ndarray] = []

    # 동작 인식 상태
    seq_buffer: List[np.ndarray] = []

    last_spoken_label = None
    last_repeat_time  = 0.0

    tcap = ThreadedCapture(cap)
    with mp_holistic.Holistic(
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6) as holistic:
        while True:
            ok, frame = tcap.read()
            if not ok or frame is None:
                continue
            frame = cv2.flip(frame, 1)
            frame_clean = frame.copy()   # 랜드마크 없는 원본 보존
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            h, w = frame.shape[:2]
            has_left  = result.left_hand_landmarks  is not None
            has_right = result.right_hand_landmarks is not None
            num_hands = int(has_left) + int(has_right)

            cur_label = ALL_LABELS[cur_idx % len(ALL_LABELS)]
            if cur_label in data_jamo:
                cur_samples = len(data_jamo[cur_label])
            elif cur_label in data_seq:
                cur_samples = len(data_seq[cur_label])
            else:
                cur_samples = 0

            lines: List[str] = []
            if SHOW_HELP:
                lines = [
                    "┌─────────── 상태 ───────────┐",
                    f"  모드    :  {mode}",
                    f"  라벨    :  {cur_label}  ({cur_samples}개)",
                    f"  손 감지 :  {num_hands}개",
                    f"  인식    :  {recog_mode_str()}",
                    f"  TTS     :  {'ON' if TTS_ENABLED else 'OFF'}",
                    f"  해상도  :  {w}x{h}  배율 {scale:.1f}x",
                    "├─────────── 기능 ───────────┤",
                    "  M       :  학습 ↔ 인식 전환",
                    "  N / P   :  라벨 이동",
                    "  G       :  인식모드 전환",
                    "  Space   :  샘플 캡처",
                    "  S / L   :  저장 / 불러오기",
                    "  X       :  현재 라벨 샘플 삭제",
                    "├─────────── 텍스트 ─────────┤",
                    "  V       :  띄어쓰기",
                    "  BackSP  :  한 글자 삭제",
                    "  Enter   :  줄바꿈",
                    "├─────────── 화면 ───────────┤",
                    "  1/2/3   :  해상도 변경",
                    "  - / =   :  화면 배율",
                    "  F       :  전체화면",
                    "  C       :  샘플 수 요약",
                    "  H       :  이 도움말 닫기",
                    "└────────────────────────────┘",
                ]
            else:
                lines = [
                    f"모드: {mode}  인식: {recog_mode_str()}  손: {num_hands}개  |  H: 도움말"
                ]

            detected_for_display = None   # 화면/tts용 최종 라벨
            detected_jamo = None
            seq_detected   = None   # 방금 끝난 동작 라벨

            # -------- 랜드마크 그리기 (항상 표시) --------
            if True:
                # 얼굴 윤곽 (자모 인식 모드에서는 미표시)
                if result.face_landmarks and recog_mode_str() != "jamo":
                    mp_draw.draw_landmarks(
                        frame, result.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                # 몸 (어깨~팔) — 자모 인식 모드에서는 미표시
                if result.pose_landmarks and recog_mode_str() != "jamo":
                    mp_draw.draw_landmarks(
                        frame, result.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                # 양손
                if result.left_hand_landmarks:
                    mp_draw.draw_landmarks(frame, result.left_hand_landmarks,  mp_hands.HAND_CONNECTIONS)
                if result.right_hand_landmarks:
                    mp_draw.draw_landmarks(frame, result.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ------------------------ 손이 있을 때 ------------------------
            if has_left or has_right:
                hand_main = result.right_hand_landmarks or result.left_hand_landmarks

                # ===== 자모 인식 (MLP) =====
                if mlp_jamo is not None and recog_mode_str() in ("both", "jamo"):
                    feat_j = extract_jamo_keypoints(hand_main)      # (12,)
                    x_j = torch.from_numpy(feat_j).float().unsqueeze(0)
                    with torch.no_grad():
                        logits_j = mlp_jamo(x_j)
                        prob_j = F.softmax(logits_j, dim=1).numpy()[0]
                    best_idx_j = int(prob_j.argmax())
                    best_prob_j = float(prob_j[best_idx_j])
                    label_j = GESTURES_JAMO[best_idx_j]

                    if SHOW_HELP:
                        lines.append(f"[자모 MLP] {label_j} | p={best_prob_j:.3f}")

                    if best_prob_j >= CONF_THRESH_JAMO:
                        detected_jamo = label_j
                    else:
                        detected_jamo = "UNKNOWN"

                    # dwell 기반 자동 커밋 (인식 모드에서만)
                    if mode == "인식" and detected_jamo != "UNKNOWN":
                        now = time.time()
                        if now >= cooldown_until:
                            if detected_jamo != last_jamo_label:
                                last_jamo_label = detected_jamo
                                last_change_ts = now
                            else:
                                if (now - last_change_ts) >= DWELL_SEC:
                                    # SPECIALS 처리 전, 자모 단어가 있었는지 기록
                                    had_word = bool(jamo_buf)

                                    if detected_jamo in SPECIALS:
                                        apply_edit_gesture(detected_jamo, committed, jamo_buf)
                                        # 자모로 만든 단어 뒤의 띄어쓰기/줄바꿈만 읽어줌
                                        if detected_jamo in ("(띄어쓰기)", "(줄바꿈)") and had_word:
                                            speak_last_word(committed)
                                    else:
                                        jamo_buf.append(detected_jamo)
                                    try:
                                        winsound.Beep(900, 60)
                                    except:
                                        pass
                                    cooldown_until = now + COOLDOWN_SEC
                                    last_change_ts  = now + 9999.0
                    else:
                        last_jamo_label = None
                else:
                    detected_jamo = None
                    if SHOW_HELP and recog_mode_str() == "seq":
                        lines.append("[자모] 현재 인식 모드: seq (자모 인식 비활성화)")

                # ===== 동작 특징 추출 =====
                feat_seq = extract_seq_keypoints_holistic(result)

                # 2-1) 학습 모드 + 녹화 중이면 capture_seq에 저장
                if mode == "학습" and is_capturing_seq and feat_seq is not None:
                    capture_seq.append(feat_seq)
                    if SHOW_HELP:
                        lines.append(
                            f"🔴 [동작 학습 중] '{cur_label}' 프레임 {len(capture_seq)}개"
                        )

                # 2-2) 인식 모드에서는 seq_buffer에 쌓아서 나중에 손이 사라질 때 인식
                if (mode == "인식"
                    and feat_seq is not None
                    and mlp_seq is not None
                    and recog_mode_str() in ("both", "seq")):
                    seq_buffer.append(feat_seq)
                    if len(seq_buffer) > SEQ_LEN * 3:
                        seq_buffer = seq_buffer[-SEQ_LEN*3:]

            # --------------------- 손이 없을 때 (동작 끝) --------------------
            elif not has_left and not has_right:
                if SHOW_HELP:
                    lines.append("손이 감지되지 않았습니다.")
                last_jamo_label = None

                # 인식 모드 & 동작 모델 존재 & 충분히 긴 시퀀스면 인식 시도
                if (mode == "인식"
                    and mlp_seq is not None
                    and len(seq_buffer) >= MIN_SEQ_FRAMES
                    and recog_mode_str() in ("both", "seq")):

                    seq_arr = np.array(seq_buffer, dtype=np.float32)   # (T,112)
                    if isinstance(mlp_seq, LSTMClassifier):
                        idx = np.linspace(0, len(seq_arr)-1, SEQ_LEN).astype(int)
                        x_s = torch.from_numpy(seq_arr[idx]).float().unsqueeze(0)  # (1,SEQ_LEN,112)
                    else:
                        vec = seq_to_fixed(seq_arr, SEQ_LEN)           # (SEQ_LEN*112,)
                        x_s = torch.from_numpy(vec).float().unsqueeze(0)

                    with torch.no_grad():
                        logits_s = mlp_seq(x_s)
                        prob_s = F.softmax(logits_s, dim=1).numpy()[0]

                    best_idx_s = int(prob_s.argmax())
                    best_prob_s = float(prob_s[best_idx_s])
                    if 0 <= best_idx_s < len(seq_label_list):
                        label_s = seq_label_list[best_idx_s]
                    else:
                        label_s = "UNKNOWN"

                    if SHOW_HELP:
                        lines.append(f"[동작 MLP] cand={label_s} | p={best_prob_s:.3f}")

                    if best_prob_s >= CONF_THRESH_SEQ and label_s != "UNKNOWN":
                        seq_detected = label_s
                        if seq_detected in SPECIALS:
                            # 동작 기반 SPECIALS에서는 단어 읽지 않음
                            apply_edit_gesture(seq_detected, committed, jamo_buf)
                        else:
                            if jamo_buf:
                                committed.append(compose_syllables(jamo_buf))
                                jamo_buf.clear()
                            committed.append(seq_detected)
                        try:
                            winsound.Beep(700, 80)
                        except:
                            pass
                    else:
                        if SHOW_HELP:
                            lines.append("[동작 MLP] 신뢰도 낮아 무시")

                seq_buffer = []  # 세그먼트 리셋

            # 화면/TTS용 최종 라벨 선택: 동작 > 자모
            if seq_detected is not None:
                detected_for_display = seq_detected
            elif detected_jamo is not None and detected_jamo != "UNKNOWN":
                detected_for_display = detected_jamo
            else:
                detected_for_display = None

            # ====== PiP: 먼저 메인/서브 프레임 결정 ======
            pip_h = max(80, h // 4)
            pip_w = max(120, w // 4)
            if pip_swapped[0]:
                main_frame = frame_clean.copy()
                pip_frame  = frame
                pip_label  = "JOINTS"
            else:
                main_frame = frame
                pip_frame  = frame_clean
                pip_label  = "CLEAN"

            # ================= UI는 main_frame에만 그리기 =================
            composed = compose_syllables(jamo_buf)
            sentence = "".join(committed) + composed

            if show_counts:
                lines += counts_summary_lines(data_jamo, data_seq)
            # 도움말/상태는 별도 창에 표시 (줄 수에 맞게 높이 동적 계산)
            LINE_GAP = 22
            MARGIN   = 12
            help_h   = max(80, MARGIN * 2 + LINE_GAP * max(len(lines), 1) + MARGIN)
            help_img = np.zeros((help_h, 420, 3), dtype=np.uint8)
            help_img = draw_panel(help_img, lines,
                                  font_size=16, color=(180,255,180),
                                  bg_color=(20,20,20), alpha=1.0,
                                  margin=MARGIN, line_gap=LINE_GAP)
            cv2.imshow(HELP_TITLE, help_img)

            # 인식 라벨 + 문장을 왼쪽 하단에 어두운 배경 + 흰 글씨로 표시
            if detected_for_display and detected_for_display != "UNKNOWN":
                if (detected_for_display in CONSONANTS or
                    detected_for_display in VOWELS or
                    detected_for_display in DIPHTHONGS):
                    show_name = detected_for_display
                else:
                    show_name = READABLE.get(detected_for_display, detected_for_display)
            else:
                show_name = ""
            main_frame = draw_bottom_left_panel(
                main_frame,
                label_text=show_name,
                sentence_text="문장: " + sentence[-40:]
            )

            # PiP 삽입 (우하단)
            pip_img = cv2.resize(pip_frame, (pip_w, pip_h))
            py1, py2 = h - pip_h, h
            px1, px2 = w - pip_w, w
            cv2.rectangle(main_frame, (px1 - 2, py1 - 2), (px2, py2), (80,80,80), 2)
            main_frame[py1:py2, px1:px2] = pip_img
            cv2.putText(main_frame, pip_label, (px1 + 4, py2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
            cv2.putText(main_frame, "click=swap", (px1 + 4, py1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)
            frame = main_frame

            # ===== TTS (실시간 인식용) =====
            if detected_for_display and TTS_ENABLED:
                valid = (detected_for_display != "UNKNOWN") or SPEAK_UNKNOWN
                now_t = time.time()
                if valid:
                    if detected_for_display != last_spoken_label:
                        speak_label(detected_for_display)
                        last_spoken_label = detected_for_display
                        last_repeat_time  = now_t
                    elif REPEAT_PERIOD > 0 and (now_t - last_repeat_time) >= REPEAT_PERIOD:
                        speak_label(detected_for_display)
                        last_repeat_time = now_t

            # ===== 표시 배율 =====
            disp = frame
            if not fullscreen and abs(scale - 1.0) > 1e-6:
                disp = cv2.resize(frame, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_LINEAR)

            cv2.imshow(WINDOW_TITLE, disp)
            key = cv2.waitKey(1) & 0xFF

            # ===== 키 입력 처리 =====
            if key in (ord('q'), 27):
                break
            elif key in (ord('m'), ord('M')):
                mode = "인식" if mode == "학습" else "학습"
                is_capturing_seq = False
                capture_seq.clear()
                seq_buffer.clear()
                print(f"[모드 전환] {mode}")
            elif key == ord('n'):
                cur_idx = (cur_idx + 1) % len(ALL_LABELS)
            elif key == ord('p'):
                cur_idx = (cur_idx - 1) % len(ALL_LABELS)

            # 인식 모드 전환 (자모/문장 분리)
            elif key in (ord('g'), ord('G')):
                recog_mode_idx = (recog_mode_idx + 1) % len(recog_modes)
                print(f"[인식 모드] {recog_mode_str()} (both/jamo/seq)")

            # ---------------- 학습 모드: 스페이스로 자모 / 동작 저장 ----------------
            elif mode == "학습" and key == ord(' '):
                lbl = ALL_LABELS[cur_idx % len(ALL_LABELS)]

                # 1) 자모(정지 포즈) 라벨: 1샷 저장
                if lbl in GESTURES_JAMO and lbl not in GESTURES_SEQ:
                    if has_left or has_right:
                        hand_main = result.right_hand_landmarks or result.left_hand_landmarks
                        feat_j = extract_jamo_keypoints(hand_main)
                        data_jamo[lbl].append(feat_j)
                        print(f"[자모 캡처] '{lbl}' 저장 ({len(data_jamo[lbl])}개)")
                    else:
                        print("[자모 캡처 실패] 손이 감지되지 않음")

                # 2) 문장 라벨: 스페이스 토글로 녹화 시작/종료
                elif lbl in GESTURES_SEQ:
                    if not is_capturing_seq:
                        if has_left or has_right:
                            is_capturing_seq = True
                            capture_seq.clear()
                            print(f"[동작 캡처 시작] '{lbl}' 동작 녹화...")
                        else:
                            print("[동작 캡처 시작 실패] 손이 감지되지 않음")
                    else:
                        is_capturing_seq = False
                        if len(capture_seq) >= 5:
                            arr = np.array(capture_seq, dtype=np.float32)
                            data_seq[lbl].append(arr)
                            print(f"[동작 캡처 완료] '{lbl}' 샘플 1개 저장 "
                                  f"(길이 {len(capture_seq)}프레임, 총 {len(data_seq[lbl])}개)")
                        else:
                            print(f"[동작 캡처 취소] 너무 짧음 ({len(capture_seq)}프레임)")
                        capture_seq.clear()

            # 저장/불러오기
            elif key in (ord('s'), ord('S')):
                save_data_jamo(DATA_FILE_JAMO, data_jamo)
                save_data_seq(DATA_FILE_SEQ, data_seq)
            elif key in (ord('l'), ord('L')):
                data_jamo = load_data_jamo(DATA_FILE_JAMO)
                data_seq  = load_data_seq(DATA_FILE_SEQ)

            # 인식 모드 편집 (키보드)
            elif mode == "인식" and key == ord('v'):
                # 키보드 띄어쓰기도 자모 단어가 있었을 때만 읽기
                had_word = bool(jamo_buf)
                apply_edit_gesture("(띄어쓰기)", committed, jamo_buf)
                if had_word:
                    speak_last_word(committed)
            elif mode == "인식" and key in (8,127):
                apply_edit_gesture("(삭제)", committed, jamo_buf)
            elif mode == "인식" and key in (10,13):
                had_word = bool(jamo_buf)
                apply_edit_gesture("(줄바꿈)", committed, jamo_buf)
                if had_word:
                    speak_last_word(committed)

            # TTS 옵션
            elif key in (ord('t'), ord('T')):
                TTS_ENABLED = not TTS_ENABLED
                print(f"[TTS] {'ON' if TTS_ENABLED else 'OFF'}")
            elif key in (ord('u'), ord('U')):
                SPEAK_UNKNOWN = not SPEAK_UNKNOWN
                print(f"[TTS UNKNOWN] {'ON' if SPEAK_UNKNOWN else 'OFF'}")

            # HUD/샘플 요약
            elif key in (ord('h'), ord('H')):
                SHOW_HELP = not SHOW_HELP
                print(f"[도움말] {'ON' if SHOW_HELP else 'OFF'}")
            elif key in (ord('c'), ord('C')):
                show_counts = not show_counts

            # 배율
            elif key in (ord('='), ord('+')):
                scale = min(SCALE_MAX, round(scale + SCALE_STEP, 2))
            elif key == ord('-'):
                scale = max(SCALE_MIN, round(scale - SCALE_STEP, 2))
            elif key == ord('0'):
                scale = SCALE_DEFAULT

            # 전체화면
            elif key in (ord('f'), ord('F')):
                fullscreen = not fullscreen
                prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, prop)

            # 표본 삭제
            elif key in (ord('x'), ord('X')):
                lbl = ALL_LABELS[cur_idx % len(ALL_LABELS)]
                if lbl in data_jamo and data_jamo[lbl]:
                    data_jamo[lbl].pop()
                    print(f"[자모 삭제] '{lbl}' 1개 삭제 (남은 {len(data_jamo[lbl])})")
                elif lbl in data_seq and data_seq[lbl]:
                    data_seq[lbl].pop()
                    print(f"[시퀀스 삭제] '{lbl}' 1개 삭제 (남은 {len(data_seq[lbl])})")
                else:
                    print(f"[삭제] '{lbl}' 표본 없음")

            # 해상도
            elif key == ord('1'):
                apply_resolution(cap, RES_PRESETS[0])
            elif key == ord('2'):
                apply_resolution(cap, RES_PRESETS[1])
            elif key == ord('3'):
                apply_resolution(cap, RES_PRESETS[2])

    winsound.PlaySound(None, 0)
    tcap.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
