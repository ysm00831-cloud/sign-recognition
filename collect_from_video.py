# collect_from_video.py
# 동영상(로컬 파일 또는 유튜브 URL)에서 수화 학습 데이터를 자동으로 수집합니다.
#
# [사용법]
# 1) 대화형 모드 (권장)
#    python collect_from_video.py
#
# 2) 타임스탬프 파일로 일괄 수집 (권장 — 긴 영상에서 구간별 자동 분할)
#    python collect_from_video.py --video URL또는파일 --timestamps timestamps.txt
#
#    timestamps.txt 형식:
#      0:17 인사                  ← 시작 시간만 (끝은 다음 구간까지 자동)
#      0:23~0:29 안녕하세요       ← 시작~끝 명시
#      0:30 건강하다
#      ...
#
# 3) 단일 라벨 모드
#    python collect_from_video.py --video 파일.mp4 --label 안녕하세요 --type seq
#    python collect_from_video.py --video 파일.mp4 --label ㄴ --type jamo --no-preview
#
# [설치 필요]
#    pip install yt-dlp

# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
import tempfile
from typing import List, Optional, Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np

from sign_lstm import (
    BASE_DIR,
    DATA_FILE_JAMO, DATA_FILE_SEQ,
    GESTURES_JAMO, GESTURES_SEQ,
    extract_jamo_keypoints,
    extract_seq_keypoints_holistic,
    load_data_jamo, save_data_jamo,
)

# =============================================================================
# 상수
# =============================================================================
mp_hands        = mp.solutions.hands
mp_holistic     = mp.solutions.holistic
mp_pose         = mp.solutions.pose
mp_draw         = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

JAMO_SAMPLE_INTERVAL = 5
MIN_SEQ_FRAMES       = 8    # 최소 프레임 수 (짧은 구간도 저장하도록 낮춤)
NO_HAND_SEC          = 0.4

# =============================================================================
# 시퀀스 데이터 IO (라벨 제한 없이 모든 키 보존)
# =============================================================================

def load_seq_raw(path: str) -> Dict[str, List[np.ndarray]]:
    """npy 파일의 모든 라벨을 필터 없이 로드합니다."""
    if not os.path.exists(path):
        return {}
    try:
        obj = np.load(path, allow_pickle=True).item()
        return {k: [np.array(a, dtype=np.float32) for a in v] for k, v in obj.items()}
    except Exception as e:
        print(f"[경고] 데이터 로드 실패: {e}")
        return {}

def save_seq_raw(path: str, data: Dict[str, List[np.ndarray]]):
    """딕셔너리를 그대로 npy 파일에 저장합니다."""
    np.save(path, data)
    total = sum(len(v) for v in data.values())
    print(f"[저장] {path}  (라벨 {len(data)}개, 샘플 총 {total}개)")

# =============================================================================
# 타임스탬프 파싱
# =============================================================================

def parse_time(t: str) -> float:
    """'MM:SS' 또는 'H:MM:SS' 문자열을 초(float)로 변환합니다."""
    parts = t.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(parts[0])

def parse_timestamp_text(text: str) -> List[Tuple[float, Optional[float], str]]:
    """
    타임스탬프 텍스트를 파싱합니다.
    지원 형식:
      MM:SS 라벨          — 시작 시간만 (끝은 다음 구간 시작으로 자동 결정)
      MM:SS~MM:SS 라벨    — 시작·끝 시간 명시
    반환: [(시작초, 끝초_or_None, 라벨), ...]
    """
    segments = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            try:
                time_part = parts[0]
                label = parts[1].strip()
                if "~" in time_part:
                    start_str, end_str = time_part.split("~", 1)
                    start = parse_time(start_str)
                    end   = parse_time(end_str)
                else:
                    start = parse_time(time_part)
                    end   = None
                segments.append((start, end, label))
            except ValueError:
                print(f"[경고] 파싱 실패, 건너뜀: {line}")
    return segments

# =============================================================================
# 유틸리티
# =============================================================================

def is_url(path: str) -> bool:
    return path.startswith(("http://", "https://", "www."))

def download_youtube(url: str) -> Optional[str]:
    tmp_dir  = tempfile.mkdtemp()
    out_path = os.path.join(tmp_dir, "video.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", out_path,
        "--no-playlist",
        "--merge-output-format", "mp4",
        url,
    ]
    print(f"[다운로드 중] {url}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print("[오류] yt-dlp 가 설치되지 않았습니다.  →  pip install yt-dlp")
        return None
    if result.returncode != 0:
        print(f"[다운로드 실패]\n{result.stderr[-600:]}")
        return None
    for fname in os.listdir(tmp_dir):
        full = os.path.join(tmp_dir, fname)
        if os.path.isfile(full):
            print(f"[다운로드 완료] {full}")
            return full
    print("[다운로드 실패] 저장된 파일을 찾을 수 없습니다.")
    return None

def _draw_progress(frame, label: str, cur: int, total: int, extra: str = ""):
    h, w = frame.shape[:2]
    pct  = min(1.0, cur / max(1, total))
    bx1, by1 = 10, h - 30
    bx2 = min(w - 10, 500)
    cv2.rectangle(frame, (bx1, by1), (bx2, h - 10), (50, 50, 50), -1)
    cv2.rectangle(frame, (bx1, by1), (bx1 + int((bx2-bx1)*pct), h-10), (0,200,100), -1)
    cv2.putText(frame, f"{label} | {extra} | {cur}/{total} ({pct*100:.0f}%)",
                (bx1, by1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

# =============================================================================
# ★ 타임스탬프 구간별 시퀀스 수집 (핵심 기능)
# =============================================================================

def collect_segments_from_video(
    video_path: str,
    segments: List[Tuple[float, Optional[float], str]],
    preview: bool = True,
) -> Dict[str, List[np.ndarray]]:
    """
    segments: [(시작초, 끝초_or_None, 라벨), ...]
    끝초가 None이면 다음 구간 시작 직전까지 자동 결정합니다.
    반환: {라벨: [시퀀스배열, ...]}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[오류] 동영상을 열 수 없습니다: {video_path}")
        return {}

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec  = total_fr / fps
    print(f"[정보] 총 {total_fr}프레임 / {fps:.1f}fps / {total_sec:.1f}초")
    print(f"[정보] 수집할 구간 {len(segments)}개\n")

    # 구간 끝 시간 계산: 명시된 끝이 없으면 다음 구간 시작 - 0.5초
    seg_with_end: List[Tuple[float, float, str]] = []
    for i, (start, explicit_end, label) in enumerate(segments):
        if explicit_end is not None:
            end = explicit_end
        elif i + 1 < len(segments):
            end = segments[i+1][0] - 0.5
        else:
            end = total_sec
        seg_with_end.append((start, end, label))

    collected: Dict[str, List[np.ndarray]] = {}
    total_saved = 0

    with mp_holistic.Holistic(
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as holistic:
        for seg_idx, (start_sec, end_sec, label) in enumerate(seg_with_end):
            start_fr = int(start_sec * fps)
            end_fr   = int(end_sec   * fps)
            dur_fr   = end_fr - start_fr
            if dur_fr <= 0:
                print(f"  [{seg_idx+1}/{len(seg_with_end)}] '{label}' 구간이 너무 짧음, 건너뜀")
                continue

            print(f"  [{seg_idx+1}/{len(seg_with_end)}] '{label}'  "
                  f"{_fmt_sec(start_sec)} ~ {_fmt_sec(end_sec)}  ({dur_fr}프레임)")

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_fr)
            seq: List[np.ndarray] = []
            frame_no = 0

            while frame_no < dur_fr:
                ok, frame = cap.read()
                if not ok:
                    break

                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb)

                feat = extract_seq_keypoints_holistic(result)
                if feat is not None:
                    seq.append(feat)

                if preview:
                    disp = frame.copy()
                    if result.face_landmarks:
                        mp_draw.draw_landmarks(
                            disp, result.face_landmarks,
                            mp_holistic.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    if result.pose_landmarks:
                        mp_draw.draw_landmarks(
                            disp, result.pose_landmarks,
                            mp_holistic.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    if result.left_hand_landmarks:
                        mp_draw.draw_landmarks(disp, result.left_hand_landmarks,  mp_hands.HAND_CONNECTIONS)
                    if result.right_hand_landmarks:
                        mp_draw.draw_landmarks(disp, result.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    _draw_progress(disp, label, frame_no, dur_fr,
                                   f"구간 {seg_idx+1}/{len(seg_with_end)}  손프레임={len(seq)}")
                    try:
                        cv2.imshow("타임스탬프 수집 (q: 현재 구간 건너뜀)", disp)
                        k = cv2.waitKey(1) & 0xFF
                        if k == ord('q'):
                            print(f"    → '{label}' 건너뜀")
                            seq = []
                            break
                    except cv2.error:
                        preview = False

                frame_no += 1

            if len(seq) >= MIN_SEQ_FRAMES:
                arr = np.array(seq, dtype=np.float32)
                if label not in collected:
                    collected[label] = []
                # 원본 1개 + 노이즈 추가 변형 4개 = 총 5개 저장
                collected[label].append(arr)
                for _ in range(4):
                    noise = np.random.normal(0, 0.005, arr.shape).astype(np.float32)
                    collected[label].append(arr + noise)
                total_saved += 5
                print(f"    → 저장 완료 (손 감지 {len(seq)}프레임, 증강 포함 5개)")
            else:
                print(f"    → 건너뜀 (손 감지 {len(seq)}프레임 < 최소 {MIN_SEQ_FRAMES})")

    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    print(f"\n[수집 완료] 총 {total_saved}개 구간 저장  (라벨 {len(collected)}종)")
    return collected

def _fmt_sec(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}:{s:02d}"

# =============================================================================
# 단일 라벨 — 자모
# =============================================================================

def collect_jamo_from_video(
    video_path: str,
    label: str,
    data_jamo: Dict,
    sample_interval: int = JAMO_SAMPLE_INTERVAL,
    preview: bool = True,
) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[오류] 동영상을 열 수 없습니다: {video_path}")
        return 0

    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[정보] 총 {total}프레임 / {fps:.1f}fps")

    added = 0; frame_no = 0

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_no % sample_interval == 0:
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if result.multi_hand_landmarks:
                    data_jamo[label].append(extract_jamo_keypoints(result.multi_hand_landmarks[0]))
                    added += 1
                if preview:
                    if result.multi_hand_landmarks:
                        for h in result.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
                    _draw_progress(frame, label, frame_no, total, f"samples={added}")
                    try:
                        cv2.imshow("동영상 자동 수집 (q: 중단)", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except cv2.error:
                        preview = False
            frame_no += 1

    cap.release()
    try: cv2.destroyAllWindows()
    except cv2.error: pass
    print(f"[자모 수집 완료] '{label}' → {added}개")
    return added

# =============================================================================
# 단일 라벨 — 시퀀스
# =============================================================================

def collect_seq_from_video(
    video_path: str,
    label: str,
    data_seq: Dict,
    min_frames: int = MIN_SEQ_FRAMES,
    preview: bool = True,
) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[오류] 동영상을 열 수 없습니다: {video_path}")
        return 0

    total          = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps            = cap.get(cv2.CAP_PROP_FPS) or 30.0
    no_hand_thresh = int(fps * NO_HAND_SEC)

    added = 0; frame_no = 0
    current_seq: List[np.ndarray] = []
    no_hand_cnt = 0

    def flush():
        nonlocal added, current_seq
        if len(current_seq) >= min_frames:
            arr = np.array(current_seq, dtype=np.float32)
            if label not in data_seq:
                data_seq[label] = []
            data_seq[label].append(arr)
            added += 1
            print(f"  → 시퀀스 {added}개 저장 (길이 {len(current_seq)}프레임)")
        current_seq = []

    with mp_holistic.Holistic(model_complexity=1,
                              min_detection_confidence=0.6,
                              min_tracking_confidence=0.5) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            has_hand = (result.left_hand_landmarks is not None or
                        result.right_hand_landmarks is not None)
            if has_hand:
                no_hand_cnt = 0
                feat = extract_seq_keypoints_holistic(result)
                if feat is not None:
                    current_seq.append(feat)
            else:
                no_hand_cnt += 1
                if no_hand_cnt >= no_hand_thresh and current_seq:
                    flush(); no_hand_cnt = 0
            if preview:
                disp = frame.copy()
                if result.face_landmarks:
                    mp_draw.draw_landmarks(
                        disp, result.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                if result.pose_landmarks:
                    mp_draw.draw_landmarks(
                        disp, result.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                if result.left_hand_landmarks:
                    mp_draw.draw_landmarks(disp, result.left_hand_landmarks,  mp_hands.HAND_CONNECTIONS)
                if result.right_hand_landmarks:
                    mp_draw.draw_landmarks(disp, result.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
                _draw_progress(disp, label, frame_no, total, f"seqs={added} buf={len(current_seq)}")
                try:
                    cv2.imshow("동영상 자동 수집 (q: 중단)", disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    preview = False
            frame_no += 1
        flush()

    cap.release()
    try: cv2.destroyAllWindows()
    except cv2.error: pass
    print(f"[시퀀스 수집 완료] '{label}' → {added}개")
    return added

# =============================================================================
# 대화형 모드
# =============================================================================

def interactive_mode():
    print("=" * 62)
    print("  동영상 자동 학습 데이터 수집 도구")
    print("=" * 62)

    # 동영상 입력
    print("\n동영상 파일 경로 또는 유튜브 URL을 입력하세요.")
    video_input = input("> ").strip().strip('"')
    if not video_input:
        print("[취소]"); return

    tmp_file = None
    if is_url(video_input):
        tmp_file = download_youtube(video_input)
        if tmp_file is None:
            return
        video_path = tmp_file
    else:
        if not os.path.exists(video_input):
            print(f"[오류] 파일이 없습니다: {video_input}"); return
        video_path = video_input

    # 수집 방식 선택
    print("\n수집 방식을 선택하세요:")
    print("  1) 타임스탬프 모드  — 구간별 자동 분할 (긴 영상에 권장)")
    print("  2) 단일 라벨 모드  — 영상 전체를 하나의 라벨로")
    mode = input("> ").strip()

    if mode == "1":
        _interactive_timestamp(video_path)
    elif mode == "2":
        _interactive_single(video_path)
    else:
        print("[취소]")

    if tmp_file and os.path.exists(tmp_file):
        os.remove(tmp_file)
        print(f"[임시 파일 삭제] {tmp_file}")


def _interactive_timestamp(video_path: str):
    print("\n타임스탬프를 입력하세요.")
    print("형식 1:  MM:SS 라벨            — 시작 시간만 (끝은 다음 구간까지)")
    print("형식 2:  MM:SS~MM:SS 라벨      — 시작·끝 시간 명시")
    print("(한 줄에 하나씩, 빈 줄로 입력 종료)")
    print("예시:")
    print("  0:17 인사")
    print("  0:23~0:29 안녕하세요")
    print("  0:30 건강하다")
    print()

    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)

    segments = parse_timestamp_text("\n".join(lines))
    if not segments:
        print("[오류] 유효한 타임스탬프가 없습니다."); return

    print(f"\n{len(segments)}개 구간 확인:")
    for t, end_t, lb in segments:
        end_str = f" ~ {_fmt_sec(end_t)}" if end_t is not None else ""
        print(f"  {_fmt_sec(t):>6}{end_str}  {lb}")

    print("\n미리보기 창을 표시하시겠습니까? (y/n, 기본 y)")
    preview = input("> ").strip().lower() not in ("n", "no")

    print("\n[수집 시작]\n")
    new_data = collect_segments_from_video(video_path, segments, preview=preview)

    if not new_data:
        print("[경고] 수집된 데이터가 없습니다."); return

    # 기존 데이터와 합산
    existing = load_seq_raw(DATA_FILE_SEQ)
    for label, seqs in new_data.items():
        if label not in existing:
            existing[label] = []
        existing[label].extend(seqs)
    save_seq_raw(DATA_FILE_SEQ, existing)

    print("\n[완료] 이제 train_seq_mlp.py 를 실행해 모델을 학습하세요.")


def _interactive_single(video_path: str):
    print("\n데이터 유형을 선택하세요:")
    print("  1) 자모  (정지 포즈 — 자음/모음)")
    print("  2) 시퀀스(동작 — 단어/문장)")
    dtype = input("> ").strip()

    if dtype == "1":
        label = _choose_label(GESTURES_JAMO, "자모")
        if label is None: return
        data = load_data_jamo(DATA_FILE_JAMO)
        n = collect_jamo_from_video(video_path, label, data)
        if n > 0:
            save_data_jamo(DATA_FILE_JAMO, data)
            print("다음 단계: train_mlp.py")
    elif dtype == "2":
        print("\n라벨 이름을 입력하세요 (자유롭게 입력 가능):")
        label = input("> ").strip()
        if not label: return
        existing = load_seq_raw(DATA_FILE_SEQ)
        n = collect_seq_from_video(video_path, label, existing)
        if n > 0:
            save_seq_raw(DATA_FILE_SEQ, existing)
            print("다음 단계: train_seq_mlp.py")
    else:
        print("[취소]")


def _choose_label(label_list: List[str], kind: str) -> Optional[str]:
    print(f"\n[{kind}] 사용 가능한 라벨:")
    for i, lb in enumerate(label_list):
        print(f"  {i:2d}: {lb}", end="   ")
        if (i + 1) % 8 == 0: print()
    print()
    raw = input("번호 또는 라벨 직접 입력: ").strip()
    try:
        return label_list[int(raw)]
    except (ValueError, IndexError):
        if raw in label_list: return raw
        print(f"[오류] 유효하지 않은 라벨: {raw}")
        return None

# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="동영상(로컬/유튜브)에서 수화 학습 데이터를 자동으로 수집합니다."
    )
    parser.add_argument("--video",       type=str, default=None,
                        help="동영상 파일 경로 또는 유튜브 URL")
    parser.add_argument("--timestamps",  type=str, default=None,
                        help="타임스탬프 파일 경로 (형식: 'MM:SS 라벨')")
    parser.add_argument("--label",       type=str, default=None,
                        help="단일 라벨 모드용 라벨")
    parser.add_argument("--type",        type=str, choices=["jamo", "seq"], default=None,
                        help="단일 라벨 모드용 유형")
    parser.add_argument("--no-preview",  action="store_true",
                        help="미리보기 창 비활성화")
    parser.add_argument("--interval",    type=int, default=JAMO_SAMPLE_INTERVAL,
                        help=f"자모 전용: N프레임마다 샘플 추출 (기본: {JAMO_SAMPLE_INTERVAL})")
    args = parser.parse_args()

    if not args.video:
        interactive_mode()
        return

    # 동영상 준비
    tmp_file = None
    if is_url(args.video):
        tmp_file = download_youtube(args.video)
        if tmp_file is None: sys.exit(1)
        video_path = tmp_file
    else:
        if not os.path.exists(args.video):
            print(f"[오류] 파일이 없습니다: {args.video}"); sys.exit(1)
        video_path = args.video

    preview = not args.no_preview

    # 타임스탬프 모드
    if args.timestamps:
        if not os.path.exists(args.timestamps):
            print(f"[오류] 타임스탬프 파일이 없습니다: {args.timestamps}"); sys.exit(1)
        with open(args.timestamps, encoding="utf-8") as f:
            text = f.read()
        segments = parse_timestamp_text(text)
        if not segments:
            print("[오류] 유효한 타임스탬프가 없습니다."); sys.exit(1)
        print(f"{len(segments)}개 구간 로드됨")
        new_data = collect_segments_from_video(video_path, segments, preview=preview)
        if new_data:
            existing = load_seq_raw(DATA_FILE_SEQ)
            for label, seqs in new_data.items():
                existing.setdefault(label, []).extend(seqs)
            save_seq_raw(DATA_FILE_SEQ, existing)
            print("이제 train_seq_mlp.py 를 실행하세요.")

    # 단일 라벨 모드
    elif args.label and args.type:
        if args.type == "jamo":
            if args.label not in GESTURES_JAMO:
                print(f"[오류] 유효하지 않은 자모 라벨: {args.label}"); sys.exit(1)
            data = load_data_jamo(DATA_FILE_JAMO)
            n = collect_jamo_from_video(video_path, args.label, data,
                                        sample_interval=args.interval, preview=preview)
            if n > 0: save_data_jamo(DATA_FILE_JAMO, data)
        else:
            existing = load_seq_raw(DATA_FILE_SEQ)
            n = collect_seq_from_video(video_path, args.label, existing, preview=preview)
            if n > 0: save_seq_raw(DATA_FILE_SEQ, existing)
    else:
        interactive_mode()

    if tmp_file and os.path.exists(tmp_file):
        os.remove(tmp_file)


if __name__ == "__main__":
    main()
