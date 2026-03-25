# app.py
# 수화 인식 웹앱 서버 (Flask + SocketIO)

# -*- coding: utf-8 -*-

import os
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import inference

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sign_app_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# 시작 시 모델 로드
print("[모델 로딩 중...]")
jamo_model,  jamo_labels            = inference.load_jamo_model()
seq_model,   seq_labels, seq_len    = inference.load_seq_model()
print("[모델 로딩 완료]")

# 연결별 문장 시퀀스 버퍼
seq_buffers: dict = {}

# =============================================================================
# 페이지 라우팅
# =============================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/jamo')
def jamo_page():
    return render_template('jamo.html')

@app.route('/sentence')
def sentence_page():
    seq_label_list = seq_labels if seq_labels else []
    return render_template('sentence.html', seq_labels=seq_label_list)

@app.route('/api/labels')
def api_labels():
    return jsonify({
        'jamo': jamo_labels,
        'seq':  seq_labels,
    })

# =============================================================================
# SocketIO — 자모 모드
# =============================================================================
@socketio.on('jamo_frame')
def handle_jamo_frame(data):
    """
    data = {
        'landmarks': [{x, y, z} × 21]  # 한 손
    }
    """
    if jamo_model is None:
        emit('jamo_result', {'label': 'UNKNOWN', 'conf': 0.0, 'error': '모델 없음'})
        return

    raw_lm = data.get('landmarks', [])
    feat = inference.extract_jamo_feat(raw_lm)
    if feat is None:
        emit('jamo_result', {'label': 'UNKNOWN', 'conf': 0.0})
        return

    label, conf = inference.predict_jamo(jamo_model, jamo_labels, feat)
    emit('jamo_result', {'label': label, 'conf': round(conf, 3)})

# =============================================================================
# SocketIO — 문장 모드
# =============================================================================
@socketio.on('connect')
def on_connect():
    from flask import request
    seq_buffers[request.sid] = []

@socketio.on('disconnect')
def on_disconnect():
    from flask import request
    seq_buffers.pop(request.sid, None)

@socketio.on('seq_frame')
def handle_seq_frame(data):
    """
    data = {
        'landmarks':   [[{x,y,z}×21], ...]  # 감지된 각 손
        'handedness':  ['Left', 'Right', ...]
    }
    """
    from flask import request
    if seq_model is None:
        return

    raw_lm_list    = data.get('landmarks', [])
    handedness_list = data.get('handedness', [])

    feat = inference.extract_seq_feat(raw_lm_list, handedness_list)
    if feat is not None:
        buf = seq_buffers.setdefault(request.sid, [])
        buf.append(feat)
        if len(buf) > seq_len * 3:
            seq_buffers[request.sid] = buf[-(seq_len * 3):]

@socketio.on('seq_end')
def handle_seq_end():
    """손이 사라졌을 때 버퍼 전체로 추론"""
    from flask import request
    buf = seq_buffers.get(request.sid, [])
    if not buf:
        emit('seq_result', {'label': 'UNKNOWN', 'conf': 0.0})
        seq_buffers[request.sid] = []
        return

    label, conf = inference.predict_seq(seq_model, seq_labels, buf, seq_len)
    seq_buffers[request.sid] = []
    emit('seq_result', {'label': label, 'conf': round(conf, 3)})

# =============================================================================
# 서버 실행
# =============================================================================
if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'='*50}")
    print(f"  수화 인식 웹앱 시작")
    print(f"  PC 접속:    http://localhost:5000")
    print(f"  모바일 접속: http://{local_ip}:5000")
    print(f"  (같은 WiFi 네트워크에서 접속)")
    print(f"{'='*50}\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
