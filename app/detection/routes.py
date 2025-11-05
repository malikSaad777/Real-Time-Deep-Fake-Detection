# app/detection/routes.py
import os
import tempfile
import numpy as np
import cv2
from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
from app.models import predict_batch, get_model
from app.config import Config
from app.detection.realtime import get_worker
from app.extensions import socketio

detection_bp = Blueprint("detection_bp", __name__)

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

def allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT

def extract_fixed_frames_from_video(path, seq_len=30, target_size=(64,64)):
    """
    Read video, extract frames (uniform sampling) -> resize -> normalize -> pad/truncate to seq_len
    returns np.array shape (seq_len, H, W, 3) dtype float32 in [0,1]
    """
    cap = cv2.VideoCapture(path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # If video cannot be opened or zero frames, return empty list
    if not cap.isOpened() or total == 0:
        cap.release()
        return None

    # We will sample frames uniformly if total > seq_len, else read all and pad
    if total <= seq_len:
        # read all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
    else:
        # pick indices
        indices = np.linspace(0, total-1, seq_len, dtype=int)
        cur = 0
        idx_set = set(indices.tolist())
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i in idx_set:
                frame = cv2.resize(frame, target_size)
                frames.append(frame)
            i += 1
            if len(frames) >= seq_len:
                break
    cap.release()

    if len(frames) == 0:
        return None

    # pad if needed
    while len(frames) < seq_len:
        frames.append(np.zeros_like(frames[-1], dtype=np.uint8))

    arr = np.stack(frames, axis=0).astype("float32") / 255.0
    return arr

@detection_bp.route("/analyze", methods=["POST"])
def analyze_video():
    """
    Upload a video file (form-data 'video') and receive classification result.
    """
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files["video"]
    filename = secure_filename(file.filename)
    if not filename or not allowed_file(filename):
        return jsonify({"error": "Unsupported file type"}), 400

    # save temporarily
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, filename)
    file.save(path)

    # extract frames
    seq = extract_fixed_frames_from_video(path, seq_len=Config.SEQUENCE_LENGTH, target_size=(Config.IMG_WIDTH, Config.IMG_HEIGHT))
    os.remove(path)
    try:
        os.rmdir(tmpdir)
    except Exception:
        pass

    if seq is None:
        return jsonify({"error": "Could not read frames from video"}), 400

    batch = np.expand_dims(seq, axis=0)  # (1, seq, H, W, 3)
    preds = predict_batch(batch)
    class_id = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    label = "real" if class_id == 0 else "fake"
    return jsonify({"label": label, "class_id": class_id, "confidence": confidence})

# SocketIO events to start/stop real-time streaming from a client's webcam or server camera source
@socketio.on("start_analysis")
def handle_start_analysis(data):
    """
    client may pass {"source": 0} or {"source": "rtsp://..."} etc.
    default source = 0 (server webcam)
    """
    src = data.get("source", 0) if data else 0
    worker = get_worker()
    # allow overriding source
    worker.source = src
    worker.start()
    socketio.emit("analysis_started", {"msg": "Real-time analysis started"})

@socketio.on("stop_analysis")
def handle_stop_analysis():
    worker = get_worker()
    worker.stop()
    socketio.emit("analysis_stopped", {"msg": "Real-time analysis stopped"})
