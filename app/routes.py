import os
import cv2
import numpy as np
import threading
import yt_dlp
import queue
import time
from flask import Blueprint, request, jsonify, Response, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from .utils import predict_video, predict_frame

main = Blueprint("main", __name__)

# =========================
# Globals
# =========================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "G:/deepfake_detection 2/models/deepfake_detection_model.h5"
model = load_model(MODEL_PATH)
print("✅ Model loaded:", model.input_shape)

cap = None
stream_stop_event = threading.Event()
lock = threading.Lock()
frame_queue = queue.Queue(maxsize=2)  # buffer small to avoid lag
stream_thread = None
capture_thread = None
latest_frame = None
frame_info = {"label": "N/A", "confidence": 0.0, "fps": 0.0}


# =========================
# Helpers
# =========================
def extract_frames(video_path, sequence_length=30, img_size=(64, 64)):
    cap_local = cv2.VideoCapture(video_path)
    if not cap_local.isOpened():
        print(f"Could not open video: {video_path}")
        return None
    frame_count = int(cap_local.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < sequence_length:
        print(f"Video too short ({frame_count} frames, need {sequence_length})")
        cap_local.release()
        return None
    indices = np.linspace(0, frame_count - 1, sequence_length, dtype=int)
    frames = []
    for idx in indices:
        cap_local.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap_local.read()
        if not ret:
            continue
        frame = cv2.resize(frame, img_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap_local.release()
    return frames


def predict_video(video_path, model, sequence_length=30):
    try:
        frames = extract_frames(video_path, sequence_length)
        if frames is None or len(frames) < sequence_length:
            return None, None
        frames = np.array(frames) / 255.0
        frames = np.expand_dims(frames, axis=0)
        prob = float(model.predict(frames)[0][0])
        return ("FAKE", prob) if prob > 0.5 else ("REAL", 1 - prob)
    except Exception as e:
        print(f"Error in predict_video: {e}")
        return None, None


def login_required(f):
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


# =========================
# YouTube Stream Extractor
# =========================
def get_youtube_stream_url(video_url: str) -> str:
    """Convert Shorts/share links to playable URLs using yt_dlp."""
    if "youtube.com/shorts/" in video_url:
        video_url = video_url.replace("youtube.com/shorts/", "youtube.com/watch?v=")
    elif "youtu.be/" in video_url:
        video_url = video_url.replace("youtu.be/", "youtube.com/watch?v=")
    print("🎬 Processed YouTube URL:", video_url)

    ydl_opts = {"format": "best[ext=mp4]/best", "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        return info["url"]

# =========================
# Background Thread
# =========================
def capture_frames():
    global cap, stream_stop_event, frame_queue

    print("🎥 Background capture thread started")
    while not stream_stop_event.is_set():
        success, frame = cap.read()
        if not success:
            break
        try:
            label, confidence = predict_frame(frame, model)
            text = f"{label} ({confidence:.2f})"
            color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        except Exception as e:
            print("Prediction error:", e)

        # Encode frame
        _, buffer = cv2.imencode(".jpg", frame)
        if not frame_queue.full():
            frame_queue.put(buffer.tobytes())

    print("🛑 Background capture stopped")
    with lock:
        if cap and cap.isOpened():
            cap.release()
            cap = None
        stream_stop_event.clear()

#=========================
def capture_loop():
    """Background capture thread for streaming"""
    global cap, latest_frame, frame_info
    print("🎥 Capture loop started")
    last_time = time.time()
    frame_count = 0
    while not stream_stop_event.is_set() and cap and cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("⚠️ Failed to read frame")
            break

        frame_count += 1
        # Calculate FPS every 1 second
        if time.time() - last_time >= 1.0:
            frame_info["fps"] = frame_count / (time.time() - last_time)
            frame_count = 0
            last_time = time.time()
            
        # Optional: draw overlay
        try:
            label, confidence = predict_frame(frame, model)
            frame_info["label"] = label
            frame_info["confidence"] = float(confidence)
            text = f"{label} ({confidence:.2f})"
            color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        except Exception as e:
            print("Prediction error:", e)

        _, buffer = cv2.imencode(".jpg", frame)
        latest_frame = buffer.tobytes()
    print("🛑 Capture loop exited")
#=========================

# =========================
# Streaming Generator
# =========================
def gen_stream():
    global cap, stream_stop_event, frame_queue, latest_frame

    print("🟢 gen_stream started (serving frames)")
    empty_count = 0

    while not stream_stop_event.is_set():
        if latest_frame is not None:
            try:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                       latest_frame + b"\r\n")
            except Exception as e:
                print("🔌 Client disconnected:", e)
                break
        else:
            empty_count += 1
            if empty_count % 20 == 0:
                print("⚠️ No frame yet...")
            time.sleep(0.05)
    print("🔴 gen_stream ended")
    cleanup_camera()

def cleanup_camera():
    global cap, capture_thread
    with lock:
        if cap and cap.isOpened():
            cap.release()
        cap = None
        capture_thread = None
        stream_stop_event.clear()
        print("📷 Camera released")

# =========================
# Routes
# =========================
@main.route("/")
@login_required
def index():
    return render_template("index.html", user=session.get("user"))


@main.route("/detection/analyze", methods=["POST"])
@login_required
def analyze_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    label, confidence = predict_video(filepath, model)
    if label is None:
        return jsonify({"error": "Not enough frames"}), 400

    return jsonify({"result": label, "confidence": round(float(confidence), 2)})


# ---------- Start Stream ----------
@main.route("/detection/live/start", methods=["POST"])
@login_required
def start_stream():
    global cap, stream_stop_event, stream_thread, capture_thread

    data = request.get_json() or {}
    source = data.get("source", "webcam")
    video_url = data.get("url", "").strip()

    # Stop any previous stream
    stream_stop_event.set()
    cleanup_camera()

    try:
        if source == "webcam":
            cap = cv2.VideoCapture(0)
        elif source == "url" and video_url:
            stream_url = get_youtube_stream_url(video_url)
            cap = cv2.VideoCapture(stream_url)
        else:
            return jsonify({"error": "Invalid source"}), 400

        if not cap.isOpened():
            return jsonify({"error": "Failed to open source"}), 500

        stream_stop_event.clear()
        capture_thread = threading.Thread(target=capture_loop, daemon=True)
        capture_thread.start()

        print("🟢 Stream started successfully")
        return jsonify({"status": "started"}), 200
    except Exception as e:
        print("❌ Error starting stream:", e)
        cleanup_camera()
        return jsonify({"error": str(e)}), 500

# ---------- MJPEG Feed ----------
@main.route("/detection/live_feed")
@login_required
def stream_video():
    global cap
    if not cap or not cap.isOpened():
        return jsonify({"error": "Stream not active"}), 400
    return Response(gen_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

#--------------------
@main.route("/detection/overlay_data")
def overlay_data():
    return jsonify(frame_info)


# ---------- Stop Stream ----------
@main.route("/detection/live/stop", methods=["POST"])
@login_required
def stop_stream():
    global stream_stop_event
    print("🛑 Stop requested (event set)")
    stream_stop_event.set()       # signal capture thread + generator
    cleanup_camera()              # release camera immediately
    return jsonify({"status": "stopped"}), 200

# ---------- Logs ----------
@main.route("/logs")
@login_required
def view_logs():
    import sqlite3
    conn = sqlite3.connect("logs.db")
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, filename, label, confidence FROM detections ORDER BY timestamp DESC")
    logs = cursor.fetchall()
    conn.close()
    return render_template("logs.html", logs=logs, user=session.get("user"))
