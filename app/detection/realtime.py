# app/detection/realtime.py
import cv2
import numpy as np
import time
import threading
from app.extensions import socketio
from app.config import Config
from app.models import predict_batch, get_model

class RealTimeWorker:
    def __init__(self, source=0):
        self.source = source
        self.thread = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None

    def _run(self):
        cap = cv2.VideoCapture(self.source)
        seq_len = Config.SEQUENCE_LENGTH
        H, W = Config.IMG_HEIGHT, Config.IMG_WIDTH
        buffer = []

        # Skip if cannot open
        if not cap.isOpened():
            socketio.emit("realtime_error", {"msg": f"Cannot open video source {self.source}"})
            self.running = False
            return

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                # preprocess frame
                frame = cv2.resize(frame, (W, H))
                frame = frame[..., ::-1]  # BGR->RGB if needed (we treat as generic)
                frame = frame.astype("float32") / 255.0
                buffer.append(frame)

                if len(buffer) >= seq_len:
                    window = np.array(buffer[-seq_len:])  # last seq_len frames
                    batch = np.expand_dims(window, axis=0)  # shape (1, seq_len, H, W, 3)
                    preds = predict_batch(batch)
                    # preds is shape (1, num_classes)
                    class_id = int(np.argmax(preds[0]))
                    confidence = float(np.max(preds[0]))
                    socketio.emit("confidence_score", {"class": int(class_id), "confidence": confidence})
                # limit CPU usage
                time.sleep(0.02)
        finally:
            cap.release()
            self.running = False

# keep a single worker per app
_worker = None
_worker_lock = threading.Lock()

def get_worker():
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = RealTimeWorker()
        return _worker
