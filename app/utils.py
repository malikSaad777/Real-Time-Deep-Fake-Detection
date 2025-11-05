import cv2
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model

# Load model function
def load_model(model_path="G:/deepfake_detection 2/deepfake_detection_model.h5"):
    """
    Loads the trained deepfake detection model.
    """
    try:
        model = keras_load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def extract_frames(video_path, max_frames=30, img_size=(64, 64)):
    """
    Extract frames from video and preprocess them for the model.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, img_size)
        frame = frame.astype("float32") / 255.0
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        return None

    frames = np.array(frames)
    # Pad if fewer than max_frames
    if frames.shape[0] < max_frames:
        pad = np.zeros((max_frames - frames.shape[0], img_size[0], img_size[1], 3))
        frames = np.vstack([frames, pad])
    return np.expand_dims(frames, axis=0)


def predict_video(video_path, model):
    """
    Run deepfake prediction on a video.
    """
    try:
        frames = extract_frames(video_path)
        if frames is None:
            print("No frames extracted.")
            return None, None

        preds = model.predict(frames)
        confidence = float(preds[0][0])
        label = "FAKE" if confidence > 0.5 else "REAL"
        return label, confidence
    except Exception as e:
        print(f"Error in predict_video: {e}")
        return None, None

frames_buffer = []

def predict_frame(frame, model):
    global frames_buffer
    frame = cv2.resize(frame, (64, 64))
    frame = frame.astype("float32") / 255.0
    frames_buffer.append(frame)

    if len(frames_buffer) == 30:
        seq = np.expand_dims(np.array(frames_buffer), axis=0)  # (1,30,64,64,3)
        pred = model.predict(seq)[0][0]
        label = "FAKE" if pred > 0.5 else "REAL"
        frames_buffer.pop(0)  # sliding window
        return label, float(pred)
    
    return "WAITING", 0.0


def open_video_stream(source: str):
    """
    Open a video stream from webcam, RTSP, HTTP, or file path.
    Example sources:
      - 0 (webcam)
      - 'rtsp://192.168.1.2:554/stream'
      - 'http://example.com/video.mp4'
      - 'videos/sample.mp4'
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    print(f"Opened video source: {source}")
    return cap
