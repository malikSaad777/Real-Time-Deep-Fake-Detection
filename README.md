# Real-Time-Deep-Fake-Detection
# Deepfake Detection 🎥🧠

This is the **Prototype Phase** of a Deepfake Detection Web Application that uses a CNN-LSTM model for detecting manipulated videos in real time.

---

## 🚀 Features
- Upload video files and detect whether they are **real or fake**
- Analyze live webcam or YouTube video streams
- Real-time frame-by-frame predictions with confidence scores
- User authentication (login/register)
- Flask + Socket.IO backend with a Bootstrap frontend

---

## 🧩 Tech Stack
- **Backend:** Flask, Flask-SocketIO, OpenCV, TensorFlow/Keras
- **Frontend:** HTML, Bootstrap, JavaScript (Fetch API)
- **Database:** SQLite
- **Streaming:** MJPEG feed via OpenCV and Socket.IO

---

## 🏗️ Project Structure

app/
├── detection/ # Real-time and batch detection logic
├── templates/ # HTML templates
├── routes.py # Flask routes
├── auth.py # Authentication module
├── utils.py # Frame extraction, model loading
├── database.py # SQLite initialization
run.py # Entry point
models/ # Trained model file

deepfake-detection/
│
├── app/
│   ├── __init__.py
│   ├── app.py
│   ├── auth.py
│   ├── routes.py
│   ├── utils.py
│   ├── database.py
│   ├── detection/
│   │   ├── realtime.py
│   │   ├── routes.py
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── login.html
│   │   ├── register.html
│   ├── static/               # (Optional: add JS/CSS later)
│
├── models/
│   └── deepfake_detection_model.h5   
│
├── uploads/                 # runtime folder
│
├── run.py                   # main entry point
├── requirements.txt         # dependencies
├── README.md                # documentation
├── .gitignore
└── LICENSE (optional)

