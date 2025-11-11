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

## 📂 Folder Structure
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



---

## ⚙️ Installation
```bash
# 1. Clone this repository
git clone https://github.com//malikSaad777/Real-Time-Deep-Fake-Detection.git
cd deepfake-detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python run.py

---
## Login
<img width="1911" height="950" alt="image" src="https://github.com/user-attachments/assets/1ba285d0-d4fc-43da-bfbe-402a37134b5a" />
## interface
<img width="1907" height="942" alt="image" src="https://github.com/user-attachments/assets/56d91f9a-cdae-46cc-9905-982a2633341f" />

