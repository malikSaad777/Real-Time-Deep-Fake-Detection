# run.py
from app import create_app, socketio

app = create_app()

if __name__ == "__main__":
    # debug=True is fine in dev; use proper server for production
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
