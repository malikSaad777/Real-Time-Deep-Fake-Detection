from flask import Flask
from flask_socketio import SocketIO
import os

socketio = SocketIO()

def create_app():
    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    app.secret_key = "supersecretkey"  # Needed for login sessions

    from .routes import main
    from .auth import auth 
    app.register_blueprint(main)
    app.register_blueprint(auth)

    socketio.init_app(app)
    return app