#app.py
from flask import Flask
from routes import main  # your blueprint in routes.py

app = Flask(__name__)
app.secret_key = "your_super_secret_key"

# Register the main blueprint
app.register_blueprint(main)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
