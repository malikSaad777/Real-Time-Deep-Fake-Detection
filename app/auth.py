from flask import Blueprint, render_template, request, redirect, url_for, flash, session
import sqlite3, os

auth = Blueprint("auth", __name__)

DB_PATH = os.path.join(os.getcwd(), "users.db")

# Initialize users table
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

init_db()

@auth.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            flash("Registration successful! Please log in.")
            return redirect(url_for("auth.login"))
        except sqlite3.IntegrityError:
            flash("Username already exists.")
        finally:
            conn.close()

    return render_template("register.html")

@auth.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session["user"] = username
            flash("Login successful!")
            return redirect(url_for("main.index"))
        else:
            flash("Invalid username or password.")

    return render_template("login.html")

@auth.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out.")
    return redirect(url_for("auth.login"))

from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Please log in to access this page.")
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated_function
