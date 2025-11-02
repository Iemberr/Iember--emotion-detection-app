from flask import Flask, render_template, request, jsonify
from model import analyze_emotion
import os
import base64
import sqlite3
from datetime import datetime

app = Flask(__name__)
STATIC_FOLDER = "static"
DB_NAME = "database.db"

os.makedirs(STATIC_FOLDER, exist_ok=True)

# --- Initialize database ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            emotion TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- Save record to database ---
def save_to_db(name, image_path, emotion):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO users (name, image_path, emotion, created_at) VALUES (?, ?, ?, ?)",
        (name, image_path, emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

# --- Home route ---
@app.route("/")
def index():
    return render_template("index.html")

# --- Detect route ---
@app.route("/detect", methods=["POST"])
def detect():
    username = request.form.get("name")
    image_path = None

    # Uploaded image
    if "image" in request.files and request.files["image"].filename != "":
        image = request.files["image"]
        image_path = os.path.join(STATIC_FOLDER, image.filename)
        image.save(image_path)

    # Webcam image (sent as blob)
    elif "webcam_image" in request.files:
        image = request.files["webcam_image"]
        filename = f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        image_path = os.path.join(STATIC_FOLDER, filename)
        image.save(image_path)

    # Analyze emotion
    emotion = "neutral"
    if image_path:
        emotion = analyze_emotion(image_path)
        save_to_db(username, image_path, emotion)  # save to database

    return jsonify({
        "username": username,
        "emotion": emotion,
        "image_path": image_path
    })

if __name__ == "__main__":
    app.run(debug=True)
