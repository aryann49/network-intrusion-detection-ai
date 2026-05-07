"""
app.py
======
Main application file. Run this after train.py.
This file does THREE things at once:
  1. Starts the Flask web server (your dashboard)
  2. Runs the IDS engine in the background (same process = shared memory)
  3. Pushes live alerts to browser via SocketIO (no manual refresh needed)

Usage:
    python app.py

Then open browser: http://127.0.0.1:5000
"""

import os
import time
import joblib
import threading
import numpy as np
import pandas as pd

from datetime import datetime
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────

app = Flask(__name__)
app.config["SECRET_KEY"] = "ids_secret_key"

# SocketIO allows server to PUSH data to browser in real time
# Without this, browser would need to refresh manually
socketio = SocketIO(app, cors_allowed_origins="*")


# ─────────────────────────────────────────────
#  SHARED DATA (same process = no bug)
#  Both the stream thread and Flask routes
#  access this same list in memory
# ─────────────────────────────────────────────

alerts      = []        # list of all alert dicts
stats       = {
    "total":   0,
    "attacks": 0,
    "normal":  0,
    "dos":     0,
    "probe":   0,
    "r2l":     0,
    "u2r":     0,
}
stream_status = {"running": False, "current_record": 0}


# ─────────────────────────────────────────────
#  LOAD MODEL FILES
#  All saved by train.py
# ─────────────────────────────────────────────

def load_models():
    """Load all saved model artifacts. Called once at startup."""

    models_exist = all(os.path.exists(f) for f in [
        "models/random_forest.pkl",
        "models/scaler.pkl",
        "models/label_encoder.pkl",
        "models/column_encoders.pkl",
        "models/feature_names.pkl",
        "models/attack_category_map.pkl",
    ])

    if not models_exist:
        print("\n[ERROR] Model files not found in models/ folder.")
        print("        Please run:  python train.py  first.\n")
        return None, None, None, None, None, None

    print("[APP] Loading model artifacts...")

    model             = joblib.load("models/random_forest.pkl")
    scaler            = joblib.load("models/scaler.pkl")
    label_encoder     = joblib.load("models/label_encoder.pkl")
    column_encoders   = joblib.load("models/column_encoders.pkl")
    feature_names     = joblib.load("models/feature_names.pkl")
    attack_category_map = joblib.load("models/attack_category_map.pkl")

    print("[APP] All models loaded successfully.")

    return model, scaler, label_encoder, column_encoders, feature_names, attack_category_map


# Load once at startup
(
    MODEL,
    SCALER,
    LABEL_ENCODER,
    COLUMN_ENCODERS,
    FEATURE_NAMES,
    ATTACK_CATEGORY_MAP
) = load_models()


# ─────────────────────────────────────────────
#  PREPROCESSING HELPER
#  Applies SAME encoders used during training
#  This fixes the LabelEncoder mismatch bug
# ─────────────────────────────────────────────

def preprocess_row(row: pd.Series) -> np.ndarray:
    """
    Takes a single row from the dataset.
    Encodes categorical columns using saved encoders.
    Returns a scaled numpy array ready for prediction.
    """
    row = row.copy()

    for col, encoder in COLUMN_ENCODERS.items():
        val = row[col]
        if val in encoder.classes_:
            row[col] = encoder.transform([val])[0]
        else:
            row[col] = 0   # unseen value → default to 0

    # Reorder columns to match training order
    row = row[FEATURE_NAMES]

    # Convert Series → single-row DataFrame keeping column names intact
    # .to_frame().T does this perfectly — scaler was fitted with named columns
    # so passing named columns back silences the sklearn warning completely
    sample_df    = row.to_frame().T.astype(float)
    sample_scaled = SCALER.transform(sample_df)

    return sample_scaled


# ─────────────────────────────────────────────
#  IDS STREAM ENGINE
#  Runs in a background thread inside Flask
#  Reads dataset row by row, predicts, generates alerts
#  Pushes each alert to browser via SocketIO
# ─────────────────────────────────────────────

def run_ids_engine():
    """
    Background thread function.
    Simulates real-time traffic by streaming dataset rows.
    """

    global alerts, stats, stream_status

    # Wait 2 seconds for Flask to fully start
    time.sleep(2)

    if MODEL is None:
        print("[IDS] Cannot start — model files missing. Run train.py first.")
        return

    print("[IDS] Loading dataset for streaming...")

    # Column names for NSL-KDD dataset
    column_names = [
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "land", "wrong_fragment",
        "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count",
        "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label", "difficulty"
    ]

    if not os.path.exists("data/KDDTrain+.txt"):
        print("[IDS] ERROR: data/KDDTrain+.txt not found.")
        return

    df = pd.read_csv("data/KDDTrain+.txt", header=None, names=column_names)
    df = df.drop(columns=["difficulty", "label"])

    stream_status["running"] = True
    print(f"[IDS] Streaming {len(df):,} traffic records...\n")

    for index, row in df.iterrows():

        stream_status["current_record"] = int(index)

        # ── Preprocess this row ──────────────────────
        try:
            sample_scaled = preprocess_row(row)
        except Exception as e:
            print(f"[IDS] Preprocessing error at row {index}: {e}")
            continue

        # ── Predict ──────────────────────────────────
        prediction_encoded = MODEL.predict(sample_scaled)
        attack_label       = LABEL_ENCODER.inverse_transform(prediction_encoded)[0]

        # ── Classify into broad category ─────────────
        attack_category = ATTACK_CATEGORY_MAP.get(
            attack_label.lower(), "Unknown"
        )

        # ── Update stats ─────────────────────────────
        stats["total"] += 1

        if attack_label == "normal":
            stats["normal"] += 1
        else:
            stats["attacks"] += 1
            cat = attack_category.lower()
            if cat in stats:
                stats[cat] += 1

        # ── Build alert object ───────────────────────
        is_attack  = attack_label != "normal"
        timestamp  = datetime.now().strftime("%H:%M:%S")

        alert = {
            "id":          int(index),
            "timestamp":   timestamp,
            "label":       attack_label,
            "category":    attack_category,
            "is_attack":   is_attack,
            "protocol":    str(row.get("protocol_type", "—")),
            "service":     str(row.get("service", "—")),
            "src_bytes":   int(row.get("src_bytes", 0)),
            "duration":    int(row.get("duration", 0)),
        }

        # ── Store alert (keep latest 100) ────────────
        alerts.append(alert)
        if len(alerts) > 100:
            alerts.pop(0)

        # ── Push to browser via SocketIO ─────────────
        # This is what makes the dashboard update live
        socketio.emit("new_alert", {
            "alert": alert,
            "stats": stats
        })

        # ── Console output ───────────────────────────
        if is_attack:
            print(
                f"[{timestamp}] Record #{index:05d} | "
                f"⚠  {attack_category} — {attack_label}"
            )
        else:
            if index % 50 == 0:   # print normal traffic every 50 records
                print(
                    f"[{timestamp}] Record #{index:05d} | "
                    f"✅ Normal"
                )

        # 0.3 seconds between records
        # Change this to make streaming faster or slower
        time.sleep(0.3)

    stream_status["running"] = False
    print("\n[IDS] Stream complete. All records processed.")
    socketio.emit("stream_complete", {"message": "All traffic records processed."})


# ─────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def dashboard():
    """Main dashboard page."""
    return render_template("index.html")


@app.route("/api/alerts")
def get_alerts():
    """
    REST endpoint — returns latest alerts as JSON.
    Used as fallback if SocketIO disconnects.
    """
    return jsonify({
        "alerts": alerts[::-1],   # newest first
        "stats":  stats,
        "status": stream_status
    })


@app.route("/api/stats")
def get_stats():
    """Returns current detection statistics."""
    return jsonify(stats)


# ─────────────────────────────────────────────
#  SOCKETIO EVENTS
# ─────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    """
    Called when a browser connects to the dashboard.
    Sends current state immediately so page loads with data.
    """
    print("[WS] Browser connected to dashboard")
    socketio.emit("initial_state", {
        "alerts": alerts[::-1][:20],  # last 20 alerts
        "stats":  stats,
        "status": stream_status
    })


@socketio.on("disconnect")
def on_disconnect():
    print("[WS] Browser disconnected")


# ─────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("  AI-Based Network Intrusion Detection System")
    print("=" * 55)
    print("[APP] Starting Flask server...")
    print("[APP] Dashboard → http://127.0.0.1:5000")
    print("[APP] Press Ctrl+C to stop\n")

    # Start the IDS engine in a background thread
    # daemon=True means it stops automatically when Flask stops
    ids_thread = threading.Thread(target=run_ids_engine, daemon=True)
    ids_thread.start()

    # Start Flask with SocketIO
    # use_reloader=False is important — prevents thread from starting twice
    socketio.run(app, debug=True, use_reloader=False, port=5000)
