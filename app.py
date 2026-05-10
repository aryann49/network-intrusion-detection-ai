"""
app.py
======
Main application. Run after train.py.

Features:
  - Live IDS stream engine (background thread)
  - Confidence scores per prediction (predict_proba)
  - SHAP explanations for every attack detection
  - SQLite persistent storage (every detection saved)
  - Pause / Resume stream control
  - Speed control (0.1s – 2.0s per record)
  - Multi-page REST API for dashboard
  - PDF report generation

Usage:
    python app.py
    Open: http://127.0.0.1:5000
"""

import os
import io
import csv
import hmac
import time
import json
import joblib
import logging
import secrets
import warnings
import threading
import numpy  as np
import pandas as pd

from functools import wraps
from html import escape as html_escape
from urllib.parse import urljoin, urlparse
from flask import session as flask_session, redirect, url_for

AUTH_USERNAME = os.environ.get("IDS_USERNAME", "admin")
AUTH_PASSWORD = os.environ.get("IDS_PASSWORD", "netguard123")

warnings.filterwarnings("ignore")

from datetime    import datetime, timedelta
from flask       import Flask, render_template, jsonify, request, send_file, send_from_directory
from flask_socketio import SocketIO, emit

import database

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def project_path(*parts):
    return os.path.join(BASE_DIR, *parts)


# ─────────────────────────────────────────────
#  LOGGER — logs/alerts.log
# ─────────────────────────────────────────────

os.makedirs(project_path("logs"), exist_ok=True)

alert_logger = logging.getLogger("ids_alerts")
alert_logger.setLevel(logging.INFO)
_fh = logging.FileHandler(project_path("logs", "alerts.log"), encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s",
                                    datefmt="%Y-%m-%d %H:%M:%S"))
alert_logger.addHandler(_fh)
alert_logger.propagate = False


# ─────────────────────────────────────────────
#  FLASK + SOCKETIO
# ─────────────────────────────────────────────

app = Flask(__name__)


def _env_int(name, default, minimum=None, maximum=None):
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


app.config["SECRET_KEY"] = os.environ.get(
    "IDS_SECRET_KEY",
    "ids-dev-secret-change-me"
)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE=os.environ.get("IDS_COOKIE_SAMESITE", "Lax"),
    SESSION_COOKIE_SECURE=os.environ.get("IDS_COOKIE_SECURE", "0").lower()
    in {"1", "true", "yes"},
    PERMANENT_SESSION_LIFETIME=timedelta(
        hours=_env_int("IDS_SESSION_HOURS", 8, minimum=1)
    ),
)


def _parse_cors_origins():
    raw = os.environ.get("IDS_CORS_ORIGINS", "").strip()
    if not raw:
        return None
    if raw == "*":
        return "*"
    return [origin.strip() for origin in raw.split(",") if origin.strip()]

socketio = SocketIO(
    app,
    cors_allowed_origins=_parse_cors_origins(),
    async_mode="threading",
)

# ─────────────────────────────────────────────
#  SHARED STATE
# ─────────────────────────────────────────────

alerts = []          # latest 100 alerts in memory
CURRENT_SESSION_ID = None
stats  = {
    # Basic counters
    "total": 0, "attacks": 0, "normal": 0,
    "dos": 0, "probe": 0, "r2l": 0, "u2r": 0,
    # Ground truth confusion matrix
    "tp": 0, "fp": 0, "fn": 0, "tn": 0,
    # Severity counters (attacks only)
    "critical": 0, "high": 0, "medium": 0, "low_sev": 0,
    # Validation outcome counters
    "correct_detections": 0, "false_positives": 0,
    "false_negatives": 0, "correct_normal": 0, "misclassifications": 0,
}

# ── Attack timeline (rolling 30-minute window) ────
from collections import OrderedDict
timeline      = OrderedDict()   # "HH:MM" → {"attacks":0,"normal":0}
TIMELINE_MAX  = 30
stream_ctrl = {
    "running":  False,
    "paused":   False,
    "speed":    0.3,        # seconds between records
    "record":   0,
    "complete": False,
}
STATE_LOCK = threading.RLock()


def make_state_payload(limit=30):
    with STATE_LOCK:
        stats_copy = dict(stats)
        return {
            "alerts": list(reversed(alerts))[:limit],
            "stats": stats_copy,
            "stream": dict(stream_ctrl),
            "session_id": CURRENT_SESSION_ID,
            "metrics": compute_live_metrics(stats_copy),
            "timeline": [{"t": k, **dict(v)} for k, v in timeline.items()],
        }


# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────

REQUIRED = [
    project_path("models", "random_forest.pkl"),
    project_path("models", "scaler.pkl"),
    project_path("models", "label_encoder.pkl"),
    project_path("models", "column_encoders.pkl"),
    project_path("models", "feature_names.pkl"),
    project_path("models", "attack_category_map.pkl"),
]

def load_models():
    if not all(os.path.exists(p) for p in REQUIRED):
        print("\n[ERROR] Missing model files. Run: python train.py\n")
        return (None,) * 7

    print("[APP] Loading models...")
    model           = joblib.load(project_path("models", "random_forest.pkl"))
    scaler          = joblib.load(project_path("models", "scaler.pkl"))
    label_enc       = joblib.load(project_path("models", "label_encoder.pkl"))
    col_encs        = joblib.load(project_path("models", "column_encoders.pkl"))
    feat_names      = joblib.load(project_path("models", "feature_names.pkl"))
    cat_map         = joblib.load(project_path("models", "attack_category_map.pkl"))

    # SHAP explainer — optional (only if train.py generated it)
    shap_exp = None
    shap_path = project_path("models", "shap_explainer.pkl")
    if os.path.exists(shap_path):
        try:
            shap_exp = joblib.load(shap_path)
            print("[APP] SHAP explainer loaded.")
        except Exception as e:
            print(f"[APP] SHAP explainer failed to load: {e}")
    else:
        try:
            import shap
            shap_exp = shap.TreeExplainer(model)
            print("[APP] SHAP explainer created from loaded model.")
        except Exception as e:
            print(f"[APP] SHAP explainer unavailable: {e}")

    print("[APP] All models loaded.")
    return model, scaler, label_enc, col_encs, feat_names, cat_map, shap_exp

MODEL, SCALER, LABEL_ENC, COL_ENCS, FEAT_NAMES, CAT_MAP, SHAP_EXP = load_models()


# ─────────────────────────────────────────────
#  PREPROCESS ONE ROW
# ─────────────────────────────────────────────

def preprocess_row(row: pd.Series) -> pd.DataFrame:
    """Encode + scale a single dataset row. Returns named DataFrame."""
    row = row.copy()
    for col, enc in COL_ENCS.items():
        val     = row[col]
        row[col] = enc.transform([val])[0] if val in enc.classes_ else 0
    row = row[FEAT_NAMES]
    df  = row.to_frame().T.astype(float)
    return pd.DataFrame(SCALER.transform(df), columns=FEAT_NAMES)


# ─────────────────────────────────────────────
#  SHAP — TOP FEATURES FOR ONE PREDICTION
# ─────────────────────────────────────────────

def get_shap_top(scaled_df: pd.DataFrame, predicted_class_idx: int, n=5):
    """
    Returns top-N feature contributions for this prediction.
    Only called for attack detections to keep stream fast.
    """
    if SHAP_EXP is None:
        return []
    try:
        shap_vals = SHAP_EXP.shap_values(scaled_df)

        if isinstance(shap_vals, list):
            sv = np.asarray(shap_vals[predicted_class_idx])[0]
        else:
            arr = np.asarray(shap_vals)
            if arr.ndim == 3:
                # Newer SHAP returns: samples x features x classes.
                if arr.shape[0] == len(scaled_df):
                    sv = arr[0, :, predicted_class_idx]
                else:
                    sv = arr[predicted_class_idx, 0, :]
            elif arr.ndim == 2:
                sv = arr[0]
            else:
                return []

        contribs = pd.Series(sv, index=FEAT_NAMES)
        top      = contribs.reindex(contribs.abs().nlargest(n).index)

        return [
            {"feature": f, "value": round(float(v), 4)}
            for f, v in top.items()
        ]
    except Exception:
        return []


# ─────────────────────────────────────────────
#  SEVERITY — based on confidence score
# ─────────────────────────────────────────────

def get_severity(confidence: float) -> str:
    """Map raw confidence (0–1) to a severity level."""
    if confidence >= 0.95:
        return "Critical"
    elif confidence >= 0.80:
        return "High"
    elif confidence >= 0.60:
        return "Medium"
    else:
        return "Low"


# ─────────────────────────────────────────────
#  LIVE METRICS — derived from TP/FP/FN/TN
# ─────────────────────────────────────────────

def compute_live_metrics(s: dict) -> dict:
    """Compute precision, recall, F1, FPR, FNR from running counters."""
    tp, fp, fn, tn = s["tp"], s["fp"], s["fn"], s["tn"]
    precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 1) \
                if (precision + recall) > 0 else 0.0
    fpr       = round(fp / (fp + tn) * 100, 1) if (fp + tn) > 0 else 0.0
    fnr       = round(fn / (fn + tp) * 100, 1) if (fn + tp) > 0 else 0.0
    return {
        "precision": precision, "recall": recall,
        "f1": f1, "fpr": fpr, "fnr": fnr,
    }


def finish_stream_session():
    with STATE_LOCK:
        stream_ctrl["running"] = False
        stream_ctrl["complete"] = True
        final_stats = dict(stats)
    database.finish_session(
        CURRENT_SESSION_ID,
        total=final_stats["total"],
        attacks=final_stats["attacks"],
        normal=final_stats["normal"],
    )


# ─────────────────────────────────────────────
#  IDS STREAM ENGINE
# ─────────────────────────────────────────────

COLUMN_NAMES = [
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
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]


def run_ids_engine():
    global alerts, stats, stream_ctrl, CURRENT_SESSION_ID

    time.sleep(2)  # wait for Flask to start

    if MODEL is None:
        print("[IDS] Cannot start - run train.py first.")
        finish_stream_session()
        return

    data_path = project_path("data", "KDDTest+.txt")
    if not os.path.exists(data_path):
        print("[IDS] ERROR: data/KDDTest+.txt not found.")
        finish_stream_session()
        return

    if CURRENT_SESSION_ID is None:
        database.init_db()
        CURRENT_SESSION_ID = database.start_session()
    with STATE_LOCK:
        stream_ctrl["session_id"] = CURRENT_SESSION_ID

    print("[IDS] Loading dataset...")
    df = pd.read_csv(data_path, header=None, names=COLUMN_NAMES)
    df = df.drop(columns=["difficulty"])   # keep 'label' for ground truth

    with STATE_LOCK:
        stream_ctrl["running"]  = True
        stream_ctrl["complete"] = False
    print(f"[IDS] Streaming {len(df):,} records...\n")

    for index, row in df.iterrows():

        # ── Pause control ─────────────────────────────
        while True:
            with STATE_LOCK:
                paused = stream_ctrl["paused"]
            if not paused:
                break
            time.sleep(0.1)

        with STATE_LOCK:
            stream_ctrl["record"] = int(index)

        # ── Ground truth ─────────────────────────────
        actual_label    = str(row.get("label", "normal")).strip()
        actual_category = CAT_MAP.get(actual_label.lower(), "Unknown")
        actual_is_attack = (actual_label != "normal")

        # ── Preprocess (drop label before passing) ────
        try:
            scaled_df = preprocess_row(row.drop("label"))
        except Exception as e:
            print(f"[IDS] Preprocess error row {index}: {e}")
            continue

        # ── Predict + Confidence ──────────────────────
        pred_input = scaled_df.to_numpy()
        pred_enc   = MODEL.predict(pred_input)
        proba      = MODEL.predict_proba(pred_input)[0]
        confidence = float(proba.max())
        label      = LABEL_ENC.inverse_transform(pred_enc)[0]
        pred_idx   = int(pred_enc[0])

        # ── Category ─────────────────────────────────
        category  = CAT_MAP.get(label.lower(), "Unknown")
        is_attack = (label != "normal")

        # ── SHAP (only for attacks) ───────────────────
        shap_top = get_shap_top(scaled_df, pred_idx) if is_attack else []
        
        # ── Severity ──────────────────────────────────
        severity = get_severity(confidence) if is_attack else "None"

        with STATE_LOCK:
            # ── Ground truth validation ───────────────
            if is_attack and actual_is_attack:
                if category == actual_category:
                    validation_status = "Correct Detection"
                    stats["tp"] += 1
                    stats["correct_detections"] += 1
                else:
                    validation_status = "Misclassification"
                    stats["tp"] += 1          # attack caught, wrong type
                    stats["misclassifications"] += 1
            elif is_attack and not actual_is_attack:
                validation_status = "False Positive"
                stats["fp"] += 1
                stats["false_positives"] += 1
            elif not is_attack and actual_is_attack:
                validation_status = "False Negative"
                stats["fn"] += 1
                stats["false_negatives"] += 1
            else:
                validation_status = "Correct Normal"
                stats["tn"] += 1
                stats["correct_normal"] += 1

            # ── Timeline bucket update ────────────────
            minute_key = datetime.now().strftime("%H:%M")
            if minute_key not in timeline:
                timeline[minute_key] = {"attacks": 0, "normal": 0}
                if len(timeline) > TIMELINE_MAX:
                    timeline.popitem(last=False)
            if is_attack:
                timeline[minute_key]["attacks"] += 1
            else:
                timeline[minute_key]["normal"]  += 1

            # ── Stats update ─────────────────────────
            stats["total"] += 1
            if is_attack:
                stats["attacks"] += 1
                cat_key = category.lower()
                if cat_key in stats:
                    stats[cat_key] += 1
                sev_key = severity.lower()
                if sev_key == "low":
                    stats["low_sev"] += 1
                elif sev_key in stats:
                    stats[sev_key] += 1
            else:
                stats["normal"] += 1

            # ── Build alert ───────────────────────────
            timestamp = datetime.now().strftime("%H:%M:%S")
            alert = {
                "id":               int(index),
                "timestamp":        timestamp,
                "label":            label,
                "category":         category,
                "is_attack":        is_attack,
                "protocol":         str(row.get("protocol_type", "—")),
                "service":          str(row.get("service",       "—")),
                "src_bytes":        int(row.get("src_bytes",     0)),
                "duration":         int(row.get("duration",      0)),
                "confidence":       round(confidence * 100, 1),
                "shap_top":         shap_top,
                "actual_label":      actual_label,
                "actual_category":   actual_category,
                "validation_status": validation_status,
                "severity":          severity,
            }

            # ── In-memory store (latest 100) ──────────
            alerts.append(alert)
            if len(alerts) > 100:
                alerts.pop(0)
            state_payload = make_state_payload()

        # ── Database (permanent) ─────────────────────
        try:
            database.insert_detection(alert, CURRENT_SESSION_ID)
        except Exception as e:
            print(f"[DB] Write error: {e}")

        # ── Log file ──────────────────────────────────
        if is_attack:
            alert_logger.warning(
                f"ATTACK  | #{index:05d} | {category:<6} | {label:<20} | "
                f"conf:{confidence*100:.1f}% | proto:{alert['protocol']} | "
                f"svc:{alert['service']} | bytes:{alert['src_bytes']}"
            )
        elif index % 100 == 0:
            alert_logger.info(
                f"NORMAL  | #{index:05d} | conf:{confidence*100:.1f}% | "
                f"proto:{alert['protocol']} | svc:{alert['service']}"
            )

        # ── SocketIO push ─────────────────────────────
        socketio.emit("new_alert", {
            "alert":    alert,
            "stats":    state_payload["stats"],
            "metrics":  state_payload["metrics"],
            "timeline": state_payload["timeline"],
        })

        # ── Console ───────────────────────────────────
        if is_attack:
            print(f"[{timestamp}] #{index:05d} | ATTACK {category} - {label} "
                  f"({confidence*100:.1f}%)")
        elif index % 100 == 0:
            print(f"[{timestamp}] #{index:05d} | OK Normal ({confidence*100:.1f}%)")

        # ── Speed control ─────────────────────────────
        with STATE_LOCK:
            speed = stream_ctrl["speed"]
        time.sleep(speed)

    finish_stream_session()
    print("\n[IDS] Stream complete.")
    socketio.emit("stream_complete", {"message": "All records processed."})

# ─────────────────────────────────────────────
#  LOGIN PROTECTION
# ─────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not flask_session.get("logged_in"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Authentication required"}), 401
            return redirect(url_for("login_page", next=request.full_path))
        return f(*args, **kwargs)
    return decorated_function


def _get_login_csrf_token():
    token = flask_session.get("_login_csrf")
    if not token:
        token = secrets.token_urlsafe(32)
        flask_session["_login_csrf"] = token
    return token


def _valid_login_csrf(token):
    expected = flask_session.get("_login_csrf", "")
    return bool(token and expected and hmac.compare_digest(token, expected))


def _valid_credentials(username, password):
    return hmac.compare_digest(username, AUTH_USERNAME) and hmac.compare_digest(
        password,
        AUTH_PASSWORD,
    )


def _is_safe_redirect(target):
    if not target:
        return False
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in {"http", "https"} and ref_url.netloc == test_url.netloc


def _show_login_hint():
    return os.environ.get("IDS_SHOW_LOGIN_HINT", "0").lower() in {"1", "true", "yes"}

# ─────────────────────────────────────────────
#  SESSION RESOLUTION
# ─────────────────────────────────────────────

def resolve_session_id():
    """
    Determines which session history APIs should use.

    Default:
        current/latest session

    Optional:
        ?session=all   → return all sessions
        ?session=5     → specific session id
    """
    session_arg = request.args.get("session", "").strip().lower()

    # all sessions
    if session_arg == "all":
        return None

    # specific numeric session
    if session_arg.isdigit():
        return int(session_arg)

    # current active session
    if CURRENT_SESSION_ID is not None:
        return CURRENT_SESSION_ID

    # fallback → latest DB session
    try:
        return database.get_latest_session_id()
    except Exception:
        return None


# ─────────────────────────────────────────────
#  ROUTES — PAGES
# ─────────────────────────────────────────────

@app.route("/")
@login_required
def dashboard():
    return render_template("index.html")


@app.route("/login", methods=["GET"])
def login_page():
    if flask_session.get("logged_in"):
        return redirect(url_for("dashboard"))
    next_url = request.args.get("next", "")
    if not _is_safe_redirect(next_url):
        next_url = ""
    return render_template(
        "login.html",
        error=None,
        csrf_token=_get_login_csrf_token(),
        next_url=next_url,
        show_login_hint=_show_login_hint(),
    )


@app.route("/login", methods=["POST"])
def login_submit():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    next_url = request.form.get("next", "")

    if not _valid_login_csrf(request.form.get("csrf_token", "")):
        return render_template(
            "login.html",
            error="Login form expired. Please try again.",
            csrf_token=_get_login_csrf_token(),
            next_url=next_url if _is_safe_redirect(next_url) else "",
            show_login_hint=_show_login_hint(),
        ), 400

    if _valid_credentials(username, password):
        flask_session["logged_in"] = True
        flask_session["username"] = username
        flask_session.permanent = True
        flask_session.pop("_login_csrf", None)

        print(f"[AUTH] Login successful: {username}")
        if _is_safe_redirect(next_url):
            return redirect(next_url)
        return redirect(url_for("dashboard"))

    print(f"[AUTH] Failed login: {username}")

    return render_template(
        "login.html",
        error="Invalid username or password.",
        csrf_token=_get_login_csrf_token(),
        next_url=next_url if _is_safe_redirect(next_url) else "",
        show_login_hint=_show_login_hint(),
    )


@app.route("/logout")
def logout():
    flask_session.clear()
    return redirect(url_for("login_page"))


# ─────────────────────────────────────────────
#  ROUTES — REST API
# ─────────────────────────────────────────────

@app.route("/api/state")
@login_required
def api_state():
    """Current in-memory state — used on browser reconnect."""
    return jsonify(make_state_payload())




@app.route("/api/history")
@login_required
def api_history():
    """
    Paginated history from SQLite.
    Query params: page, limit, category, search, attacks_only
    """
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    try:
        limit = max(1, min(200, int(request.args.get("limit", 50))))
    except ValueError:
        limit = 50
    category    = request.args.get("category",    None)
    search      = request.args.get("search",      None)
    attacks_only = request.args.get("attacks_only", "false").lower() == "true"
    session_id = resolve_session_id()

    offset = (page - 1) * limit
    try:
        rows = database.get_detections(
            limit, offset, category, search, attacks_only, session_id
        )
        total = database.get_total_count(category, search, attacks_only, session_id)
    except Exception as exc:
        return jsonify({"error": f"History query failed: {exc}"}), 500

    return jsonify({
        "rows":        rows,
        "total":       total,
        "page":        page,
        "total_pages": max(1, -(-total // limit)),  # ceiling division
        "session_id":  session_id,
    })


@app.route("/api/db_stats")
@login_required
def api_db_stats():
    """Full DB statistics for Reports page."""
    session_id = resolve_session_id()
    try:
        payload = database.get_db_stats(session_id)
    except Exception as exc:
        return jsonify({"error": f"Statistics query failed: {exc}"}), 500
    payload["session_id"] = session_id
    return jsonify(payload)


@app.route("/api/models")
@login_required
def api_models():
    """Model comparison data for Models page."""
    comparison_path = project_path("models", "model_comparison.json")

    if not os.path.exists(comparison_path):
        return jsonify({"error": "Run train.py first"}), 404

    try:
        with open(comparison_path, encoding="utf-8") as f:
            comparison = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return jsonify({"error": f"Model comparison file is unreadable: {exc}"}), 500

    if not isinstance(comparison, dict) or not isinstance(comparison.get("models"), list):
        return jsonify({"error": "Model comparison file has an invalid structure"}), 500

    return jsonify({
        "comparison": comparison,
        "best": comparison.get("best_model", ""),
    })


def _safe_csv_value(value):
    if value is None:
        return ""
    text = str(value)
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text


# ─────────────────────────────────────────────
#  ROUTES — EXPORT
# ─────────────────────────────────────────────

@app.route("/api/export/csv")
@login_required
def export_csv():
    """
    Export detection history as CSV.
    Respects the same filters as /api/history:
      category, search, attacks_only, session
    No pagination — returns ALL matching rows.

    Usage examples:
      /api/export/csv                         ← all current session
      /api/export/csv?attacks_only=true       ← attacks only
      /api/export/csv?category=DoS            ← DoS only
      /api/export/csv?search=neptune          ← neptune only
      /api/export/csv?session=all             ← all sessions ever
    """
    category     = request.args.get("category",     None)
    search       = request.args.get("search",        None)
    attacks_only = request.args.get("attacks_only", "false").lower() == "true"
    session_id   = resolve_session_id()

    # Fetch ALL matching rows (no limit)
    try:
        rows = database.get_detections(
            limit=100_000,
            offset=0,
            category=category,
            search=search,
            attacks_only=attacks_only,
            session_id=session_id,
        )
    except Exception as exc:
        return jsonify({"error": f"CSV export failed: {exc}"}), 500

    if not rows:
        return jsonify({"error": "No records found matching filters"}), 404

    # ── Build CSV in memory ───────────────────────
    # csv.writer handles commas, quotes, and newlines correctly.
    output = io.StringIO()

    # Column order — clean and readable
    columns = [
        "id", "record_id", "timestamp", "is_attack",
        "label", "category", "confidence",
        "severity", "validation_status",
        "actual_label", "actual_category",
        "protocol", "service", "src_bytes", "duration",
    ]

    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        values = [_safe_csv_value(row.get(col, "")) for col in columns]
        writer.writerow(values)

    output.seek(0)

    # ── Build filename ────────────────────────────
    parts  = ["IDS_detections"]
    if category:
        parts.append(category)
    if attacks_only:
        parts.append("attacks_only")
    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    filename = "_".join(parts) + ".csv"

    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/api/export/json")
@login_required
def export_json():
    """
    Export detection history as JSON.
    Same filters as /api/export/csv.
    Includes metadata block at the top.

    Usage examples:
      /api/export/json
      /api/export/json?attacks_only=true
      /api/export/json?category=Probe&session=all
    """
    category     = request.args.get("category",     None)
    search       = request.args.get("search",        None)
    attacks_only = request.args.get("attacks_only", "false").lower() == "true"
    session_id   = resolve_session_id()

    try:
        rows = database.get_detections(
            limit=100_000,
            offset=0,
            category=category,
            search=search,
            attacks_only=attacks_only,
            session_id=session_id,
        )
    except Exception as exc:
        return jsonify({"error": f"JSON export failed: {exc}"}), 500

    if not rows:
        return jsonify({"error": "No records found matching filters"}), 404

    # ── Clean rows for export ─────────────────────
    # shap_top is already a list (parsed by database.py)
    # Remove internal DB id to keep export clean
    export_rows = []
    for row in rows:
        clean = {
            "record_id":         row.get("record_id"),
            "timestamp":         row.get("timestamp"),
            "is_attack":         bool(row.get("is_attack")),
            "label":             row.get("label"),
            "category":          row.get("category"),
            "confidence_pct":    round(float(row.get("confidence", 0)), 1),
            "severity":          row.get("severity"),
            "validation_status": row.get("validation_status"),
            "actual_label":      row.get("actual_label"),
            "actual_category":   row.get("actual_category"),
            "protocol":          row.get("protocol"),
            "service":           row.get("service"),
            "src_bytes":         row.get("src_bytes"),
            "duration":          row.get("duration"),
            "shap_top_features": row.get("shap_top", []),
        }
        export_rows.append(clean)

    # ── Build metadata block ──────────────────────
    payload = {
        "metadata": {
            "exported_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset":        "NSL-KDD",
            "model":          "Random Forest",
            "session_id":     session_id,
            "total_records":  len(export_rows),
            "filters": {
                "category":     category,
                "search":       search,
                "attacks_only": attacks_only,
            },
        },
        "detections": export_rows,
    }

    # ── Build filename ────────────────────────────
    parts  = ["IDS_detections"]
    if category:
        parts.append(category)
    if attacks_only:
        parts.append("attacks_only")
    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    filename = "_".join(parts) + ".json"

    json_bytes = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")

    return send_file(
        io.BytesIO(json_bytes),
        mimetype="application/json",
        as_attachment=True,
        download_name=filename,
    )


# Report images used by the dashboard and generated PDF preview.
@app.route("/static/reports/<path:filename>")
@login_required
def serve_report_fixed(filename):
    return send_from_directory(project_path("static", "reports"), filename)



# ─────────────────────────────────────────────
#  ROUTES — STREAM CONTROLS
# ─────────────────────────────────────────────

@app.route("/api/control/pause", methods=["POST"])
@login_required
def control_pause():
    with STATE_LOCK:
        stream_ctrl["paused"] = True
    return jsonify({"paused": True})


@app.route("/api/control/resume", methods=["POST"])
@login_required
def control_resume():
    with STATE_LOCK:
        stream_ctrl["paused"] = False
    return jsonify({"paused": False})


@app.route("/api/control/speed", methods=["POST"])
@login_required
def control_speed():
    data  = request.get_json(silent=True) or {}
    try:
        speed = float(data.get("speed", 0.3))
    except (TypeError, ValueError):
        speed = 0.3
    speed = max(0.05, min(2.0, speed))   # clamp 0.05s – 2.0s
    with STATE_LOCK:
        stream_ctrl["speed"] = speed
    return jsonify({"speed": speed})


# ─────────────────────────────────────────────
#  ROUTES — PDF REPORT
# ─────────────────────────────────────────────

@app.route("/api/reports/pdf")
@login_required
def generate_pdf():
    """
    Generates a professional PDF report with:
      - Summary statistics
      - Top attack types table
      - Embedded confusion matrix, feature importance,
        model comparison, and SHAP images
    """
    try:
        from reportlab.lib.pagesizes  import A4
        from reportlab.lib.units      import cm
        from reportlab.lib            import colors
        from reportlab.lib.styles     import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums      import TA_CENTER, TA_LEFT, TA_RIGHT
        from reportlab.platypus       import (
            SimpleDocTemplate, Paragraph, Spacer, Table,
            TableStyle, Image as RLImage, PageBreak, HRFlowable
        )
    except ImportError:
        return jsonify({"error": "reportlab not installed. Run: pip install reportlab"}), 500

    session_id = resolve_session_id()
    try:
        db_stats = database.get_db_stats(session_id)
    except Exception as exc:
        return jsonify({"error": f"PDF statistics query failed: {exc}"}), 500
    with STATE_LOCK:
        live_stats = dict(stats)
    generated  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm,   bottomMargin=2*cm,
        title="IDS Report"
    )

    style  = getSampleStyleSheet()
    DARK_C = colors.HexColor("#0d1117")
    BLUE_C = colors.HexColor("#38bdf8")
    RED_C  = colors.HexColor("#f87171")
    GRN_C  = colors.HexColor("#34d399")
    TEXT_C = colors.HexColor("#1e293b")

    h1 = ParagraphStyle("h1", parent=style["Title"],
                         fontSize=20, textColor=DARK_C,
                         spaceAfter=6, alignment=TA_CENTER)
    h2 = ParagraphStyle("h2", parent=style["Heading2"],
                         fontSize=13, textColor=DARK_C,
                         spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle("body", parent=style["Normal"],
                           fontSize=10, textColor=TEXT_C,
                           spaceAfter=4)
    caption = ParagraphStyle("caption", parent=style["Normal"],
                              fontSize=9, textColor=colors.grey,
                              alignment=TA_CENTER, spaceAfter=10)

    story = []

    # ── Title ─────────────────────────────────
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(
        "AI-Based Network Intrusion Detection System", h1))
    story.append(Paragraph(
        "Automated Security Analysis Report", h1))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=BLUE_C, spaceAfter=10))
    story.append(Paragraph(f"<b>Generated:</b> {generated}", body))
    story.append(Paragraph("<b>Dataset:</b> NSL-KDD (125,973 training samples)", body))
    story.append(Paragraph("<b>Algorithm:</b> Random Forest — 100 estimators, max_depth=20", body))
    story.append(Spacer(1, 0.5*cm))

    # ── Summary Stats Table ───────────────────
    story.append(Paragraph("1. Detection Summary", h2))

    total     = db_stats["total"]   or live_stats["total"]
    attacks   = db_stats["attacks"] or live_stats["attacks"]
    normal    = db_stats["normal"]  or live_stats["normal"]
    rate      = f"{(attacks/total*100):.1f}%" if total > 0 else "—"
    avg_conf  = f"{db_stats['avg_confidence']}%"

    summary_data = [
        ["Metric",            "Value"],
        ["Total Records Analyzed", f"{total:,}"],
        ["Attacks Detected",       f"{attacks:,}"],
        ["Normal Traffic",         f"{normal:,}"],
        ["Attack Detection Rate",  rate],
        ["Avg Attack Confidence",  avg_conf],
        ["DoS Attacks",   str(db_stats["by_category"].get("DoS", live_stats.get("dos", 0)))],
        ["Probe Attacks", str(db_stats["by_category"].get("Probe", live_stats.get("probe", 0)))],
        ["R2L Attacks",   str(db_stats["by_category"].get("R2L", live_stats.get("r2l", 0)))],
        ["U2R Attacks",   str(db_stats["by_category"].get("U2R", live_stats.get("u2r", 0)))],
    ]

    tbl = Table(summary_data, colWidths=[9*cm, 8*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), BLUE_C),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTSIZE",    (0,0), (-1,0), 11),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.HexColor("#f8fafc"), colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("FONTSIZE",    (0,1), (-1,-1), 10),
        ("TOPPADDING",  (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",(0,0),(-1,-1), 7),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.4*cm))

    # ── Top Attacks Table ─────────────────────
    if db_stats["top_attacks"]:
        story.append(Paragraph("2. Top Attack Types Detected", h2))
        atk_data = [["Attack Label", "Count"]]
        for a in db_stats["top_attacks"]:
            atk_data.append([html_escape(str(a["label"])), str(a["count"])])
        atk_tbl = Table(atk_data, colWidths=[12*cm, 5*cm])
        atk_tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.HexColor("#fef2f2"), colors.white]),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ("FONTSIZE",    (0,0), (-1,-1), 10),
            ("TOPPADDING",  (0,0), (-1,-1), 6),
            ("BOTTOMPADDING",(0,0),(-1,-1), 6),
            ("LEFTPADDING", (0,0), (-1,-1), 10),
        ]))
        story.append(atk_tbl)
        story.append(Spacer(1, 0.4*cm))

    # ── Charts ────────────────────────────────
    chart_pages = [
        (project_path("static", "reports", "confusion_matrix.png"),   "3. Confusion Matrix"),
        (project_path("static", "reports", "feature_importance.png"), "4. Top Feature Importances"),
        (project_path("static", "reports", "model_comparison.png"),   "5. Model Comparison"),
        (project_path("static", "reports", "shap_summary.png"),       "6. SHAP Feature Impact"),
    ]

    for path, title in chart_pages:
        if os.path.exists(path):
            story.append(PageBreak())
            story.append(Paragraph(title, h2))
            story.append(Spacer(1, 0.2*cm))
            img = RLImage(path, width=16*cm, height=11*cm,
                          kind="proportional")
            story.append(img)
            story.append(Paragraph(
                f"Source: {html_escape(os.path.basename(path))}", caption))

    # ── Footer note ───────────────────────────
    story.append(PageBreak())
    story.append(Spacer(1, 2*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE_C))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "AI-Based Network Intrusion Detection System<br/>"
        "NSL-KDD Dataset · Random Forest Classifier · Flask + SocketIO<br/>"
        "MCA Project · Amity University, Noida",
        ParagraphStyle("footer", parent=style["Normal"],
                       fontSize=9, textColor=colors.grey,
                       alignment=TA_CENTER)
    ))

    try:
        doc.build(story)
    except Exception as exc:
        return jsonify({"error": f"PDF generation failed: {exc}"}), 500
    buffer.seek(0)

    filename = f"IDS_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


# ─────────────────────────────────────────────
#  SOCKETIO EVENTS
# ─────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    if not flask_session.get("logged_in"):
        raise ConnectionRefusedError("Authentication required")

    print("[WS] Browser connected")
    emit("initial_state", make_state_payload())


@socketio.on("disconnect")
def on_disconnect():
    print("[WS] Browser disconnected")


# ─────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────

def main():
    global CURRENT_SESSION_ID

    print("=" * 60)
    print("  AI-Based Network Intrusion Detection System")
    print("  NSL-KDD | Random Forest | Flask + SocketIO")
    print("=" * 60)

    # Init SQLite
    database.init_db()
    print("[DB] Database initialised -> logs/ids_database.db")
    CURRENT_SESSION_ID = database.start_session()
    with STATE_LOCK:
        stream_ctrl["session_id"] = CURRENT_SESSION_ID
    print(f"[DB] Session started -> #{CURRENT_SESSION_ID}")

    autostart_stream = os.environ.get("IDS_AUTOSTART_STREAM", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    if autostart_stream:
        t = threading.Thread(target=run_ids_engine, daemon=True)
        t.start()
    else:
        print("[IDS] Stream autostart disabled by IDS_AUTOSTART_STREAM=0")

    debug = os.environ.get("IDS_DEBUG", "0").lower() in {"1", "true", "yes"}
    host = os.environ.get("IDS_HOST", "0.0.0.0")
    port = _env_int("IDS_PORT", 5000, minimum=1, maximum=65535)

    print(f"[APP] Dashboard -> http://127.0.0.1:{port}")
    print("[APP] Press Ctrl+C to stop\n")

    socketio.run(
        app,
        host=host,
        port=port,
        debug=debug,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
