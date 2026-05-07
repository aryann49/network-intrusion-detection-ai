"""
train.py
========
Run this file ONCE before starting the application.
It trains the Random Forest model on NSL-KDD dataset
and saves all model artifacts into the models/ folder.

Usage:
    python train.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # non-interactive — works without a display window
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ─────────────────────────────────────────────
#  COLUMN DEFINITIONS
#  NSL-KDD has 41 features + 1 label + 1 difficulty
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
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

# These 3 columns contain text values — must be encoded
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

# ─────────────────────────────────────────────
#  ATTACK CATEGORY MAPPING
#  Maps specific attack names → broader category
#  Used for cleaner dashboard display
# ─────────────────────────────────────────────

ATTACK_CATEGORY_MAP = {
    "normal":        "Normal",

    # DoS attacks
    "back":          "DoS",
    "land":          "DoS",
    "neptune":       "DoS",
    "pod":           "DoS",
    "smurf":         "DoS",
    "teardrop":      "DoS",
    "apache2":       "DoS",
    "udpstorm":      "DoS",
    "processtable":  "DoS",
    "mailbomb":      "DoS",

    # Probe attacks
    "satan":         "Probe",
    "ipsweep":       "Probe",
    "nmap":          "Probe",
    "portsweep":     "Probe",
    "mscan":         "Probe",
    "saint":         "Probe",

    # R2L attacks (Remote to Local)
    "guess_passwd":  "R2L",
    "ftp_write":     "R2L",
    "imap":          "R2L",
    "phf":           "R2L",
    "multihop":      "R2L",
    "warezmaster":   "R2L",
    "warezclient":   "R2L",
    "spy":           "R2L",
    "xlock":         "R2L",
    "xsnoop":        "R2L",
    "snmpguess":     "R2L",
    "snmpgetattack": "R2L",
    "httptunnel":    "R2L",
    "sendmail":      "R2L",
    "named":         "R2L",

    # U2R attacks (User to Root)
    "buffer_overflow": "U2R",
    "loadmodule":    "U2R",
    "rootkit":       "U2R",
    "perl":          "U2R",
    "sqlattack":     "U2R",
    "xterm":         "U2R",
    "ps":            "U2R",
}


def get_attack_category(label):
    """Convert specific attack name to broad category."""
    return ATTACK_CATEGORY_MAP.get(label.lower(), "Unknown")


# ─────────────────────────────────────────────
#  STEP 1: LOAD DATA
# ─────────────────────────────────────────────

def load_data():
    train_path = "data/KDDTrain+.txt"
    test_path  = "data/KDDTest+.txt"

    if not os.path.exists(train_path):
        print(f"\n[ERROR] Training file not found: {train_path}")
        print("Please place KDDTrain+.txt inside the data/ folder.")
        sys.exit(1)

    if not os.path.exists(test_path):
        print(f"\n[ERROR] Test file not found: {test_path}")
        print("Please place KDDTest+.txt inside the data/ folder.")
        sys.exit(1)

    print("[1/5] Loading dataset...")

    df_train = pd.read_csv(train_path, header=None, names=COLUMN_NAMES)
    df_test  = pd.read_csv(test_path,  header=None, names=COLUMN_NAMES)

    print(f"      Training samples : {len(df_train):,}")
    print(f"      Test samples     : {len(df_test):,}")

    return df_train, df_test


# ─────────────────────────────────────────────
#  STEP 2: PREPROCESS
# ─────────────────────────────────────────────

def preprocess(df_train, df_test):
    print("\n[2/5] Preprocessing data...")

    # Drop difficulty column — not a feature
    df_train = df_train.drop(columns=["difficulty"])
    df_test  = df_test.drop(columns=["difficulty"])

    # Remove duplicates
    before = len(df_train)
    df_train = df_train.drop_duplicates()
    removed = before - len(df_train)
    print(f"      Removed {removed:,} duplicate rows from training set")

    # Separate features and labels
    X_train = df_train.drop(columns=["label"]).copy()
    y_train = df_train["label"].copy()

    X_test  = df_test.drop(columns=["label"]).copy()
    y_test  = df_test["label"].copy()

    # ── Encode categorical columns ──────────────────
    # IMPORTANT: Fit encoders on TRAIN only, transform both
    # Save each encoder so stream.py uses identical mappings
    column_encoders = {}

    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])

        # Test set may have unseen labels — map them to 0
        X_test[col] = X_test[col].apply(
            lambda val: le.transform([val])[0]
            if val in le.classes_ else 0
        )

        column_encoders[col] = le
        print(f"      Encoded '{col}' — {len(le.classes_)} unique values")

    # ── Encode target labels ────────────────────────
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc  = label_encoder.transform(
        y_test.apply(
            lambda val: val if val in label_encoder.classes_ else "normal"
        )
    )

    print(f"      Attack types found: {list(label_encoder.classes_)}")

    # ── Scale numerical features ────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print("      Feature scaling complete")

    return (
        X_train_scaled, y_train_enc,
        X_test_scaled,  y_test_enc,
        scaler, label_encoder, column_encoders,
        list(X_train.columns)
    )


# ─────────────────────────────────────────────
#  STEP 3: TRAIN MODEL
# ─────────────────────────────────────────────

def train_model(X_train, y_train):
    print("\n[3/5] Training Random Forest model...")
    print("      This may take 1-3 minutes...")

    model = RandomForestClassifier(
        n_estimators=100,       # 100 decision trees
        max_depth=20,           # prevents overfitting
        min_samples_split=5,
        n_jobs=-1,              # use all CPU cores
        random_state=42,
        verbose=0
    )

    model.fit(X_train, y_train)

    print("      Training complete!")

    return model


# ─────────────────────────────────────────────
#  STEP 4: EVALUATE
# ─────────────────────────────────────────────

def evaluate(model, X_test, y_test, label_encoder, feature_names):
    print("\n[4/5] Evaluating model — generating charts...")

    os.makedirs("reports", exist_ok=True)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n      ✅ Accuracy : {accuracy * 100:.2f}%")

    # Only use labels present in test set or predictions
    present_labels = sorted(set(y_test) | set(y_pred))
    target_names   = label_encoder.inverse_transform(present_labels)

    report = classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=target_names,
        zero_division=0
    )
    print("\n      Classification Report:")
    print(report)

    # ── Chart 1: Confusion Matrix ─────────────────
    print("      Generating confusion matrix...")

    cm = confusion_matrix(y_test, y_pred, labels=present_labels)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
        linewidths=0.5,
        linecolor="#1a2332",
        annot_kws={"size": 9, "color": "white"},
    )

    ax.set_title(
        "Confusion Matrix — Random Forest on NSL-KDD",
        fontsize=14, color="white", pad=16, fontweight="bold"
    )
    ax.set_xlabel("Predicted Label", fontsize=11, color="#94a3b8", labelpad=10)
    ax.set_ylabel("True Label",      fontsize=11, color="#94a3b8", labelpad=10)
    ax.tick_params(colors="#94a3b8", labelsize=9)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = "reports/confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"      Saved: {cm_path}")

    # ── Chart 2: Feature Importance ───────────────
    print("      Generating feature importance chart...")

    importances   = model.feature_importances_
    feat_series   = pd.Series(importances, index=feature_names)
    top20         = feat_series.sort_values(ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    colors = ["#38bdf8" if v < top20.max() * 0.5
              else "#f87171" for v in top20.values]

    bars = ax.barh(top20.index, top20.values, color=colors,
                   edgecolor="none", height=0.6)

    # Value labels on bars
    for bar, val in zip(bars, top20.values):
        ax.text(
            val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left",
            color="#94a3b8", fontsize=8
        )

    ax.set_title(
        "Top 20 Feature Importances — Random Forest",
        fontsize=13, color="white", pad=14, fontweight="bold"
    )
    ax.set_xlabel("Importance Score", fontsize=10,
                  color="#94a3b8", labelpad=8)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.xaxis.grid(True, color="#1a2332", linewidth=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    fi_path = "reports/feature_importance.png"
    plt.savefig(fi_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"      Saved: {fi_path}")


# ─────────────────────────────────────────────
#  STEP 5: SAVE MODEL ARTIFACTS
# ─────────────────────────────────────────────

def save_models(model, scaler, label_encoder, column_encoders, feature_names):
    print("[5/5] Saving model artifacts to models/ folder...")

    os.makedirs("models", exist_ok=True)

    # Save the trained model
    joblib.dump(model, "models/random_forest.pkl")
    print("      Saved: models/random_forest.pkl")

    # Save the scaler
    joblib.dump(scaler, "models/scaler.pkl")
    print("      Saved: models/scaler.pkl")

    # Save the label encoder (for decoding predictions)
    joblib.dump(label_encoder, "models/label_encoder.pkl")
    print("      Saved: models/label_encoder.pkl")

    # Save ALL column encoders in one file
    # This is the fix for the LabelEncoder mismatch bug
    joblib.dump(column_encoders, "models/column_encoders.pkl")
    print("      Saved: models/column_encoders.pkl")

    # Save feature column names (for correct ordering in stream.py)
    joblib.dump(feature_names, "models/feature_names.pkl")
    print("      Saved: models/feature_names.pkl")

    # Save attack category map (for dashboard display)
    joblib.dump(ATTACK_CATEGORY_MAP, "models/attack_category_map.pkl")
    print("      Saved: models/attack_category_map.pkl")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  NSL-KDD Intrusion Detection — Training Pipeline")
    print("=" * 55)

    df_train, df_test = load_data()

    (
        X_train, y_train,
        X_test,  y_test,
        scaler, label_encoder,
        column_encoders, feature_names
    ) = preprocess(df_train, df_test)

    model = train_model(X_train, y_train)

    evaluate(model, X_test, y_test, label_encoder, feature_names)

    save_models(
        model, scaler, label_encoder,
        column_encoders, feature_names
    )

    print("\n" + "=" * 55)
    print("  ✅ Training complete!")
    print("  You can now run:  python app.py")
    print("=" * 55)
