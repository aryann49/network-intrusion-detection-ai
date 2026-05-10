"""
Standalone prediction helper.

Runs one NSL-KDD record through the same artifacts used by app.py.
Usage:
    python scripts/predict.py
"""

import os
import sys

import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train import COLUMN_NAMES


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


REQUIRED = [
    project_path("models", "random_forest.pkl"),
    project_path("models", "scaler.pkl"),
    project_path("models", "label_encoder.pkl"),
    project_path("models", "column_encoders.pkl"),
    project_path("models", "feature_names.pkl"),
    project_path("models", "attack_category_map.pkl"),
]
DATA_PATH = project_path("data", "KDDTest+.txt")


def load_artifacts():
    missing = [path for path in REQUIRED if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts. Run `python train.py` first: "
            + ", ".join(missing)
        )

    return {
        "model": joblib.load(project_path("models", "random_forest.pkl")),
        "scaler": joblib.load(project_path("models", "scaler.pkl")),
        "label_encoder": joblib.load(project_path("models", "label_encoder.pkl")),
        "column_encoders": joblib.load(project_path("models", "column_encoders.pkl")),
        "feature_names": joblib.load(project_path("models", "feature_names.pkl")),
        "category_map": joblib.load(project_path("models", "attack_category_map.pkl")),
    }


def preprocess_row(row, artifacts):
    row = row.copy()
    for col, encoder in artifacts["column_encoders"].items():
        value = row[col]
        row[col] = encoder.transform([value])[0] if value in encoder.classes_ else 0

    row = row[artifacts["feature_names"]]
    frame = row.to_frame().T.astype(float)
    return pd.DataFrame(
        artifacts["scaler"].transform(frame),
        columns=artifacts["feature_names"],
    )


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("data/KDDTest+.txt not found.")

    artifacts = load_artifacts()
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)
    row = df.iloc[0].drop(labels=["label", "difficulty"])

    scaled = preprocess_row(row, artifacts)
    pred_input = scaled.to_numpy()
    pred = artifacts["model"].predict(pred_input)
    proba = artifacts["model"].predict_proba(pred_input)[0]
    label = artifacts["label_encoder"].inverse_transform(pred)[0]
    category = artifacts["category_map"].get(label.lower(), "Unknown")

    print("Prediction:", label)
    print("Category:", category)
    print(f"Confidence: {proba.max() * 100:.1f}%")


if __name__ == "__main__":
    main()
