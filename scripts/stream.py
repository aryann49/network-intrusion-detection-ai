"""
Standalone console stream for NSL-KDD detections.

The main dashboard stream lives in app.py. This script is only a lightweight
CLI helper for checking that saved model artifacts can process dataset rows.
Usage:
    python scripts/stream.py
"""

import os
import time

import pandas as pd

from predict import DATA_PATH, load_artifacts, preprocess_row
from train import COLUMN_NAMES


def main(limit=20, delay=0.3):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("data/KDDTest+.txt not found.")

    artifacts = load_artifacts()
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)

    for index, row in df.head(limit).iterrows():
        actual_label = str(row["label"]).strip()
        features = row.drop(labels=["label", "difficulty"])
        scaled = preprocess_row(features, artifacts)

        pred_input = scaled.to_numpy()
        pred = artifacts["model"].predict(pred_input)
        proba = artifacts["model"].predict_proba(pred_input)[0]
        label = artifacts["label_encoder"].inverse_transform(pred)[0]
        category = artifacts["category_map"].get(label.lower(), "Unknown")

        status = "ATTACK" if label != "normal" else "NORMAL"
        print(
            f"#{index:05d} {status:<6} predicted={label:<18} "
            f"category={category:<7} actual={actual_label:<18} "
            f"confidence={proba.max() * 100:.1f}%"
        )
        time.sleep(delay)


if __name__ == "__main__":
    main()
