import pandas as pd
import numpy as np
import time
import joblib
from sklearn.preprocessing import LabelEncoder

from scripts.shared_data import alerts

# load trained files
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# load dataset
df = pd.read_csv(
    "data/KDDTrain+.txt",
    header=None
)

# separate features only
X = df.iloc[:, :-2]

# encode categorical columns
for column in X.select_dtypes(
    include=['object', 'string']
).columns:

    le = LabelEncoder()

    X[column] = le.fit_transform(X[column])

# simulate real-time traffic
for index, row in X.iterrows():

    print(f"\nTraffic Record #{index}")

    # convert row into array
    sample = np.array([row.values])

    # scale
    sample_scaled = scaler.transform(sample)

    # predict
    prediction = model.predict(sample_scaled)

    # decode label
    result = encoder.inverse_transform(prediction)

    attack_type = result[0]

    print("Prediction:", attack_type)

    # create live alert
    alert = {
        "traffic_id": index,
        "prediction": attack_type
    }

    alerts.append(alert)

    # keep only latest 20 alerts
    if len(alerts) > 20:
        alerts.pop(0)

    # terminal alert
    if attack_type != "normal":

        print("⚠ ALERT: Intrusion Detected!")

        print("Attack Type:", attack_type)

    else:

        print("Status: Normal Traffic")

    print("-" * 50)

    time.sleep(1)