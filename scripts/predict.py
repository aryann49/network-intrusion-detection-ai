import joblib
import numpy as np

# load model
model = joblib.load("models/random_forest_model.pkl")

# load scaler
scaler = joblib.load("models/scaler.pkl")

# load label encoder
encoder = joblib.load("models/label_encoder.pkl")

# sample input
sample = np.array([[0]*41])

# scale input
sample_scaled = scaler.transform(sample)

# predict
prediction = model.predict(sample_scaled)

# decode label
decoded_prediction = encoder.inverse_transform(prediction)

print("Prediction:", decoded_prediction)