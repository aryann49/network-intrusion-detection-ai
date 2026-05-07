# AI-Based Real-Time Network Intrusion Detection and Threat Classification System

A machine learning powered Intrusion Detection System (IDS) that monitors network traffic in real time, detects malicious activity, classifies attack types, and displays live results on a web dashboard.

Built using the **NSL-KDD** dataset and a **Random Forest** classifier with a **Flask + SocketIO** real-time dashboard.

---

## Features

- Real-time traffic simulation using the NSL-KDD dataset
- Random Forest ML model with **86% accuracy**
- Detects 4 attack categories: DoS, Probe, R2L, U2R
- Live dashboard — alerts update automatically, no page refresh needed
- Attack distribution chart, detection rate, and threat counters
- Clean architecture — stream engine runs as background thread inside Flask

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| Machine Learning | scikit-learn (Random Forest) |
| Data Processing | pandas, numpy |
| Model Storage | joblib |
| Web Framework | Flask |
| Real-Time Communication | Flask-SocketIO |
| Frontend | HTML, CSS, JavaScript |
| Dataset | NSL-KDD |

---

## Project Structure

```
intrusion_detection_project/
│
├── data/
│   ├── KDDTrain+.txt           ← training dataset
│   └── KDDTest+.txt            ← test dataset
│
├── models/                     ← created automatically by train.py
│   ├── random_forest.pkl       ← trained Random Forest model
│   ├── scaler.pkl              ← StandardScaler parameters
│   ├── label_encoder.pkl       ← attack label decoder
│   ├── column_encoders.pkl     ← categorical feature encoders
│   ├── feature_names.pkl       ← column order reference
│   └── attack_category_map.pkl ← maps specific attacks to categories
│
├── templates/
│   └── index.html              ← live dashboard frontend
│
├── train.py                    ← Step 1: train and save the model
├── app.py                      ← Step 2: run the full system
├── requirements.txt            ← all dependencies
└── README.md                   ← this file
```

---

## Setup Instructions

### Step 1 — Download the Dataset

Go to: https://www.unb.ca/cic/datasets/nsl.html

Download `KDDTrain+.txt` and `KDDTest+.txt` and place both inside the `data/` folder.

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Train the Model (run only once)

```bash
python train.py
```

This will load the dataset, preprocess it, train the Random Forest model, print accuracy results, and save all model files into `models/`.

Expected output:
```
✅ Accuracy : 86.03%
✅ Training complete! You can now run: python app.py
```

### Step 4 — Run the System

```bash
python app.py
```

Open your browser and go to:
```
http://127.0.0.1:5000
```

The dashboard will immediately begin showing live detections.

---

## Model Performance

| Metric | Value |
|---|---|
| Accuracy | 86.03% |
| Dataset | NSL-KDD |
| Algorithm | Random Forest (100 trees, max_depth=20) |
| Training samples | 125,973 |
| Test samples | 22,544 |

### Attack Categories Detected

| Category | Description | Example Attacks |
|---|---|---|
| DoS | Denial of Service | neptune, smurf, back, pod, teardrop |
| Probe | Network Scanning | ipsweep, nmap, portsweep, satan |
| R2L | Remote to Local | guess_passwd, ftp_write, imap, multihop |
| U2R | User to Root | buffer_overflow, rootkit, loadmodule, perl |

---

## System Architecture

```
NSL-KDD Dataset
      ↓
train.py
  Preprocessing → Encoding → Scaling → Train RF → Save 6 model files
      ↓
app.py
  Load models → Start Flask server → Launch background thread
      ↓
Background Thread (same process = shared memory)
  Stream dataset rows one by one → preprocess → scale → predict
      ↓
Random Forest Prediction
  Decode label → map to category (DoS / Probe / R2L / U2R / Normal)
      ↓
SocketIO push → Browser updates live (no refresh)
      ↓
Dashboard: stats counter + alert feed + doughnut chart
```

---

## Key Architecture Decisions

**Why Random Forest?**
Random Forest handles tabular network data extremely well. It manages class imbalance, delivers high accuracy without heavy hyperparameter tuning, and is fast enough for simulated real-time use.

**Why simulated real-time traffic instead of live packet capture?**
Live packet capture with Scapy requires administrator privileges and Npcap drivers, making it unreliable across machines. Dataset streaming produces identical IDS behavior and is the standard academic approach for IDS research systems.

**Why Flask-SocketIO instead of page refresh?**
SocketIO pushes each detection from server to browser the instant it happens. A page refresh approach loses alerts and adds delay. All production security dashboards use WebSockets for this reason.

**Why run the stream engine inside Flask as a thread?**
Running `stream.py` as a separate process gives each process its own copy of the alerts list in memory — the dashboard never receives the alerts. A background thread inside Flask shares the same memory space, which is the correct architecture.

---

## Future Improvements

- Deep learning models (LSTM for sequence-based anomaly detection)
- Live packet capture with proper privilege handling
- SQLite database for persistent alert logging
- Historical attack trend charts
- Automated response simulation (IP block logging)
- Cloud deployment on AWS or GCP

---

## Author

**Aryan**  
MCA Student · Amity University, Noida  
Interests: Cybersecurity, AI/ML, Penetration Testing

---

## Dataset Reference

NSL-KDD Dataset  
Canadian Institute for Cybersecurity, University of New Brunswick  
https://www.unb.ca/cic/datasets/nsl.html
