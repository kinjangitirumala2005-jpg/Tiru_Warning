# 🌲 Wildfire Smoke Early-Warning Detector (CNN + OpenCV + Flask)

An intelligent, real-time early-warning computer vision system designed to detect wildfire smoke signatures from forest surveillance watchtowers, analyze frames using a Deep Convolutional Neural Network (**MobileNetV2** with Transfer Learning), and dispatch automated alert notifications with location telemetry to forest department officials.

---

## 🚀 Key Features

* **CNN Inference Core:** MobileNetV2 architecture with Transfer Learning (pre-trained on ImageNet, fine-tuned on wildfire smoke imagery) providing the ideal trade-off between inference latency and classification precision.
* **OpenCV Image & Live Stream Processor:** Accepts static photo uploads, recorded video footage, or continuous live camera streams (Webcam / RTSP watchtower IP cameras).
* **Temporal Smoothing (False-Alarm Reduction):** Filters out fleeting atmospheric noise (clouds, dust, fog, sunlight reflections) by requiring $k$ consecutive positive frames before triggering an alert.
* **Automated SMTP Notification Dispatcher:** Dispatches formatted emergency notification emails with surveillance snapshots, watchtower identity, and GPS coordinates to authorities, equipped with alert cooldown timers to prevent spam.
* **Modern Bootstrap 5 Dashboard:** Real-time statistics, diagnostics HUD, incident history table, and system parameter controls.
* **Zero-Setup Database Portability:** Seamlessly defaults to SQLite for local development and easily switches to MySQL or PostgreSQL in production via environment configuration.

---

## 📂 Project Architecture

```text
Early-Warning-Detector/
├── app.py                      # Flask application entrypoint & API routes
├── requirements.txt            # Dependency specification
├── .env.example                # Configuration & credentials template
├── notebooks/
│   └── train_wildfire_smoke.ipynb # GPU training notebook ready for Google Colab
├── src/
│   ├── config.py               # Global settings & environment variables
│   ├── database.py             # SQLAlchemy ORM models (Detections & Alerts)
│   ├── detector.py             # OpenCV inference engine with temporal smoothing
│   ├── alert_service.py        # Background SMTP notification dispatcher
│   └── train.py                # Standalone Python model training pipeline
└── web/
    ├── static/
    │   ├── css/style.css       # Dark-mode dashboard styling
    │   ├── js/dashboard.js     # Drag-and-drop & metrics polling
    │   ├── uploads/            # Scanned image snapshots
    │   └── captures/           # Live alert frame captures
    └── templates/
        ├── base.html           # Layout shell with navigation & telemetry pill
        ├── index.html          # Upload interface & analysis diagnostics
        ├── live_feed.html      # Real-time webcam / RTSP monitoring feed
        ├── alerts.html         # Historical detection records & audit logs
        └── settings.html       # Sensitivity & SMTP configuration
```

---

## 🛠️ Quickstart Installation

### 1. Clone & Set Up Virtual Environment

Using Python 3.10:

```bash
# Create virtual environment
py -3.10 -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```

*(Optional)* Configure your Gmail SMTP credentials inside `.env` to receive real email alerts. When left unconfigured, the system automatically runs in **simulation mode**, cleanly logging all events to the database without throwing connection errors.

### 3. Launch the Application

```bash
python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 🧠 Model Training (Google Colab or Local GPU)

### Using Google Colab (Recommended):
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `notebooks/train_wildfire_smoke.ipynb`.
3. Set runtime to **GPU** (`Runtime > Change runtime type > T4 GPU`).
4. Place or mount your dataset into `smoke/` and `no_smoke/` folders.
5. Run the cells to train MobileNetV2 with Early Stopping and Augmentation.
6. Download the generated `smoke_detector_mobilenetv2.keras` and place it in the `models/` directory of this project.

### Using Local Training Script:
```bash
python src/train.py --data data/processed --model mobilenetv2 --epochs 20 --batch 32
```

---

## 🌐 Deploying to Render / Cloud

1. Set Environment Variable `DATABASE_URL` to your production MySQL or PostgreSQL database.
2. In the Render Web Service settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app` (or `python app.py`)

