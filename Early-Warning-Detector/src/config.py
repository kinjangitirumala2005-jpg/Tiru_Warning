import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory of Project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present
load_dotenv(BASE_DIR / ".env")

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "wildfire-smoke-detector-secret-key-2026")
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'wildfire_detector.db'}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Directories
    BASE_DIR = BASE_DIR
    MODELS_DIR = BASE_DIR / "models"
    UPLOAD_FOLDER = BASE_DIR / "web" / "static" / "uploads"
    CAPTURES_FOLDER = BASE_DIR / "web" / "static" / "captures"
    
    # Model configuration
    DEFAULT_MODEL_NAME = "smoke_detector_mobilenetv2.keras"
    LEGACY_MODEL_NAME = "smoke_detector_mobilenetv2.h5"
    INPUT_SIZE = (224, 224)
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.75))
    
    # False Alarm Reduction (Temporal Smoothing)
    # Require N consecutive frames classified as smoke before issuing an alert
    CONSECUTIVE_FRAMES_REQUIRED = int(os.getenv("CONSECUTIVE_FRAMES_REQUIRED", 5))
    
    # Alert Cooldown: Suppress repeated alerts for the same camera for N seconds
    ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", 300))

    # Email SMTP Alert Settings
    EMAIL_ALERTS_ENABLED = os.getenv("EMAIL_ALERTS_ENABLED", "False").lower() in ("true", "1", "t", "yes")
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    ALERT_RECIPIENT_EMAIL = os.getenv("ALERT_RECIPIENT_EMAIL", "")
    
    # Camera Metadata
    DEFAULT_CAMERA_NAME = os.getenv("DEFAULT_CAMERA_NAME", "Watchtower 4 - Sector Alpha")
    DEFAULT_LOCATION = os.getenv("DEFAULT_LOCATION", "Ridge Forest Range (28.61° N, 77.20° E)")

    @classmethod
    def ensure_directories(cls):
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        cls.CAPTURES_FOLDER.mkdir(parents=True, exist_ok=True)

