from datetime import datetime
from typing import Any
from flask_sqlalchemy import SQLAlchemy

db: Any = SQLAlchemy()

class DetectionLog(db.Model):
    """Stores every processed detection event (image upload, video clip, or live camera alert)."""
    __tablename__ = "detection_logs"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    source_type = db.Column(db.String(20), nullable=False) # 'image', 'video', 'live'
    camera_name = db.Column(db.String(100), default="Surveillance Station 1")
    location = db.Column(db.String(150), default="Sector Alpha")
    
    # Classification Results
    is_smoke = db.Column(db.Boolean, nullable=False, index=True)
    confidence = db.Column(db.Float, nullable=False) # 0.0 to 1.0
    label = db.Column(db.String(30), nullable=False) # 'Smoke Detected' or 'No Smoke'
    
    # Media paths (relative to static)
    image_path = db.Column(db.String(255), nullable=True)
    annotated_path = db.Column(db.String(255), nullable=True)

    # Status / Notes
    alert_triggered = db.Column(db.Boolean, default=False)
    notes = db.Column(db.Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "source_type": self.source_type,
            "camera_name": self.camera_name,
            "location": self.location,
            "is_smoke": self.is_smoke,
            "confidence": round(self.confidence * 100, 2),
            "label": self.label,
            "image_path": self.image_path,
            "annotated_path": self.annotated_path,
            "alert_triggered": self.alert_triggered
        }


class AlertRecord(db.Model):
    """Tracks email and SMS notifications dispatched to forest officers."""
    __tablename__ = "alert_records"

    id = db.Column(db.Integer, primary_key=True)
    detection_id = db.Column(db.Integer, db.ForeignKey("detection_logs.id"), nullable=True)
    sent_at = db.Column(db.DateTime, default=datetime.utcnow)
    recipient = db.Column(db.String(120), nullable=False)
    channel = db.Column(db.String(20), default="EMAIL") # 'EMAIL', 'SMS'
    status = db.Column(db.String(20), default="SENT")    # 'SENT', 'FAILED', 'SIMULATED'
    subject = db.Column(db.String(200), nullable=False)
    message_body = db.Column(db.Text, nullable=False)
    error_message = db.Column(db.Text, nullable=True)

    detection = db.relationship("DetectionLog", backref="alerts", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "sent_at": self.sent_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "recipient": self.recipient,
            "channel": self.channel,
            "status": self.status,
            "subject": self.subject,
            "detection_id": self.detection_id
        }

