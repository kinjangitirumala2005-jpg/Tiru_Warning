import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime, timedelta
from pathlib import Path
from src.config import Config
from src.database import db, AlertRecord

class AlertService:
    """Manages dispatching early-warning wildfire notifications via Email SMTP."""
    
    _last_alert_time = {} # camera_name -> datetime
    _lock = threading.Lock()

    @classmethod
    def can_send_alert(cls, camera_name: str) -> bool:
        """Enforces alert cooldown window to avoid spamming officials."""
        with cls._lock:
            now = datetime.utcnow()
            last_time = cls._last_alert_time.get(camera_name)
            if last_time and (now - last_time) < timedelta(seconds=Config.ALERT_COOLDOWN_SECONDS):
                return False
            cls._last_alert_time[camera_name] = now
            return True

    @classmethod
    def dispatch_alert_async(cls, app, detection_id, camera_name, location, confidence, image_full_path=None):
        """Dispatches an alert in a background thread so the inference loop remains real-time."""
        thread = threading.Thread(
            target=cls._send_alert_worker,
            args=(app, detection_id, camera_name, location, confidence, image_full_path),
            daemon=True
        )
        thread.start()

    @classmethod
    def _send_alert_worker(cls, app, detection_id, camera_name, location, confidence, image_full_path):
        with app.app_context():
            subject = f"🚨 URGENT: Wildfire Smoke Detected at {camera_name}"
            timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            
            body_text = (
                f"WILDFIRE SMOKE EARLY-WARNING ALERT\n"
                f"===================================\n"
                f"Camera: {camera_name}\n"
                f"Location: {location}\n"
                f"Confidence: {confidence * 100:.2f}%\n"
                f"Timestamp: {timestamp_str}\n\n"
                f"Immediate field verification recommended. Check the central monitoring dashboard."
            )

            body_html = f"""
            <html>
            <body style="font-family: Arial, sans-serif; background-color: #f7f9fa; padding: 20px;">
                <div style="max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 8px; border-left: 6px solid #dc3545; padding: 25px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
                    <h2 style="color: #dc3545; margin-top: 0;">🚨 Wildfire Smoke Early Warning</h2>
                    <p style="color: #495057; font-size: 15px;">
                        The automated CNN computer vision system has detected smoke signatures matching a potential wildfire outbreak.
                    </p>
                    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                        <tr style="border-bottom: 1px solid #dee2e6;">
                            <td style="padding: 8px 0; font-weight: bold; color: #212529;">Watch Station:</td>
                            <td style="padding: 8px 0; color: #495057;">{camera_name}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #dee2e6;">
                            <td style="padding: 8px 0; font-weight: bold; color: #212529;">Coordinates / Sector:</td>
                            <td style="padding: 8px 0; color: #495057;">{location}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #dee2e6;">
                            <td style="padding: 8px 0; font-weight: bold; color: #212529;">Detection Confidence:</td>
                            <td style="padding: 8px 0; color: #dc3545; font-weight: bold;">{confidence * 100:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #dee2e6;">
                            <td style="padding: 8px 0; font-weight: bold; color: #212529;">Timestamp:</td>
                            <td style="padding: 8px 0; color: #495057;">{timestamp_str}</td>
                        </tr>
                    </table>
                    <div style="background-color: #fff3cd; border: 1px solid #ffeeba; border-radius: 4px; padding: 12px; margin-top: 15px; color: #856404; font-size: 14px;">
                        <strong>Action Required:</strong> Dispatch ground reconnaissance or verify via adjacent watchtower feeds.
                    </div>
                </div>
            </body>
            </html>
            """

            recipient = Config.ALERT_RECIPIENT_EMAIL or "forest-duty-officer@local"
            
            # If email alerts are disabled or credentials are dummy, log as simulated
            if not Config.EMAIL_ALERTS_ENABLED or not Config.SMTP_USERNAME or not Config.ALERT_RECIPIENT_EMAIL:
                print(f"[AlertService: SIMULATED] Alert logged for {camera_name} (Confidence: {confidence * 100:.1f}%)")
                record = AlertRecord(
                    detection_id=detection_id,
                    recipient=recipient,
                    channel="EMAIL",
                    status="SIMULATED",
                    subject=subject,
                    message_body=body_text
                )
                db.session.add(record)
                db.session.commit()
                return

            # Live SMTP dispatch
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = Config.SMTP_USERNAME
                msg["To"] = recipient

                msg.attach(MIMEText(body_text, "plain"))
                msg.attach(MIMEText(body_html, "html"))

                if image_full_path and Path(image_full_path).exists():
                    try:
                        with open(image_full_path, "rb") as f:
                            img_data = f.read()
                            img_part = MIMEImage(img_data, name=Path(image_full_path).name)
                            msg.attach(img_part)
                    except Exception as e:
                        print(f"[AlertService] Could not attach capture image: {e}")

                server = smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT, timeout=10)
                server.starttls()
                server.login(Config.SMTP_USERNAME, Config.SMTP_PASSWORD)
                server.sendmail(Config.SMTP_USERNAME, [recipient], msg.as_string())
                server.quit()

                record = AlertRecord(
                    detection_id=detection_id,
                    recipient=recipient,
                    channel="EMAIL",
                    status="SENT",
                    subject=subject,
                    message_body=body_text
                )
                db.session.add(record)
                db.session.commit()
                print(f"[AlertService] Email successfully sent to {recipient}")

            except Exception as e:
                print(f"[AlertService Error] Failed to send email: {e}")
                record = AlertRecord(
                    detection_id=detection_id,
                    recipient=recipient,
                    channel="EMAIL",
                    status="FAILED",
                    subject=subject,
                    message_body=body_text,
                    error_message=str(e)
                )
                db.session.add(record)
                db.session.commit()

