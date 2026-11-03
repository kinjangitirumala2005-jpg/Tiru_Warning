import os
import cv2
import time
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify

from src.config import Config
from src.database import db, DetectionLog, AlertRecord
from src.alert_service import AlertService
from src.detector import detector_instance

# Initialize Flask App
app = Flask(
    __name__,
    template_folder="web/templates",
    static_folder="web/static"
)
app.config.from_object(Config)

# Ensure required storage folders exist
Config.ensure_directories()

# Initialize Database
db.init_app(app)
with app.app_context():
    db.create_all()

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


@app.context_processor
def inject_global_data():
    """Injects dynamic telemetry info into all templates."""
    engine_name = "MobileNetV2 CNN" if detector_instance.model_loaded else "Computer Vision Heuristic (Baseline)"
    return {
        "active_engine": engine_name,
        "config_data": Config
    }


def get_dashboard_stats():
    """Calculates overall incident totals."""
    total_detections = DetectionLog.query.count()
    total_smoke = DetectionLog.query.filter_by(is_smoke=True).count()
    total_clear = DetectionLog.query.filter_by(is_smoke=False).count()
    total_alerts = AlertRecord.query.count()
    return {
        "total_detections": total_detections,
        "total_smoke": total_smoke,
        "total_clear": total_clear,
        "total_alerts": total_alerts
    }


# ==============================================================
# Routes
# ==============================================================

@app.route("/")
def index():
    stats = get_dashboard_stats()
    recent_logs = [log.to_dict() for log in DetectionLog.query.order_by(DetectionLog.timestamp.desc()).limit(8).all()]
    return render_template(
        "index.html",
        stats=stats,
        recent_logs=recent_logs,
        latest_result=None
    )


@app.route("/upload", methods=["POST"])
def upload_media():
    if "file" not in request.files:
        flash("No file part selected.", "error")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file chosen.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a JPG or PNG image.", "error")
        return redirect(url_for("index"))

    camera_name = request.form.get("camera_name", Config.DEFAULT_CAMERA_NAME)
    location = request.form.get("location", Config.DEFAULT_LOCATION)

    # Save original uploaded image
    filename = secure_filename(f"{int(time.time())}_{file.filename}")
    original_save_path = Config.UPLOAD_FOLDER / filename
    file.save(str(original_save_path))

    # Read image into OpenCV
    image_bgr = cv2.imread(str(original_save_path))
    if image_bgr is None:
        flash("Could not decode image file.", "error")
        return redirect(url_for("index"))

    # Run inference
    result = detector_instance.predict_frame(image_bgr)
    
    # Generate annotated image
    annotated_bgr = detector_instance.annotate_frame(image_bgr, result, camera_name)
    annotated_filename = f"annotated_{filename}"
    annotated_save_path = Config.UPLOAD_FOLDER / annotated_filename
    cv2.imwrite(str(annotated_save_path), annotated_bgr)

    # Check alert dispatch eligibility
    alert_triggered = False
    if result["is_smoke"] and result["confidence"] >= Config.CONFIDENCE_THRESHOLD:
        if AlertService.can_send_alert(camera_name):
            alert_triggered = True

    # Record detection into DB
    log_entry = DetectionLog(
        source_type="image",
        camera_name=camera_name,
        location=location,
        is_smoke=result["is_smoke"],
        confidence=result["confidence"],
        label=result["label"],
        image_path=f"uploads/{filename}",
        annotated_path=f"uploads/{annotated_filename}",
        alert_triggered=alert_triggered
    )
    db.session.add(log_entry)
    db.session.commit()

    # Trigger asynchronous alert if qualified
    if alert_triggered:
        AlertService.dispatch_alert_async(
            app=app,
            detection_id=log_entry.id,
            camera_name=camera_name,
            location=location,
            confidence=result["confidence"],
            image_full_path=annotated_save_path
        )

    stats = get_dashboard_stats()
    recent_logs = [l.to_dict() for l in DetectionLog.query.order_by(DetectionLog.timestamp.desc()).limit(8).all()]

    return render_template(
        "index.html",
        stats=stats,
        recent_logs=recent_logs,
        latest_result=log_entry.to_dict()
    )


@app.route("/live")
def live_view():
    camera_source = request.args.get("camera_source", "0")
    return render_template("live_feed.html", camera_source=camera_source)


def generate_video_stream(source="0"):
    """MJPEG streaming generator with graceful synthetic test pattern fallback."""
    # Convert numeric camera source strings (e.g. '0') to int
    try:
        source_val = int(source)
    except ValueError:
        source_val = source

    cap = cv2.VideoCapture(source_val)
    use_synthetic = not cap.isOpened()

    frame_counter = 0

    try:
        while True:
            if not use_synthetic:
                success, frame = cap.read()
                if not success:
                    # Loop back or switch to test pattern
                    cap.release()
                    use_synthetic = True
                    continue
            else:
                # Generate animated test surveillance frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:, :] = (25, 30, 38) # Dark monitoring slate
                # Draw synthetic grid lines
                for y in range(0, 480, 80):
                    cv2.line(frame, (0, y), (640, y), (35, 42, 54), 1)
                for x in range(0, 640, 80):
                    cv2.line(frame, (x, 0), (x, 480), (35, 42, 54), 1)

                # Simulated surveillance text
                cv2.putText(frame, "SIMULATED SURVEILLANCE FEED", (140, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 140, 160), 2)
                cv2.putText(frame, "No Physical Camera / Feed Attached", (155, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 105, 120), 1)
                time.sleep(0.04)

            frame_counter += 1

            # Run inference every 3rd frame to optimize CPU load
            if frame_counter % 3 == 0:
                result = detector_instance.predict_frame(frame)
                
                # If an alert is confirmed by the temporal smoother, log it and dispatch
                if result.get("confirmed_alert", False):
                    cam_name = Config.DEFAULT_CAMERA_NAME
                    if AlertService.can_send_alert(cam_name):
                        # Save capture snapshot
                        snap_filename = f"live_alert_{int(time.time())}.jpg"
                        snap_path = Config.CAPTURES_FOLDER / snap_filename
                        annotated_snapshot = detector_instance.annotate_frame(frame, result, cam_name)
                        cv2.imwrite(str(snap_path), annotated_snapshot)

                        with app.app_context():
                            log_entry = DetectionLog(
                                source_type="live",
                                camera_name=cam_name,
                                location=Config.DEFAULT_LOCATION,
                                is_smoke=True,
                                confidence=result["confidence"],
                                label=result["label"],
                                annotated_path=f"captures/{snap_filename}",
                                alert_triggered=True
                            )
                            db.session.add(log_entry)
                            db.session.commit()

                            AlertService.dispatch_alert_async(
                                app=app,
                                detection_id=log_entry.id,
                                camera_name=cam_name,
                                location=Config.DEFAULT_LOCATION,
                                confidence=result["confidence"],
                                image_full_path=snap_path
                            )

                annotated_frame = detector_instance.annotate_frame(frame, result, Config.DEFAULT_CAMERA_NAME)
            else:
                # Draw overlay with current status
                annotated_frame = frame

            # Encode frame to JPEG
            ret, buffer = cv2.imencode(".jpg", annotated_frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    finally:
        if cap.isOpened():
            cap.release()


@app.route("/video_feed")
def video_feed():
    camera_source = request.args.get("source", "0")
    return Response(
        generate_video_stream(camera_source),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/alerts")
def alerts_view():
    filter_type = request.args.get("filter", "all")
    query = DetectionLog.query.order_by(DetectionLog.timestamp.desc())

    if filter_type == "smoke":
        query = query.filter_by(is_smoke=True)
    elif filter_type == "alerts":
        query = query.filter_by(alert_triggered=True)

    logs = query.limit(100).all()
    return render_template("alerts.html", logs=logs, current_filter=filter_type)


@app.route("/settings")
def settings_view():
    return render_template("settings.html")


@app.route("/settings/save", methods=["POST"])
def save_settings():
    try:
        conf_thresh = float(request.form.get("confidence_threshold", Config.CONFIDENCE_THRESHOLD))
        consec_frames = int(request.form.get("consecutive_frames", Config.CONSECUTIVE_FRAMES_REQUIRED))
        cooldown = int(request.form.get("alert_cooldown", Config.ALERT_COOLDOWN_SECONDS))

        Config.CONFIDENCE_THRESHOLD = conf_thresh
        Config.CONSECUTIVE_FRAMES_REQUIRED = consec_frames
        Config.ALERT_COOLDOWN_SECONDS = cooldown
        detector_instance.smoother.required_consecutive = consec_frames

        flash("Settings saved successfully.", "success")
    except Exception as e:
        flash(f"Error saving settings: {e}", "error")

    return redirect(url_for("settings_view"))


@app.route("/api/stats")
def api_stats():
    return jsonify(get_dashboard_stats())


@app.route("/api/test-alert", methods=["POST"])
def api_test_alert():
    try:
        cam_name = Config.DEFAULT_CAMERA_NAME
        loc = Config.DEFAULT_LOCATION
        simulated = not Config.EMAIL_ALERTS_ENABLED or not Config.SMTP_USERNAME

        log_entry = DetectionLog(
            source_type="test",
            camera_name=cam_name,
            location=loc,
            is_smoke=True,
            confidence=0.92,
            label="Smoke Detected (Test Drill)",
            alert_triggered=True
        )
        db.session.add(log_entry)
        db.session.commit()

        AlertService.dispatch_alert_async(
            app=app,
            detection_id=log_entry.id,
            camera_name=cam_name,
            location=loc,
            confidence=0.92
        )

        mode_text = "simulated & logged to DB (enable live SMTP in .env to send real emails)" if simulated else f"dispatched to {Config.ALERT_RECIPIENT_EMAIL}"
        return jsonify({"success": True, "message": f"Test alert successfully {mode_text}."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

