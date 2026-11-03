import cv2
import numpy as np
from collections import deque
from src.config import Config


# Try to import TensorFlow; if not installed or model not present, we use intelligent CV fallback
TF_AVAILABLE = False
try:
    import tensorflow as tf  # type: ignore
    TF_AVAILABLE = True
except ImportError:
    pass


class TemporalSmoother:
    """Reduces false alarms by requiring consecutive positive frames before triggering an alert."""
    def __init__(self, required_consecutive=5, window_size=10):
        self.required_consecutive = required_consecutive
        self.history = deque(maxlen=window_size)
        self.consecutive_count = 0

    def update(self, is_smoke: bool, confidence: float) -> tuple[bool, bool]:
        """
        Returns:
            (raw_detection, confirmed_alert)
        """
        self.history.append(is_smoke)
        if is_smoke and confidence >= Config.CONFIDENCE_THRESHOLD:
            self.consecutive_count += 1
        else:
            self.consecutive_count = max(0, self.consecutive_count - 1)
            
        confirmed_alert = self.consecutive_count >= self.required_consecutive
        return is_smoke, confirmed_alert

    def reset(self):
        self.history.clear()
        self.consecutive_count = 0


class SmokeDetector:
    """Core inference engine for Wildfire Smoke Detection."""

    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.model_path = None
        self.smoother = TemporalSmoother(
            required_consecutive=Config.CONSECUTIVE_FRAMES_REQUIRED
        )
        self.load_model()

    def load_model(self):
        """Attempts to load a trained Keras/TensorFlow model if available."""
        if not TF_AVAILABLE:
            print("[Detector] TensorFlow is not installed in the active environment. Running with Computer Vision Heuristic Engine.")
            return

        candidate_files = [
            Config.MODELS_DIR / Config.DEFAULT_MODEL_NAME,
            Config.MODELS_DIR / Config.LEGACY_MODEL_NAME,
            Config.MODELS_DIR / "smoke_detector.h5"
        ]

        for path in candidate_files:
            if path.exists():
                try:
                    self.model = tf.keras.models.load_model(str(path))
                    self.model_loaded = True
                    self.model_path = str(path)
                    print(f"[Detector] Successfully loaded CNN model from: {path}")
                    return
                except Exception as e:
                    print(f"[Detector] Error loading model from {path}: {e}")

        print("[Detector] No pre-trained .keras/.h5 model found in models/ directory. Using CV Heuristic fallback for testing.")

    def _heuristic_smoke_analysis(self, image_bgr: np.ndarray) -> tuple[bool, float]:
        """
        Intelligent fallback heuristic based on chrominance & luminance characteristics of smoke:
        Smoke in forest canopies typically exhibits low color saturation (gray/white/light-brown),
        high luminance dispersion, and high edge diffusion compared to green foliage.
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Foliage mask (greens)
        green_mask = cv2.inRange(hsv, (25, 40, 20), (85, 255, 255))
        foliage_ratio = np.count_nonzero(green_mask) / float(image_bgr.shape[0] * image_bgr.shape[1])
        
        # Smoke candidates: Low saturation (desaturated) with moderate to high brightness
        smoke_mask = cv2.inRange(hsv, (0, 0, 110), (180, 55, 245))
        # Mask out foliage regions
        smoke_mask = cv2.bitwise_and(smoke_mask, smoke_mask, mask=cv2.bitwise_not(green_mask))
        smoke_ratio = np.count_nonzero(smoke_mask) / float(image_bgr.shape[0] * image_bgr.shape[1])

        # Baseline confidence estimation
        if smoke_ratio > 0.15:
            conf = min(0.96, 0.65 + (smoke_ratio * 0.8))
            is_smoke = conf >= Config.CONFIDENCE_THRESHOLD
        elif smoke_ratio > 0.08:
            conf = 0.50 + (smoke_ratio * 1.5)
            is_smoke = False
        else:
            conf = max(0.04, 1.0 - (foliage_ratio * 0.9))
            is_smoke = False

        return is_smoke, float(conf)

    def predict_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Infers whether the given frame contains smoke.
        Returns: {
            'is_smoke': bool,
            'confidence': float (0.0 to 1.0),
            'label': str,
            'confirmed_alert': bool
        }
        """
        if self.model_loaded and self.model is not None:
            try:
                # Preprocess for MobileNetV2 (input shape: 224x224x3, RGB normalized)
                resized = cv2.resize(frame_bgr, Config.INPUT_SIZE)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                input_arr = np.expand_dims(rgb, axis=0).astype(np.float32) / 255.0
                
                # Single binary output (Sigmoid): 1 = Smoke, 0 = No Smoke
                pred = self.model.predict(input_arr, verbose=0)[0]
                raw_prob = float(pred[0]) if isinstance(pred, (list, np.ndarray)) else float(pred)
                
                is_smoke = raw_prob >= Config.CONFIDENCE_THRESHOLD
                confidence = raw_prob if is_smoke else (1.0 - raw_prob)
            except Exception as e:
                print(f"[Detector] Model prediction failed: {e}. Falling back to heuristic.")
                is_smoke, confidence = self._heuristic_smoke_analysis(frame_bgr)
        else:
            is_smoke, confidence = self._heuristic_smoke_analysis(frame_bgr)

        _, confirmed_alert = self.smoother.update(is_smoke, confidence)
        label = "Smoke Detected" if is_smoke else "Clear / No Smoke"

        return {
            "is_smoke": is_smoke,
            "confidence": confidence,
            "label": label,
            "confirmed_alert": confirmed_alert,
            "consecutive_count": self.smoother.consecutive_count,
            "engine": "MobileNetV2 CNN" if self.model_loaded else "CV Heuristic (Baseline)"
        }

    def annotate_frame(self, frame_bgr: np.ndarray, result: dict, camera_name: str = "Live Feed") -> np.ndarray:
        """Draws HUD diagnostic overlay onto the frame."""
        annotated = frame_bgr.copy()
        h, w, _ = annotated.shape

        is_smoke = result["is_smoke"]
        conf = result["confidence"]
        confirmed = result["confirmed_alert"]
        engine = result.get("engine", "CNN")

        # Top banner styling
        if confirmed:
            banner_color = (0, 0, 200) # Deep Red
            status_text = f"🚨 WILDFIRE SMOKE ALERT! ({conf*100:.1f}%)"
        elif is_smoke:
            banner_color = (0, 140, 255) # Orange Warning
            status_text = f"⚠️ POSSIBLE SMOKE ({conf*100:.1f}%) [Consecutive: {result['consecutive_count']}/{Config.CONSECUTIVE_FRAMES_REQUIRED}]"
        else:
            banner_color = (40, 160, 40) # Green Clear
            status_text = f"NORMAL / CLEAR ({conf*100:.1f}%)"

        # Draw semi-transparent header bar
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), banner_color, -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # Status text
        cv2.putText(annotated, status_text, (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Bottom info bar
        overlay_bottom = annotated.copy()
        cv2.rectangle(overlay_bottom, (0, h - 35), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay_bottom, 0.7, annotated, 0.3, 0, annotated)

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_str = f"Station: {camera_name} | Engine: {engine} | Time: {timestamp_str}"
        cv2.putText(annotated, info_str, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # Draw border outline if in alert
        if confirmed:
            cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

        return annotated


# Global detector singleton
detector_instance = SmokeDetector()

