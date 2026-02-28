import os
import cv2
import base64
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO

# File is at: backend/app/services/yolo_engine.py
# Go up 3 levels: services -> app -> backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PPE_MODEL_PATH = os.path.join(
    BASE_DIR, "runs", "detect", "ppe_training2", "weights", "best.pt"
)
FIRE_MODEL_PATH = os.path.join(
    BASE_DIR, "runs", "detect", "fire_training", "weights", "best.pt"
)

# Camera ID -> Zone mapping for real deployments
CAMERA_ZONE_MAP = {
    "cam_01":       "zone_a",
    "cam_02":       "zone_b",
    "cam_03":       "zone_c",
    "cam_04":       "zone_d",
    "image_upload": "zone_a",
    "live_cam":     "zone_b",
    "manual_push":  "zone_a",
    "default":      "zone_a",
}

# Normalize raw YOLO label -> standard violation_type
LABEL_MAP = {
    "no-hardhat":     "no_helmet",
    "no hardhat":     "no_helmet",
    "no-safety vest": "no_vest",
    "no safety vest": "no_vest",
    "no-mask":        "no_mask",
    "no mask":        "no_mask",
    "fire":           "fire",
    "smoke":          "smoke",
    "hardhat":        "hardhat",
    "safety vest":    "safety_vest",
    "safety cone":    "safety_cone",
    "person":         "person",
    "machinery":      "machinery",
    "vehicle":        "vehicle",
    "mask":           "mask",
}

# Bounding box colors per violation type (BGR for OpenCV)
VIOLATION_COLORS = {
    "no_helmet":       (0, 0, 255),      # Red
    "no_vest":         (0, 165, 255),    # Orange
    "fire":            (0, 69, 255),     # Deep red
    "smoke":           (128, 128, 128),  # Gray
    "no_mask":         (255, 0, 255),    # Magenta
    "restricted_zone": (0, 0, 200),      # Dark red
    "hardhat":         (0, 255, 0),      # Green
    "safety_vest":     (0, 255, 128),    # Light green
    "person":          (255, 255, 0),    # Cyan
    "machinery":       (255, 0, 128),    # Purple
    "vehicle":         (255, 128, 0),    # Blue
    "default":         (255, 255, 255),  # White
}


def normalize_label(raw: str) -> str:
    key = raw.lower().strip()
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    return key.replace(" ", "_").replace("-", "_")


def get_zone_for_camera(camera_id: str) -> str:
    return CAMERA_ZONE_MAP.get(camera_id, CAMERA_ZONE_MAP["default"])


def draw_annotations(image_path: str, detections: List[Dict]) -> str:
    """
    Draw bounding boxes + labels on image.
    Returns base64-encoded annotated JPEG string.
    """
    img = cv2.imread(image_path)
    if img is None:
        return ""

    h, w = img.shape[:2]

    for det in detections:
        bbox = det.get("bbox", [])
        label = det.get("violation_type", "unknown")
        conf = det.get("confidence", 0.0)
        color = VIOLATION_COLORS.get(label, VIOLATION_COLORS["default"])

        if len(bbox) == 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Label background
            label_text = f"{label.replace('_', ' ').upper()} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)

            # Label text
            text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
            cv2.putText(img, label_text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

    # Timestamp watermark
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(img, f"SURAKSHA AI | {ts}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 148), 1)

    # Encode to base64
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


class YoloEngine:
    def __init__(self) -> None:
        self.ppe_model = None
        self.fire_model = None

        print(f"[YOLO] BASE_DIR resolved to: {BASE_DIR}")

        if os.path.exists(PPE_MODEL_PATH):
            try:
                self.ppe_model = YOLO(PPE_MODEL_PATH)
                print(f"[YOLO] ✅ PPE model loaded")
                print(f"[YOLO]    PPE classes: {self.ppe_model.names}")
            except Exception as e:
                print(f"[YOLO] ⚠️  PPE model failed: {e}")
        else:
            print(f"[YOLO] ⚠️  PPE model not found at: {PPE_MODEL_PATH}")

        if os.path.exists(FIRE_MODEL_PATH):
            try:
                self.fire_model = YOLO(FIRE_MODEL_PATH)
                print(f"[YOLO] ✅ Fire model loaded")
                print(f"[YOLO]    Fire classes: {self.fire_model.names}")
            except Exception as e:
                print(f"[YOLO] ⚠️  Fire model failed: {e}")
        else:
            print(f"[YOLO] ⚠️  Fire model not found at: {FIRE_MODEL_PATH}")

        if not self.ppe_model and not self.fire_model:
            print("[YOLO] ⚠️  No models loaded — detection will return empty results.")

    def run_detection(
        self,
        image_path: str,
        camera_id: str = "default",
        annotate: bool = False,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Returns (detections, annotated_image_base64).
        annotated_image_base64 is empty string if annotate=False.
        """
        results: List[Dict[str, Any]] = []
        timestamp: str = datetime.now(timezone.utc).isoformat()
        zone = get_zone_for_camera(camera_id)

        if self.ppe_model:
            try:
                ppe_results = self.ppe_model(image_path, verbose=False)
                for result in ppe_results:
                    for box in result.boxes:
                        raw_label = result.names[int(box.cls)]
                        violation_type = normalize_label(raw_label)
                        results.append({
                            "violation_type": violation_type,
                            "confidence":     float(box.conf),
                            "bbox":           [round(x) for x in box.xyxy[0].tolist()],
                            "zone":           zone,
                            "camera_id":      camera_id,
                            "timestamp":      timestamp,
                        })
            except Exception as e:
                print(f"[YOLO] PPE detection error: {e}")

        if self.fire_model:
            try:
                fire_results = self.fire_model(image_path, verbose=False)
                for result in fire_results:
                    for box in result.boxes:
                        raw_label = result.names[int(box.cls)]
                        violation_type = normalize_label(raw_label)
                        results.append({
                            "violation_type": violation_type,
                            "confidence":     float(box.conf),
                            "bbox":           [round(x) for x in box.xyxy[0].tolist()],
                            "zone":           zone,
                            "camera_id":      camera_id,
                            "timestamp":      timestamp,
                        })
            except Exception as e:
                print(f"[YOLO] Fire detection error: {e}")

        annotated_b64 = ""
        if annotate and results:
            try:
                annotated_b64 = draw_annotations(image_path, results)
            except Exception as e:
                print(f"[YOLO] Annotation error: {e}")

        return results, annotated_b64