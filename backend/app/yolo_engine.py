import os
from datetime import datetime, timezone
from typing import List, Dict, Any
class YoloEngine:
    def __init__(self):
        self.model = None

    def load_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
        return self.model


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PPE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "ppe_training2",
    "weights",
    "best.pt"
)

FIRE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "fire_training",
    "weights",
    "best.pt"
)


class YoloEngine:
    def __init__(self) -> None:
        self.ppe_model: YOLO = YOLO(PPE_MODEL_PATH)
        self.fire_model: YOLO = YOLO(FIRE_MODEL_PATH)

    def run_detection(self, image_path: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        timestamp: str = datetime.now(timezone.utc).isoformat()

        ppe_results = self.ppe_model(image_path, verbose=False)
        fire_results = self.fire_model(image_path, verbose=False)

        for result in ppe_results:
            for box in result.boxes:
                results.append({
                    "timestamp": timestamp,
                    "violation_type": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "zone": "construction_site"
                })

        for result in fire_results:
            for box in result.boxes:
                results.append({
                    "timestamp": timestamp,
                    "violation_type": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "zone": "construction_site"
                })

        return results
