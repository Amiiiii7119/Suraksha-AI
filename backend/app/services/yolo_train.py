from ultralytics import YOLO
import os
import shutil

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RUNS_DIR = os.path.join(BASE_DIR, "backend", "runs")

os.makedirs(MODELS_DIR, exist_ok=True)

PPE_DATA = os.path.join(DATASETS_DIR, "ppe", "data.yaml")
FIRE_DATA = os.path.join(DATASETS_DIR, "fire", "data.yaml")

def train_ppe():
    run_name = "ppe_training"
    run_path = os.path.join(RUNS_DIR, "detect", run_name)
    best_path = os.path.join(run_path, "weights", "best.pt")

    if os.path.exists(best_path):
        model = YOLO(best_path)
        model.train(
            data=PPE_DATA,
            epochs=50,
            imgsz=640,
            batch=8,
            workers=32,
            project=os.path.join(RUNS_DIR, "detect"),
            name=run_name,
            resume=True
        )
    else:
        model = YOLO("yolov8n.pt")
        model.train(
            data=PPE_DATA,
            epochs=50,
            imgsz=640,
            batch=8,
            workers=16,
            project=os.path.join(RUNS_DIR, "detect"),
            name=run_name
        )

    final_best = os.path.join(run_path, "weights", "best.pt")
    if os.path.exists(final_best):
        shutil.copy(final_best, os.path.join(MODELS_DIR, "ppe_best.pt"))

def train_fire():
    run_name = "fire_training"
    run_path = os.path.join(RUNS_DIR, "detect", run_name)
    best_path = os.path.join(run_path, "weights", "best.pt")

    if os.path.exists(best_path):
        model = YOLO(best_path)
        model.train(
            data=FIRE_DATA,
            epochs=35,
            imgsz=640,
            batch=16,
            workers=12,
            project=os.path.join(RUNS_DIR, "detect"),
            name=run_name,
            resume=True
        )
    else:
        model = YOLO("yolov8m.pt")
        model.train(
            data=FIRE_DATA,
            epochs=35,
            imgsz=640,
            batch=16,
            workers=12,
            project=os.path.join(RUNS_DIR, "detect"),
            name=run_name
        )

    final_best = os.path.join(run_path, "weights", "best.pt")
    if os.path.exists(final_best):
        shutil.copy(final_best, os.path.join(MODELS_DIR, "fire_best.pt"))