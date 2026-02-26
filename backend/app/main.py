
import os
import shutil
import threading
import cv2
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import func, desc
from sqlalchemy.orm import Session
from pydantic import BaseModel

# Services
from app.services.yolo_engine import YoloEngine
from app.services.simulator import (
    start_simulator,
    stop_simulator,
    simulator_status,
)

# Core
from app.core.risk_engine import (
    start_pathway,
    latest_risk_state,
    state_lock,
    subject,
)

# Database
from app.db.database import engine as db_engine, Base, SessionLocal
from app.db.models import User, Incident

# API utilities
from api.auth_routes import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    get_db,
)

# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────
class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    role: str = "viewer"

class LoginJSON(BaseModel):
    email: str
    password: str

class EventPayload(BaseModel):
    violation_type: str
    confidence: float = 0.0
    zone: str = "zone_a"
    camera_id: str = "manual"
    timestamp: str | None = None


# ─────────────────────────────────────────────
# Create DB tables
# ─────────────────────────────────────────────
Base.metadata.create_all(bind=db_engine)


# ─────────────────────────────────────────────
# Default Admin
# ─────────────────────────────────────────────
def create_default_admin():
    db = SessionLocal()
    try:
        existing = db.query(User).filter(
            User.email == "amteshwarrajsingh@gmail.com"
        ).first()
        if not existing:
            admin = User(
                name="Amteshwar",
                email="amteshwarrajsingh@gmail.com",
                password_hash=hash_password("admin123"),
                role="admin",
            )
            db.add(admin)
            db.commit()
            print("[AUTH] Default admin created.")
        else:
            print("[AUTH] Default admin already exists.")
    except Exception as e:
        print(f"[AUTH] Error creating admin: {e}")
        db.rollback()
    finally:
        db.close()


# ─────────────────────────────────────────────
# Lifespan — auto-start Pathway + Simulator
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_default_admin()

    # Start Pathway risk engine
    threading.Thread(target=start_pathway, daemon=True).start()
    print("[STARTUP] Pathway risk engine started.")

    # Auto-start simulator so dashboard has live data immediately
    start_simulator()
    print("[STARTUP] Safety event simulator started.")

    yield
    print("[SHUTDOWN] FastAPI shutting down.")


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Suraksha AI",
    description="AI-powered workplace safety monitoring system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

yolo = YoloEngine()
live_running = False


# ─────────────────────────────────────────────
# General
# ─────────────────────────────────────────────
@app.get("/", tags=["General"])
def root():
    return {"message": "Suraksha AI is running 🚀"}


@app.get("/health", tags=["General"])
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────
@app.post("/signup", tags=["Auth"])
def signup(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = User(
        name=user.name,
        email=user.email,
        password_hash=hash_password(user.password),
        role=user.role,
    )
    db.add(new_user)
    db.commit()
    return {"message": "User created successfully"}


@app.post("/login", tags=["Auth"])
def login_json(user: LoginJSON, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, str(db_user.password_hash)):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": db_user.email})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/login/form", tags=["Auth"])
def login_form(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    db_user = db.query(User).filter(User.email == form_data.username).first()
    if not db_user or not verify_password(form_data.password, str(db_user.password_hash)):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": db_user.email})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/me", tags=["Auth"])
def me(current_user: User = Depends(get_current_user)):
    return {
        "name":  current_user.name,
        "email": current_user.email,
        "role":  current_user.role,
    }


# ─────────────────────────────────────────────
# Analytics
# ─────────────────────────────────────────────
@app.get("/analytics", tags=["Analytics"])
def analytics(current_user: User = Depends(get_current_user)):
    with state_lock:
        return dict(latest_risk_state)


@app.get("/risk/summary", tags=["Analytics"])
def risk_summary(current_user: User = Depends(get_current_user)):
    with state_lock:
        state = dict(latest_risk_state)

    level = state["risk_level"]
    level_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "CRITICAL": "🔴"}.get(level, "⚪")

    return {
        "summary":                  f"{level_emoji} Risk Level: {level}",
        "score":                    state["risk_score"],
        "alert":                    state["alert_triggered"],
        "predicted_risk":           state["predicted_risk"],
        "velocity":                 state["velocity"],
        "growth_rate_pct":          f"{state['growth_rate']}%",
        "accident_probability_pct": f"{round(state['accident_probability'] * 100, 2)}%",
        "top_violations": {
            "fire_smoke": state["fire_count"],
            "no_helmet":  state["helmet_count"],
            "no_vest":    state["vest_count"],
            "intrusion":  state["intrusion_count"],
        },
        "mitigation": state["mitigation_text"],
    }


# ─────────────────────────────────────────────
# Leaderboard
# ─────────────────────────────────────────────
@app.get("/leaderboard", tags=["Analytics"])
def leaderboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    zones = (
        db.query(
            Incident.zone,
            func.avg(Incident.risk_impact).label("risk_score"),
            func.count().label("violation_count"),
        )
        .group_by(Incident.zone)
        .order_by(desc("risk_score"))
        .all()
    )
    return [
        {
            "zone":            z.zone,
            "risk_score":      round(float(z.risk_score), 2),
            "velocity":        z.violation_count,
            "violation_count": z.violation_count,
            "alert":           float(z.risk_score) > 70,
        }
        for z in zones
    ]


# ─────────────────────────────────────────────
# Zones
# ─────────────────────────────────────────────
@app.get("/zones", tags=["Analytics"])
def get_zones(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    zone_rows = (
        db.query(
            Incident.zone,
            func.avg(Incident.risk_impact).label("risk_score"),
            func.count().label("violation_count"),
        )
        .group_by(Incident.zone)
        .all()
    )
    return [
        {
            "zone":            z.zone,
            "risk_score":      round(float(z.risk_score), 2),
            "velocity":        z.violation_count,
            "violation_count": z.violation_count,
            "alert_triggered": float(z.risk_score) > 70,
        }
        for z in zone_rows
    ]


# ─────────────────────────────────────────────
# Incidents
# ─────────────────────────────────────────────
@app.get("/incidents", tags=["Analytics"])
def get_incidents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    incidents = (
        db.query(Incident)
        .order_by(desc(Incident.timestamp))
        .limit(100)
        .all()
    )
    return [
        {
            "id":             i.id,
            "timestamp":      i.timestamp.isoformat(),
            "zone":           i.zone,
            "violation_type": i.violation_type,
            "confidence":     i.confidence,
            "risk_impact":    i.risk_impact,
        }
        for i in incidents
    ]


# ─────────────────────────────────────────────
# Manual Event Push
# ─────────────────────────────────────────────
@app.post("/event", tags=["Detection"])
def push_event(
    payload: EventPayload,
    current_user: User = Depends(get_current_user),
):
    if payload.timestamp:
        ts = datetime.fromisoformat(payload.timestamp.replace("Z", "+00:00"))
    else:
        ts = datetime.now(timezone.utc)

    subject.push(
        timestamp=ts,
        violation_type=payload.violation_type,
        confidence=payload.confidence,
        zone=payload.zone,
        camera_id=payload.camera_id,
    )
    return {
        "status":         "accepted",
        "pushed_at":      ts.isoformat(),
        "violation_type": payload.violation_type,
        "zone":           payload.zone,
    }


# ─────────────────────────────────────────────
# Image Detection
# ─────────────────────────────────────────────
@app.post("/detect/image", tags=["Detection"])
async def detect_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        detections = yolo.run_detection(temp_path)
        violations = []
        risk_score = 0.0

        for det in detections:
            confidence = float(det.get("confidence", 0.0))
            zone = det.get("zone", "zone_a")
            violation_type = det.get("violation_type", "unknown")

            violations.append({
                "type":       violation_type,
                "confidence": confidence,
                "bbox":       det.get("bbox", [0, 0, 0, 0]),
                "zone":       zone,
            })
            risk_score += confidence * 100

            subject.push(
                timestamp=datetime.now(timezone.utc),
                violation_type=violation_type,
                confidence=confidence,
                zone=zone,
                camera_id="image_upload",
            )
            db.add(Incident(
                timestamp=datetime.now(timezone.utc),
                zone=zone,
                violation_type=violation_type,
                confidence=confidence,
                risk_impact=confidence * 100,
            ))

        db.commit()

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "violations":  violations,
        "risk_score":  round(risk_score, 2),
        "frame_count": 1,
    }


# ─────────────────────────────────────────────
# Video Detection
# ─────────────────────────────────────────────
@app.post("/detect/video", tags=["Detection"])
async def detect_video(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    temp_path = f"temp_{file.filename}"
    violations = []
    risk_score = 0.0
    frame_count = 0

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        cap = cv2.VideoCapture(temp_path)
        FRAME_SKIP = 5

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            frame_path = "temp_frame.jpg"
            cv2.imwrite(frame_path, frame)
            detections = yolo.run_detection(frame_path)
            if os.path.exists(frame_path):
                os.remove(frame_path)

            for det in detections:
                confidence = float(det.get("confidence", 0.0))
                zone = det.get("zone", "zone_a")
                violation_type = det.get("violation_type", "unknown")

                violations.append({
                    "type":       violation_type,
                    "confidence": confidence,
                    "bbox":       det.get("bbox", [0, 0, 0, 0]),
                    "zone":       zone,
                })
                risk_score += confidence * 100

                subject.push(
                    timestamp=datetime.now(timezone.utc),
                    violation_type=violation_type,
                    confidence=confidence,
                    zone=zone,
                    camera_id=file.filename,
                )
                db.add(Incident(
                    timestamp=datetime.now(timezone.utc),
                    zone=zone,
                    violation_type=violation_type,
                    confidence=confidence,
                    risk_impact=confidence * 100,
                ))

        db.commit()
        cap.release()

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "violations":  violations,
        "risk_score":  round(risk_score, 2),
        "frame_count": frame_count // FRAME_SKIP,
    }


# ─────────────────────────────────────────────
# Live Camera
# ─────────────────────────────────────────────
def live_camera_loop():
    global live_running
    cap = cv2.VideoCapture(0)
    frame_count = 0
    FRAME_SKIP = 5
    print("[LIVE] Camera started.")

    while live_running:
        ret, frame = cap.read()
        if not ret:
            print("[LIVE] Camera read failed — stopping.")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame_path = "temp_live.jpg"
        cv2.imwrite(frame_path, frame)
        detections = yolo.run_detection(frame_path)
        if os.path.exists(frame_path):
            os.remove(frame_path)

        db = SessionLocal()
        try:
            for det in detections:
                confidence = float(det.get("confidence", 0.0))
                zone = det.get("zone", "zone_a")
                violation_type = det.get("violation_type", "unknown")

                subject.push(
                    timestamp=datetime.now(timezone.utc),
                    violation_type=violation_type,
                    confidence=confidence,
                    zone=zone,
                    camera_id="live_cam",
                )
                db.add(Incident(
                    timestamp=datetime.now(timezone.utc),
                    zone=zone,
                    violation_type=violation_type,
                    confidence=confidence,
                    risk_impact=confidence * 100,
                ))
            db.commit()
        finally:
            db.close()

    cap.release()
    print("[LIVE] Camera released.")


@app.post("/detect/live/start", tags=["Live Camera"])
def start_live(current_user: User = Depends(get_current_user)):
    global live_running
    if live_running:
        return {"message": "Live detection already running"}
    live_running = True
    threading.Thread(target=live_camera_loop, daemon=True).start()
    return {"message": "Live detection started ✅"}


@app.post("/detect/live/stop", tags=["Live Camera"])
def stop_live(current_user: User = Depends(get_current_user)):
    global live_running
    live_running = False
    return {"message": "Live detection stopped 🛑"}


@app.get("/detect/live/status", tags=["Live Camera"])
def live_status(current_user: User = Depends(get_current_user)):
    return {
        "active":      live_running,
        "event_count": latest_risk_state.get("total_events", 0),
    }


# ─────────────────────────────────────────────
# Simulator Routes
# ─────────────────────────────────────────────
@app.post("/simulator/start", tags=["Simulator"])
def sim_start(current_user: User = Depends(get_current_user)):
    started = start_simulator()
    return {"message": "Simulator started ✅" if started else "Already running"}


@app.post("/simulator/stop", tags=["Simulator"])
def sim_stop(current_user: User = Depends(get_current_user)):
    stop_simulator()
    return {"message": "Simulator stopped 🛑"}


@app.get("/simulator/status", tags=["Simulator"])
def sim_status(current_user: User = Depends(get_current_user)):
    return simulator_status()