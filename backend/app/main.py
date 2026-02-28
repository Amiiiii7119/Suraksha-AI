import os
import shutil
import smtplib
import threading
import cv2
import httpx
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import desc
from sqlalchemy.orm import Session
from pydantic import BaseModel


import pathway as pw
from pathway.xpacks.llm.llms import LiteLLMChat
from pathway.xpacks.llm.embedders import LiteLLMEmbedder
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.stdlib.indexing.nearest_neighbors import BruteForceKnnFactory
from pathway.engine import BruteForceKnnMetricKind  # type: ignore[attr-defined]


from app.services.yolo_engine import YoloEngine
from app.services.simulator import (
    start_simulator,
    stop_simulator,
    simulator_status,
)


from app.core.risk_engine import (
    start_pathway,
    latest_risk_state,
    state_lock,
    subject,
    ZONE_MULTIPLIERS,
    RISK_WEIGHTS,
    ALERT_THRESHOLD,
)


from app.db.database import engine as db_engine, Base, SessionLocal
from app.db.models import User, Incident


from api.auth_routes import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    get_db,
)


OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

ALERT_EMAIL_FROM    = os.getenv("ALERT_EMAIL_FROM", "")
ALERT_EMAIL_TO      = os.getenv("ALERT_EMAIL_TO", "")
ALERT_EMAIL_PASS    = os.getenv("ALERT_EMAIL_PASS", "")
SMTP_HOST           = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT           = int(os.getenv("SMTP_PORT", "587"))

SAFETY_DOCS_DIR     = os.getenv("SAFETY_DOCS_DIR", "./safety_docs")


_llm: LiteLLMChat | None = None
_doc_store: DocumentStore | None = None
_rag_lock = threading.Lock()


def _make_llm() -> LiteLLMChat:
    return LiteLLMChat(
        model=f"openai/{OPENROUTER_MODEL}",
        api_key=OPENROUTER_API_KEY,
        api_base=OPENROUTER_BASE_URL,
        temperature=0.3,
        max_tokens=400,
    )


def init_rag():
    global _llm, _doc_store

    if not OPENROUTER_API_KEY:
        print("[RAG] ⚠️  OPENROUTER_API_KEY not set — AI disabled.")
        return

    os.makedirs(SAFETY_DOCS_DIR, exist_ok=True)
    builtin = os.path.join(SAFETY_DOCS_DIR, "osha_builtin_rules.txt")
    if not os.path.exists(builtin):
        with open(builtin, "w") as f:
            f.write("""OSHA CONSTRUCTION SITE SAFETY RULES

1. PERSONAL PROTECTIVE EQUIPMENT (PPE)
   - All workers must wear hard hats (helmets) at all times. (OSHA 29 CFR 1926.100)
   - High-visibility safety vests are mandatory near vehicles/machinery.
   - Workers without PPE must be removed from the hazard zone immediately.

2. FIRE AND SMOKE SAFETY
   - Any fire or smoke detection requires immediate site evacuation. (OSHA 29 CFR 1926.150)
   - Call emergency services (112 / 101) on first detection.
   - Fire extinguishers must be within 100 feet of any combustible material.

3. RESTRICTED ZONE INTRUSIONS
   - Unauthorized entry into heavy machinery zones is strictly prohibited.
   - Intrusion events must be reported to safety officer within 15 minutes.
   - Barricades must be maintained at all restricted zone perimeters.

4. RISK SCORE ACTION THRESHOLDS
   - Score 0-4  (LOW)      : Monitor continuously, no immediate action.
   - Score 4-8  (MEDIUM)   : Issue verbal warnings, increase supervisor patrols.
   - Score 8+   (CRITICAL) : Halt operations in affected zone, evacuate if fire present.

5. ZONE RISK MULTIPLIERS
   - Zone A (1.0x): General site area — standard precautions.
   - Zone B (1.5x): Active construction floor — elevated vigilance.
   - Zone C (2.0x): Heavy machinery area — strict PPE enforcement.
   - Zone D (3.0x): Confined spaces / electrical / explosives — highest alert.

6. INCIDENT REPORTING
   - All violations must be logged in the safety incident register within 1 hour.
   - Critical incidents must be reported to project manager immediately.
""")
        print(f"[RAG] ✅ Built-in OSHA rulebook written to {builtin}")

    try:
        documents = pw.io.fs.read(
            SAFETY_DOCS_DIR,
            format="binary",
            mode="streaming",
            with_metadata=True,
        )

        embedder = LiteLLMEmbedder(
            capacity=5,
            retry_strategy=None,
            cache_strategy=None,
            model="huggingface/sentence-transformers/all-MiniLM-L6-v2",
        )

        
        retriever_factory = BruteForceKnnFactory(
            reserved_space=1000,
            embedder=embedder,
            metric=BruteForceKnnMetricKind.COS, 
            dimensions=384,
        )

        doc_store = DocumentStore(
            docs=documents,
            retriever_factory=retriever_factory,
            splitter=TokenCountSplitter(max_tokens=300),
        )

        with _rag_lock:
            _llm = _make_llm()
            _doc_store = doc_store

        print("[RAG] ✅ Pathway LLM xPack pipeline initialised.")
        print(f"[RAG]    LLM     : {OPENROUTER_MODEL} via OpenRouter")
        print(f"[RAG]    Embedder: all-MiniLM-L6-v2 (local)")
        print(f"[RAG]    Docs dir: {SAFETY_DOCS_DIR}")

    except Exception as e:
        print(f"[RAG] ❌ DocumentStore init failed: {e}")
        print("[RAG]    Falling back to direct LLM (no retrieval).")
        try:
            with _rag_lock:
                _llm = _make_llm()
            print("[RAG] ✅ Direct LLM fallback ready.")
        except Exception as e2:
            print(f"[RAG] ❌ LLM fallback also failed: {e2}")



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

class ExplainRequest(BaseModel):
    force: bool = False



zone_tracker: dict = {
    zone: {
        "risk_score":      0.0,
        "violation_count": 0,
        "velocity":        0.0,
        "previous_risk":   0.0,
        "alert":           False,
    }
    for zone in ZONE_MULTIPLIERS
}
zone_tracker_lock = threading.Lock()

last_alert_email_time: datetime | None = None
alert_email_lock = threading.Lock()

ai_explanation_cache: dict = {
    "text": "",
    "generated_at": "",
    "risk_score": 0.0,
    "risk_level": "",
}


def update_zone_tracker(violation_type: str, zone: str, confidence: float):
    normalized_zone = zone if zone in ZONE_MULTIPLIERS else "zone_a"
    weight = RISK_WEIGHTS.get(violation_type, 1.0)
    multiplier = ZONE_MULTIPLIERS.get(normalized_zone, 1.0)
    risk_contribution = weight * multiplier * confidence

    with zone_tracker_lock:
        prev = zone_tracker[normalized_zone]["risk_score"]
        new_score = round(prev + risk_contribution, 4)
        zone_tracker[normalized_zone]["risk_score"]      = new_score
        zone_tracker[normalized_zone]["violation_count"] += 1
        zone_tracker[normalized_zone]["velocity"]        = round(new_score - prev, 4)
        zone_tracker[normalized_zone]["previous_risk"]   = prev
        zone_tracker[normalized_zone]["alert"]           = new_score > ALERT_THRESHOLD



def send_alert_email(state: dict, explanation: str = ""):
    global last_alert_email_time
    if not ALERT_EMAIL_FROM or not ALERT_EMAIL_TO or not ALERT_EMAIL_PASS:
        print("[EMAIL] Email credentials not configured — skipping alert.")
        return

    with alert_email_lock:
        now = datetime.now(timezone.utc)
        if last_alert_email_time:
            diff = (now - last_alert_email_time).total_seconds()
            if diff < 300:
                print(f"[EMAIL] Skipping — last alert was {diff:.0f}s ago.")
                return
        last_alert_email_time = now

    def _send():
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"🚨 SURAKSHA AI CRITICAL ALERT — Risk Score: {state['risk_score']}"
            msg["From"]    = ALERT_EMAIL_FROM
            msg["To"]      = ALERT_EMAIL_TO
            html_body = f"""
<html><body style="font-family:monospace;background:#0a0a0a;color:#e8eaf0;padding:24px;">
<div style="border:2px solid #FF2D2D;border-radius:8px;padding:24px;max-width:600px;">
  <h2 style="color:#FF2D2D;margin:0 0 16px;">🚨 CRITICAL SAFETY ALERT</h2>
  <p style="color:#FFD600;font-size:18px;margin:0 0 8px;">
    Risk Score: <strong>{state['risk_score']}</strong> | Level: <strong>{state['risk_level']}</strong>
  </p>
  <p style="color:#e8eaf0;margin:0 0 16px;">
    Accident Probability: <strong>{round(state['accident_probability'] * 100, 2)}%</strong>
  </p>
  <hr style="border-color:#1E2535;margin:16px 0;">
  <h3 style="color:#FFD600;margin:0 0 8px;">VIOLATIONS DETECTED</h3>
  <table style="width:100%;border-collapse:collapse;">
    <tr><td style="padding:4px 0;color:#FF2D2D;">🔥 Fire / Smoke</td>
        <td style="text-align:right;font-weight:bold;">{state['fire_count']}</td></tr>
    <tr><td style="padding:4px 0;color:#FF2D2D;">🪖 No Helmet</td>
        <td style="text-align:right;font-weight:bold;">{state['helmet_count']}</td></tr>
    <tr><td style="padding:4px 0;color:#FFD600;">🦺 No Vest</td>
        <td style="text-align:right;font-weight:bold;">{state['vest_count']}</td></tr>
    <tr><td style="padding:4px 0;color:#A855F7;">⛔ Intrusion</td>
        <td style="text-align:right;font-weight:bold;">{state['intrusion_count']}</td></tr>
  </table>
  <hr style="border-color:#1E2535;margin:16px 0;">
  <h3 style="color:#00FF94;margin:0 0 8px;">MITIGATION</h3>
  <p style="color:#e8eaf0;">{state['mitigation_text']}</p>
  {"<hr style='border-color:#1E2535;margin:16px 0;'><h3 style='color:#00FF94;margin:0 0 8px;'>AI ANALYSIS</h3><p style='color:#e8eaf0;'>" + explanation + "</p>" if explanation else ""}
  <hr style="border-color:#1E2535;margin:16px 0;">
  <p style="color:#5A6480;font-size:11px;">
    Generated by Suraksha AI | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
  </p>
</div>
</body></html>"""
            msg.attach(MIMEText(html_body, "html"))
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.starttls()
                server.login(ALERT_EMAIL_FROM, ALERT_EMAIL_PASS)
                server.sendmail(ALERT_EMAIL_FROM, ALERT_EMAIL_TO, msg.as_string())
            print(f"[EMAIL] ✅ Critical alert sent to {ALERT_EMAIL_TO}")
        except Exception as e:
            print(f"[EMAIL] ❌ Failed to send alert: {e}")

    threading.Thread(target=_send, daemon=True).start()



async def get_ai_explanation(state: dict) -> str:
    with zone_tracker_lock:
        zones_sorted = sorted(
            zone_tracker.items(),
            key=lambda x: x[1]["risk_score"],
            reverse=True,
        )
        top_zone = zones_sorted[0] if zones_sorted else (
            "unknown", {"risk_score": 0, "violation_count": 0}
        )

    with _rag_lock:
        llm = _llm
        doc_store = _doc_store

    rag_context = ""
    rag_used = False
    if doc_store is not None:
        try:
            results = list(doc_store.retrieve(  # type: ignore[attr-defined]
                query="PPE helmet vest fire intrusion OSHA construction safety rules",
                k=3,
            ))
            if results:
                chunks = [r.get("text", r.get("content", "")) for r in results if r]
                rag_context = "\n\n---\n\n".join(c for c in chunks if c)
                rag_used = bool(rag_context)
        except Exception as e:
            print(f"[RAG] Retrieval skipped: {e}")

    rag_section = (
        f"\n\nRELEVANT OSHA RULES (from live safety rulebook):\n{rag_context}\n"
        if rag_context else ""
    )

    prompt = f"""You are Suraksha AI, an expert workplace safety analyst monitoring a construction site in real time.

CURRENT LIVE DATA:
- Risk Score: {state['risk_score']} (scale: 0-20+)
- Risk Level: {state['risk_level']} (LOW / MEDIUM / CRITICAL)
- Alert Triggered: {'YES - IMMEDIATE ACTION REQUIRED' if state['alert_triggered'] else 'NO'}
- Accident Probability: {round(state['accident_probability'] * 100, 2)}%
- Velocity (risk change rate): {state['velocity']:+.4f}
- Predicted Risk (next window): {state['predicted_risk']:.2f}

VIOLATIONS IN CURRENT 10-SECOND WINDOW:
- Fire / Smoke detections: {state['fire_count']}
- Workers without helmets: {state['helmet_count']}
- Workers without safety vests: {state['vest_count']}
- Unauthorized zone intrusions: {state['intrusion_count']}

HIGHEST RISK ZONE: {top_zone[0].upper()} (score: {top_zone[1]['risk_score']:.2f}, violations: {top_zone[1]['violation_count']})
CURRENT MITIGATION: {state['mitigation_text']}{rag_section}
Write a concise 3-sentence safety report for the site supervisor. Be direct, urgent if needed, and actionable."""

    if llm is not None:
        try:
            import litellm
            response = await litellm.acompletion(
                model=f"openai/{OPENROUTER_MODEL}",
                api_key=OPENROUTER_API_KEY,
                api_base=OPENROUTER_BASE_URL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3,
            )
            choices = getattr(response, "choices", [])
            first_choice = choices[0] if choices else None
            message = getattr(first_choice, "message", None)
            content = getattr(message, "content", "") or ""
            result = content.strip()
            prefix = "[RAG-powered — Pathway xPack + DocumentStore]" if rag_used else "[Pathway LLM xPack]"
            return f"{prefix}\n\n{result}"
        except Exception as e:
            print(f"[AI] Pathway xPack error: {e} — falling back to raw OpenRouter")

    if not OPENROUTER_API_KEY:
        return "AI analysis unavailable. Add OPENROUTER_API_KEY to your .env file."
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://suraksha-ai.app",
                    "X-Title": "Suraksha AI Safety Platform",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 250,
                    "temperature": 0.4,
                }
            )
            data = response.json()
            if "choices" not in data:
                error_msg = data.get("error", {}).get("message", str(data))
                return f"AI Error: {error_msg}"
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"



Base.metadata.create_all(bind=db_engine)



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
            print("[AUTH] ✅ Default admin created.")
        else:
            print("[AUTH] ✅ Default admin exists.")
    except Exception as e:
        print(f"[AUTH] Error: {e}")
        db.rollback()
    finally:
        db.close()



@asynccontextmanager
async def lifespan(app: FastAPI):
    create_default_admin()
    threading.Thread(target=start_pathway, daemon=True).start()
    print("[STARTUP] ✅ Pathway risk engine started.")
    threading.Thread(target=init_rag, daemon=True).start()
    print("[STARTUP] ✅ Pathway LLM xPack RAG initialising...")
    start_simulator()
    print("[STARTUP] ✅ Simulator started.")
    yield
    print("[SHUTDOWN] FastAPI shutting down.")



app = FastAPI(
    title="Suraksha AI",
    description="AI-powered workplace safety monitoring system",
    version="2.0.0",
    lifespan=lifespan,
)
__all__ = ["app"]

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



@app.get("/", tags=["General"])
def root():
    return {"message": "Suraksha AI v2.0 is running 🚀"}


@app.get("/health", tags=["General"])
def health():
    with _rag_lock:
        rag_ready = _doc_store is not None
        llm_ready = _llm is not None
    return {
        "status":           "ok",
        "ppe_model":        yolo.ppe_model is not None,
        "fire_model":       yolo.fire_model is not None,
        "ai_configured":    llm_ready,
        "rag_configured":   rag_ready,
        "email_configured": bool(ALERT_EMAIL_FROM),
    }



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



@app.get("/analytics", tags=["Analytics"])
def analytics(_current_user: User = Depends(get_current_user)):
    with state_lock:
        return dict(latest_risk_state)


@app.get("/risk/summary", tags=["Analytics"])
def risk_summary(_current_user: User = Depends(get_current_user)):
    with state_lock:
        state = dict(latest_risk_state)
    level = state["risk_level"]
    emoji = {"LOW": "🟢", "MEDIUM": "🟡", "CRITICAL": "🔴"}.get(level, "⚪")
    return {
        "summary":                  f"{emoji} Risk Level: {level}",
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



@app.get("/ai/explain", tags=["AI"])
async def ai_explain(_current_user: User = Depends(get_current_user)):
    with state_lock:
        state = dict(latest_risk_state)

    explanation = await get_ai_explanation(state)

    ai_explanation_cache.update({
        "text":         explanation,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "risk_score":   state["risk_score"],
        "risk_level":   state["risk_level"],
    })

    if state["alert_triggered"]:
        send_alert_email(state, explanation)

    with _rag_lock:
        rag_ready = _doc_store is not None

    return {
        "explanation":  explanation,
        "risk_level":   state["risk_level"],
        "risk_score":   state["risk_score"],
        "generated_at": ai_explanation_cache["generated_at"],
        "email_sent":   state["alert_triggered"] and bool(ALERT_EMAIL_FROM),
        "rag_used":     rag_ready and "[RAG-powered" in explanation,
    }


@app.get("/ai/cached", tags=["AI"])
def ai_cached(_current_user: User = Depends(get_current_user)):
    return ai_explanation_cache


@app.get("/ai/status", tags=["AI"])
def ai_status(_current_user: User = Depends(get_current_user)):
    with _rag_lock:
        rag_ready = _doc_store is not None
        llm_ready = _llm is not None
    return {
        "rag_pipeline_ready": rag_ready,
        "llm_ready":          llm_ready,
        "model":              OPENROUTER_MODEL,
        "safety_docs_dir":    SAFETY_DOCS_DIR,
        "api_key_set":        bool(OPENROUTER_API_KEY),
    }



@app.get("/leaderboard", tags=["Analytics"])
def leaderboard(_current_user: User = Depends(get_current_user)):
    with zone_tracker_lock:
        zones = [
            {
                "zone":            zone,
                "risk_score":      round(data["risk_score"], 2),
                "velocity":        round(data["velocity"], 4),
                "violation_count": data["violation_count"],
                "alert":           data["alert"],
            }
            for zone, data in zone_tracker.items()
        ]
    zones.sort(key=lambda z: z["risk_score"], reverse=True)
    return zones



@app.get("/zones", tags=["Analytics"])
def get_zones(_current_user: User = Depends(get_current_user)):
    with zone_tracker_lock:
        zones = [
            {
                "zone":            zone,
                "risk_score":      round(data["risk_score"], 2),
                "velocity":        round(data["velocity"], 4),
                "violation_count": data["violation_count"],
                "alert_triggered": data["alert"],
            }
            for zone, data in zone_tracker.items()
        ]
    zones.sort(key=lambda z: z["risk_score"], reverse=True)
    return zones



@app.get("/incidents", tags=["Analytics"])
def get_incidents(
    _current_user: User = Depends(get_current_user),
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



@app.post("/event", tags=["Detection"])
def push_event(
    payload: EventPayload,
    _current_user: User = Depends(get_current_user),
):
    ts = (
        datetime.fromisoformat(payload.timestamp.replace("Z", "+00:00"))
        if payload.timestamp
        else datetime.now(timezone.utc)
    )
    subject.push(
        timestamp=ts,
        violation_type=payload.violation_type,
        confidence=payload.confidence,
        zone=payload.zone,
        camera_id=payload.camera_id,
    )
    update_zone_tracker(payload.violation_type, payload.zone, payload.confidence)
    return {
        "status":         "accepted",
        "pushed_at":      ts.isoformat(),
        "violation_type": payload.violation_type,
        "zone":           payload.zone,
    }


@app.post("/detect/image", tags=["Detection"])
async def detect_image(
    file: UploadFile = File(...),
    _current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        camera_id = "image_upload"
        detections, annotated_b64 = yolo.run_detection(
            temp_path, camera_id=camera_id, annotate=True
        )
        violations = []
        risk_score = 0.0

        for det in detections:
            confidence     = float(det.get("confidence", 0.0))
            zone           = det.get("zone", "zone_a")
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
                camera_id=camera_id,
            )
            update_zone_tracker(violation_type, zone, confidence)
            db.add(Incident(
                timestamp=datetime.now(timezone.utc),
                zone=zone,
                violation_type=violation_type,
                confidence=confidence,
                risk_impact=confidence * 100,
            ))

        db.commit()

        with state_lock:
            current_state = dict(latest_risk_state)
        if current_state.get("alert_triggered"):
            send_alert_email(current_state)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "violations":      violations,
        "risk_score":      round(risk_score, 2),
        "frame_count":     1,
        "annotated_image": annotated_b64,
    }



@app.post("/detect/video", tags=["Detection"])
async def detect_video(
    file: UploadFile = File(...),
    _current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    temp_path     = f"temp_{file.filename}"
    violations    = []
    risk_score    = 0.0
    frame_count   = 0
    annotated_b64 = ""

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

            do_annotate = not annotated_b64
            detections, b64 = yolo.run_detection(
                frame_path,
                camera_id=file.filename or "uploaded_video",
                annotate=do_annotate,
            )
            if b64:
                annotated_b64 = b64
            if os.path.exists(frame_path):
                os.remove(frame_path)

            for det in detections:
                confidence     = float(det.get("confidence", 0.0))
                zone           = det.get("zone", "zone_a")
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
                update_zone_tracker(violation_type, zone, confidence)
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
        "violations":      violations,
        "risk_score":      round(risk_score, 2),
        "frame_count":     frame_count // FRAME_SKIP,
        "annotated_image": annotated_b64,
    }



def live_camera_loop():
    global live_running
    cap = cv2.VideoCapture(0)
    frame_count = 0
    FRAME_SKIP  = 5
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
        detections, _ = yolo.run_detection(frame_path, camera_id="live_cam")
        if os.path.exists(frame_path):
            os.remove(frame_path)

        db = SessionLocal()
        try:
            for det in detections:
                confidence     = float(det.get("confidence", 0.0))
                zone           = det.get("zone", "zone_a")
                violation_type = det.get("violation_type", "unknown")
                subject.push(
                    timestamp=datetime.now(timezone.utc),
                    violation_type=violation_type,
                    confidence=confidence,
                    zone=zone,
                    camera_id="live_cam",
                )
                update_zone_tracker(violation_type, zone, confidence)
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
def start_live(_current_user: User = Depends(get_current_user)):
    global live_running
    if live_running:
        return {"message": "Live detection already running"}
    live_running = True
    threading.Thread(target=live_camera_loop, daemon=True).start()
    return {"message": "Live detection started ✅"}


@app.post("/detect/live/stop", tags=["Live Camera"])
def stop_live(_current_user: User = Depends(get_current_user)):
    global live_running
    live_running = False
    return {"message": "Live detection stopped 🛑"}


@app.get("/detect/live/status", tags=["Live Camera"])
def live_status(_current_user: User = Depends(get_current_user)):
    return {
        "active":      live_running,
        "event_count": latest_risk_state.get("total_events", 0),
    }



@app.post("/simulator/start", tags=["Simulator"])
def sim_start(_current_user: User = Depends(get_current_user)):
    started = start_simulator()
    return {"message": "Simulator started ✅" if started else "Already running"}


@app.post("/simulator/stop", tags=["Simulator"])
def sim_stop(_current_user: User = Depends(get_current_user)):
    stop_simulator()
    return {"message": "Simulator stopped 🛑"}


@app.get("/simulator/status", tags=["Simulator"])
def sim_status(_current_user: User = Depends(get_current_user)):
    return simulator_status()