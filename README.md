# Suraksha AI
**Real-Time AI Safety Intelligence for Construction & Industrial Sites**

> Suraksha AI detects workplace safety violations the moment they happen, scores site-wide risk in real time using stream processing, and gives safety teams the intelligence they need to prevent accidents before they occur.

---

## The Problem

Every year, thousands of workers are injured or killed on construction sites. The root cause is almost never bad luck — it is a failure to act on visible warning signs in time.

Workers remove helmets. Vests go unworn. People enter restricted zones. Fires start small. None of these are invisible. The problem is that no one is watching everything, all the time.

Existing safety systems are reactive. They record what happened after the fact. Suraksha AI is built to be proactive — detecting, scoring, and surfacing risk in real time so that supervisors can act before an incident becomes an accident.

---

## What Suraksha AI Does

Suraksha AI is an end-to-end safety intelligence platform. It watches camera feeds, detects violations using computer vision, streams those events into a real-time processing engine, computes a dynamic risk score for the site, and surfaces everything through a live dashboard.

The system does five things continuously:

- Detects safety violations using a trained YOLO model
- Pushes each detection into a Pathway streaming engine
- Aggregates events over a sliding time window and computes a weighted risk score
- Classifies the site as LOW, MEDIUM, or CRITICAL risk in real time
- Stores every incident in a PostgreSQL database for audit and analysis

The result is a safety system that thinks in real time, not in retrospect.

---

## Why Pathway

Most backend systems process data in batches or respond to individual API calls. Neither approach is suitable for safety monitoring, where the risk profile of a site can change in seconds.

Pathway is a stream processing engine that treats data as a continuous flow rather than isolated events. In Suraksha AI, Pathway receives every violation detection as it happens, groups events into a 10-second sliding window with a 2-second hop, and produces an updated risk score every 2 seconds.

This means the dashboard does not show you what the risk was 30 seconds ago. It shows you what the risk is now.

```
Violation Detected
       ↓
EventSubject.push() → Pathway Stream
       ↓
Sliding Window (10s duration, 2s hop)
       ↓
Weighted Aggregation
  fire × 3.0 | smoke × 2.5 | restricted_zone × 1.5 | no_helmet × 1.2 | no_vest × 1.0
       ↓
Zone Multiplier Applied
  Zone A × 1.0 | Zone B × 1.5 | Zone C × 2.0 | Zone D × 3.0
       ↓
Risk Score → Level → Accident Probability → Alert
       ↓
on_change() updates global state → Frontend polls every 2 seconds
```

Pathway handles the entire streaming layer. FastAPI handles auth, detection, and storage. The two systems are deliberately separated so each can scale independently.

---

## Risk Model

Risk is not binary. A missing helmet in a low-traffic area is not the same as a fire in a critical zone. Suraksha AI reflects this through a weighted, zone-adjusted scoring model.

**Violation Weights**

| Violation | Weight |
|---|---|
| Fire | 3.0 |
| Smoke | 2.5 |
| Restricted Zone Intrusion | 1.5 |
| No Helmet | 1.2 |
| No Safety Vest | 1.0 |

**Zone Multipliers**

| Zone | Multiplier |
|---|---|
| Zone A — General Area | 1.0× |
| Zone B — Elevated Risk | 1.5× |
| Zone C — Hazardous | 2.0× |
| Zone D — Critical | 3.0× |

**Risk Classification**

| Score | Level | Response |
|---|---|---|
| 0 – 4.9 | LOW | Continue standard monitoring |
| 5 – 9.9 | MEDIUM | Supervisor review, PPE reminder |
| 10+ | CRITICAL | Evacuate, alert safety officer, dispatch response |

**Accident Probability** is computed using a sigmoid function over the risk score:

```
P(accident) = 1 / (1 + e^(-score/5))
```

This gives a smooth, interpretable probability that rises steeply as violations accumulate.

---

## System Architecture

```
┌─────────────────────────────────────────────┐
│              Frontend (HTML/JS)             │
│  Dashboard · Detection · Leaderboard ·      │
│  Pathway Live View                          │
└────────────────────┬────────────────────────┘
                     │ HTTP / REST
┌────────────────────▼────────────────────────┐
│           FastAPI Backend                   │
│  Auth · Detection · Events · Analytics      │
│  Incidents · Zones · Simulator              │
└──────┬─────────────────────┬────────────────┘
       │                     │
┌──────▼──────┐    ┌─────────▼──────────────┐
│  YOLO       │    │  Pathway Stream Engine  │
│  Detection  │    │  Sliding Window · Risk  │
│  Engine     │    │  Score · Alert Logic    │
└──────┬──────┘    └─────────┬──────────────┘
       │                     │
┌──────▼─────────────────────▼──────────────┐
│           PostgreSQL Database              │
│       Users · Incidents · Zones            │
└────────────────────────────────────────────┘
```

---

## Detection Capabilities

Suraksha AI supports three detection modes:

**Image Detection**
Upload any workplace image. The YOLO model runs inference and returns all violations with bounding boxes, confidence scores, zone classification, and risk impact. Results are immediately pushed into the Pathway stream.

**Video Detection**
Upload a video file. The system samples every 5th frame, runs YOLO on each, and aggregates violations across the entire clip. Frame count and total risk score are returned.

**Live Camera Detection**
The frontend captures frames from the user's webcam every 3 seconds, sends them to the detection endpoint, receives violation data back, and draws bounding boxes directly on the live video canvas. Every detection is simultaneously pushed into Pathway and displayed in real time.

---

## Backend Structure

```
backend/
├── app/
│   ├── main.py          — FastAPI app, routes, lifespan, CORS
│   ├── risk_engine.py   — Pathway stream engine, sliding window, risk scoring
│   ├── yolo_engine.py   — YOLO inference wrapper
│   ├── simulator.py     — Safety event simulator for live demo data
│   ├── models.py        — SQLAlchemy User and Incident models
│   ├── database.py      — PostgreSQL connection and session management
│   ├── schemas.py       — Pathway DetectionSchema
│   ├── schemas_auth.py  — Pydantic auth schemas
│   └── auth.py          — JWT token creation, password hashing, user auth
├── requirements.txt
└── .env
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | /signup | Register new user |
| POST | /login | Authenticate and receive JWT |
| GET | /me | Get current user profile |
| POST | /detect/image | Run YOLO on uploaded image |
| POST | /detect/video | Run YOLO on uploaded video |
| POST | /detect/live/start | Start live camera detection |
| POST | /detect/live/stop | Stop live camera detection |
| GET | /detect/live/status | Get live detection status |
| POST | /event | Push manual event into Pathway |
| GET | /analytics | Get current Pathway risk state |
| GET | /risk/summary | Get formatted risk summary |
| GET | /incidents | Get last 100 incidents |
| GET | /leaderboard | Get zone risk leaderboard |
| GET | /zones | Get all zone risk data |
| GET | /health | System health check |
| POST | /simulator/start | Start safety event simulator |
| POST | /simulator/stop | Stop simulator |
| GET | /simulator/status | Get simulator status |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Stream Processing | Pathway 0.29.0 |
| Backend Framework | FastAPI |
| Object Detection | YOLOv8 |
| Database | PostgreSQL |
| ORM | SQLAlchemy |
| Authentication | JWT (python-jose) + bcrypt |
| Frontend | Vanilla HTML, CSS, JavaScript |
| Model Training | Ultralytics YOLO |

---

## Installation

**Clone the repository**

```bash
git clone https://github.com/Amiiiii7119/Suraksha-AI.git
cd Suraksha-AI
```

**Create and activate virtual environment**

```bash
cd backend
python -m venv venv
source venv/bin/activate
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Configure environment**

Create `backend/.env`:

```
DATABASE_URL=postgresql://postgres:admin123@localhost:5432/suraksha
SECRET_KEY=supersecretkey123456789
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

**Create the database**

```bash
psql -U postgres -c "CREATE DATABASE suraksha;"
```

**Start everything**

```bash
cd ..
bash run.sh
```

---

## Access

| Service | URL |
|---|---|
| Frontend Dashboard | http://localhost:3000/app.html |
| Backend API | http://localhost:8000 |
| Interactive API Docs | http://localhost:8000/docs |

**Default login**

```
Email:    amteshwarrajsingh@gmail.com
Password: admin123
```

---

## How the Pathway Integration Works

Pathway is not used as a simple message queue. It is the risk brain of the system.

When any detection occurs — from YOLO, from the simulator, or from a manual API push — the event is passed to `EventSubject.push()`. This feeds directly into a live Pathway stream.

The stream is windowed using `pw.temporal.sliding` with a 10-second duration and 2-second hop. Every 2 seconds, Pathway re-aggregates all events in the current window, applies the weighted scoring model, derives the risk level, computes accident probability, and triggers the `on_change` observer.

The observer updates a shared in-memory state dictionary under a thread lock. The FastAPI `/analytics` endpoint reads from this dictionary. The frontend polls `/analytics` every 2 seconds. The result is a dashboard that reflects site risk with a maximum 4-second lag from detection to display.

The Pathway Live page in the dashboard visualises this entire pipeline — showing the terminal output, sliding window parameters, current velocity, growth rate, predicted risk, and accident probability — so the stream processing layer is not a black box but a visible, explainable part of the system.

---

## Datasets

Models were trained on two custom datasets:

**PPE Dataset**
- Classes: helmet, no helmet, safety vest, no vest, person
- Split: train / val / test
- Format: YOLO annotation format

**Fire and Smoke Dataset**
- Classes: fire, smoke
- Split: train / val / test
- Format: YOLO annotation format

Training runs are stored under `runs/detect/` with weights saved as `.pt` files.

---

## What Makes This Different

Most safety systems are cameras with recording. Suraksha AI is a risk intelligence engine.

The combination of YOLO-based detection, Pathway stream processing, weighted zone-aware risk scoring, and a live dashboard that updates every 2 seconds creates a system that can genuinely inform real-time decisions — not just log what went wrong.

The architecture is also built to scale. The detection layer, stream layer, and storage layer are independent. Each can be upgraded or replaced without touching the others. Live CCTV integration, multi-site deployment, and cloud-native hosting are natural next steps, not rewrites.

---

## Roadmap

- Live CCTV and IP camera stream integration
- Real-time SMS and email alerts on CRITICAL events
- Risk heatmap overlay on site floor plans
- Multi-site dashboard with unified risk view
- Role-based access control for supervisors and admins
- Cloud deployment on AWS or GCP with containerisation

