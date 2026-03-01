import threading
import random
import time
from datetime import datetime, timezone

from app.core.risk_engine import subject
from app.db.database import SessionLocal
from app.db.models import Incident

# ── Import update_zone_tracker from main to keep leaderboard in sync ──
# We import lazily inside the loop to avoid circular import at module load
# time (main imports simulator, simulator would import main).

MIN_INTERVAL = 2.0
MAX_INTERVAL = 6.0

# FIXED: violation types now match RISK_WEIGHTS keys in risk_engine.py
VIOLATION_POOL = [
    ("no_helmet",       0.30),
    ("no_vest",         0.25),
    ("restricted_zone", 0.15),
    ("fire",            0.12),
    ("smoke",           0.08),
]

# FIXED: zones now match ZONE_MULTIPLIERS keys in risk_engine.py
ZONES = ["zone_a", "zone_b", "zone_c", "zone_d"]

CONFIDENCE_RANGE = {
    "no_helmet":       (0.55, 0.97),
    "no_vest":         (0.50, 0.95),
    "restricted_zone": (0.60, 0.92),
    "fire":            (0.40, 0.88),
    "smoke":           (0.45, 0.90),
}

_simulator_running = False
_simulator_thread: threading.Thread | None = None
_events_generated = 0


def _weighted_choice(pool):
    types, weights = zip(*pool)
    return random.choices(types, weights=weights, k=1)[0]


def _generate_event():
    violation_type = _weighted_choice(VIOLATION_POOL)
    zone = random.choice(ZONES)
    conf_min, conf_max = CONFIDENCE_RANGE.get(violation_type, (0.50, 0.95))
    confidence = round(random.uniform(conf_min, conf_max), 3)
    ts = datetime.now(timezone.utc)
    return violation_type, zone, confidence, ts


def _simulator_loop():
    global _simulator_running, _events_generated

    # Lazy import to avoid circular import
    from app.main import update_zone_tracker

    print("[SIMULATOR] 🟢 Simulation started.")

    while _simulator_running:
        try:
            violation_type, zone, confidence, ts = _generate_event()

            # Push to Pathway stream
            subject.push(
                timestamp=ts,
                violation_type=violation_type,
                confidence=confidence,
                zone=zone,
                camera_id="simulator",
            )

            # FIXED: also update zone_tracker so leaderboard stays in sync
            update_zone_tracker(violation_type, zone, confidence)

            # Save to DB
            db = SessionLocal()
            try:
                db.add(Incident(
                    timestamp=ts,
                    zone=zone,
                    violation_type=violation_type,
                    confidence=confidence,
                    risk_impact=confidence * 100,
                ))
                db.commit()
            finally:
                db.close()

            _events_generated += 1
            print(
                f"[SIMULATOR] Event #{_events_generated:04d} | "
                f"{violation_type:<15} | zone: {zone:<10} | conf: {confidence:.2f}"
            )

        except Exception as e:
            print(f"[SIMULATOR] ⚠️  Error: {e}")

        time.sleep(random.uniform(MIN_INTERVAL, MAX_INTERVAL))

    print("[SIMULATOR] 🔴 Simulation stopped.")


def start_simulator():
    global _simulator_running, _simulator_thread
    if _simulator_running:
        return False
    _simulator_running = True
    _simulator_thread = threading.Thread(target=_simulator_loop, daemon=True)
    _simulator_thread.start()
    return True


def stop_simulator():
    global _simulator_running
    _simulator_running = False
    return True


def simulator_status():
    return {
        "running":          _simulator_running,
        "events_generated": _events_generated,
        "interval_range":   f"{MIN_INTERVAL}–{MAX_INTERVAL}s",
        "zones":            ZONES,
        "violation_types":  [v for v, _ in VIOLATION_POOL],
    }