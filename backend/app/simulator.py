# app/simulator.py
"""
Suraksha AI — Live Safety Event Simulator
Continuously pushes realistic violation events into the Pathway risk stream
so dashboards, leaderboard, zones, and incident log stay populated with live data.
"""

import threading
import random
import time
from datetime import datetime, timezone

from app.risk_engine import subject
from app.database import SessionLocal
from app.models import Incident

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

# How often a new event fires (seconds)
MIN_INTERVAL = 2.0
MAX_INTERVAL = 6.0

# Violation types and their realistic weights
VIOLATION_POOL = [
    ("no_helmet",    0.30),   # most common
    ("no_vest",      0.25),
    ("person",       0.15),
    ("intrusion",    0.12),
    ("fire_smoke",   0.08),
    ("hardhat",      0.06),
    ("fire",         0.04),
]

ZONES = [
    "construction_site",
    "warehouse_a",
    "loading_bay",
    "electrical_room",
    "rooftop_zone",
    "boiler_room",
    "entry_gate",
]

# confidence range per violation type (min, max)
CONFIDENCE_RANGE = {
    "no_helmet":   (0.55, 0.97),
    "no_vest":     (0.50, 0.95),
    "person":      (0.70, 0.99),
    "intrusion":   (0.60, 0.92),
    "fire_smoke":  (0.45, 0.90),
    "hardhat":     (0.65, 0.98),
    "fire":        (0.40, 0.88),
}

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
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

    print("[SIMULATOR] 🟢 Simulation started.")

    while _simulator_running:
        try:
            violation_type, zone, confidence, ts = _generate_event()

            # Push into Pathway stream
            subject.push(
                timestamp=ts,
                violation_type=violation_type,
                confidence=confidence,
                zone=zone,
                camera_id="simulator",
            )

            # Save to DB so incident log, leaderboard, zones all populate
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
                f"{violation_type:<15} | zone: {zone:<20} | conf: {confidence:.2f}"
            )

        except Exception as e:
            print(f"[SIMULATOR] ⚠️  Error: {e}")

        # Random delay between events
        time.sleep(random.uniform(MIN_INTERVAL, MAX_INTERVAL))

    print("[SIMULATOR] 🔴 Simulation stopped.")


# ─────────────────────────────────────────────
# Public API — called from main.py
# ─────────────────────────────────────────────

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