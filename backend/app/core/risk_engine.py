import pathway as pw
from datetime import timedelta
import math
import threading
import queue


latest_risk_state: dict = {
    "risk_score": 0.0,
    "previous_risk": 0.0,
    "velocity": 0.0,
    "predicted_risk": 0.0,
    "growth_rate": 0.0,
    "risk_level": "LOW",
    "accident_probability": 0.0,
    "helmet_count": 0,
    "vest_count": 0,
    "fire_count": 0,
    "intrusion_count": 0,
    "zone_risk": {},
    "alert_triggered": False,
    "mitigation_text": "",
}
state_lock = threading.Lock()


RISK_WEIGHTS = {
    "no_helmet": 1.2,
    "no_vest":   1.0,
    "restricted_zone": 1.5,
    "fire":  3.0,
    "smoke": 2.5,
}

ZONE_MULTIPLIERS = {
    "zone_a": 1.0,
    "zone_b": 1.5,
    "zone_c": 2.0,
    "zone_d": 3.0,
}

ALERT_THRESHOLD = 8.0

MITIGATION_MAP = {
    "LOW":      "✅ No immediate action required. Continue standard monitoring.",
    "MEDIUM":   "⚠️ Supervisor should review live feed. Remind workers of PPE compliance.",
    "CRITICAL": "🚨 IMMEDIATE ACTION: Evacuate at-risk zones, alert safety officer, and dispatch response team.",
}


from app.schemas.schemas import DetectionSchema


class EventSubject(pw.io.python.ConnectorSubject):
    def __init__(self):
        super().__init__()
        self._queue = queue.Queue()

    def run(self):
        """
        Blocks forever so Pathway treats stream as open/live.
        Without this, pw.run() finishes immediately and thread dies.
        """
        print("[PATHWAY] EventSubject.run() started — stream is live.")
        while True:
            try:
                item = self._queue.get(timeout=1)
                if item is None:   # shutdown signal
                    print("[PATHWAY] EventSubject shutting down.")
                    break
            except queue.Empty:
                continue  

    def push(self, **kwargs):
        """Call this to send an event into the Pathway stream."""
        self.next(**kwargs)
        self._queue.put(kwargs)   
    def stop(self):
        """Gracefully stop the subject."""
        self._queue.put(None)


subject = EventSubject()


events = pw.io.python.read(subject, schema=DetectionSchema)


pw.io.subscribe(
    events,
    lambda key, row, time, is_addition: print(f"[RAW EVENT] {row}")
)


def weighted_score(v: str, zone: str) -> float:
    return RISK_WEIGHTS.get(v, 1.0) * ZONE_MULTIPLIERS.get(zone, 1.0)

def is_helmet(v: str) -> int:
    return 1 if v == "no_helmet" else 0

def is_vest(v: str) -> int:
    return 1 if v == "no_vest" else 0

def is_fire(v: str) -> int:
    return 1 if v in ("fire", "smoke") else 0

def is_intrusion(v: str) -> int:
    return 1 if v == "restricted_zone" else 0

def sigmoid(x: float) -> float:
    return round(1 / (1 + math.exp(-x / 5)), 4)

def classify_risk(score: float) -> str:
    if score < 5:
        return "LOW"
    elif score < 10:
        return "MEDIUM"
    return "CRITICAL"

def calc_velocity(curr: float, prev: float) -> float:
    return round(curr - prev, 4)

def calc_predicted(curr: float, vel: float) -> float:
    return round(curr + vel * 2, 4)

def calc_growth_rate(curr: float, prev: float) -> float:
    if prev == 0:
        return 0.0
    return round((curr - prev) / prev * 100, 2)

def get_mitigation(score: float) -> str:
    return MITIGATION_MAP.get(classify_risk(score), "")

def get_alert(score: float) -> bool:
    return score > ALERT_THRESHOLD


windowed = events.windowby(
    events.timestamp,
    window=pw.temporal.sliding(
        duration=timedelta(seconds=10),
        hop=timedelta(seconds=2),
    ),
    behavior=pw.temporal.common_behavior(
        delay=timedelta(seconds=2),
        cutoff=timedelta(seconds=5),
        keep_results=False,
    ),
)


aggregated = windowed.groupby().reduce(
    risk_score      = pw.reducers.sum(pw.apply(weighted_score, pw.this.violation_type, pw.this.zone)),
    helmet_count    = pw.reducers.sum(pw.apply(is_helmet,      pw.this.violation_type)),
    vest_count      = pw.reducers.sum(pw.apply(is_vest,        pw.this.violation_type)),
    fire_count      = pw.reducers.sum(pw.apply(is_fire,        pw.this.violation_type)),
    intrusion_count = pw.reducers.sum(pw.apply(is_intrusion,   pw.this.violation_type)),
)


final_table = aggregated.with_columns(
    accident_probability = pw.apply(sigmoid,        aggregated.risk_score),
    risk_level           = pw.apply(classify_risk,  aggregated.risk_score),
    alert_triggered      = pw.apply(get_alert,      aggregated.risk_score),
    mitigation_text      = pw.apply(get_mitigation, aggregated.risk_score),
)

def on_change(key, row, time, is_addition):
    print(f"[ON_CHANGE] is_addition={is_addition} | row={row}")

    if not is_addition:
        return

    risk_score      = float(row.get("risk_score", 0))
    helmet_count    = int(row.get("helmet_count", 0))
    vest_count      = int(row.get("vest_count", 0))
    fire_count      = int(row.get("fire_count", 0))
    intrusion_count = int(row.get("intrusion_count", 0))
    level           = row.get("risk_level", "LOW")
    prob            = float(row.get("accident_probability", 0))
    alert           = bool(row.get("alert_triggered", False))
    mitigation      = row.get("mitigation_text", "")

    with state_lock:
        prev_risk = latest_risk_state["risk_score"]
        vel       = calc_velocity(risk_score, prev_risk)
        predicted = calc_predicted(risk_score, vel)
        growth    = calc_growth_rate(risk_score, prev_risk)

        latest_risk_state.update({
            "risk_score":           round(risk_score, 4),
            "previous_risk":        round(prev_risk, 4),
            "velocity":             vel,
            "predicted_risk":       predicted,
            "growth_rate":          growth,
            "risk_level":           level,
            "accident_probability": prob,
            "helmet_count":         helmet_count,
            "vest_count":           vest_count,
            "fire_count":           fire_count,
            "intrusion_count":      intrusion_count,
            "alert_triggered":      alert,
            "mitigation_text":      mitigation,
        })

    print(f"[RISK ENGINE] score={risk_score:.2f} | level={level} | "
          f"vel={vel:+.2f} | predicted={predicted:.2f} | "
          f"alert={'🚨 YES' if alert else '✅ NO'}")


pw.io.subscribe(final_table, on_change)


def start_pathway():
    print("[PATHWAY] pw.run() starting...")
    pw.run()
    print("[PATHWAY] pw.run() exited.")   