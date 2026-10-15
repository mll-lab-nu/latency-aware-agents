"""hello's low-level reference skill (robot and mode rule 4): aim(x, y) built from the toolkit only (the throw
interface and /opt/ro/lib/flight.py), the way a low-level player would. Yaw straight at the point; pitch 35 deg; the
speed found by bisection on the flight model's landing distance (the flight model is forward only: the search is the
skill). Deterministic, standard library only."""

import math
import sys

sys.path.insert(0, "/opt/ro/lib")
import flight  # noqa: E402

PITCH = 35.0


def _distance(yaw, speed):
    p = flight.land(yaw, PITCH, speed)
    return math.hypot(p["x"] - flight.RELEASE[0], p["y"] - flight.RELEASE[1])


def aim(x, y, *, state=None, spec=None):
    yaw = math.degrees(math.atan2(y - flight.RELEASE[1], x - flight.RELEASE[0]))
    yaw = min(max(yaw, flight.LIMITS["yaw_deg"][0]), flight.LIMITS["yaw_deg"][1])
    want = math.hypot(x - flight.RELEASE[0], y - flight.RELEASE[1])
    lo, hi = flight.LIMITS["speed_m_s"]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _distance(yaw, mid) < want:
            lo = mid
        else:
            hi = mid
    return {"type": "throw", "yaw": round(yaw, 6), "pitch": PITCH, "speed": round(0.5 * (lo + hi), 6)}
