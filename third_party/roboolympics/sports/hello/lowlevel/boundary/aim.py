"""hello's boundary baseline (robot and mode rule 5): the toolkit used WITHOUT developing the skill. It points the
launcher at the requested point (the yaw is plain geometry) and throws at the launcher's nominal setting (pitch
35 deg, 7 m/s): no aiming in depth. Its API-test success must stay at or below the manifest's threshold (0.05);
otherwise the contract's tolerance is tightened (the tools stay)."""

import math


def aim(x, y, *, state=None, spec=None):
    yaw = min(max(math.degrees(math.atan2(y, x)), -20.0), 20.0)
    return {"type": "throw", "yaw": round(yaw, 6), "pitch": 35.0, "speed": 7.0}
