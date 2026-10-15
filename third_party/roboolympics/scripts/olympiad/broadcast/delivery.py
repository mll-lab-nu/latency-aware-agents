"""Shared, bounded JPEG renditions of a spectator feed; never calls the simulator."""
import io
import threading
import time

from PIL import Image

# Original retains the renderer's configured resolution/FPS. Lower tiers also
# reduce frame rate and JPEG payload; scaling an <img> in CSS saves no bandwidth.
PROFILES = {
    'original': None,
    '720p': ((1280, 720), 24, 78),
    '540p': ((960, 540), 20, 70),
    '360p': ((640, 360), 12, 60),
}


class Rendition:
    def __init__(self, size, fps, quality):
        self.size, self.fps, self.quality = size, fps, quality
        self.lock = threading.Lock()
        self.seq, self.jpeg, self.due = -1, None, 0.0

    def frame(self, seq, jpeg):
        # One cache per feed/tier, shared across viewers. Conversion is performed
        # in a request thread outside the source renderer/condition lock.
        with self.lock:
            now = time.monotonic()
            if self.jpeg is not None and (seq <= self.seq or now < self.due):
                return self.seq, self.jpeg
            with Image.open(io.BytesIO(jpeg)) as im:
                im.draft('RGB', self.size)
                im.thumbnail(self.size, getattr(Image, 'Resampling', Image).BILINEAR)
                out = io.BytesIO()
                im.convert('RGB').save(out, 'JPEG', quality=self.quality, subsampling=2)
            self.seq, self.jpeg = seq, out.getvalue()
            period = 1.0 / self.fps
            # Keep a cadence across source ticks (24 FPS on a 30 FPS source),
            # rather than rounding every interval up to two source frames.
            self.due = now + period if now - self.due > period else self.due + period
            return self.seq, self.jpeg


class Delivery:
    def __init__(self):
        self.tiers = {key: Rendition(*spec) for key, spec in PROFILES.items() if spec}

    def frame(self, profile, seq, jpeg):
        if profile == 'original':
            return seq, jpeg
        return self.tiers[profile].frame(seq, jpeg)
