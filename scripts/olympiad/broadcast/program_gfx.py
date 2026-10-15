"""On-screen graphics of the program feed (the TV-style broadcast with an automatic director, broadcast.py /program/).

Pure Pillow, drawn into the frame by the render helper (so the MJPEG stream itself carries them): the score bug (top
left), the LIVE / REPLAY tag with the speed note and the shot label (top right), the result banner of a replay, the
agents' lower thirds (one per seat, seat colour, model name, latest public message / action / review) and the
"RoboOlympics" stinger wipe between live and replay.

The overlay dict (all keys optional) built by broadcast.Program:
  bug      {"sport": "Table tennis", "unit": "Point 3", "rows": [[seat, name, score(, note)], ...]}  (note: a small
           line before the score, e.g. racing "CP 9/16 · 1 warning", air hockey "1 foul")
  tag      "LIVE" | "REPLAY"        speed  "World slowed ×10" / "Slow motion 0.25×"  shot "Close-up"
  banner   result text of the replayed point                                   phase  small status line ("Review pause")
  lower    [{"seat": "A", "name": "Luna", "kind": "Review", "text": "...", "age": "12 s ago"}, ...]
  seat_rgb {seat: rgb} colours for seats beyond A / B (soccer: A1, A2, B1, B2) or overriding them
  stinger  0..1 (progress of the wipe; at 0.5 the frame is fully covered)       blank  True: no scene under it
Turn-based sports (billiards) also: bug["turn"] (the seat to play, marked in the bug), call "text" or [label, text]
(a chip at the top centre: the called shot, ball in hand), rack [{"seat", "label", "balls": ["1", ...]}] (rows under
the bug: group and the balls left, drawn with ball_rgb {ball: rgb}, which the billiards helper fills in).
Fonts: Noto Sans CJK from BC_FONT_DIR (the host's /usr/share/fonts/opentype/noto mounted read-only), else any CJK
font found, else Pillow's default font (Latin only).
"""
from __future__ import annotations

import glob
from functools import lru_cache
import math
import os

from PIL import Image, ImageDraw, ImageFont
import audience

SEAT_RGB = audience.SEAT_RGB
NAVY, BLUE, GOLD, RED = (11, 22, 52), (33, 85, 239), (242, 183, 5), (214, 40, 40)
BRAND, BRAND_CN = "RoboOlympics", "Embodied agents at play"

_FONTS: dict = {}
_FILES: dict = {}


def _font_file(weight):
    if weight in _FILES:
        return _FILES[weight]
    dirs = [os.environ.get("BC_FONT_DIR", "/fonts"), "/usr/share/fonts/opentype/noto", "/usr/share/fonts"]
    want = {"bold": ("NotoSansCJK-Bold", "NotoSansCJK-Black"), "black": ("NotoSansCJK-Black", "NotoSansCJK-Bold"),
            "regular": ("NotoSansCJK-Medium", "NotoSansCJK-Regular", "NotoSansCJK-DemiLight")}[weight]
    path = None
    for d in dirs:
        for w in want:
            hits = sorted(glob.glob(os.path.join(d, "**", w + ".tt[cf]"), recursive=True))
            if hits:
                path = hits[0]
                break
        if path:
            break
    if path is None:                                    # any CJK font (wqy, droid, arphic)
        for d in dirs:
            hits = [p for p in glob.glob(os.path.join(d, "**", "*.tt[cf]"), recursive=True)
                    if any(x in p.lower() for x in ("cjk", "wqy", "droidsansfallback", "uming", "ukai"))]
            if hits:
                path = sorted(hits)[0]
                break
    _FILES[weight] = path
    return path


def font(size, weight="regular"):
    key = (size, weight)
    if key not in _FONTS:
        path, f = _font_file(weight), None
        if path:
            for idx in ((2, 0) if path.endswith(".ttc") else (0,)):     # .ttc face 2 = Simplified Chinese in Noto
                try:
                    f = ImageFont.truetype(path, size, index=idx)
                    break
                except (OSError, ValueError):
                    continue
        if f is None:
            try:
                f = ImageFont.load_default(size=size)
            except TypeError:
                f = ImageFont.load_default()
        _FONTS[key] = f
    return _FONTS[key]


def ease(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


_MEASURE = ImageDraw.Draw(Image.new("RGB", (1, 1)))


@lru_cache(maxsize=4096)
def textlength(text, *, font):
    return _MEASURE.textlength(text, font=font)


def fit(dr, text, f, width):
    return _fit(text, f, width)


@lru_cache(maxsize=1024)
def _fit(text, f, width):
    """text cut to one line of the given width (with an ellipsis)."""
    if textlength(text, font=f) <= width:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if textlength(text[:mid] + "…", font=f) <= width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + "…"


def wrap(dr, text, f, width, lines=2):
    # No frame-specific image/draw object in the key: layout depends on text, font and width.
    return list(_wrap(text, f, width, lines))


@lru_cache(maxsize=512)
def _wrap(text, f, width, lines=2):
    """Greedy wrap (CJK by character, Latin by word) into at most `lines` lines; the last one ellipsized."""
    out, cur = [], ""
    tokens, buf = [], ""
    for ch in text:                                     # words for Latin, single characters for CJK
        if ch == " ":
            if buf:
                tokens.append(buf)
            tokens.append(" ")
            buf = ""
        elif ord(ch) > 0x2E80:
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(ch)
        else:
            buf += ch
    if buf:
        tokens.append(buf)
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if textlength(cur + t, font=f) <= width or not cur.strip():
            if not cur.strip() and textlength(t, font=f) > width:          # one very long token
                out.append(fit(_MEASURE, t, f, width))
                cur = ""
            else:
                cur += t
            i += 1
        else:
            out.append(cur.rstrip())
            cur = ""
            if t == " ":
                i += 1
            if len(out) == lines:
                break
    if len(out) < lines and cur.strip():
        out.append(cur.strip())
    if i < len(tokens) and out:
        rest = out[-1] + " " + "".join(tokens[i:])
        out[-1] = fit(_MEASURE, rest, f, width) if textlength(rest, font=f) > width else rest
    return tuple(x for x in out[:lines] if x)


def rbox(dr, xy, fill, r=6):
    try:
        dr.rounded_rectangle(xy, r, fill=fill)
    except AttributeError:
        dr.rectangle(xy, fill=fill)


def decorate_program(im, ov):
    W, H = im.size
    u = H / 540.0                                       # layout unit: 1 px at 960x540
    dr = ImageDraw.Draw(im, "RGBA")
    rgb = {**SEAT_RGB, **(ov.get("seat_rgb") or {})}       # e.g. soccer: the four robots' torso colours
    fs, fm, fb, fl = font(int(13 * u)), font(int(16 * u)), font(int(17 * u), "bold"), font(int(22 * u), "black")
    # ---- score bug (top left)
    bug = ov.get("bug")
    bug_right = 0
    if bug:
        x0, y0 = int(14 * u), int(12 * u)
        rows = bug.get("rows") or []
        rh = int(30 * u)
        names = [fit(dr, str(r[1]), fb, 190 * u) for r in rows]
        notes = [fit(dr, str(r[3]), fs, 150 * u) if len(r) > 3 and r[3] else "" for r in rows]   # e.g. "CP 9/16"
        wnote = max([textlength(x, font=fs) for x in notes] + [0]) + (12 * u if any(notes) else 0)
        wn = max([textlength(n, font=fb) for n in names] + [80 * u]) + 22 * u + wnote
        ws = max([textlength(str(r[2]), font=fl) for r in rows] + [20 * u]) + 22 * u
        head = f"{bug.get('sport', '')}" + (f"  ·  {bug['unit']}" if bug.get("unit") else "")
        hw = max(wn + ws + 8 * u, textlength(head, font=fs) + 22 * u)
        rbox(dr, [x0, y0, x0 + hw, y0 + int(22 * u)], (*NAVY, 235), int(4 * u))
        dr.rectangle([x0, y0, x0 + int(4 * u), y0 + int(22 * u)], fill=(*GOLD, 255))
        dr.text((x0 + 12 * u, y0 + 3 * u), head, font=fs, fill=(235, 240, 255, 255))
        y = y0 + int(24 * u)
        bug_right = x0 + hw
        for r, n, note in zip(rows, names, notes):
            seat, score = r[0], r[2]
            col = rgb.get(seat, (180, 180, 180))
            dr.rectangle([x0, y, x0 + hw, y + rh], fill=(12, 18, 36, 215))
            dr.rectangle([x0, y, x0 + int(6 * u), y + rh], fill=(*col, 255))
            dr.text((x0 + 14 * u, y + (rh - 17 * u) / 2 - 3 * u), n, font=fb, fill=(255, 255, 255, 255))
            if note:
                nw = textlength(note, font=fs)
                dr.text((x0 + hw - ws - nw - 9 * u, y + (rh - 13 * u) / 2 - 2 * u), note, font=fs, fill=(255, 225, 140, 255))
            dr.rectangle([x0 + hw - ws, y, x0 + hw, y + rh], fill=(*GOLD, 240))
            sw = textlength(str(score), font=fl)
            dr.text((x0 + hw - ws / 2 - sw / 2, y + (rh - 22 * u) / 2 - 5 * u), str(score), font=fl, fill=(*NAVY, 255))
            y += rh + int(2 * u)
            if bug.get("turn") == seat:                    # turn-based sports: whose turn it is
                tx = x0 + hw - ws - 12 * u
                dr.polygon([(tx - 7 * u, y - rh - 2 * u + rh / 2 - 6 * u), (tx - 7 * u, y - rh - 2 * u + rh / 2 + 6 * u),
                            (tx + 2 * u, y - rh - 2 * u + rh / 2)], fill=(*GOLD, 255))
        for r in ov.get("rack") or []:                     # billiards: each player's group and the balls left
            y = rack_row(dr, ov, x0, y, hw, r, u, fs)
        if ov.get("phase"):
            pw = textlength(ov["phase"], font=fs) + 18 * u
            rbox(dr, [x0, y + 2 * u, x0 + pw, y + 22 * u], (0, 0, 0, 150), int(4 * u))
            dr.text((x0 + 9 * u, y + 4 * u), ov["phase"], font=fs, fill=(255, 225, 140, 255))
    if ov.get("call"):                                     # turn-based: the called shot / ball in hand (top centre)
        lab, txt = ov["call"] if isinstance(ov["call"], (list, tuple)) else ("CALL", ov["call"])
        lw = textlength(lab, font=fs) + 16 * u
        tag_left = W - 14 * u                            # keep clear of the LIVE / REPLAY tag (drawn below)
        if ov.get("tag"):
            tag_left -= textlength("● LIVE" if ov["tag"] == "LIVE" else "▶ REPLAY", font=fb) + 22 * u
            tag_left -= (textlength(ov["speed"], font=fm) + 20 * u) if ov.get("speed") else 0
        x0c = bug_right + 12 * u
        txt = fit(dr, txt, fm, max(60 * u, tag_left - 12 * u - x0c - lw - 20 * u))
        tw_ = textlength(txt, font=fm) + 20 * u
        x = min(max(W / 2 - (lw + tw_) / 2, x0c), tag_left - 12 * u - lw - tw_)
        rbox(dr, [x, int(12 * u), x + lw + tw_, int(40 * u)], (*NAVY, 225), int(4 * u))
        rbox(dr, [x, int(12 * u), x + lw, int(40 * u)], (*GOLD, 245), int(4 * u))
        dr.text((x + 8 * u, int(18 * u)), lab, font=fs, fill=(*NAVY, 255))
        dr.text((x + lw + 10 * u, int(15 * u)), txt, font=fm, fill=(255, 255, 255, 255))
    # ---- LIVE / REPLAY tag, speed note, shot label, brand (top right)
    tag = ov.get("tag")
    xr, yt = W - int(14 * u), int(12 * u)
    if tag:
        live = tag == "LIVE"
        txt = "● LIVE" if live else "▶ REPLAY"
        tw = textlength(txt, font=fb) + 22 * u
        speed = ov.get("speed") or ""
        sw = textlength(speed, font=fm) + 20 * u if speed else 0
        rbox(dr, [xr - tw - sw, yt, xr, yt + int(28 * u)], (0, 0, 0, 165), int(4 * u))
        rbox(dr, [xr - tw - sw, yt, xr - sw, yt + int(28 * u)], (*(RED if live else BLUE), 245), int(4 * u))
        dr.text((xr - tw - sw + 11 * u, yt + 3 * u), txt, font=fb, fill=(255, 255, 255, 255))
        if speed:
            dr.text((xr - sw + 10 * u, yt + 4 * u), speed, font=fm, fill=(255, 225, 140, 255))
        yt += int(32 * u)
        if ov.get("shot"):
            shw = textlength(ov["shot"], font=fm) + 20 * u
            rbox(dr, [xr - shw, yt, xr, yt + int(25 * u)], (*NAVY, 220), int(4 * u))
            dr.text((xr - shw + 10 * u, yt + 2 * u), ov["shot"], font=fm, fill=(255, 255, 255, 255))
            yt += int(29 * u)
    brand = BRAND
    bw = textlength(brand, font=fs)
    dr.text((xr - bw, yt + 1 * u), brand, font=fs, fill=(255, 255, 255, 170), stroke_width=1, stroke_fill=(0, 0, 0, 120))
    # ---- lower thirds (one per seat) and the replay banner above them
    lower = [x for x in (ov.get("lower") or []) if x.get("text")]
    ybase = H - int(10 * u)
    if lower:
        n = len(lower)
        gap = int(10 * u)
        bw_ = (W - int(28 * u) - gap * (n - 1)) / n
        tops = []
        blocks = []
        for i, c in enumerate(lower):
            lines = []
            for clause in audience.caption_lines(c):
                lines.extend(wrap(dr, clause, fm, bw_ - 26 * u, 2))
            lines = lines[:4]
            h = int(26 * u) + int(21 * u) * len(lines) + int(8 * u)
            blocks.append((lines, h))
            tops.append(ybase - h)
        top = min(tops)
        for i, (c, (lines, h)) in enumerate(zip(lower, blocks)):
            x = int(14 * u) + i * (bw_ + gap)
            col = rgb.get(c.get("seat"), (150, 150, 150))
            dr.rectangle([x, top, x + bw_, ybase], fill=(10, 14, 28, 205))
            dr.rectangle([x, top, x + int(5 * u), ybase], fill=(*col, 255))
            plate = c.get("name", "") + (" · " + c["seat"] if len(c.get("seat", "")) > 1 else "")
            pw = textlength(plate, font=fb) + 20 * u
            dr.rectangle([x + int(5 * u), top, x + int(5 * u) + pw, top + int(24 * u)], fill=(*col, 235))
            dr.text((x + 14 * u, top + 1 * u), plate, font=fb, fill=(255, 255, 255, 255))
            kx = x + int(5 * u) + pw + 8 * u
            if c.get("kind"):
                label = {"Review": "Model review · summary" if c.get("summary") else "Model's review", "Action": "Activity", "Says": "Model says"}.get(c["kind"], c["kind"])
                kc = (255, 210, 90, 255) if c["kind"] == "Review" else (120, 230, 170, 255) if c["kind"].startswith("Team") \
                    else (190, 205, 230, 255)
                dr.text((kx, top + 4 * u), label, font=fs, fill=kc)
                kx += textlength(label, font=fs) + 8 * u
            if c.get("age"):
                aw = textlength(c["age"], font=fs)
                dr.text((x + bw_ - aw - 10 * u, top + 4 * u), c["age"], font=fs, fill=(150, 160, 180, 255))
            for j, ln in enumerate(lines):
                dr.text((x + 14 * u, top + 28 * u + j * 21 * u), ln, font=fm, fill=(245, 247, 252, 255))
        ybase = top - int(6 * u)
    if ov.get("banner"):
        txt = fit(dr, ov["banner"], fb, W - 80 * u)
        tw = textlength(txt, font=fb) + 30 * u
        x = (W - tw) / 2
        rbox(dr, [x, ybase - int(30 * u), x + tw, ybase], (*GOLD, 240), int(4 * u))
        dr.text((x + 15 * u, ybase - 27 * u), txt, font=fb, fill=(*NAVY, 255))
    if ov.get("stinger") is not None:
        stinger(im, float(ov["stinger"]))
    return im


def rack_row(dr, ov, x0, y, hw, r, u, fs):
    """One line under the score bug: "A · Solids" and a small ball per ball left (stripes with white caps)."""
    rgb = ov.get("ball_rgb") or {}
    rh = int(21 * u)
    label = f"{r.get('seat', '')} · {r.get('label', '')}" if r.get("seat") else r.get("label", "")
    balls = r.get("balls") or []
    rad = 7.5 * u
    lw = textlength(label, font=fs) + 14 * u
    w = max(hw, lw + len(balls) * (2 * rad + 3 * u) + 10 * u)
    dr.rectangle([x0, y, x0 + w, y + rh], fill=(12, 18, 36, 200))
    dr.rectangle([x0, y, x0 + int(3 * u), y + rh], fill=(*SEAT_RGB.get(r.get("seat"), (150, 150, 150)), 255))
    dr.text((x0 + 9 * u, y + 3 * u), label, font=fs, fill=(220, 228, 245, 255))
    fn = font(max(7, int(8 * u)), "bold")
    cx = x0 + lw + rad
    for b in balls:
        col = tuple(rgb.get(str(b), (200, 200, 200)))
        cy = y + rh / 2
        box = [cx - rad, cy - rad, cx + rad, cy + rad]
        dr.ellipse(box, fill=(*col, 255), outline=(0, 0, 0, 255))
        if str(b).isdigit() and int(b) >= 9:
            dr.chord(box, 210, 330, fill=(246, 246, 240, 255))
            dr.chord(box, 30, 150, fill=(246, 246, 240, 255))
            dr.ellipse(box, outline=(0, 0, 0, 255))
        dr.ellipse([cx - rad * 0.55, cy - rad * 0.55, cx + rad * 0.55, cy + rad * 0.55], fill=(250, 250, 250, 255))
        tw = textlength(str(b), font=fn)
        dr.text((cx - tw / 2, cy - 5.5 * u), str(b), font=fn, fill=(0, 0, 0, 255))
        cx += 2 * rad + 3 * u
    return y + rh + int(2 * u)


def stinger(im, p):
    """The branded wipe: three slanted bands (gold, blue, navy) sweep in from the left, cover the frame with the
    RoboOlympics logo around p = 0.5 (the cut happens under it) and sweep out to the right."""
    W, H = im.size
    u = H / 540.0
    dr = ImageDraw.Draw(im, "RGBA")
    slant = 0.35 * H
    span = W + 2 * slant

    def band(x_from, x_to, col):
        if x_to <= x_from:
            return
        dr.polygon([(x_from + slant, 0), (x_to + slant, 0), (x_to, H), (x_from, H)], fill=col)

    cols = [(*GOLD, 255), (*BLUE, 255), (*NAVY, 255)]  # drawn in this order: navy on top
    for i, col in enumerate(cols):
        if p < 0.5:                                     # in: gold leads, navy closes the frame
            t = ease((p / 0.5 - 0.12 * i) / 0.76)
            band(-slant - 40, -slant - 40 + (span + 40) * t, col)
        else:                                           # out: navy leaves first, revealing blue, then gold
            t = ease(((p - 0.5) / 0.5 - 0.12 * (2 - i)) / 0.76)
            band(-slant - 40 + (span + 40) * t, W + 40, col)
    cover = 1.0 - min(1.0, abs(p - 0.5) / 0.17)          # the logo, while the navy band covers the middle
    if cover > 0:
        a = int(255 * min(1.0, cover * 1.8))
        s = 0.92 + 0.08 * cover
        fbig, fsub = font(int(64 * u * s), "black"), font(int(21 * u * s), "bold")
        sub = BRAND_CN.upper()
        tw, sw = textlength(BRAND, font=fbig), textlength(sub, font=fsub)
        r = 46 * u * s                                   # emblem: a hexagon with a stylised robot-arm stroke
        total = 2 * r + 28 * u + max(tw, sw)
        ex, ey = W / 2 - total / 2 + r, H / 2
        cy = H / 2
        hexp = [(ex + r * math.cos(math.pi / 3 * k + math.pi / 6), ey + r * math.sin(math.pi / 3 * k + math.pi / 6))
                for k in range(6)]
        dr.polygon(hexp, fill=(*GOLD, a))
        dr.line([(ex - 20 * u, ey + 22 * u), (ex - 4 * u, ey - 8 * u), (ex + 18 * u, ey - 2 * u)],
                fill=(*NAVY, a), width=max(2, int(8 * u)), joint="curve")
        dr.ellipse([ex + 12 * u, ey - 22 * u, ex + 28 * u, ey - 6 * u], fill=(*NAVY, a))
        x = ex + r + 28 * u
        dr.text((x, cy - 58 * u * s), BRAND, font=fbig, fill=(255, 255, 255, a))
        dr.rectangle([x, cy + 24 * u * s, x + max(tw, sw) * cover, cy + 27 * u * s], fill=(255, 255, 255, a))
        dr.text((x + 2 * u, cy + 32 * u * s), sub, font=fsub, fill=(*GOLD, a))
    return im


def blank(size):
    return Image.new("RGB", tuple(size), NAVY)
