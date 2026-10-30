"""
Effects + director for the haunted mirror.

The Director is handed the mirrored frame and the people in it, and decides
what to show: a style per passer-by, an escalating routine for anyone who
stops to watch, a parting stare when they leave, and the odd apparition when
the street is empty. Everything is classical compositing on the CPU.

Knobs via env (all optional):
  HM_GRADE           1/0 horror colour grade (default 1)
  HM_GRAIN           grain strength 0-30 (default 10)
  HM_STYLES          passer-by styles to pick from (default follower,echo,both)
  HM_FOLLOWER_SIZE   wraith height as a fraction of the person (default 0.85)
  HM_FOLLOWER_GAP    person-heights it hovers behind them (default 0.6)
  HM_ZOOM            push-in on a passer-by (default 1.5, 1 = off)
  HM_ZOOM_WATCH      push-in on someone who has stopped (default 1.9)
  HM_BINOS           chance a passer-by is watched through binoculars, and
                     that a blink cuts to them for a watcher (default 0.35,
                     0 = off)
  HM_ZOOM_BINOS      how far the binoculars push in (default 2.4)
  HM_ATTRACT         1/0 apparitions when nobody is around (default 1)
  HM_CLOSE           1/0 close-range extras when someone fills a third of the
                     frame: wraith behind them, draining, hollow eyes
                     (default 1; experimental, see close.py)
  HM_DEBUG           1/0 draw what the tracker sees (default 0)
Phantoms (earlier passers-by replayed into the scene) have theirs in
phantoms.py.
"""
import os
import random
from collections import deque

import cv2
import numpy as np

from . import close
from . import phantoms
from .vision import STILL_SPEED

DO_GRADE = os.environ.get("HM_GRADE", "1") == "1"
GRAIN = int(os.environ.get("HM_GRAIN", "10"))
STYLES = [s for s in os.environ.get(
    "HM_STYLES", "follower,echo,both").split(",") if s]
FOLLOWER_SIZE = float(os.environ.get("HM_FOLLOWER_SIZE", "0.85"))
FOLLOWER_GAP = float(os.environ.get("HM_FOLLOWER_GAP", "0.6"))
ZOOM = max(1.0, float(os.environ.get("HM_ZOOM", "1.5")))
ZOOM_WATCH = max(ZOOM, float(os.environ.get("HM_ZOOM_WATCH", "1.9")))
BINOS = float(os.environ.get("HM_BINOS", "0.35"))
ZOOM_BINOS = float(os.environ.get("HM_ZOOM_BINOS", "2.4"))
DO_ATTRACT = os.environ.get("HM_ATTRACT", "1") == "1"
DO_CLOSE = os.environ.get("HM_CLOSE", "1") == "1"
DEBUG = os.environ.get("HM_DEBUG", "0") == "1"

MIN_FIGURE_H = 0.08     # of frame height; the wraith never shrinks below this
TRAIL_SECS = 1.2        # how long the wraith's smoke hangs around
CLOSE_PERSON = 0.33     # of frame height; from here up the close-range extras run
SMALL_PERSON = 0.12     # of frame height; below this an echo is too faint to see
WATCH_AFTER = 2.5       # seconds standing still before the watcher routine starts
APPROACH_SECS = 18.0    # how long the wraith takes to creep to their shoulder
BLINK_SECS = 0.3        # length of the glitch that hides a jump
STARE_SECS = 2.5        # how long it stays looking at where they stood
ECHOES = ((0.45, 0.5), (0.9, 0.3))  # (seconds behind, strength) per ghost copy
HISTORY_SECS = 1.2
# seconds (min, max) before a phantom walks through: after someone stops to
# watch, between phantoms while they stay, and the same for an empty street
PHANTOM_WATCH_WAIT = (6, 12)
PHANTOM_WATCH_GAP = (20, 40)
PHANTOM_IDLE_WAIT = (60, 150)
PHANTOM_IDLE_GAP = (90, 200)
PHANTOM_REVERSE = 0.2   # chance a phantom walks its path backwards


def make_vignette(shape, strength=0.75):
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    d = d / d.max()
    v = 1 - strength * (d ** 2.2)
    return np.clip(v, 0, 1).astype(np.float32)[..., None]


_grain = []     # pre-rolled noise frames, see horror_grade


def horror_grade(frame, vig, t):
    f = frame.astype(np.float32)
    # desaturate
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
    f = f * 0.55 + gray * 0.45
    # cold sickly tint (BGR): lift blue/green, drop red
    f *= np.array([1.12, 1.03, 0.80], dtype=np.float32)
    # contrast crush toward shadow
    f = (f - 110) * 1.18 + 95
    # vignette
    f *= vig
    # film grain
    if GRAIN > 0:
        if not _grain or _grain[0].shape != frame.shape:
            _grain[:] = [np.random.normal(0, GRAIN, frame.shape)
                         .astype(np.float32) for _ in range(8)]
        f += random.choice(_grain)
    f = np.clip(f, 0, 255).astype(np.uint8)
    # occasional analog glitch: channel/ row offset
    if random.random() < 0.015:
        sh = random.randint(3, 10)
        f[:, :, 2] = np.roll(f[:, :, 2], sh, axis=1)   # red bleed
    if random.random() < 0.01:
        gh = random.randint(4, 16)
        y0 = random.randint(0, f.shape[0] - gh)
        f[y0:y0 + gh] = np.roll(f[y0:y0 + gh], random.randint(-30, 30), axis=1)
    return f


def draw_figure(frame, x, ground_y, fh, alpha, t, trail=(), now=0.0,
                lean=0.0, eyes=0.0):
    """Hovering wraith over (x, ground_y), fh px tall, darkening the frame
    beneath it. `trail` crumbs (ts, x, ground_y) leave fading smoke; `lean`
    (-1..1) tips it toward its prey and streams the tail out behind. `eyes`
    (0..1) holds them open; at 0 they only open now and then."""
    H, Wd = frame.shape[:2]
    pts = [(x, ground_y)] + [(c[1], c[2]) for c in trail]
    pad = int(fh * 0.9) + 4
    x0 = max(0, int(min(p[0] for p in pts)) - pad)
    x1 = min(Wd, int(max(p[0] for p in pts)) + pad)
    y0 = max(0, int(min(p[1] for p in pts) - fh * 1.2) - pad)
    y1 = min(H, int(max(p[1] for p in pts)) + pad)
    if alpha <= 0 or x1 <= x0 or y1 <= y0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), np.float32)
    # smoke: older crumbs are smaller, fainter and drift upward
    for ts, tx, ty in trail:
        age = (now - ts) / TRAIL_SECS
        if age >= 1:
            continue
        wob = np.sin(ts * 9) * fh * 0.05
        cv2.circle(mask,
                   (int(tx + wob - x0),
                    int(ty - fh * (0.5 + 0.2 * age) - y0)),
                   max(1, int(fh * 0.15 * (1 - 0.5 * age))),
                   0.6 * (1 - age) ** 2, -1)
    # floats clear of the ground, bobbing; head + shoulders taper to a tail
    bob = np.sin(t * 1.3) * fh * 0.04
    bottom = ground_y - fh * 0.15 + bob - y0
    head_r = max(2, int(fh * 0.09))
    hx = int(x + lean * fh * 0.05 - x0)
    hy = int(bottom - fh + head_r)
    sh = hy + head_r * 1.2
    mid = bottom - fh * 0.35
    tail_x = x - x0 - lean * fh * 0.22 + np.sin(t * 2.3) * fh * 0.06
    body = np.array([[hx - fh * 0.15, sh], [hx + fh * 0.15, sh],
                     [x - x0 + fh * 0.10, mid], [tail_x, bottom],
                     [x - x0 - fh * 0.10, mid]], np.int32)
    cv2.fillPoly(mask, [body], 1.0)
    cv2.circle(mask, (hx, hy), head_r, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, fh * 0.035))
    # a wide soft pool of dimness around it, so it still reads as a presence
    # where black-on-dark-road would vanish
    aura = np.zeros_like(mask)
    cv2.ellipse(aura, (int(x - x0), int(bottom - fh * 0.5)),
                (int(fh * 0.45), int(fh * 0.7)), 0, 0, 360, 0.45, -1)
    aura = cv2.GaussianBlur(aura, (0, 0), sigmaX=max(2.0, fh * 0.18))
    mask = np.maximum(mask, aura)
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi * (1 - mask[..., None] * alpha)).astype(np.uint8)
    # eyes: two dim pinpricks that only open now and then
    glow = max(eyes, max(0.0, np.sin(t * 0.9)) ** 6) * alpha
    if glow > 0.1:
        er = max(1, int(head_r * 0.2))
        ex = max(2, int(head_r * 0.45))
        for dx in (-ex, ex):
            cv2.circle(frame, (hx + x0 + dx, hy + y0), er,
                       (30, 30, int(220 * glow)), -1)


def glitch(frame, k):
    """The cut that hides a jump. k runs 1 -> 0 over the blink: a dark beat,
    then the picture tearing back together."""
    sh = int(6 + 18 * k)
    frame[:, :, 2] = np.roll(frame[:, :, 2], sh, axis=1)
    frame[:, :, 0] = np.roll(frame[:, :, 0], -sh, axis=1)
    for _ in range(4):
        gh = random.randint(6, 40)
        y0 = random.randint(0, frame.shape[0] - gh)
        frame[y0:y0 + gh] = np.roll(frame[y0:y0 + gh],
                                    random.randint(-80, 80), axis=1)
    if k > 0.5:
        frame[:] = (frame * 0.25).astype(np.uint8)


def make_binoculars(shape):
    """Mask for the view through binoculars: two overlapping circles with a
    soft, slightly dim rim, and a faint reticle."""
    H, Wd = shape[:2]
    r = int(min(H * 0.49, Wd / 3.6))
    m = np.zeros((H, Wd), np.float32)
    for cx in (Wd / 2 - r * 0.75, Wd / 2 + r * 0.75):
        cv2.circle(m, (int(cx), H // 2), r, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=r * 0.035)
    rim = cv2.GaussianBlur(m, (0, 0), sigmaX=r * 0.22)
    m *= 0.55 + 0.45 * np.clip(rim * 1.6 - 0.6, 0, 1)
    for d in range(-3, 4):
        if d:
            x, y, n = Wd // 2 + d * r // 8, H // 2 + d * r // 8, 4 + 3 * (d % 2 == 0)
            cv2.line(m, (x, H // 2 - n), (x, H // 2 + n), 0.35, 1)
            cv2.line(m, (Wd // 2 - n, y), (Wd // 2 + n, y), 0.35, 1)
    return m[..., None]


def at_or_before(items, when):
    """Latest (t, value) entry at or before `when`, or None."""
    found = None
    for item in items:
        if item[0] > when:
            break
        found = item
    return found


def draw_echo(frame, history, boxes, now):
    """Ghost copies of the person from moments ago, where they were then. The
    camera is fixed, so blending an old patch back in shows only what has
    changed since: them. Walking, it trails behind; standing, it is their
    reflection running late."""
    H, Wd = frame.shape[:2]
    for delay, strength in ECHOES:
        old = at_or_before(history, now - delay)
        box = at_or_before(boxes, now - delay)
        if old is None or box is None:
            continue
        x1, y1, x2, y2 = box[1]
        mx, my = (x2 - x1) * 0.25, (y2 - y1) * 0.1
        x1, x2 = int(max(0, x1 - mx)), int(min(Wd, x2 + mx))
        y1, y2 = int(max(0, y1 - my)), int(min(H, y2 + my))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        w, h = x2 - x1, y2 - y1
        mask = np.zeros((h, w), np.float32)
        cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.42), int(h * 0.46)),
                    0, 0, 360, strength, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, w * 0.08))
        mask = mask[..., None]
        ghost = old[1][y1:y2, x1:x2] * np.array([1.08, 1.0, 0.85], np.float32)
        roi = frame[y1:y2, x1:x2]
        roi[:] = np.clip(roi * (1 - mask) + ghost * mask, 0, 255).astype(
            np.uint8)


class Wraith:
    """A hovering figure: where it is, how solid, and its smoke trail."""

    def __init__(self):
        self.pos = None         # (x, ground_y) it hovers over
        self.alpha = 0.0
        self.trail = deque()    # (t, x, ground_y) of its own recent positions
        self.lean = 0.0
        self.eyes = 0.0
        self.scale = FOLLOWER_SIZE

    def seek(self, tx, ty, ease=0.15):
        if self.pos is None:
            self.pos = [tx, ty]
        else:
            self.pos[0] += (tx - self.pos[0]) * ease
            self.pos[1] += (ty - self.pos[1]) * ease

    def jump(self, tx, ty):
        self.pos = [tx, ty]
        self.trail.clear()

    def tick(self, now, visible):
        if visible and self.pos is not None:
            self.alpha = min(1.0, self.alpha + 0.08)
        else:
            self.alpha = max(0.0, self.alpha - 0.03)
        if self.pos is not None:
            self.trail.append((now, self.pos[0], self.pos[1]))
        while self.trail and now - self.trail[0][0] > TRAIL_SECS:
            self.trail.popleft()

    def draw(self, frame, person_h, t, now):
        if self.pos is None or self.alpha <= 0:
            return
        fh = max(MIN_FIGURE_H * frame.shape[0], person_h * self.scale)
        draw_figure(frame, self.pos[0], self.pos[1], fh, self.alpha, t,
                    trail=self.trail, now=now, lean=self.lean, eyes=self.eyes)


class Encounter:
    """Everything the window does to one person, from first sight until the
    wraith has finished with the spot where they stood."""

    def __init__(self, person, style):
        self.person = person
        self.style = style              # follower | echo | both
        self.wraith = Wraith()
        self.dir = None                 # eased heading, length <= 1
        self.gap = FOLLOWER_GAP         # person-heights the wraith keeps back
        self.h = person.h
        self.last = (person.x, person.y)
        self.drift = (0.0, 0.0)         # px/s the wraith carries on at
        self.boxes = deque()            # (t, box) for the echo
        self.present = True
        self.watching = False
        self.cycle = 0.0                # when the current approach began
        self.next_blink = 0.0
        self.hidden_until = 0.0
        self.lost_at = None
        self.moves = 0

    def done(self, now):
        return (self.lost_at is not None and self.wraith.alpha <= 0
                and now - self.lost_at > 0.5)

    def behind(self):
        """Ground spot `gap` person-heights behind them."""
        return (self.last[0] - self.dir[0] * self.gap * self.h,
                self.last[1] - self.dir[1] * self.gap * self.h)

    def update(self, now, present):
        """Advance one frame. Returns True when a blink cut should play."""
        p, w = self.person, self.wraith
        self.present = present
        if not present:
            if self.lost_at is None:
                self.lost_at = now
                w.eyes = 1.0
            gone = now - self.lost_at
            # carries on the way they went for a moment, looking after them
            if w.pos is not None and gone < STARE_SECS:
                w.pos[0] += self.drift[0] / 14
                w.pos[1] += self.drift[1] / 14
            w.tick(now, gone < STARE_SECS)
            return False

        self.lost_at = None
        self.h += (p.h - self.h) * 0.2
        self.last = (p.x, p.y)
        self.boxes.append((now, tuple(p.box)))
        while self.boxes and now - self.boxes[0][0] > HISTORY_SECS + 0.3:
            self.boxes.popleft()
        speed = np.hypot(p.vx, p.vy)
        moving = speed > STILL_SPEED * p.h
        if moving:
            d = [p.vx / speed, p.vy / speed]
            if self.dir is not None:
                # eased, and left un-normalised so a U-turn shrinks the offset
                # through zero: they walk through it and it is behind again
                d = [self.dir[0] * 0.9 + d[0] * 0.1,
                     self.dir[1] * 0.9 + d[1] * 0.1]
            self.dir = d
            self.drift = (p.vx * 0.5, p.vy * 0.5)
        elif self.dir is None:
            self.dir = [random.choice((-1.0, 1.0)), 0.0]    # found standing
        if not moving:
            self.drift = (0.0, 0.0)

        blink = False
        was = self.watching
        self.watching = p.still_for(now) > WATCH_AFTER
        if self.watching and not was:
            self.cycle = now
            self.next_blink = now + random.uniform(5, 8)
        if self.watching:
            # creeps to their shoulder, growing, eyes opening
            k = min(1.0, (now - self.cycle) / APPROACH_SECS)
            want = FOLLOWER_GAP * (1 - 0.7 * k)
            w.scale = FOLLOWER_SIZE + 0.3 * k
            w.eyes = min(1.0, (now - self.cycle) / 8)
            if now >= self.next_blink:
                blink = True
                self.next_blink = now + random.uniform(6, 11)
                self.jump(now, k)
        else:
            want = FOLLOWER_GAP
            w.scale = FOLLOWER_SIZE
            w.eyes = 0.0
        self.gap += (want - self.gap) * 0.05
        w.seek(*self.behind())
        w.lean = max(-1.0, min(1.0, self.dir[0]))
        w.tick(now, (self.watching or self.style != "echo")
               and now >= self.hidden_until)
        return blink

    def jump(self, now, k):
        """Under cover of a blink, be somewhere else. Once it has reached
        their shoulder, start again from a distance, so there is always
        something coming for as long as they stand there."""
        move = "reset" if k >= 1.0 else random.choice(
            ("closer", "swap", "vanish"))
        self.moves += 1
        if move == "closer":
            self.cycle -= APPROACH_SECS * 0.3
            k = min(1.0, (now - self.cycle) / APPROACH_SECS)
            self.gap = FOLLOWER_GAP * (1 - 0.7 * k)
        elif move == "swap":
            self.dir = [-self.dir[0], -self.dir[1]]
        elif move == "vanish":
            self.hidden_until = now + random.uniform(1.2, 2.5)
            self.wraith.alpha = 0.0
        else:
            self.cycle = now
            self.gap = 2.2
            self.dir = [-self.dir[0], -self.dir[1]]
            self.wraith.eyes = 0.0
        self.wraith.jump(*self.behind())


class Director:
    """One mirrored frame and the people in it in, one finished frame out."""

    def __init__(self, passes=None, phantoms_on=None):
        self.passes = passes or phantoms.PASSES
        self.do_phantoms = (phantoms.DO_PHANTOMS if phantoms_on is None
                            else phantoms_on)
        self.enc = {}               # person id -> Encounter
        self.history = deque()      # (t, graded frame) for the echo
        self.spots = deque(maxlen=200)  # (x, y, h) where real people have stood
        self.vig = None
        self.t0 = None
        self.zoom = 1.0
        self.focus = None           # eased (x, y) the zoom pushes in on
        self.glitch_until = 0.0
        self.last_style = None
        self.closeup = None         # close.CloseUp, loaded on first use
        self.eyes = None            # (t, eyes relative to their box)
        self.idle_since = None
        self.next_attract = None
        self.apparition = None      # (Wraith, height, until)
        self.plate = phantoms.Plate()
        self.recorder = phantoms.Recorder(self.passes)
        self.phantom = None         # phantoms.Phantom being played
        self.phantom_mode = None    # watch | idle: what the next one is for
        self.next_phantom = None
        self.bino_mask = None
        self.binos = False          # the view is through binoculars right now
        self.binos_who = None       # person id being watched for their whole pass
        self.binos_until = 0.0      # or: watched until this time

    def pick_style(self, person, H):
        if person.h < SMALL_PERSON * H or not STYLES:
            return "follower"
        pool = [s for s in STYLES if s != self.last_style] or STYLES
        self.last_style = random.choice(pool)
        return self.last_style

    def step(self, raw, people, now):
        H = raw.shape[0]
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0
        if self.vig is None or self.vig.shape[:2] != raw.shape[:2]:
            self.vig = make_vignette(raw.shape)
        scene = self.haunt(raw, people, now)
        graded = horror_grade(scene, self.vig, t) if DO_GRADE else scene.copy()
        self.history.append((now, graded))
        while now - self.history[0][0] > HISTORY_SECS:
            self.history.popleft()
        frame = graded.copy()

        here = {p.id for p in people}
        for p in people:
            if p.id not in self.enc:
                self.enc[p.id] = Encounter(p, self.pick_style(p, H))
                print(f"[director] person {p.id} h={p.h:.0f} -> "
                      f"{self.enc[p.id].style}", flush=True)
                if len(self.enc) == 1 and random.random() < BINOS:
                    self.binos_who = p.id
        for pid, e in list(self.enc.items()):
            was = e.watching
            if e.update(now, pid in here):
                self.glitch_until = now + BLINK_SECS
                if now > self.binos_until and random.random() < BINOS:
                    self.binos_until = now + random.uniform(5, 8)
            if e.watching and not was:
                print(f"[director] person {pid} stopped to watch", flush=True)
            if e.present and random.random() < 0.07:
                self.spots.append((e.last[0], e.last[1], e.h))
            if e.done(now):
                del self.enc[pid]

        for e in self.enc.values():
            if e.present and e.style != "follower":
                draw_echo(frame, self.history, e.boxes, now)
        near = next((e for e in self.enc.values()
                     if e.present and e.h > CLOSE_PERSON * H), None)
        if near is not None and DO_CLOSE:
            self.close_range(frame, raw, near, t, now)
        else:
            for e in self.enc.values():
                e.wraith.draw(frame, e.h, t, now)
        self.attract(frame, t, now)

        if DEBUG:
            for p in people:
                cv2.rectangle(frame, (int(p.box[0]), int(p.box[1])),
                              (int(p.box[2]), int(p.box[3])), (0, 255, 0), 1)
            cv2.putText(frame, " ".join(
                f"{pid}:{e.style}{'/watch' if e.watching else ''}"
                for pid, e in self.enc.items()) or "nobody", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # binoculars: something out there is watching them. Not while a
        # phantom needs the whole scene, or with someone right at the camera
        binos = (near is None and self.phantom is None
                 and (self.binos_who in self.enc or now < self.binos_until))
        if binos != self.binos:
            self.binos = binos
            self.glitch_until = now + BLINK_SECS
            if binos:
                self.zoom = ZOOM_BINOS      # a cut, not a push
                self.focus = None
                print("[director] binoculars", flush=True)
        frame = self.push_in(frame, t)
        if self.binos and self.zoom > 1.01:
            if self.bino_mask is None or self.bino_mask.shape[:2] != frame.shape[:2]:
                self.bino_mask = make_binoculars(frame.shape)
            frame = (frame * self.bino_mask).astype(np.uint8)
        if now < self.glitch_until:
            glitch(frame, (self.glitch_until - now) / BLINK_SECS)
        return frame

    def close_range(self, frame, raw, near, t, now):
        """Someone is big in frame: drain them toward shadow the longer they
        stay, stand the wraiths behind them, and hollow out their eyes."""
        if self.closeup is None:
            self.closeup = close.CloseUp()
            print("[director] close-range effects on", flush=True)
        p = near.person
        stay = min(1.0, p.still_for(now) / 15) if near.watching else 0.0
        masked = self.closeup.mask(raw, p.box)
        if masked is not None:
            region, mask = masked
            close.drain(frame, region, mask, 0.15 + 0.6 * stay)
            clean = frame.copy()
            for e in self.enc.values():
                e.wraith.draw(frame, e.h, t, now)
            close.put_behind(frame, clean, region, mask)

        # the face detector blinks in and out; keep the last eyes for a moment,
        # stored relative to their box so they travel with the head
        bx, by, bh = p.x, p.box[1], max(p.h, 1.0)
        found = self.closeup.eyes(raw, p.box)
        if found is not None:
            a, b, d = found
            self.eyes = (now, ((a[0] - bx) / bh, (a[1] - by) / bh),
                         ((b[0] - bx) / bh, (b[1] - by) / bh), d / bh)
        if self.eyes is not None and now - self.eyes[0] < 0.6:
            _, a, b, d = self.eyes
            close.hollow_eyes(
                frame, [(bx + a[0] * bh, by + a[1] * bh),
                        (bx + b[0] * bh, by + b[1] * bh), d * bh],
                0.55 + 0.45 * stay)

    def haunt(self, raw, people, now):
        """Record whoever is walking through, and now and then walk an
        earlier passer-by back through the scene: for someone who has stopped
        to watch, or along an empty street. Returns the frame to grade."""
        self.plate.update(raw, now, not people and not self.enc)
        if not self.do_phantoms:
            return raw
        self.recorder.observe(raw, self.plate.value, people, now)
        mode = ("watch" if any(e.watching for e in self.enc.values())
                else None if self.enc else "idle")
        if mode != self.phantom_mode:
            self.phantom_mode = mode
            self.next_phantom = mode and now + random.uniform(
                *(PHANTOM_WATCH_WAIT if mode == "watch"
                  else PHANTOM_IDLE_WAIT))
        if (self.phantom is None and self.next_phantom
                and now >= self.next_phantom):
            path = phantoms.choose(self.passes)
            self.next_phantom = now + random.uniform(
                *(PHANTOM_WATCH_GAP if mode == "watch" else PHANTOM_IDLE_GAP))
            if path:
                # give the zoom a moment to pull back so it is in shot
                self.phantom = phantoms.Phantom(
                    path, now + (1.5 if self.zoom > 1.2 else 0.0),
                    reverse=random.random() < PHANTOM_REVERSE)
                self.next_phantom += self.phantom.length
                print(f"[director] phantom {self.phantom.name}"
                      f"{' (backwards)' if self.phantom.reverse else ''}"
                      f" for {mode}", flush=True)
        if self.phantom is None:
            return raw
        scene = raw.copy()
        if not self.phantom.draw(scene, raw, self.plate.value, people, now):
            self.phantom = None
        return scene

    def attract(self, frame, t, now):
        """With the street empty, very occasionally something stands for a
        few seconds where a person once stood."""
        if self.enc or not DO_ATTRACT:
            self.idle_since = self.next_attract = self.apparition = None
            return
        if self.idle_since is None:
            self.idle_since = now
            self.next_attract = now + random.uniform(40, 100)
        if self.apparition is None and now >= self.next_attract and self.spots:
            x, y, h = random.choice(self.spots)
            w = Wraith()
            w.jump(x, y)
            w.eyes = 1.0
            self.apparition = (w, h, now + random.uniform(3, 5))
            self.next_attract = now + random.uniform(45, 120)
            print("[director] apparition", flush=True)
        if self.apparition is not None:
            w, h, until = self.apparition
            w.tick(now, now < until)
            w.draw(frame, h, t, now)
            if now >= until and w.alpha <= 0:
                self.apparition = None

    def push_in(self, frame, t=0.0):
        """Slow zoom toward whoever has been here longest, framed to keep
        their wraith in shot; tighter if they have stopped to watch."""
        H, Wd = frame.shape[:2]
        main = next((e for e in self.enc.values() if e.present), None) or next(
            (e for e in self.enc.values() if e.wraith.alpha > 0.3), None)
        want = 1.0
        if main is not None:
            want = ZOOM_BINOS if self.binos else (
                ZOOM_WATCH if main.watching else ZOOM)
            # never so tight that a wraith towering over them loses its head
            tall = main.h * max(1.0, main.wraith.scale + 0.25)
            want = max(1.0, min(want, 0.8 * H / max(tall, 1.0)))
            tx, ty = main.last[0], main.last[1] - tall * 0.5
            if main.wraith.pos is not None and main.wraith.alpha > 0:
                tx = (tx + main.wraith.pos[0]) / 2
            if self.focus is None:
                self.focus = [tx, ty]
            self.focus[0] += (tx - self.focus[0]) * 0.06
            self.focus[1] += (ty - self.focus[1]) * 0.06
        if self.phantom is not None:
            want = 1.0      # it could be anywhere in the scene: show all of it
        if self.binos and main is not None:
            self.zoom = min(self.zoom, want)
        self.zoom += (want - self.zoom) * 0.03
        if self.zoom < 1.01 or self.focus is None:
            if self.zoom < 1.01:
                self.focus = None
            return frame
        cw, ch = Wd / self.zoom, H / self.zoom
        fx, fy = self.focus
        if self.binos:
            # held by an unsteady hand
            fx += (np.sin(t * 1.1) * 7 + np.sin(t * 2.9) * 3) / self.zoom
            fy += (np.sin(t * 0.8 + 1) * 5 + np.sin(t * 3.4) * 2) / self.zoom
        x0 = int(min(max(fx - cw / 2, 0), Wd - cw))
        y0 = int(min(max(fy - ch / 2, 0), H - ch))
        return cv2.resize(frame[y0:y0 + int(ch), x0:x0 + int(cw)], (Wd, H),
                          interpolation=cv2.INTER_LINEAR)
