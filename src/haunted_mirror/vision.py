"""
Person detection + tracking for the haunted mirror.

A small SSD person detector (the same OpenVINO model Frigate runs) is pointed
at square regions of the frame: a slow sweep of tiles to find people, plus a
tight region around everyone already being tracked. Detecting people rather
than motion means someone who stops to watch stays tracked, and a swaying
branch or decoration never is.
"""
import os
import threading
import time

import cv2
import numpy as np
import openvino as ov

MODEL = os.environ.get("HM_MODEL", os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "ssdlite_mobilenet_v2.xml"))
PERSON = 1              # label index in the COCO-91 map the model was trained on
MIN_SCORE = float(os.environ.get("HM_MIN_SCORE", "0.5"))
CONFIRM_HITS = 2        # detections before a new track counts as a person
# Undetected this long and the track is dropped. Generous so a track coasts
# through the detector blinking out (common in low light) and is re-matched as
# the same person, instead of dying and re-spawning a fresh wraith each time.
LOST_SECS = float(os.environ.get("HM_LOST_SECS", "3.0"))
COAST_DECAY = 0.94      # per-step easing of a coasting track's speed to zero
STILL_SPEED = 0.25      # person-heights per second; slower than this is "still"
MAX_PEOPLE = 4
REFIND_SECS = 15.0      # a lost person re-found near the same spot within this
# A small figure that is detected in one spot for this many seconds (within
# the last quarter hour or so) is a fixture: a decoration, a statue, a sign.
# Only small ones, so that someone standing at the window is never written off.
FIXTURE_SECS = float(os.environ.get("HM_FIXTURE_SECS", "40"))
FIXTURE_H = 0.15        # of frame height
FIXTURE_CELL = 24       # px
FIXTURE_FADE = 900.0    # seconds for a spot's tally to fade once it is empty


def _parse_zones(spec):
    """`x1,y1,x2,y2;x1,y1,x2,y2` of normalised (0-1) boxes -> list of tuples.
    Used to blank out spots where a static decoration stands (skeletons,
    scarecrows) that the detector keeps calling a person."""
    zones = []
    for part in (spec or "").split(";"):
        nums = [float(v) for v in part.replace(" ", "").split(",") if v]
        if len(nums) == 4:
            x1, y1, x2, y2 = nums
            zones.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return zones


# Spots where a static decoration stands (skeletons, scarecrows). A track that
# sits *still* with its feet inside one of these normalised boxes is a prop and
# is suppressed; a real person who walks through or behind that spot keeps
# moving, so they are never dropped. Set via HM_EXCLUDE. This is deliberately
# gated on stillness (unlike the old hard filter, which blanked the ground and
# so lost anyone walking past the props).
EXCLUDE = _parse_zones(os.environ.get("HM_EXCLUDE", ""))


class Detector:
    def __init__(self, threads=2):
        core = ov.Core()
        self.net = core.compile_model(
            core.read_model(MODEL), "CPU",
            {"INFERENCE_NUM_THREADS": threads})
        inp = self.net.input(0)
        shape = list(inp.shape)
        self.nhwc = shape[3] == 3
        self.size = shape[1] if self.nhwc else shape[2]
        self.dtype = np.uint8 if "u8" in str(inp.element_type) else np.float32
        self.out = self.net.output(0)

    def detect(self, frame, region):
        """People inside the square `region` (x0, y0, side) of `frame`, as
        (x1, y1, x2, y2, score) in frame pixels."""
        x0, y0, s = region
        H, Wd = frame.shape[:2]
        blob = cv2.resize(frame[y0:y0 + s, x0:x0 + s], (self.size, self.size))
        if not self.nhwc:
            blob = blob.transpose(2, 0, 1)
        res = self.net(blob[None].astype(self.dtype))[self.out]
        found = []
        for _, label, score, xa, ya, xb, yb in res.reshape(-1, 7):
            if int(label) != PERSON or score < MIN_SCORE:
                continue
            # cut off by the edge of the region (not of the frame): this is
            # half a person, and a better view comes from another region
            e = 0.02
            if ((xa < e and x0 > 0) or (xb > 1 - e and x0 + s < Wd)
                    or (ya < e and y0 > 0) or (yb > 1 - e and y0 + s < H)):
                continue
            found.append((x0 + xa * s, y0 + ya * s,
                          x0 + xb * s, y0 + yb * s, float(score)))
        return found


def sweep_regions(Wd, H):
    """Overlapping squares covering the frame: small ones for people out on
    the street, frame-height ones for people close to the camera."""
    regions = []
    s = min(H, max(160, int(H * 0.6)))
    nx = max(2, int(np.ceil((Wd - s) / (s * 0.66))) + 1)
    for y0 in sorted({max(0, H - s - int(s * 0.5)), H - s}):
        for i in range(nx):
            regions.append((int(i * (Wd - s) / (nx - 1)), y0, s))
    nb = max(2, int(np.ceil(Wd / H)))
    for i in range(nb):
        regions.append((int(i * (Wd - H) / (nb - 1)), 0, H))
    return regions


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


class Person:
    """One tracked person. `x, y` is where their feet are; `h` their height."""
    _next = 1

    def __init__(self, box, now):
        self.id = Person._next
        Person._next += 1
        self.box = list(box[:4])
        self.hits = 1
        self.born = now
        self.seen = now
        self.vx = self.vy = 0.0         # px/s, eased
        self.still_since = now          # when they last stopped moving
        self.origin = (self.x, self.y)
        self.moved = False              # has walked somewhere since first seen

    @property
    def x(self):
        return (self.box[0] + self.box[2]) / 2

    @property
    def y(self):
        return self.box[3]

    @property
    def h(self):
        return self.box[3] - self.box[1]

    @property
    def confirmed(self):
        return self.hits >= CONFIRM_HITS

    def still_for(self, now):
        return now - self.still_since

    def update(self, box, now):
        dt = max(now - self.seen, 1e-3)
        ox, oy = self.x, self.y
        a = 0.5
        for i in range(4):
            self.box[i] += (box[i] - self.box[i]) * a
        if dt < 0.6:
            # normal cadence: re-estimate the eased velocity. After a longer
            # gap we were coasting on the last heading, which is a better guess
            # than a near-zero velocity derived over the whole gap, so keep it.
            k = min(1.0, dt / 0.5)
            self.vx += ((self.x - ox) / dt - self.vx) * k
            self.vy += ((self.y - oy) / dt - self.vy) * k
        self.hits += 1
        self.seen = now
        if np.hypot(self.vx, self.vy) > STILL_SPEED * self.h:
            self.still_since = now
        else:
            # While still, let the origin drift slowly toward the current
            # position so jitter or sway in one spot never accumulates into
            # "has moved": a decoration that twitches in the wind, or whose
            # box the detector nudges a few px each frame, never reads as a
            # walker. A person actually walking outruns this slow trail and
            # still latches `moved` within a stride.
            k = min(1.0, dt / 4.0)
            self.origin = (self.origin[0] + (self.x - self.origin[0]) * k,
                           self.origin[1] + (self.y - self.origin[1]) * k)
        if np.hypot(self.x - self.origin[0],
                    self.y - self.origin[1]) > max(30.0, 0.5 * self.h):
            self.moved = True

    def coast(self, dt):
        """Not detected this frame: drift along the last heading (speed easing
        toward zero) so the track stays where the person probably is. Keeps the
        wraith trailing them and, crucially, keeps the predicted box near where
        they reappear, so they are re-matched as the same person rather than
        dropped and re-spawned when the detector blinks out in the dark."""
        self.box[0] += self.vx * dt
        self.box[2] += self.vx * dt
        self.box[1] += self.vy * dt
        self.box[3] += self.vy * dt
        self.vx *= COAST_DECAY
        self.vy *= COAST_DECAY

    def region(self, Wd, H, now):
        """Square to search for them next: around where they should be by now."""
        dt = min(now - self.seen, 0.5)
        cx = self.x + self.vx * dt
        cy = self.y - self.h / 2 + self.vy * dt
        s = int(min(H, max(160, self.h * 2.2)))
        return (int(min(max(cx - s / 2, 0), Wd - s)),
                int(min(max(cy - s / 2, 0), H - s)), s)


class People:
    """Tracks people across frames. `step` does one round of detection;
    `run` loops it on the newest frame in a background thread."""

    def __init__(self, threads=2, exclude=None):
        self.detector = Detector(threads)
        self.exclude = EXCLUDE if exclude is None else exclude
        self.tracks = []
        self.lock = threading.Lock()
        self._regions = []
        self._tile = 0
        self._lost = []         # (t, x, y) where real people dropped out
        self._dwell = {}        # grid cell -> [seconds a small figure stood there, when]
        self._stepped = None
        self._H = 0
        self._Wd = 0
        self.rate = 0.0         # detection rounds per second, for the log

    def step(self, frame, now, tiles=2):
        H, Wd = frame.shape[:2]
        self._H = H
        self._Wd = Wd
        dt = min(now - self._stepped, 0.5) if self._stepped else 0.0
        self._stepped = now
        if not self._regions or self._regions[0][2] > H:
            self._regions = sweep_regions(Wd, H)
        regions = [t.region(Wd, H, now) for t in self.tracks]
        for _ in range(tiles):
            regions.append(self._regions[self._tile % len(self._regions)])
            self._tile += 1
        boxes = []
        for r in regions:
            boxes += self.detector.detect(frame, r)
        # the same person seen through two regions: keep the best sighting
        boxes.sort(key=lambda b: -b[4])
        kept = []
        for b in boxes:
            if all(iou(b, k) < 0.4 for k in kept):
                kept.append(b)

        tracks = list(self.tracks)
        for t in sorted(tracks, key=lambda t: -t.hits):
            best = None
            for b in kept:
                cx, cy = (b[0] + b[2]) / 2, b[3]
                d = np.hypot(cx - t.x, cy - t.y)
                if (d <= max(40.0, 0.8 * t.h)
                        and 0.5 < (b[3] - b[1]) / max(t.h, 1) < 2.0
                        and (best is None or d < best[0])):
                    best = (d, b)
            if best:
                kept.remove(best[1])
                t.update(best[1], now)
            elif dt:
                t.coast(dt)     # lost this frame: drift on the last heading
        for b in kept:
            # a second box on someone already tracked is not a second person
            if (len(tracks) < MAX_PEOPLE * 2
                    and all(iou(b, t.box) < 0.2 for t in tracks)):
                new = Person(b, now)
                # re-found where a real person just dropped out (they stood
                # still behind something): still the same real person
                new.moved = any(
                    now - t0 < REFIND_SECS
                    and np.hypot(new.x - x, new.y - y) < 1.5 * new.h
                    for t0, x, y in self._lost)
                tracks.append(new)
        for t in tracks:
            if t.confirmed and t.seen == now and t.h < FIXTURE_H * H:
                cell = self._dwell.setdefault(
                    (int(t.x // FIXTURE_CELL), int(t.y // FIXTURE_CELL)),
                    [0.0, now])
                cell[0] = cell[0] * np.exp((cell[1] - now) / FIXTURE_FADE) + dt
                cell[1] = now
        for t in tracks:
            if (now - t.seen >= LOST_SECS and t.confirmed and t.moved
                    and not self.fixture(t)):
                self._lost.append((now, t.x, t.y))
        del self._lost[:-8]
        tracks = [t for t in tracks if now - t.seen < LOST_SECS
                  and (t.confirmed or now - t.seen < 0.6)]
        with self.lock:
            self.tracks = tracks

    def fixture(self, t):
        """Is this small figure standing where one always stands?"""
        if t.h >= FIXTURE_H * self._H:
            return False
        cx, cy = int(t.x // FIXTURE_CELL), int(t.y // FIXTURE_CELL)
        return sum(self._dwell.get((cx + i, cy + j), (0.0,))[0]
                   for i in (-1, 0, 1) for j in (-1, 0, 1)) > FIXTURE_SECS

    def parked(self, t):
        """A track anchored inside a decoration zone (HM_EXCLUDE): a prop, not
        a person. Gated on the track's *origin* (where it was first seen and
        has settled), not on momentary stillness — a fixed decoration is born
        in its spot and stays there, so a waving arm, a changed pose or a
        fluttering flag beside it never lets it count. A real person is born on
        the path (origin outside the zone), so walking through or behind a prop
        never suppresses them. Both origin and current feet must be in the
        same zone, so the instant someone anchored in a zone walks out of it
        they are tracked again."""
        if not self.exclude or self._H <= 0 or self._Wd <= 0:
            return False
        ox, oy = t.origin[0] / self._Wd, t.origin[1] / self._H
        fx, fy = t.x / self._Wd, t.y / self._H
        return any(x1 <= ox <= x2 and y1 <= oy <= y2
                   and x1 <= fx <= x2 and y1 <= fy <= y2
                   for x1, y1, x2, y2 in self.exclude)

    def current(self):
        """Real people, longest-tracked first. Something person-shaped that
        has never moved (a statue, a prop, a false detection) is not one, nor
        is a small figure standing where one has stood for ages, nor anything
        parked still inside a decoration zone."""
        with self.lock:
            people = [t for t in self.tracks if t.confirmed and t.moved
                      and not self.fixture(t) and not self.parked(t)]
        return sorted(people, key=lambda t: t.born)[:MAX_PEOPLE]

    def run(self, get_frame, stopped):
        last, n, t0 = None, 0, time.time()
        while not stopped():
            frame = get_frame()
            if frame is None or frame is last:
                time.sleep(0.005)
                continue
            last = frame
            self.step(frame, time.time())
            n += 1
            if not self.current():
                # nobody about: sweeping at half pace still finds a walker
                # within a second, for half the CPU
                time.sleep(0.08)
            if time.time() - t0 > 10:
                self.rate, n, t0 = n / (time.time() - t0), 0, time.time()
