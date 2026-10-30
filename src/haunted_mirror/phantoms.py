"""
Phantoms for the haunted mirror: people who really walked past earlier,
played back into the live picture where they walked.

The camera is fixed, so a person cut out of an old frame drops straight back
into the same spot of a new one. Everyone who walks through alone is recorded
as a "take" (their patch of the picture plus a mask of what differed from the
empty scene), and takes are later replayed into the scene before it is graded,
so the phantom gets the same grain and colour as everything else.

Knobs via env (all optional):
  HM_PHANTOMS        1/0 record and replay passers-by (default 0)
  HM_PASSES          where takes are kept (default ~/.haunted-mirror/passes; takes in its
                     keep/ subfolder are never pruned)
  HM_PHANTOM_KEEP    how many recorded takes to keep (default 40)
  HM_PHANTOM_ALPHA   how solid a phantom is, 0-1 (default 0.92)
"""
import glob
import os
import random
import time
from collections import deque

import cv2
import numpy as np

# Off by default: it records strangers walking past. Turn on with care.
DO_PHANTOMS = os.environ.get("HM_PHANTOMS", "0") == "1"
PASSES = os.environ.get("HM_PASSES", os.path.expanduser(
    "~/.haunted-mirror/passes"))
KEEP = int(os.environ.get("HM_PHANTOM_KEEP", "40"))
ALPHA = float(os.environ.get("HM_PHANTOM_ALPHA", "0.92"))

DIFF = 26               # 0-255; a pixel this far from the empty scene is "them"
MIN_SECS = 2.5          # a take shorter than this is not worth replaying
MAX_SECS = 15.0         # and one longer than this is cut here
MIN_TRAVEL = 0.15       # of frame width they must cross: a pass, not a twitch
MIN_H = 0.1             # of frame height; smaller people are specks
FADE = 0.6              # seconds a phantom takes to appear and to go
MIN_AGE = 120.0         # seconds before a new take may be replayed


def crop_box(box, Wd, H):
    """The person's box with room for swinging arms, strides and shadow."""
    x1, y1, x2, y2 = box[:4]
    mx, my = (x2 - x1) * 0.5, (y2 - y1) * 0.15
    return (int(max(0, x1 - mx)), int(max(0, y1 - my)),
            int(min(Wd, x2 + mx)), int(min(H, y2 + my)))


def foreground(patch, back, core=None):
    """0/255 mask of where `patch` differs from the empty scene `back`. With
    `core` (x1, y1, x2, y2 in patch pixels) only blobs reaching into it are
    kept, which drops a car passing behind the person."""
    m = (cv2.absdiff(patch, back).max(axis=2) > DIFF).astype(np.uint8) * 255
    k = max(3, int(patch.shape[0] * 0.04) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    if core is not None:
        n, lab = cv2.connectedComponents(m)
        if n > 2:
            inside = lab[core[1]:core[3], core[0]:core[2]]
            m = np.where(np.isin(lab, np.unique(inside[inside > 0])),
                         255, 0).astype(np.uint8)
    # clothes the colour of what is behind them leave holes: fill them in
    outline, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(m, outline, -1, 255, -1)
    return m


class Plate:
    """The scene with nobody in it: the median of the last three frames
    sampled while the street was empty, so a passing car does not stick."""

    def __init__(self, every=3.0):
        self.every = every
        self.samples = deque(maxlen=3)
        self.value = None
        self._next = 0.0

    def update(self, raw, now, empty):
        if not empty or now < self._next:
            return
        self._next = now + self.every
        if self.samples and self.samples[0].shape != raw.shape:
            self.samples.clear()
        self.samples.append(raw)
        if len(self.samples) < 3:
            self.value = raw
        else:
            a, b, c = self.samples
            self.value = cv2.max(cv2.min(a, b), cv2.min(cv2.max(a, b), c))


class Recorder:
    """Records everyone who walks through alone, and saves the good takes."""

    def __init__(self, folder=PASSES, prefix=""):
        self.folder = folder
        self.prefix = prefix
        self.takes = {}         # person id -> [(t, x, y, feet, light, h, patch, mask)]
        self.spoiled = set()    # ids that shared the frame with someone else
        self.saved = []

    def observe(self, raw, plate, people, now):
        H, Wd = raw.shape[:2]
        here = {p.id for p in people}
        if len(people) > 1:
            self.spoiled |= here
        if plate is not None and plate.shape == raw.shape:
            for p in people:
                take = self.takes.setdefault(p.id, [])
                # a track coasting between sightings has a stale box
                if (p.id in self.spoiled or now - p.seen > 0.25
                        or (take and now - take[0][0] > MAX_SECS)):
                    continue
                x1, y1, x2, y2 = crop_box(p.box, Wd, H)
                if x2 - x1 < 8 or y2 - y1 < 16:
                    continue
                patch = raw[y1:y2, x1:x2].copy()
                back = plate[y1:y2, x1:x2]
                core = (int(p.box[0]) - x1, int(p.box[1]) - y1,
                        int(p.box[2]) - x1, int(p.box[3]) - y1)
                mask = cv2.GaussianBlur(foreground(patch, back, core),
                                        (0, 0), sigmaX=1.0)
                take.append((now, x1, y1, p.box[3], float(back.mean()), p.h,
                             patch, mask))
        for pid in list(self.takes):
            if pid not in here:
                take = self.takes.pop(pid)
                if pid not in self.spoiled:
                    self.save(take, Wd, H)
        self.spoiled &= here

    def save(self, take, Wd, H):
        if len(take) < 20 or take[-1][0] - take[0][0] < MIN_SECS:
            return None
        xs = [f[1] + f[6].shape[1] / 2 for f in take]
        cover = float(np.median([(f[7] > 127).mean() for f in take]))
        if (max(xs) - min(xs) < MIN_TRAVEL * Wd
                or np.median([f[5] for f in take]) < MIN_H * H
                or not 0.04 < cover < 0.6):
            return None
        arrays = {"meta": np.array([f[:6] for f in take], np.float32),
                  "size": np.array([Wd, H])}
        for i, f in enumerate(take):
            arrays[f"j{i}"] = cv2.imencode(
                ".jpg", f[6], [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
            arrays[f"m{i}"] = cv2.imencode(".png", f[7])[1]
        os.makedirs(self.folder, exist_ok=True)
        path = os.path.join(
            self.folder, f"{self.prefix}{int(time.time() * 1000)}.npz")
        np.savez(path + ".tmp.npz", **arrays)
        os.replace(path + ".tmp.npz", path)
        self.saved.append(path)
        print(f"[phantoms] kept a take: {len(take)} frames, "
              f"{take[-1][0] - take[0][0]:.1f}s -> {os.path.basename(path)}",
              flush=True)
        for old in sorted(glob.glob(os.path.join(self.folder, "*.npz")),
                          key=os.path.getmtime)[:-KEEP]:
            os.remove(old)
        return path


def choose(folder=PASSES):
    """A take to replay, or None. One just recorded has to wait: its owner
    may still be standing there."""
    paths = [p for p in glob.glob(os.path.join(folder, "*.npz"))
             + glob.glob(os.path.join(folder, "keep", "*.npz"))
             if not p.endswith(".tmp.npz")
             and time.time() - os.path.getmtime(p) > MIN_AGE]
    return random.choice(paths) if paths else None


class Phantom:
    """One take being played back into the scene."""

    def __init__(self, path, start, reverse=False):
        z = np.load(path)
        self.name = os.path.basename(path)
        self.meta = z["meta"]
        self.size = tuple(int(v) for v in z["size"])
        self.frames = [
            (cv2.imdecode(z[f"j{i}"], cv2.IMREAD_COLOR),
             cv2.imdecode(z[f"m{i}"], cv2.IMREAD_GRAYSCALE))
            for i in range(len(self.meta))]
        self.t = self.meta[:, 0] - self.meta[0, 0]
        self.length = float(self.t[-1])
        self.start = start
        self.reverse = reverse      # walking backwards

    def draw(self, frame, raw, plate, people, now):
        """Composite onto `frame` (a copy of `raw`). False once it is over."""
        H, Wd = raw.shape[:2]
        e = now - self.start
        if e < 0:
            return True
        if e > self.length or self.size != (Wd, H):
            return False
        at = self.length - e if self.reverse else e
        i = max(0, int(np.searchsorted(self.t, at, side="right")) - 1)
        _, x1, y1, feet, light, _ = self.meta[i]
        x1, y1 = int(x1), int(y1)
        patch, mask = self.frames[i]
        h, w = mask.shape
        live = raw[y1:y1 + h, x1:x1 + w]
        m = mask.astype(np.float32) / 255 * (
            ALPHA * min(1.0, e / FADE, (self.length - e) / FADE))
        back = plate[y1:y1 + h, x1:x1 + w] if (
            plate is not None and plate.shape == raw.shape) else None
        # lit as the scene is now, not as it was when they walked through
        now_light = float((live if back is None else back).mean())
        ghost = patch.astype(np.float32) * float(
            np.clip(now_light / max(light, 1.0), 0.3, 1.5))
        if (live.max(axis=2).astype(np.int16) - live.min(axis=2)).mean() < 6:
            # the camera is in night (infrared) mode: no colour in the scene
            ghost[:] = ghost.mean(axis=2, keepdims=True)
        if back is not None:
            # anyone real standing nearer the camera stays in front of it
            for p in people:
                if p.y <= feet + 3:
                    continue
                px1, py1, px2, py2 = crop_box(p.box, Wd, H)
                ax, ay = max(px1, x1) - x1, max(py1, y1) - y1
                bx, by = min(px2, x1 + w) - x1, min(py2, y1 + h) - y1
                if bx - ax < 2 or by - ay < 2:
                    continue
                fg = foreground(live[ay:by, ax:bx], back[ay:by, ax:bx])
                m[ay:by, ax:bx] *= 1 - cv2.GaussianBlur(
                    fg, (0, 0), sigmaX=1.0).astype(np.float32) / 255
        m = m[..., None]
        roi = frame[y1:y1 + h, x1:x1 + w]
        roi[:] = np.clip(roi * (1 - m) + ghost * m, 0, 255).astype(np.uint8)
        return True
