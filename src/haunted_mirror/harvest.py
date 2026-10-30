"""
Turn recorded clips from the same camera into phantom takes.

  haunted-mirror harvest [--keep] clip.mp4 [clip.mp4 ...]

Each clip is run through the person tracker, and anyone who walks through
alone is saved to the passes folder (see phantoms.py). --keep saves into its
keep/ subfolder, which is never pruned: use it for staged walk-pasts.
The clip must be framed exactly as the live camera is.
"""
import os

import cv2
import numpy as np

from . import phantoms
from . import vision

W = int(os.environ.get("HM_WIDTH", "960"))
MIRROR = os.environ.get("HM_MIRROR", "1") == "1"


def frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            return
        f = cv2.resize(f, (W, int(f.shape[0] * W / f.shape[1])))
        yield n / fps, cv2.flip(f, 1) if MIRROR else f
        n += 1


def harvest(path, folder):
    # the empty scene: what each pixel shows for most of the clip
    every = max(1, sum(1 for _ in frames(path)) // 9)
    plate = np.median(np.stack(
        [f for i, (_, f) in enumerate(frames(path)) if i % every == 0]),
        axis=0).astype(np.uint8)
    people = vision.People()
    name = os.path.splitext(os.path.basename(path))[0]
    rec = phantoms.Recorder(folder, prefix=name[-6:] + "-")
    now = 0.0
    for now, f in frames(path):
        people.step(f, now)
        rec.observe(f, plate, people.current(), now)
    rec.observe(f, plate, [], now + 1)
    print(f"{name}: {len(rec.saved)} take(s)")


def main(args):
    """harvest [--feed NAME] [--keep] clip.mp4 ...: takes go to the passes
    folder of feed NAME (default: webcam), as the live feed would save them."""
    args = list(args)
    feed = "webcam"
    if "--feed" in args:
        i = args.index("--feed")
        feed = args[i + 1]
        del args[i:i + 2]
    keep = "--keep" in args
    folder = os.path.join(phantoms.PASSES, feed)
    if keep:
        folder = os.path.join(folder, "keep")
    for clip in args:
        if clip != "--keep":
            harvest(clip, folder)
