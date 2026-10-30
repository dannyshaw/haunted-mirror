"""
Close-camera extras for the haunted mirror.

When someone is big in frame (camera at eye level, a few metres away) there
is enough of them to work on directly: a silhouette mask, so the wraith can
stand *behind* them and their own image can be drained toward shadow, and
face landmarks, so their eyes can be hollowed out. EXPERIMENTAL: tuned on
footage cropped to simulate a closer camera; best with a webcam at eye level.
"""
import os

import cv2
import numpy as np

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SEG_MODEL = os.path.join(_DIR, "human_segmentation_pphumanseg_2023mar.onnx")
FACE_MODEL = os.path.join(_DIR, "face_detection_yunet_2023mar.onnx")
SEG_SIZE = 192


class CloseUp:
    def __init__(self):
        self.seg = cv2.dnn.readNet(SEG_MODEL)
        self.face = cv2.FaceDetectorYN.create(FACE_MODEL, "", (320, 320),
                                              0.6, 0.3, 20)

    def region(self, frame, box, pad=0.15):
        H, Wd = frame.shape[:2]
        x1, y1, x2, y2 = box
        mx, my = (x2 - x1) * pad, (y2 - y1) * pad
        return (int(max(0, x1 - mx)), int(max(0, y1 - my)),
                int(min(Wd, x2 + mx)), int(min(H, y2 + my)))

    def mask(self, frame, box):
        """(region, mask): 0..1 float silhouette of the person in `box`,
        the size of the padded region (x1, y1, x2, y2) it covers."""
        x1, y1, x2, y2 = r = self.region(frame, box)
        # a coasting track can drift off the frame edge: nothing to mask
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = frame[y1:y2, x1:x2]
        blob = cv2.dnn.blobFromImage(
            crop, 1 / 127.5, (SEG_SIZE, SEG_SIZE), (127.5, 127.5, 127.5),
            swapRB=True)
        self.seg.setInput(blob)
        out = self.seg.forward()[0]             # (2, 192, 192): bg, person
        if out.min() >= 0 and out.max() <= 1.01:    # already probabilities
            prob = out[1] / (out[0] + out[1] + 1e-6)
        else:
            prob = 1 / (1 + np.exp(out[0] - out[1]))
        # sharpen: anything the model is unsure of is not the person
        prob = np.clip((prob - 0.5) * 4 + 0.5, 0, 1)
        m = cv2.resize(prob.astype(np.float32), (x2 - x1, y2 - y1))
        return r, cv2.GaussianBlur(m, (0, 0), sigmaX=1.5)

    def eyes(self, frame, box):
        """[(x, y), (x, y), spacing] for the eyes of the face in the top of
        `box`, in frame pixels, or None."""
        x1, y1, x2, y2 = self.region(frame, box, pad=0.1)
        y2 = y1 + int((y2 - y1) * 0.45)         # heads are at the top
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 20 or crop.shape[1] < 20:
            return None
        # small faces: give the detector a bigger picture of them
        k = max(1.0, 240 / crop.shape[0])
        if k > 1:
            crop = cv2.resize(crop, None, fx=k, fy=k,
                              interpolation=cv2.INTER_CUBIC)
        self.face.setInputSize((crop.shape[1], crop.shape[0]))
        _, faces = self.face.detect(crop)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda f: f[14])
        a = (x1 + f[4] / k, y1 + f[5] / k)
        b = (x1 + f[6] / k, y1 + f[7] / k)
        return [a, b, float(np.hypot(a[0] - b[0], a[1] - b[1]))]


def put_behind(frame, clean, region, mask):
    """Paste the person from `clean` (the frame before anything was drawn on
    it) back over `frame`, so whatever was drawn ends up behind them."""
    x1, y1, x2, y2 = region
    m = mask[..., None]
    roi = frame[y1:y2, x1:x2]
    roi[:] = (roi * (1 - m) + clean[y1:y2, x1:x2] * m).astype(np.uint8)


def drain(frame, region, mask, k):
    """Pull the person toward a cold, dark silhouette of themselves. k 0..1."""
    x1, y1, x2, y2 = region
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    gray = cv2.cvtColor(frame[y1:y2, x1:x2],
                        cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
    dead = gray * np.array([0.75, 0.62, 0.5], np.float32) * (1 - 0.55 * k)
    m = mask[..., None] * k
    frame[y1:y2, x1:x2] = np.clip(roi * (1 - m) + dead * m, 0,
                                  255).astype(np.uint8)


def hollow_eyes(frame, eyes, k):
    """Black out the eye sockets. k 0..1."""
    a, b, d = eyes
    r = max(2, int(d * 0.33))
    pad = r * 3
    for ex, ey in (a, b):
        x1, y1 = int(ex) - pad, int(ey) - pad
        x2, y2 = int(ex) + pad, int(ey) + pad
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            continue
        m = np.zeros((y2 - y1, x2 - x1), np.float32)
        cv2.ellipse(m, (pad, pad + r // 4), (r, int(r * 1.15)), 0, 0, 360,
                    1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=max(1.0, r * 0.35))[..., None]
        roi = frame[y1:y2, x1:x2]
        roi[:] = (roi * (1 - m * k)).astype(np.uint8)
