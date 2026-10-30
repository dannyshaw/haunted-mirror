"""
One haunted feed: a source (camera stream, webcam or video file) in, the
mirrored picture with people tracked and haunted out, as JPEGs for the web
server and optionally as H.264 pushed to an RTSP server.

Three threads per feed: capture (newest frame only), people (the detector,
which runs as fast as it can and is allowed to lag), and process (effects on
every captured frame, using whatever the detector last saw).
"""
import os
import queue
import subprocess
import threading
import time
import traceback

import cv2

from .effects import Director
from .vision import People

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def open_source(source):
    """cv2.VideoCapture for a webcam index ("0", 0, "webcam") or any URL /
    file path ffmpeg can read (rtsp://, http:// MJPEG, .mp4, ...)."""
    if source == "webcam":
        source = 0
    if isinstance(source, int) or str(source).isdigit():
        return cv2.VideoCapture(int(source)), True
    return cv2.VideoCapture(str(source), cv2.CAP_FFMPEG), False


class Feed:
    def __init__(self, name, source, width=960, mirror=True, exclude=None,
                 phantoms=None, passes=None, publish=None, jpeg_quality=80):
        self.name = name
        self.source = source
        self.width = width
        self.mirror = mirror
        self.exclude = exclude or []
        self.phantoms = phantoms
        self.passes = passes
        self.publish = publish
        self.quality = jpeg_quality
        self.people = None
        self.raw = None             # newest mirrored, resized source frame
        self.out = None             # newest finished frame (BGR)
        self.jpg = None             # ... and as JPEG
        self.fps = 0.0
        self._lock = threading.Lock()
        self._stop = False
        self._pub = None

    # ---- lifecycle --------------------------------------------------------
    def start(self):
        self.people = People(exclude=self.exclude)
        for target, args in ((self._capture, ()),
                             (self.people.run, (self.latest_raw,
                                                lambda: self._stop)),
                             (self._process, ())):
            threading.Thread(target=target, args=args, daemon=True,
                             name=f"{self.name}-{target.__name__}").start()
        if self.publish:
            self._pub = Publisher(self.publish, self.name)
        print(f"[{self.name}] source {redact(self.source)}"
              + (f", publishing {redact(self.publish)}" if self.publish
                 else ""), flush=True)
        return self

    def stop(self):
        self._stop = True
        if self._pub:
            self._pub.close()

    def latest_raw(self):
        with self._lock:
            return self.raw

    def latest(self):
        """(BGR frame, JPEG bytes) of the newest finished frame."""
        with self._lock:
            return self.out, self.jpg

    # ---- capture ----------------------------------------------------------
    def _capture(self):
        while not self._stop:
            cap, live = open_source(self.source)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if not cap.isOpened():
                print(f"[{self.name}] cannot open {redact(self.source)}, "
                      "retry in 3s", flush=True)
                time.sleep(3)
                continue
            print(f"[{self.name}] source open", flush=True)
            # a file plays at its own frame rate, looped; live sources as fast
            # as they come
            is_file = not live and os.path.exists(str(self.source))
            gap = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 15) if is_file else 0
            fails, due = 0, time.time()
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    if is_file:
                        break           # end of file: reopen to loop
                    fails += 1
                    if fails > 30:
                        print(f"[{self.name}] source stalled, reconnecting",
                              flush=True)
                        break
                    time.sleep(0.03)
                    continue
                fails = 0
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (self.width,
                                           int(h * self.width / w)))
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                with self._lock:
                    self.raw = frame
                if gap:
                    due += gap
                    time.sleep(max(0.0, due - time.time()))
            cap.release()

    # ---- effects ----------------------------------------------------------
    def _process(self):
        director = Director(passes=self.passes, phantoms_on=self.phantoms)
        last, n, t0 = None, 0, time.time()
        while not self._stop:
            raw = self.latest_raw()
            if raw is None or raw is last:
                time.sleep(0.005)
                continue
            last = raw
            # one bad effect frame must never kill the feed: show it raw
            try:
                frame = director.step(raw, self.people.current(), time.time())
            except Exception:
                traceback.print_exc()
                frame = raw
            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            with self._lock:
                self.out = frame
                if ok:
                    self.jpg = buf.tobytes()
            if self._pub:
                self._pub.put(frame)
            n += 1
            if time.time() - t0 >= 5:
                self.fps, n, t0 = n / (time.time() - t0), 0, time.time()

    # ---- debugging --------------------------------------------------------
    def tracks_jpg(self):
        """The raw frame with every confirmed track boxed, for tuning
        exclusion zones. Red = counted as a real person, green = ignored;
        exclusion zones in blue."""
        frame = self.latest_raw()
        if frame is None:
            return None
        frame = frame.copy()
        H, W = frame.shape[:2]
        for x1, y1, x2, y2 in self.exclude:
            cv2.rectangle(frame, (int(x1 * W), int(y1 * H)),
                          (int(x2 * W), int(y2 * H)), (255, 128, 0), 1)
        p = self.people
        for t in list(p.tracks):
            if not t.confirmed:
                continue
            x1, y1, x2, y2 = (int(v) for v in t.box)
            real = t.moved and not p.fixture(t) and not p.parked(t)
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 0, 255) if real else (0, 255, 0), 1)
            cv2.putText(frame, f"{t.id} feet {t.x / W:.2f},{t.y / H:.2f}",
                        (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 255, 255), 1)
        return cv2.imencode(".jpg", frame)[1].tobytes()


class Publisher:
    """Pushes finished frames to an RTSP server (mediamtx, go2rtc, ...) as
    H.264 through an ffmpeg subprocess. Frames are dropped, never queued up,
    if ffmpeg falls behind; ffmpeg is restarted if it dies."""

    def __init__(self, url, name, fps=15):
        self.url, self.name, self.fps = url, name, fps
        self.q = queue.Queue(maxsize=2)
        self.proc = None
        self.size = None
        self._closed = False
        threading.Thread(target=self._run, daemon=True,
                         name=f"{name}-publish").start()

    def put(self, frame):
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            pass

    def _spawn(self, w, h):
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-use_wallclock_as_timestamps", "1", "-i", "-",
               "-c:v", "libx264", "-preset", "veryfast", "-tune",
               "zerolatency", "-g", str(self.fps * 2), "-pix_fmt", "yuv420p",
               "-f", "rtsp", "-rtsp_transport", "tcp", self.url]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def _run(self):
        while not self._closed:
            frame = self.q.get()
            h, w = frame.shape[:2]
            if self.proc is None or self.proc.poll() is not None \
                    or self.size != (w, h):
                if self.proc is not None:
                    print(f"[{self.name}] publisher restarting", flush=True)
                    self.close_proc()
                    time.sleep(2)
                self.proc, self.size = self._spawn(w, h), (w, h)
            try:
                self.proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                self.close_proc()

    def close_proc(self):
        if self.proc:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            self.proc.kill()
            self.proc = None

    def close(self):
        self._closed = True
        self.close_proc()


def redact(url):
    """A source URL with any password hidden, for logs."""
    s = str(url)
    if "://" in s and "@" in s.split("://", 1)[1].split("/", 1)[0]:
        scheme, rest = s.split("://", 1)
        cred, host = rest.split("@", 1)
        return f"{scheme}://{cred.split(':', 1)[0]}:***@{host}"
    return s
