"""Fullscreen window for the turnkey webcam mode. Keys: f toggles
fullscreen, q or Esc quits."""
import time

import cv2

TITLE = "Haunted Mirror"


def show(feed):
    try:
        cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
    except cv2.error:
        raise SystemExit(
            "This OpenCV build has no GUI (opencv-python-headless?). Install "
            "opencv-python, or run --headless and open the web page instead.")
    full = True
    cv2.setWindowProperty(TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    last = None
    while True:
        frame, _ = feed.latest()
        if frame is not None and frame is not last:
            cv2.imshow(TITLE, frame)
            last = frame
        key = cv2.waitKey(10) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("f"):
            full = not full
            cv2.setWindowProperty(
                TITLE, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
        try:
            if cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break           # window closed with the mouse
        except cv2.error:
            break
        if frame is None:
            time.sleep(0.02)
    cv2.destroyAllWindows()
