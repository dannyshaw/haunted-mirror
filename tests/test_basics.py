import time

import cv2
import numpy as np
import pytest

from haunted_mirror.cli import parse_source, load_config
from haunted_mirror.feed import Feed, redact
from haunted_mirror.vision import _parse_zones


def test_parse_source_names():
    assert parse_source("rtsp://cam/stream", 0) == ("feed1", "rtsp://cam/stream")
    assert parse_source("front=rtsp://u:p@cam/s", 0) == ("front", "rtsp://u:p@cam/s")
    assert parse_source("0", 0) == ("webcam", "0")
    # an = inside the URL is not a name
    assert parse_source("http://cam/x?a=b", 1) == ("feed2", "http://cam/x?a=b")


def test_zones():
    assert _parse_zones("0.5,0.5,0.1,0.1; 0,0,1,1") == [
        (0.1, 0.1, 0.5, 0.5), (0.0, 0.0, 1.0, 1.0)]
    assert _parse_zones("") == []


def test_redact():
    assert redact("rtsp://admin:hunter2@10.0.0.5:554/s") == \
        "rtsp://admin:***@10.0.0.5:554/s"
    assert redact("rtsp://10.0.0.5/s") == "rtsp://10.0.0.5/s"
    assert redact(0) == "0"


def test_config(tmp_path):
    p = tmp_path / "feeds.yaml"
    p.write_text("width: 640\nfeeds:\n  front:\n    source: a.mp4\n"
                 "    exclude: [[0.1, 0.2, 0.3, 0.4]]\n  back: b.mp4\n")
    cfg, feeds = load_config(p)
    assert cfg["width"] == 640
    assert [f["name"] for f in feeds] == ["front", "back"]
    assert feeds[0]["exclude"] == [(0.1, 0.2, 0.3, 0.4)]
    assert feeds[1]["source"] == "b.mp4"


@pytest.fixture
def clip(tmp_path):
    """Three seconds of a grey box drifting across a dark scene."""
    path = str(tmp_path / "clip.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                          (640, 360))
    for i in range(45):
        f = np.full((360, 640, 3), 30, np.uint8)
        cv2.rectangle(f, (50 + i * 10, 100), (110 + i * 10, 300),
                      (150, 150, 150), -1)
        out.write(f)
    out.release()
    return path


def test_feed_end_to_end(clip, tmp_path):
    feed = Feed("t", clip, width=480, passes=str(tmp_path / "p")).start()
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            frame, jpg = feed.latest()
            if jpg is not None:
                break
            time.sleep(0.1)
        assert jpg is not None and jpg[:2] == b"\xff\xd8"
        assert frame.shape[1] == 480
        assert feed.tracks_jpg()[:2] == b"\xff\xd8"
    finally:
        feed.stop()
