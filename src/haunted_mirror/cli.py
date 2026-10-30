"""
haunted-mirror: a live camera feed that shows people their own reflection,
with something standing behind them that isn't there.

  haunted-mirror                       your webcam, fullscreen window
  haunted-mirror rtsp://cam/stream     one camera, served on :8080
  haunted-mirror front=rtsp://a back=rtsp://b --publish rtsp://mediamtx:8554
  haunted-mirror --config feeds.yaml
  haunted-mirror harvest [--keep] clip.mp4 ...   (see phantoms)
"""
import argparse
import os
import signal
import sys
import threading

from . import __version__


def parse_zones(spec):
    from .vision import _parse_zones
    return _parse_zones(spec)


def parse_source(arg, i):
    """`name=url` or just `url` (named feed1, feed2, ... or webcam)."""
    if "=" in arg and "://" not in arg.split("=", 1)[0]:
        name, src = arg.split("=", 1)
    else:
        name, src = None, arg
    if name is None:
        name = "webcam" if src in ("webcam",) or src.isdigit() else \
            f"feed{i + 1}"
    return name, src


def load_config(path):
    import yaml
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    feeds = []
    for name, f in (cfg.get("feeds") or {}).items():
        if isinstance(f, str):
            f = {"source": f}
        ex = f.get("exclude") or []
        if isinstance(ex, str):
            ex = parse_zones(ex)
        feeds.append(dict(name=str(name), source=f["source"],
                          mirror=f.get("mirror"), exclude=[tuple(z) for z in ex],
                          phantoms=f.get("phantoms"), publish=f.get("publish")))
    return cfg, feeds


def build_parser():
    p = argparse.ArgumentParser(
        prog="haunted-mirror",
        description="Haunt a live camera feed. With no sources, uses your "
                    "webcam and opens a fullscreen window.",
        epilog="Effect knobs are environment variables (HM_GRADE, HM_GRAIN, "
               "HM_ZOOM, ...); see the README.")
    p.add_argument("sources", nargs="*", metavar="SOURCE",
                   help="[name=]URL, file or webcam index (0, 1, ...)")
    p.add_argument("-c", "--config", help="YAML file of feeds (see README)")
    p.add_argument("-p", "--port", type=int,
                   default=int(os.environ.get("HM_PORT", "8080")),
                   help="web server port (default 8080, 0 = no server)")
    p.add_argument("--host", default=os.environ.get("HM_HOST", "0.0.0.0"))
    p.add_argument("-w", "--width", type=int,
                   default=int(os.environ.get("HM_WIDTH", "960")),
                   help="working width in pixels (default 960)")
    p.add_argument("--no-mirror", action="store_true",
                   help="don't flip the picture left-right")
    p.add_argument("--exclude", default=os.environ.get("HM_EXCLUDE", ""),
                   help="decoration zones to ignore, for every CLI feed: "
                        "'x1,y1,x2,y2;...' in 0-1 coordinates")
    p.add_argument("--phantoms", action="store_true",
                   default=os.environ.get("HM_PHANTOMS", "0") == "1",
                   help="record passers-by and replay them later as phantoms "
                        "(stores footage of whoever walks past; off by "
                        "default)")
    p.add_argument("--publish", default=os.environ.get("HM_PUBLISH"),
                   help="also push each feed as H.264 to this RTSP server "
                        "base URL, at <base>/<name> (needs ffmpeg)")
    win = p.add_mutually_exclusive_group()
    win.add_argument("--window", action="store_true",
                     help="show the first feed in a fullscreen window")
    win.add_argument("--headless", action="store_true",
                     help="never open a window")
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {__version__}")
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["harvest"]:
        from . import harvest
        return harvest.main(argv[1:])

    args = build_parser().parse_args(argv)
    cfg, specs = ({}, [])
    if args.config:
        cfg, specs = load_config(args.config)
    for i, s in enumerate(args.sources):
        name, src = parse_source(s, i)
        specs.append(dict(name=name, source=src, mirror=None,
                          exclude=parse_zones(args.exclude), phantoms=None,
                          publish=None))
    if not specs:
        specs.append(dict(name="webcam", source="0", mirror=None,
                          exclude=parse_zones(args.exclude), phantoms=None,
                          publish=None))
    names = [s["name"] for s in specs]
    if len(set(names)) != len(names):
        sys.exit(f"feed names must be unique: {names}")

    from .feed import Feed
    width = cfg.get("width", args.width)
    publish = cfg.get("publish", args.publish)
    feeds = []
    for s in specs:
        mirror = s["mirror"] if s["mirror"] is not None else \
            cfg.get("mirror", not args.no_mirror)
        phantoms = s["phantoms"] if s["phantoms"] is not None else \
            cfg.get("phantoms", args.phantoms)
        pub = s["publish"] or (publish and
                               f"{publish.rstrip('/')}/{s['name']}")
        passes = os.path.join(os.environ.get("HM_PASSES", os.path.expanduser(
            "~/.haunted-mirror/passes")), s["name"])
        feeds.append(Feed(s["name"], s["source"], width=width, mirror=mirror,
                          exclude=s["exclude"], phantoms=phantoms,
                          passes=passes, publish=pub).start())

    port = cfg.get("port", args.port)
    if port:
        from .server import make_app
        app = make_app(feeds)
        threading.Thread(
            target=lambda: app.run(host=args.host, port=port, threaded=True,
                                   use_reloader=False),
            daemon=True).start()
        print(f"[haunted-mirror] http://localhost:{port}/", flush=True)

    window = args.window or (not args.headless and _has_display()
                             and any(str(s["source"]).isdigit()
                                     or s["source"] == "webcam"
                                     for s in specs))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        if window:
            from .window import show
            show(feeds[0])
        else:
            threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        for f in feeds:
            f.stop()


def _has_display():
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


if __name__ == "__main__":
    main()
