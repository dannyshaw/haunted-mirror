"""
Web server for the haunted feeds.

  /                  the only feed fullscreen, or an index of all of them
  /<name>/           one feed, fullscreen (put this on the TV)
  /<name>/stream     MJPEG, for browsers, go2rtc, Frigate, VLC, ...
  /<name>/snapshot.jpg   newest finished frame
  /<name>/tracks.jpg     what the tracker sees, for placing exclusion zones
  /stream            the first feed's MJPEG
"""
import html
import time

from flask import Flask, Response, abort

FULLSCREEN = """<!doctype html><html><head><title>{title}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>html,body{{margin:0;background:#000;height:100%;overflow:hidden}}
img{{width:100vw;height:100vh;object-fit:{fit}}}</style></head>
<body><img src="{src}" ondblclick="document.documentElement.requestFullscreen()">
</body></html>"""

INDEX = """<!doctype html><html><head><title>Haunted Mirror</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{margin:0;padding:16px;background:#0b0b0d;color:#ddd;
font:15px system-ui,sans-serif}}h1{{font-weight:500;color:#e8e2d0}}
.grid{{display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(320px,1fr))}}
a{{color:#e8e2d0;text-decoration:none}}img{{width:100%;border-radius:6px;background:#000}}
small{{color:#888}}</style></head><body><h1>Haunted Mirror</h1>
<div class="grid">{cards}</div></body></html>"""

CARD = """<div><a href="/{n}/"><img src="/{n}/stream"></a><div>{n}
<small>{fps:.1f} fps &middot; <a href="/{n}/tracks.jpg">tracks</a></small></div></div>"""


def make_app(feeds, fit="cover"):
    app = Flask(__name__)
    by_name = {f.name: f for f in feeds}

    def feed(name):
        f = by_name.get(name)
        if f is None:
            abort(404)
        return f

    def mjpeg(f):
        last = None
        while True:
            _, jpg = f.latest()
            # each frame is sent once, so consumers see the real frame rate
            if jpg is None or jpg is last:
                time.sleep(0.005)
                continue
            last = jpg
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpg)).encode()
                   + b"\r\n\r\n" + jpg + b"\r\n")

    def stream_of(f):
        return Response(mjpeg(f),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/")
    def index():
        if len(feeds) == 1:
            return page(feeds[0].name)
        cards = "".join(CARD.format(n=html.escape(f.name), fps=f.fps)
                        for f in feeds)
        return INDEX.format(cards=cards)

    @app.route("/<name>/")
    def page(name):
        f = feed(name)
        return FULLSCREEN.format(title=html.escape(f.name), fit=fit,
                                 src=f"/{html.escape(f.name)}/stream")

    @app.route("/<name>/stream")
    def stream(name):
        return stream_of(feed(name))

    @app.route("/stream")
    def first_stream():
        return stream_of(feeds[0])

    @app.route("/<name>/snapshot.jpg")
    def snapshot(name):
        _, jpg = feed(name).latest()
        if jpg is None:
            return "no frame yet", 503
        return Response(jpg, mimetype="image/jpeg")

    @app.route("/<name>/tracks.jpg")
    def tracks(name):
        jpg = feed(name).tracks_jpg()
        if jpg is None:
            return "no frame yet", 503
        return Response(jpg, mimetype="image/jpeg")

    @app.route("/healthz")
    def healthz():
        stale = [f.name for f in feeds if f.fps < 0.5]
        return ({"ok": not stale, "feeds": {f.name: round(f.fps, 1)
                                             for f in feeds}},
                503 if stale else 200)

    return app
