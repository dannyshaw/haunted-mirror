# Haunted Mirror

Point it at a camera, put the result on a screen in your window, and people
walking past see their own reflection, with something standing behind them
that isn't there.

It's built for Halloween: a live camera feed, mirrored so it reads as a
reflection, with a found-footage colour grade, and a director that reacts to
real people:

- **The follower.** A dark wraith drifts along behind each passer-by and
  trails smoke.
- **The echo.** Their own image follows them a beat late.
- **The watcher routine.** Stop to look and the wraith slowly creeps up to your
  shoulder, the picture pushes in, and it escalates the longer you stay.
- **Binoculars.** Now and then the camera "watches" someone through a
  vignette.
- **Apparitions.** When the street is empty, something occasionally shows up
  anyway.
- **Close-range extras** (experimental). When someone fills the frame, the
  wraith stands *behind* them, they drain toward shadow, and their eyes go
  hollow.
- **Phantoms** (opt-in). Real passers-by from earlier are replayed back into
  the scene where they walked.

It's all classical compositing driven by a small person detector, so it runs
in real time on a CPU: no GPU and no cloud. One feed at 960 px uses roughly
1-3 cores of a modern x86 CPU, depending on how busy the scene is.

## Quick start: your webcam

Needs Python 3.9+. With [uv](https://docs.astral.sh/uv/):

```bash
uvx --from git+https://github.com/dannyshaw/haunted-mirror haunted-mirror
```

or with pipx:

```bash
pipx install git+https://github.com/dannyshaw/haunted-mirror
haunted-mirror
```

That opens your webcam in a fullscreen window (**f** toggles fullscreen, **q**
quits) and also serves it at <http://localhost:8080/>. Use `haunted-mirror 1`
for a second camera.

It works on macOS, Windows and Linux. On macOS the first run asks for camera
permission for your terminal app. Stand back a couple of metres: the effects
are tuned for full-height people, not a face filling the frame.

## Cameras: Docker

For one or more IP cameras (RTSP, MJPEG over HTTP, anything ffmpeg can open):

```bash
docker run -d --name haunted-mirror -p 8080:8080 \
  ghcr.io/dannyshaw/haunted-mirror \
  front=rtsp://user:pass@192.168.1.50:554/stream1
```

Then open `http://<host>:8080/` on the screen in your window, fullscreen.

Each feed is served at:

| URL | What |
|---|---|
| `/<name>/` | fullscreen page (double-click for browser fullscreen) |
| `/<name>/stream` | MJPEG: browsers, VLC, go2rtc, Frigate, Home Assistant |
| `/<name>/snapshot.jpg` | newest frame |
| `/<name>/tracks.jpg` | what the tracker sees (for placing exclusion zones) |
| `/healthz` | per-feed fps; 503 if a feed has stalled |

### RTSP output

To get each feed back out over RTSP (for an NVR, a media player, or a Raspberry
Pi driving the TV), run it next to [mediamtx](https://github.com/bluenviron/mediamtx).
`docker-compose.yml` in this repo does exactly that:

```bash
cp feeds.example.yaml feeds.yaml    # edit your cameras in
docker compose up -d
# rtsp://<host>:8554/<feed name>
```

`--publish rtsp://server:8554` pushes H.264 to any RTSP server (mediamtx,
go2rtc), at `<base>/<feed name>`.

### Frigate / go2rtc users

Many cameras only allow two or three simultaneous streams. If you already run
Frigate, read the camera through go2rtc's restream rather than directly, so
haunted-mirror shares the connection Frigate already has:

```bash
haunted-mirror front=rtsp://<frigate-host>:8554/<camera>
```

To record the haunted feed in Frigate, add it as a go2rtc source:

```yaml
go2rtc:
  ffmpeg:
    # MJPEG carries no timestamps: tell ffmpeg the format, stamp on arrival
    mjpeg_in: "-fflags nobuffer -flags low_delay -f mpjpeg -i {input}"
  streams:
    haunted: ffmpeg:http://<host>:8080/front/stream#input=mjpeg_in#video=h264
```

Or skip the re-encode and use `--publish` (above) into go2rtc directly.

## Configuration

Small setups fit on the command line:

```
haunted-mirror [SOURCE ...] [--port 8080] [--width 960] [--no-mirror]
               [--exclude 'x1,y1,x2,y2;...'] [--phantoms]
               [--publish rtsp://host:8554] [--window | --headless]
```

`SOURCE` is `[name=]url`, a video file (looped), or a webcam index. For
several cameras, or per-camera settings, use a config file (`--config
feeds.yaml`). See [`feeds.example.yaml`](feeds.example.yaml).

### Decorations the detector thinks are people

Life-size skeletons and scarecrows look like people to the detector, and they
will collect wraiths. List the spots where they stand as exclusion zones:

1. Open `http://<host>:8080/<name>/tracks.jpg`. Red boxes are counted as
   people; green ones are ignored. Each box is labelled with its feet
   position in 0-1 coordinates.
2. Add a box around each prop, including its feet, to `exclude:` in the
   config (or `--exclude`). Zones show in blue on `tracks.jpg`.

A track is only ignored if it *started* inside a zone. Someone walking through
or behind your props is still tracked.

Small figures that stand in one spot for a long time are written off
automatically, as are tracks that never move.

### Effect settings

All are optional environment variables:

| Variable | Default | |
|---|---|---|
| `HM_GRADE` | `1` | horror colour grade |
| `HM_GRAIN` | `10` | film grain, 0-30 |
| `HM_STYLES` | `follower,echo,both` | passer-by styles to pick from |
| `HM_FOLLOWER_SIZE` | `0.85` | wraith height, as a fraction of the person |
| `HM_FOLLOWER_GAP` | `0.6` | person-heights it hovers behind them |
| `HM_ZOOM` / `HM_ZOOM_WATCH` | `1.5` / `1.9` | push-in on passers-by / watchers (1 = off) |
| `HM_BINOS` | `0.35` | chance of the binoculars bit (0 = off) |
| `HM_ATTRACT` | `1` | apparitions when nobody's around |
| `HM_CLOSE` | `1` | close-range extras |
| `HM_MIN_SCORE` | `0.5` | detector confidence threshold |
| `HM_LOST_SECS` | `3.0` | how long a track coasts through detector dropouts |
| `HM_DEBUG` | `0` | draw tracker boxes on the output |

### Phantoms (off by default)

With `--phantoms` (or `phantoms: true` per feed), anyone who walks through
alone is saved as a short cut-out "take" (to `~/.haunted-mirror/passes/<feed>`,
or `/data/passes` in Docker). Later it's replayed into the scene, walking the
same path. It's the creepiest effect, and it also means **storing footage of
strangers**, often children on Halloween. Check that's OK where you live before
turning it on. `HM_PHANTOM_KEEP` (default 40) caps how many takes are kept.

`haunted-mirror harvest --feed <name> clip.mp4 ...` turns recorded clips from
the same camera (framed identically) into takes. With `--keep` they're never
pruned, which is good for staged walk-pasts.

## Tips

- **Framing matters more than effects.** Eye-level, tight, life-size on the
  screen, after dark. The mirror illusion is the whole trick.
- **Driving the TV from a Raspberry Pi.** Play the RTSP output with `mpv` on
  the console (`--vo=gpu --gpu-context=drm`). No desktop is needed. H.264
  decodes far more cheaply than MJPEG on a Pi, and a proper power supply
  matters: an under-volted Pi throttles to a crawl.
- The web server is Flask's built-in one. It's fine for a few screens on your
  LAN; don't expose it to the internet directly.

## How it works

`capture → mirror → person detector (SSDLite MobileNet V2 on OpenVINO, swept
over tiles) → tracker (coasting, prop/fixture rejection) → director picks
effects per person → composite → grade → MJPEG / RTSP`

Detection runs in its own thread and is allowed to lag. Effects run on every
frame using the latest tracks, so the picture stays smooth even when the
detector is slow.

## Licence

MIT. The bundled models keep their own licences (Apache-2.0 / MIT); see
[`src/haunted_mirror/models/NOTICE.md`](src/haunted_mirror/models/NOTICE.md).
