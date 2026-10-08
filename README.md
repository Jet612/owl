# owl

A backyard wildlife camera on a Raspberry Pi 5 with a Camera Module 3 and the
Hailo AI HAT. It streams live video, records a clip whenever a person or an
animal shows up, works out which species it was, keeps clips for 30 days, and
pushes a notification with a snapshot to every subscribed phone. A website (in
its own repo, on Vercel) shows the live feed and the clips through this Pi's API.

```
Camera ─► owl-vision ─┬─► MediaMTX ──(HLS)──► owl-api ─► Tailscale Funnel ─► website / browsers
                      ├─► clips in /var/lib/owl/clips ─┘
                      └─► ntfy.sh ─► phones
```

## How it decides what it saw

1. **Trigger.** The Hailo chip runs YOLOv8 ten times a second looking for people
   and animal-like objects (bird, cat, dog, horse, sheep, cow, bear), and a
   motion detector watches for anything that moves. Deer, foxes, rabbits and the
   like aren't classes the Hailo model knows, so they are usually caught by
   movement. A trigger starts recording at once, with the previous 5 seconds
   included, so the clip begins before the animal arrived.
2. **Identify.** About once a second the camera crops what moved and asks
   [BioCLIP](https://huggingface.co/imageomics/bioclip), a model trained on
   biodiversity photos, which species it is. It chooses from `species.txt` (73
   northern Virginia species) plus a dozen "nothing there" options (leaves,
   fence, snow, an insect on the lens...). It runs on the Pi's CPU, about 0.8 s
   per frame.
3. **Confirm.** A species counts only when the classifier is at least 95% sure
   in two frames, and in more frames than it saw nothing. That sends the
   notification. People are confirmed by the Hailo detector directly, which is
   much better at them, so a person notification arrives in under a second.
4. **Clean up.** If movement turned out to be nothing (wind, shadows, a bug),
   the clip is deleted and no notification goes out. If the classifier thought
   it saw an animal but never got sure, the clip is kept as "unidentified
   animal" for 3 days, with its best guesses, so a real visitor isn't lost.

### How reliable is it?

Measured on stock photos of 18 local animals and people, pasted into the real
camera frame at several sizes, and on 224 crops of outdoor scenes with no
animals in them:

- At 95% confidence or more the species was right 97% of the time. Below 90% it
  was right under 40% of the time, which is why the bar is so high.
- Close or medium-distance animals in daylight: about 4 in 5 named correctly.
  Small and far away animals (under about 90 px wide in the frame): about 2 in 3.
- Animal-free outdoor scenes were mistaken for a confident animal in about 2% of
  crops. Requiring two frames and a majority makes a false notification much
  rarer, but not impossible.
- **Night is the weak spot.** In a harsh simulation (dark, grayscale, noisy)
  almost nothing was named correctly, so night visitors will often be
  "unidentified animal". Real infrared footage is brighter than the simulation,
  but it hasn't been tested outdoors yet. An IR light and a NoIR camera help most.
- Mistakes were mostly between similar species (red-tailed vs Cooper's hawk, a
  small coyote called a wolf). Red fox, deer, rabbit, raccoon, opossum, squirrel,
  groundhog and the common birds were right in the tests. A black bear and a gray
  fox photo were never named correctly, but their crops were poor, so treat those
  two as untested.
- The model can only name what is on the list. An animal that isn't on it gets
  called something similar, or "unidentified animal" if it isn't sure.

Wolves aren't found in Virginia, but `gray wolf` is on the list. Coyotes can be
mistaken for wolves, so delete that line if it happens.

## Install

On the Pi (Raspberry Pi OS Trixie, 64-bit) with the camera, AI HAT, `hailo-all`,
`uv` and Tailscale installed:

```sh
git clone https://github.com/Jet612/owl.git
cd owl
uv sync
sudo ./scripts/install-mediamtx.sh --force-config
sudo ./scripts/install-owl.sh
sudo tailscale funnel --bg 8080
```

The first run of `install-owl.sh` creates `/etc/owl/owl.env` with a random API
secret and ntfy topic, installs `/etc/owl/species.txt`, downloads the species
model (about 600 MB, once), and prints the topic and the public API URL
(`https://<pi-name>.<tailnet>.ts.net`). Rerunning keeps your settings.

## Phone notifications

1. Install **ntfy** ([iOS](https://apps.apple.com/app/ntfy/id1625396347),
   [Android](https://play.google.com/store/apps/details?id=io.heckel.ntfy)) on
   each phone.
2. Tap **+** and subscribe to the topic in `OWL_NTFY_TOPIC` (server `ntfy.sh`).

Each confirmed sighting sends one notification with the snapshot ("Red fox
spotted"). There is at most one per species (or per person) every 5 minutes, so
a feeder full of cardinals doesn't flood your phone; a fox right after a deer
still notifies. To record something without notifying, add it to
`OWL_NOTIFY_MUTE`, for example `eastern gray squirrel,person`. Tapping a
notification opens `OWL_SITE_URL`. Anyone who knows the topic can subscribe, so
keep it random.

## Settings

Everything lives in `/etc/owl/owl.env`; `owl.env.example` documents each option.
After editing it:

```sh
sudo systemctl restart owl-vision owl-api
```

The most useful options:

| Variable | Default | What it controls |
| --- | --- | --- |
| `OWL_DETECT_PEOPLE` | `true` | Record and notify for people |
| `OWL_NOTIFY_MUTE` | none | Species or `person` to record without notifying |
| `OWL_MOTION_MIN_AREA` | `0.0015` | Smallest moving blob that triggers, as a fraction of the frame. Lower it to catch smaller or farther animals, at the cost of more false triggers |
| `OWL_SPECIES_MIN_SCORE` | `0.95` | Confidence needed to name a species |
| `OWL_UNIDENTIFIED_DAYS` | `3` | How long to keep clips of an animal nothing could name (`0` deletes them) |
| `OWL_SITE_URL` | none | The website's address. Tapping a notification opens it, so set it once the site is deployed |
| `OWL_RETENTION_DAYS` | `30` | Days named clips are kept, counted from when they start |
| `OWL_LOW_DISK_PERCENT` | `10` | Below this much free disk you get a warning notification (at most one a day) |
| `OWL_MIN_FREE_GB` | `5` | Below this much free space the oldest clips are deleted early, and a notification says how many |
| `OWL_PRE_ROLL` / `OWL_POST_ROLL` | `5` / `10` | Seconds kept before and after |
| `OWL_HFLIP` / `OWL_VFLIP` | `false` | Flip the image for odd camera mounts |
| `OWL_VIDEO_WIDTH` / `HEIGHT` / `FPS` / `BITRATE` | 1280 / 720 / 30 / 2.5 Mb/s | Stream and clip quality |

### The species list

`/etc/owl/species.txt` has one line per species: `category | common name |
scientific name`. Add a line to teach the camera a new animal, or delete lines
for animals that don't live near you (fewer choices means fewer mix-ups), then
restart `owl-vision`. The first start after a change spends about 30 seconds
encoding the list.

## Recovery and disk space

- If the camera stops delivering frames (unplugged, crashed), `owl-vision`
  notices within 30 seconds and exits; systemd restarts it, and it keeps
  retrying every ~15 seconds until the camera is back. Meanwhile
  `/api/status` says `camera_online: false` and the live route returns 502.
- Clips are the only copy. Named clips expire 30 days after they start and
  unidentified ones after `OWL_UNIDENTIFIED_DAYS`; each clip's `expires_at`
  says when. If the disk gets short of space the oldest clips are deleted
  early, and you are always sent a notification saying so.
- Leftovers from a power cut (unfinished recordings, orphaned thumbnails) are
  removed after an hour.

## Checking it from outside

`scripts/smoke-test.sh` runs the API's acceptance checks the way viewers reach
it: credentials and tokens, clip paging and filters, range requests, the live
playlist and its CORS headers, and that nothing but the API answers through
Funnel. Run it from a machine that is not on the tailnet:

```sh
OWL_API_URL=https://<pi-name>.<tailnet>.ts.net OWL_API_SECRET=... ./scripts/smoke-test.sh
```

Add `OWL_EXPECT_CAMERA=1` to make "camera offline" a failure. It changes
nothing on the Pi.

## If `owl-vision` crashes right after a restart

Seen once on kernel 6.18 with HailoRT 4.23 (which logs a `find_vma` warning
every time the service starts): the service died with `HAILO_HEF_FILE_CORRUPTED`
or with an illegal instruction / segfault while importing `cv2`. The files on
the SD card were fine. Only their cached copies in RAM were damaged, which
`dpkg -V hailo-models python3-opencv` reveals (it reads through the cache) and
reading the file with `dd iflag=direct` does not. The cause is not confirmed.
Clearing the cache fixed it:

```sh
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
sudo systemctl restart owl-vision
```

If it keeps happening, reboot the Pi instead of restarting the service.

## Development

```sh
.venv/bin/python -m unittest discover -s tests -t .
```

The tests need no camera. They cover the token check (including tokens minted
by the website's own function under Node), every API route, the live proxy,
retention, the clip shapes the recorder stores, and the settings file.

## Managing the services

```sh
systemctl status owl-vision owl-api mediamtx
journalctl -u owl-vision -f          # triggers, identifications, notifications
journalctl -u owl-api -f
sudo systemctl restart owl-vision
tailscale funnel status
```

Clips are stored in `/var/lib/owl/clips` as `<id>.mp4`, `<id>.jpg` (snapshot
with the box and label drawn) and `<id>.json`. The species model and its
encoded list are in `/var/lib/owl/models`.

## API for the website

Base URL: `https://<pi-name>.<tailnet>.ts.net` (from `tailscale funnel status`).

### Authentication

The website's **server** keeps `OWL_API_SECRET` (copy it from
`/etc/owl/owl.env` into the Vercel environment variables). Two ways to use it:

- Server to API: `Authorization: Bearer <OWL_API_SECRET>`.
- Browser to API: `<video>` tags and HLS players can't send headers. After
  someone logs in with the site password, have the server mint a media token
  and give it to the browser, which passes it as `?t=<token>`:

```js
import crypto from "node:crypto";

export function mintOwlToken(secret, ttlSeconds = 6 * 3600) {
  const exp = Math.floor(Date.now() / 1000) + ttlSeconds;
  const sig = crypto.createHmac("sha256", secret).update(`owl-media:${exp}`).digest("base64url");
  return `${exp}.${sig}`;
}
```

Tokens are valid on every route until `exp`. Never send the secret itself to
the browser.

### Routes

| Method & path | Returns |
| --- | --- |
| `GET /api/health` | `{"ok": true}`. No auth needed. |
| `GET /api/status` | Whether the camera is online, streaming, recording or still identifying a sighting, the last detection, clip count and disk space |
| `GET /api/clips?limit=50&before=<id>&label=red%20fox&category=bird` | `{"clips": [...], "next_before": "<id>" \| null}`, newest first. `label` and `category` filter. Pass `next_before` back as `before` for the next page. |
| `GET /api/labels` | `{"labels": [{"label", "category", "count"}]}`, every label in the stored clips, for building filters |
| `GET /api/clips/<id>` | One clip's metadata |
| `GET /api/clips/<id>/video` | MP4 for `<video src>`. Supports range requests, so seeking works. |
| `GET /api/clips/<id>/download` | The same MP4 with `Content-Disposition: attachment`, so it saves to the device |
| `GET /api/clips/<id>/thumb` | JPEG snapshot |
| `DELETE /api/clips/<id>` | Deletes a clip |
| `GET /live/<token>/index.m3u8` | Live low-latency HLS. The token goes in the path. |

Errors are JSON, like `{"error": "unauthorized"}` (401), `{"error": "not found"}`
(404) or `{"error": "limit must be a number"}` (400). CORS allows any origin
(`OWL_CORS_ORIGINS`), on live-stream responses too. The live route answers
`502 {"error": "camera offline"}` or `{"error": "stream unavailable"}` whenever
there is no stream, which the site treats as normal; playlists are sent with
`Cache-Control: no-store`. Thumbnails are about 640 px wide.

A clip looks like:

```json
{
  "id": "20261007T215601Z",
  "started_at": 1791410158.582,
  "ended_at": 1791410169.293,
  "expires_at": 1794002169.293,
  "duration": 10.7,
  "label": "red fox",
  "scientific": "Vulpes vulpes",
  "category": "mammal",
  "identified": true,
  "labels": {"red fox": 0.99},
  "label_categories": {"red fox": "mammal"},
  "categories": ["mammal"],
  "size_bytes": 3192209
}
```

Times are Unix seconds. `label` is the main thing seen: a species name, `person`,
or `unidentified animal` (then `identified` is false, `category` is `unknown`
and `labels` holds low-confidence guesses). `category` is one of mammal, bird,
reptile, amphibian, pet, person, unknown. `labels` has every label in the clip
with its best confidence, and `categories` lists the distinct categories, so a
clip with a person and a fox appears under both. If both are present, `label` is
the animal. Species names come from `species.txt`, so don't hardcode them; build
filters from `/api/labels`.

### Playing the live feed

Safari (including iOS) plays HLS natively. Other browsers need
[hls.js](https://github.com/video-dev/hls.js):

```js
const src = `${OWL_API_URL}/live/${token}/index.m3u8`;
if (video.canPlayType("application/vnd.apple.mpegurl")) {
  video.src = src;
} else {
  const hls = new Hls({ lowLatencyMode: true });
  hls.loadSource(src);
  hls.attachMedia(video);
}
```

Expect 2–4 seconds of delay. WebRTC would be faster, but Funnel only carries
HTTPS. On the tailnet you can watch with sub-second delay at
`http://<pi-tailscale-address>:8889/owl`.
