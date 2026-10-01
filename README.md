# owl
Raspi5 Camera

Each owl camera is a Raspberry Pi 5 that streams its camera with
[MediaMTX](https://github.com/bluenviron/mediamtx) over Tailscale. The owl-nest
dashboard plays the stream over WebRTC.

## Camera streaming service

`scripts/install-mediamtx.sh` installs MediaMTX and runs it as a systemd
service (`mediamtx`) that starts on boot and restarts if it crashes.

What you need on the Pi:

- Raspberry Pi OS Bookworm or Trixie (64-bit recommended) with the camera connected
- Tailscale installed and logged in

Install (or upgrade) on the Pi:

```sh
git clone https://github.com/Jet612/owl.git
cd owl
sudo ./scripts/install-mediamtx.sh
```

The script downloads a pinned MediaMTX release (checksum verified), installs it
to `/usr/local/bin/mediamtx`, copies `mediamtx/mediamtx.yml` to
`/usr/local/etc/mediamtx.yml`, creates a `mediamtx` system user in the `video`
group, and enables `systemd/mediamtx.service`. Rerunning it is safe. If you've
edited `/usr/local/etc/mediamtx.yml` it leaves your copy alone and writes the
repo version next to it as `mediamtx.yml.new`; pass `--force-config` to
overwrite. Set `MEDIAMTX_VERSION=vX.Y.Z` to install a different release.

### Watching the stream

The camera is published on path `owl`:

| What | Address |
| --- | --- |
| WebRTC (WHEP), used by owl-nest | `http://<pi-tailscale-address>:8889/owl/whep` |
| Browser test page | `http://<pi-tailscale-address>:8889/owl` |
| RTSP, for debugging | `rtsp://<pi-tailscale-address>:8554/owl` |

WebRTC media also uses UDP port 8189.

Viewing is only allowed from the Pi itself and from Tailscale addresses. To
allow your LAN as well, add its range (for example `192.168.1.0/24`) to the
`ips` list in `/usr/local/etc/mediamtx.yml`. MediaMTX reloads its config when
the file changes, so no restart is needed.

### Managing the service

```sh
systemctl status mediamtx       # is it running?
journalctl -u mediamtx -f       # live logs
sudo systemctl restart mediamtx
sudo systemctl disable --now mediamtx   # stop it and don't start on boot
```

Camera settings (resolution, FPS, flip) live under `paths.owl` in the config.
The full list of options is in the
[upstream reference config](https://github.com/bluenviron/mediamtx/blob/main/mediamtx.yml).
