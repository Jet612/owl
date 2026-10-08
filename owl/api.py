"""Public HTTP API, exposed to the internet through Tailscale Funnel.

Serves the clip library and proxies MediaMTX's HLS live stream. Every route
except /api/health needs either `Authorization: Bearer <OWL_API_SECRET>` or a
media token (see auth.py), passed as `?t=<token>` or, for the live stream, as a
path segment so the playlist's relative segment URLs keep it.
"""

import json
import logging
import shutil
import time

import aiohttp
from aiohttp import web

from . import auth, clips
from .config import Config

LOG = logging.getLogger("owl.api")

CONFIG = web.AppKey("config", Config)
HTTP = web.AppKey("http", aiohttp.ClientSession)

# Headers worth passing back from MediaMTX's HLS server.
HLS_HEADERS = ("Content-Type", "Cache-Control")


def _authorized(request: web.Request, token: str | None = None) -> bool:
    secret = request.app[CONFIG].api_secret
    token = token or request.query.get("t", "")
    if token and auth.check_token(secret, token):
        return True
    return auth.check_bearer(secret, request.headers.get("Authorization", ""))


@web.middleware
async def cors_and_auth(request: web.Request, handler):
    origins = request.app[CONFIG].cors_origins
    origin = request.headers.get("Origin", "")
    allow = "*" if "*" in origins else (origin if origin in origins else "")

    if request.method == "OPTIONS":
        response = web.Response(status=204)
    elif request.path == "/api/health" or (
        request.match_info.route.name == "live" or _authorized(request)
    ):
        try:
            response = await handler(request)
        except web.HTTPException as exc:
            response = exc
    else:
        response = web.json_response({"error": "unauthorized"}, status=401)

    if allow:
        response.headers["Access-Control-Allow-Origin"] = allow
        response.headers["Access-Control-Allow-Headers"] = "Authorization, Range"
        response.headers["Access-Control-Allow-Methods"] = "GET, DELETE, OPTIONS"
        response.headers["Access-Control-Expose-Headers"] = "Content-Length, Content-Range"
        if allow != "*":
            response.headers["Vary"] = "Origin"
    return response


def _clip_path(request: web.Request, ext: str):
    clip_id = request.match_info["clip_id"]
    if not clips.CLIP_ID.match(clip_id):
        raise web.HTTPNotFound()
    path = request.app[CONFIG].clips_dir / f"{clip_id}{ext}"
    if not path.exists():
        raise web.HTTPNotFound()
    return path


async def health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def status(request: web.Request) -> web.Response:
    config = request.app[CONFIG]
    try:
        vision = json.loads(config.status_path.read_text())
    except (OSError, ValueError):
        vision = {}
    online = time.time() - vision.get("updated_at", 0) < 15
    disk = shutil.disk_usage(config.clips_dir)
    return web.json_response({
        "camera_online": online,
        "streaming": online and vision.get("streaming", False),
        "recording": online and vision.get("recording", False),
        "identifying": online and vision.get("identifying", False),
        "classifier": vision.get("classifier") if online else None,
        "current_clip": vision.get("current_clip") if online else None,
        "last_detection": vision.get("last_detection"),
        "retention_days": config.retention_days,
        "clip_count": len(list(config.clips_dir.glob("*.json"))),
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
        "server_time": time.time(),
    })


async def list_clips(request: web.Request) -> web.Response:
    config = request.app[CONFIG]
    try:
        limit = min(max(int(request.query.get("limit", 50)), 1), 500)
    except ValueError:
        raise web.HTTPBadRequest(text="limit must be a number")
    before = request.query.get("before")
    label = request.query.get("label")
    category = request.query.get("category")
    result = clips.list_clips(config.clips_dir)
    if before:
        result = [c for c in result if c["id"] < before]
    if label:
        result = [c for c in result if label in c.get("labels", {})]
    if category:
        result = [c for c in result if category in c.get("categories", [])]
    page = result[:limit]
    return web.json_response({
        "clips": page,
        "next_before": page[-1]["id"] if len(result) > limit else None,
    })


async def labels(request: web.Request) -> web.Response:
    """Every label that appears in a stored clip, with its category and clip count, for filters."""
    found: dict[str, dict] = {}
    for clip in clips.list_clips(request.app[CONFIG].clips_dir):
        for name in clip.get("labels", {}):
            entry = found.setdefault(name, {
                "label": name, "count": 0,
                "category": clip.get("label_categories", {}).get(name, "unknown"),
            })
            entry["count"] += 1
    return web.json_response({"labels": sorted(found.values(), key=lambda e: (-e["count"], e["label"]))})


async def get_clip(request: web.Request) -> web.Response:
    return web.json_response(json.loads(_clip_path(request, ".json").read_text()))


async def clip_video(request: web.Request) -> web.FileResponse:
    return web.FileResponse(_clip_path(request, ".mp4"), headers={"Content-Type": "video/mp4"})


async def clip_download(request: web.Request) -> web.FileResponse:
    path = _clip_path(request, ".mp4")
    meta = json.loads(_clip_path(request, ".json").read_text())
    name = f"owl-{meta['label'].replace(' ', '-').replace(chr(39), '')}-{path.stem}.mp4"
    return web.FileResponse(path, headers={
        "Content-Type": "video/mp4",
        "Content-Disposition": f'attachment; filename="{name}"',
    })


async def clip_thumb(request: web.Request) -> web.FileResponse:
    return web.FileResponse(
        _clip_path(request, ".jpg"),
        headers={"Content-Type": "image/jpeg", "Cache-Control": "private, max-age=86400"},
    )


async def delete_clip(request: web.Request) -> web.Response:
    _clip_path(request, ".json")
    clips.delete_clip(request.app[CONFIG].clips_dir, request.match_info["clip_id"])
    return web.json_response({"deleted": request.match_info["clip_id"]})


async def live(request: web.Request) -> web.StreamResponse:
    """Proxy /live/<token>/<file> to MediaMTX's HLS server, which only listens on localhost."""
    if not _authorized(request, request.match_info["token"]):
        return web.json_response({"error": "unauthorized"}, status=401)
    path = request.match_info["path"] or "index.m3u8"
    if ".." in path:
        raise web.HTTPNotFound()
    upstream = f"{request.app[CONFIG].hls_url}/{path}"
    try:
        async with request.app[HTTP].get(upstream, params=request.query) as resp:
            headers = {k: resp.headers[k] for k in HLS_HEADERS if k in resp.headers}
            response = web.StreamResponse(status=resp.status, headers=headers)
            await response.prepare(request)
            async for chunk in resp.content.iter_chunked(64 * 1024):
                await response.write(chunk)
            await response.write_eof()
            return response
    except aiohttp.ClientError as exc:
        LOG.warning("HLS proxy error for %s: %s", path, exc)
        return web.json_response({"error": "live stream unavailable"}, status=502)


async def _http_session(app: web.Application):
    # LL-HLS playlist requests block until the next part is ready, so allow some time.
    app[HTTP] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
    yield
    await app[HTTP].close()


def make_app(config: Config) -> web.Application:
    if len(config.api_secret) < 32:
        raise SystemExit("OWL_API_SECRET must be set to a random string of at least 32 characters")
    config.clips_dir.mkdir(parents=True, exist_ok=True)
    app = web.Application(middlewares=[cors_and_auth])
    app[CONFIG] = config
    app.cleanup_ctx.append(_http_session)
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/status", status)
    app.router.add_get("/api/clips", list_clips)
    app.router.add_get("/api/labels", labels)
    app.router.add_get("/api/clips/{clip_id}", get_clip)
    app.router.add_delete("/api/clips/{clip_id}", delete_clip)
    app.router.add_get("/api/clips/{clip_id}/video", clip_video)
    app.router.add_get("/api/clips/{clip_id}/download", clip_download)
    app.router.add_get("/api/clips/{clip_id}/thumb", clip_thumb)
    app.router.add_get("/live/{token}/{path:.*}", live, name="live")
    return app


def main() -> None:
    config = Config()
    logging.basicConfig(level=config.log_level, format="%(levelname)s %(name)s: %(message)s")
    web.run_app(make_app(config), host=config.api_host, port=config.api_port, access_log=None)


if __name__ == "__main__":
    main()
