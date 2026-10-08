"""Public HTTP API, exposed to the internet through Tailscale Funnel.

Serves the clip library and proxies MediaMTX's HLS live stream. Every route
except /api/health needs either `Authorization: Bearer <OWL_API_SECRET>` or a
media token (see auth.py), passed as `?t=<token>` or, for the live stream, as a
path segment so the playlist's relative segment URLs keep it.

Errors are JSON, like {"error": "unauthorized"}.
"""

import asyncio
import json
import logging
import re
import shutil
import time

import aiohttp
from aiohttp import web

from . import auth, clips
from .config import Config

LOG = logging.getLogger("owl.api")

CONFIG = web.AppKey("config", Config)
HTTP = web.AppKey("http", aiohttp.ClientSession)

# The camera counts as offline once owl-vision's status file is this many seconds old.
CAMERA_TIMEOUT = 15


def _error(status: int, message: str, headers: dict | None = None) -> web.Response:
    return web.json_response({"error": message}, status=status, headers=headers)


def _authorized(request: web.Request, token: str | None = None) -> bool:
    secret = request.app[CONFIG].api_secret
    token = token or request.query.get("t", "")
    if token and auth.check_token(secret, token):
        return True
    return auth.check_bearer(secret, request.headers.get("Authorization", ""))


def _add_cors(request: web.Request, response: web.StreamResponse) -> None:
    """Let the website's origin call the API straight from the browser."""
    if response.prepared:  # the headers are already on the wire
        return
    origins = request.app[CONFIG].cors_origins
    origin = request.headers.get("Origin", "").lower()
    allow = "*" if "*" in origins else (origin if origin in origins else "")
    if not allow:
        return
    response.headers["Access-Control-Allow-Origin"] = allow
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Range"
    response.headers["Access-Control-Allow-Methods"] = "GET, DELETE, OPTIONS"
    response.headers["Access-Control-Expose-Headers"] = (
        "Content-Length, Content-Range, Content-Disposition"
    )
    response.headers["Access-Control-Max-Age"] = "600"
    if allow != "*":
        response.headers["Vary"] = "Origin"


@web.middleware
async def cors_and_auth(request: web.Request, handler):
    if request.method == "OPTIONS":
        response = web.Response(status=204)
    elif (
        request.path == "/api/health"
        or request.match_info.route.name == "live"  # checks the token from its path itself
        or _authorized(request)
    ):
        try:
            response = await handler(request)
        except web.HTTPException as exc:
            allow = exc.headers.get("Allow")
            response = _error(exc.status, exc.reason.lower(), {"Allow": allow} if allow else None)
        except Exception:  # whatever went wrong, the client still gets JSON
            LOG.exception("Unhandled error for %s %s", request.method, request.path)
            response = _error(500, "internal error")
    else:
        response = _error(401, "unauthorized")
    _add_cors(request, response)
    return response


def _clip_path(request: web.Request, ext: str):
    clip_id = request.match_info["clip_id"]
    if not clips.CLIP_ID.match(clip_id):
        raise web.HTTPNotFound()
    path = request.app[CONFIG].clips_dir / f"{clip_id}{ext}"
    if not path.exists():
        raise web.HTTPNotFound()
    return path


def _read_vision(config: Config) -> tuple[dict, bool]:
    """What owl-vision last reported, and whether that is recent enough to call the camera online."""
    try:
        vision = json.loads(config.status_path.read_text())
        online = time.time() - float(vision["updated_at"]) < CAMERA_TIMEOUT
    except (OSError, ValueError, KeyError, TypeError):
        return {}, False
    return vision, online


async def health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def status(request: web.Request) -> web.Response:
    config = request.app[CONFIG]
    vision, online = _read_vision(config)
    disk = shutil.disk_usage(config.clips_dir)
    clip_count = await asyncio.to_thread(lambda: sum(1 for _ in config.clips_dir.glob("*.json")))
    return web.json_response({
        "camera_online": online,
        "streaming": online and vision.get("streaming", False),
        "recording": online and vision.get("recording", False),
        "identifying": online and vision.get("identifying", False),
        "classifier": vision.get("classifier") if online else None,
        "current_clip": vision.get("current_clip") if online else None,
        "last_detection": vision.get("last_detection"),
        "retention_days": config.retention_days,
        "clip_count": clip_count,
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
        "server_time": time.time(),
    })


async def list_clips(request: web.Request) -> web.Response:
    config = request.app[CONFIG]
    try:
        limit = min(max(int(request.query.get("limit", 50)), 1), 500)
    except ValueError:
        return _error(400, "limit must be a number")
    before = request.query.get("before")
    label = request.query.get("label")
    category = request.query.get("category")
    result = await asyncio.to_thread(clips.list_clips, config.clips_dir)
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
    for clip in await asyncio.to_thread(clips.list_clips, request.app[CONFIG].clips_dir):
        categories = clip.get("label_categories", {})
        for name in clip.get("labels", {}):
            category = categories.get(name, "unknown")
            entry = found.setdefault(name, {"label": name, "category": category, "count": 0})
            # An unidentified clip lists its guesses as "unknown"; a real sighting of the same
            # label decides the category, whichever clip comes first.
            if entry["category"] == "unknown":
                entry["category"] = category
            entry["count"] += 1
    return web.json_response({"labels": sorted(found.values(), key=lambda e: (-e["count"], e["label"]))})


async def get_clip(request: web.Request) -> web.Response:
    path = _clip_path(request, ".json")
    try:
        return web.json_response(json.loads(await asyncio.to_thread(path.read_text)))
    except (OSError, ValueError):  # deleted while we were looking
        raise web.HTTPNotFound()


async def clip_video(request: web.Request) -> web.FileResponse:
    # FileResponse answers Range requests with 206, which Safari needs to play a video at all.
    return web.FileResponse(_clip_path(request, ".mp4"), headers={"Content-Type": "video/mp4"})


async def clip_download(request: web.Request) -> web.FileResponse:
    path = _clip_path(request, ".mp4")
    try:
        label = str(json.loads(_clip_path(request, ".json").read_text())["label"])
    except (OSError, ValueError, KeyError, TypeError):
        label = "clip"
    name = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "clip"
    return web.FileResponse(path, headers={
        "Content-Type": "video/mp4",
        "Content-Disposition": f'attachment; filename="owl-{name}-{path.stem}.mp4"',
    })


async def clip_thumb(request: web.Request) -> web.FileResponse:
    return web.FileResponse(
        _clip_path(request, ".jpg"),
        headers={"Content-Type": "image/jpeg", "Cache-Control": "private, max-age=86400"},
    )


async def delete_clip(request: web.Request) -> web.Response:
    _clip_path(request, ".json")
    clip_id = request.match_info["clip_id"]
    await asyncio.to_thread(clips.delete_clip, request.app[CONFIG].clips_dir, clip_id)
    return web.json_response({"deleted": clip_id})


async def live(request: web.Request) -> web.StreamResponse:
    """Proxy /live/<token>/<file> to MediaMTX's HLS server, which only listens on localhost.

    The site shows "camera offline" and retries quietly on 502, so that is the answer
    whenever there is no stream to hand out.
    """
    config = request.app[CONFIG]
    if not _authorized(request, request.match_info["token"]):
        return _error(401, "unauthorized")
    _, online = _read_vision(config)
    if not online:
        return _error(502, "camera offline")
    path = request.match_info["path"] or "index.m3u8"
    if ".." in path:
        raise web.HTTPNotFound()
    playlist = path.endswith(".m3u8")

    response = None
    try:
        # The query goes through untouched: low-latency HLS uses it for blocking playlist reloads.
        async with request.app[HTTP].get(f"{config.hls_url}/{path}", params=request.query) as upstream:
            # MediaMTX answers 404 for the playlist while nothing is publishing. A missing
            # segment is an ordinary 404: it has just rolled out of the window.
            if upstream.status >= 500 or upstream.status in (401, 403) \
                    or (upstream.status == 404 and playlist):
                LOG.debug("HLS upstream answered %d for %s", upstream.status, path)
                return _error(502, "stream unavailable")
            headers = {"Content-Type": upstream.headers.get("Content-Type", "application/octet-stream")}
            if playlist:
                headers["Cache-Control"] = "no-store"
            elif "Cache-Control" in upstream.headers:
                headers["Cache-Control"] = upstream.headers["Cache-Control"]
            response = web.StreamResponse(status=upstream.status, headers=headers)
            _add_cors(request, response)  # has to happen before the headers go out
            await response.prepare(request)
            async for chunk in upstream.content.iter_chunked(64 * 1024):
                await response.write(chunk)
            await response.write_eof()
            return response
    except (aiohttp.ClientError, TimeoutError) as exc:
        if response is not None and response.prepared:
            # Either the viewer left or MediaMTX went away mid-response; the connection just ends.
            LOG.debug("HLS proxy stopped for %s: %s", path, exc)
            return response
        LOG.warning("HLS proxy error for %s: %s", path, exc)
        return _error(502, "stream unavailable")


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
