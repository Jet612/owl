"""The HTTP API, checked against the Pi build spec's routes and acceptance list."""

import tempfile
import time
import unittest
from pathlib import Path
from typing import ClassVar
from urllib.parse import parse_qs

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from owl import auth, clips
from owl.api import make_app
from owl.config import Config

from .support import SECRET, clip_id_at, make_config, write_clip

NOW = 1_791_400_000.0  # a fixed "recent" time for clip ids
PLAYLIST = b"#EXTM3U\n#EXT-X-VERSION:9\nmain.m3u8\n"
SEGMENT = bytes(range(256)) * 8


class ApiTestCase(unittest.IsolatedAsyncioTestCase):
    """Starts the API on a temporary data directory, with a fake MediaMTX behind its live route."""

    upstream_down = False  # subclasses can start without the HLS server

    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.upstream_requests: list[tuple[str, str]] = []
        self.upstream_status = {}

        upstream = web.Application()
        upstream.router.add_get("/owl/{tail:.*}", self._hls)
        self.upstream = TestServer(upstream)
        await self.upstream.start_server()
        self.addAsyncCleanup(self.upstream.close)

        self.config = make_config(
            self.dir, hls_url=f"http://127.0.0.1:{self.upstream.port}/owl", **self.config_overrides()
        )
        self.client = TestClient(TestServer(make_app(self.config)))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.addCleanup(self._tmp.cleanup)
        self.set_vision()

    def config_overrides(self) -> dict:
        return {}

    async def _hls(self, request: web.Request) -> web.Response:
        tail = request.match_info["tail"]
        self.upstream_requests.append((tail, request.query_string))
        status = self.upstream_status.get(tail)
        if status:
            return web.Response(status=status, text="nope")
        if tail.endswith(".m3u8"):
            return web.Response(body=PLAYLIST, content_type="application/vnd.apple.mpegurl")
        return web.Response(body=SEGMENT, content_type="video/mp4")

    # -- helpers ---------------------------------------------------------------------------

    def set_vision(self, age: float = 0.0, **fields) -> None:
        """What owl-vision would have written to status.json `age` seconds ago."""
        state = {
            "updated_at": time.time() - age, "streaming": True, "classifier": "ready",
            "recording": False, "current_clip": None, "last_detection": None, "identifying": False,
        }
        clips.write_json(self.config.status_path, {**state, **fields})

    def add_clip(self, offset: int, **kwargs) -> dict:
        started_at = NOW + offset
        return write_clip(
            self.config.clips_dir, clip_id_at(started_at), started_at=started_at, **kwargs
        )

    @property
    def bearer(self) -> dict:
        return {"Authorization": f"Bearer {SECRET}"}

    @staticmethod
    def token(ttl: int = 3600) -> str:
        return auth.make_token(SECRET, ttl)

    async def get_json(self, path: str, *, expect: int = 200, **kwargs):
        response = await self.client.get(path, **kwargs)
        self.assertEqual(response.status, expect, await response.text())
        return await response.json()


class AuthTest(ApiTestCase):
    async def test_health_needs_no_credentials(self):
        self.assertEqual(await self.get_json("/api/health"), {"ok": True})

    async def test_every_other_route_answers_401_json_without_credentials(self):
        self.add_clip(0)
        clip = clip_id_at(NOW)
        paths = [
            "/api/status", "/api/clips", "/api/labels", f"/api/clips/{clip}",
            f"/api/clips/{clip}/video", f"/api/clips/{clip}/download", f"/api/clips/{clip}/thumb",
            "/api/nothing-here", "/live/abc/index.m3u8", "/live/1.2/index.m3u8",
        ]
        for path in paths:
            with self.subTest(path=path):
                response = await self.client.get(path)
                self.assertEqual(response.status, 401)
                self.assertEqual(await response.json(), {"error": "unauthorized"})
        response = await self.client.delete(f"/api/clips/{clip}")
        self.assertEqual((response.status, await response.json()), (401, {"error": "unauthorized"}))

    async def test_the_secret_and_a_media_token_both_open_every_read_route(self):
        self.add_clip(0)
        clip = clip_id_at(NOW)
        for path in ("/api/status", "/api/clips", "/api/labels", f"/api/clips/{clip}",
                     f"/api/clips/{clip}/video", f"/api/clips/{clip}/thumb"):
            with self.subTest(path=path):
                self.assertEqual((await self.client.get(path, headers=self.bearer)).status, 200)
                self.assertEqual((await self.client.get(path, params={"t": self.token()})).status, 200)

    async def test_bad_credentials_are_rejected(self):
        expired = auth.make_token(SECRET, -1)
        exp, sig = self.token().split(".")
        credentials = [
            {"params": {"t": expired}},
            {"params": {"t": f"{exp}.{'A' * len(sig)}"}},
            {"params": {"t": auth.make_token("another-secret" * 4)}},
            {"params": {"t": "garbage"}},
            {"params": {"t": f"{exp}.é"}},
            {"params": {"t": "²." + sig}},
            {"params": {"t": "9" * 5000 + "." + sig}},
            {"headers": {"Authorization": "Bearer wrong"}},
            {"headers": {"Authorization": "Bearer é"}},
            {"headers": {"Authorization": f"Basic {SECRET}"}},
            {"headers": {"Authorization": f"Bearer {self.token()}"}},  # a token is not the secret
        ]
        for kwargs in credentials:
            with self.subTest(kwargs=str(kwargs)[:60]):
                response = await self.client.get("/api/status", **kwargs)
                self.assertEqual(response.status, 401)
                self.assertEqual(await response.json(), {"error": "unauthorized"})

    async def test_a_secret_shorter_than_32_characters_is_refused_at_startup(self):
        with self.assertRaises(SystemExit):
            make_app(Config(data_dir=self.dir, api_secret="short"))


class StatusTest(ApiTestCase):
    async def test_the_status_object(self):
        self.add_clip(0)
        status = await self.get_json("/api/status", headers=self.bearer)
        self.assertEqual(set(status), {
            "camera_online", "streaming", "recording", "identifying", "classifier", "current_clip",
            "last_detection", "retention_days", "clip_count", "disk_free_bytes", "disk_total_bytes",
            "server_time",
        })
        self.assertTrue(status["camera_online"])
        self.assertEqual(status["retention_days"], 30)
        self.assertEqual(status["clip_count"], 1)
        self.assertGreater(status["disk_total_bytes"], status["disk_free_bytes"] - 1)
        self.assertAlmostEqual(status["server_time"], time.time(), delta=5)

    async def test_nothing_to_report_is_null(self):
        status = await self.get_json("/api/status", headers=self.bearer)
        self.assertIsNone(status["current_clip"])
        self.assertIsNone(status["last_detection"])

    async def test_a_sighting_shows_identifying_and_recording_then_current_clip(self):
        self.set_vision(recording=True, identifying=True)
        status = await self.get_json("/api/status", headers=self.bearer)
        self.assertTrue(status["recording"] and status["identifying"])
        self.assertIsNone(status["current_clip"])

        self.set_vision(recording=True, identifying=False, current_clip="20261007T215601Z",
                        last_detection={"label": "red fox", "score": 0.99, "at": time.time()})
        status = await self.get_json("/api/status", headers=self.bearer)
        self.assertTrue(status["recording"])
        self.assertFalse(status["identifying"])
        self.assertEqual(status["current_clip"], "20261007T215601Z")
        self.assertEqual(status["last_detection"]["label"], "red fox")
        # ...and the clip is not listed until it has finished.
        self.assertEqual((await self.get_json("/api/clips", headers=self.bearer))["clips"], [])

    async def test_the_camera_is_offline_when_vision_stops_reporting(self):
        self.set_vision(age=60, recording=True, current_clip="x", streaming=True)
        status = await self.get_json("/api/status", headers=self.bearer)
        self.assertFalse(status["camera_online"])
        self.assertFalse(status["streaming"] or status["recording"] or status["identifying"])
        self.assertIsNone(status["current_clip"])
        self.config.status_path.unlink()
        self.assertFalse((await self.get_json("/api/status", headers=self.bearer))["camera_online"])
        self.config.status_path.write_text("{ not json")
        self.assertFalse((await self.get_json("/api/status", headers=self.bearer))["camera_online"])


class ClipListTest(ApiTestCase):
    async def test_paging_returns_every_clip_once_newest_first(self):
        expected = [self.add_clip(i * 60)["id"] for i in range(10)][::-1]
        seen, before = [], None
        for _ in range(10):
            params = {"limit": "3", **({"before": before} if before else {})}
            page = await self.get_json("/api/clips", params=params, headers=self.bearer)
            seen += [c["id"] for c in page["clips"]]
            before = page["next_before"]
            if before is None:
                break
            self.assertEqual(before, page["clips"][-1]["id"])
        self.assertEqual(seen, expected)
        self.assertIsNone(before)

    async def test_the_last_page_has_no_cursor_even_when_it_is_exactly_full(self):
        for i in range(6):
            self.add_clip(i * 60)
        page = await self.get_json("/api/clips", params={"limit": "3"}, headers=self.bearer)
        self.assertIsNotNone(page["next_before"])
        page = await self.get_json(
            "/api/clips", params={"limit": "3", "before": page["next_before"]}, headers=self.bearer
        )
        self.assertEqual(len(page["clips"]), 3)
        self.assertIsNone(page["next_before"])

    async def test_before_is_exclusive(self):
        ids = [self.add_clip(i * 60)["id"] for i in range(3)]
        page = await self.get_json("/api/clips", params={"before": ids[1]}, headers=self.bearer)
        self.assertEqual([c["id"] for c in page["clips"]], [ids[0]])

    async def test_deleting_a_clip_mid_scroll_does_not_break_the_next_page(self):
        ids = [self.add_clip(i * 60)["id"] for i in range(9)][::-1]
        first = await self.get_json("/api/clips", params={"limit": "3"}, headers=self.bearer)
        cursor = first["next_before"]
        self.assertEqual(cursor, ids[2])
        for gone in (cursor, ids[3]):  # the cursor clip itself, and the next one
            self.assertEqual(
                (await self.client.delete(f"/api/clips/{gone}", headers=self.bearer)).status, 200
            )
        rest = await self.get_json("/api/clips", params={"before": cursor}, headers=self.bearer)
        self.assertEqual([c["id"] for c in rest["clips"]], ids[4:])

    async def test_two_clips_in_one_second_page_correctly(self):
        base = clip_id_at(NOW)
        write_clip(self.config.clips_dir, base, started_at=NOW)
        write_clip(self.config.clips_dir, base + "-1", started_at=NOW + 0.5)
        write_clip(self.config.clips_dir, clip_id_at(NOW - 60), started_at=NOW - 60)
        page = await self.get_json("/api/clips", params={"limit": "1"}, headers=self.bearer)
        self.assertEqual([c["id"] for c in page["clips"]], [base + "-1"])
        page = await self.get_json(
            "/api/clips", params={"limit": "1", "before": page["next_before"]}, headers=self.bearer
        )
        self.assertEqual([c["id"] for c in page["clips"]], [base])

    async def test_limit_is_clamped_to_1_through_500_and_must_be_a_number(self):
        for i in range(3):
            self.add_clip(i * 60)
        for value, count in (("0", 1), ("-5", 1), ("2", 2), ("500", 3), ("100000", 3)):
            page = await self.get_json("/api/clips", params={"limit": value}, headers=self.bearer)
            self.assertEqual(len(page["clips"]), count, value)
        for value in ("abc", "1.5", ""):
            body = await self.get_json(
                "/api/clips", params={"limit": value}, headers=self.bearer, expect=400
            )
            self.assertIn("error", body)

    async def test_the_clip_object_is_returned_as_stored(self):
        stored = self.add_clip(0)
        page = await self.get_json("/api/clips", headers=self.bearer)
        self.assertEqual(page["clips"], [stored])
        self.assertEqual(await self.get_json(f"/api/clips/{stored['id']}", headers=self.bearer), stored)


class FilterTest(ApiTestCase):
    def add_all_kinds(self):
        self.fox = self.add_clip(0, label="red fox")
        self.person = self.add_clip(
            60, label="person", scientific="", category="person", labels={"person": 0.97}
        )
        self.both = self.add_clip(
            120, label="red fox",
            labels={"red fox": 0.99, "person": 0.97},
            label_categories={"red fox": "mammal", "person": "person"},
        )
        self.unnamed = self.add_clip(
            180, label="unidentified animal", scientific="", category="unknown", identified=False,
            labels={"red fox": 0.41, "dog": 0.4}, label_categories={"red fox": "unknown", "dog": "unknown"},
        )
        self.bird = self.add_clip(
            240, label="northern cardinal", scientific="Cardinalis cardinalis", category="bird"
        )

    async def ids(self, **params) -> set[str]:
        page = await self.get_json("/api/clips", params=params, headers=self.bearer)
        return {c["id"] for c in page["clips"]}

    async def test_label_matches_any_label_in_the_clip(self):
        self.add_all_kinds()
        self.assertEqual(await self.ids(label="person"), {self.person["id"], self.both["id"]})
        self.assertEqual(
            await self.ids(label="red fox"), {self.fox["id"], self.both["id"], self.unnamed["id"]}
        )
        self.assertEqual(await self.ids(label="nothing"), set())

    async def test_category_filters_use_the_categories_list(self):
        self.add_all_kinds()
        self.assertEqual(await self.ids(category="person"), {self.person["id"], self.both["id"]})
        self.assertEqual(await self.ids(category="mammal"), {self.fox["id"], self.both["id"]})
        self.assertEqual(await self.ids(category="bird"), {self.bird["id"]})
        self.assertEqual(await self.ids(category="unknown"), {self.unnamed["id"]})

    async def test_label_and_category_combine(self):
        self.add_all_kinds()
        self.assertEqual(await self.ids(label="red fox", category="person"), {self.both["id"]})

    async def test_each_label_count_equals_what_the_label_filter_returns(self):
        self.add_all_kinds()
        labels = (await self.get_json("/api/labels", headers=self.bearer))["labels"]
        self.assertEqual(
            {e["label"] for e in labels}, {"red fox", "person", "dog", "northern cardinal"}
        )
        for entry in labels:
            self.assertEqual(set(entry), {"label", "category", "count"})
            self.assertEqual(len(await self.ids(label=entry["label"])), entry["count"], entry)
        counts = [e["count"] for e in labels]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(labels[0]["label"], "red fox")

    async def test_a_label_keeps_its_real_category_next_to_an_unidentified_guess(self):
        self.add_clip(0, label="red fox")
        # The newest clip is an unidentified one that guessed "red fox".
        self.add_clip(
            600, label="unidentified animal", scientific="", category="unknown", identified=False,
            labels={"red fox": 0.4}, label_categories={"red fox": "unknown"},
        )
        labels = (await self.get_json("/api/labels", headers=self.bearer))["labels"]
        self.assertEqual(labels, [{"label": "red fox", "category": "mammal", "count": 2}])

    async def test_no_clips_means_no_labels(self):
        self.assertEqual(await self.get_json("/api/labels", headers=self.bearer), {"labels": []})


class ClipFileTest(ApiTestCase):
    async def test_unknown_ids_are_404_json(self):
        for path in ("/api/clips/20200101T000000Z", "/api/clips/20200101T000000Z/video",
                     "/api/clips/20200101T000000Z/download", "/api/clips/20200101T000000Z/thumb",
                     "/api/clips/not-an-id", "/api/clips/..%2Fetc"):
            with self.subTest(path=path):
                body = await self.get_json(path, headers=self.bearer, expect=404)
                self.assertEqual(body, {"error": "not found"})
        response = await self.client.delete("/api/clips/20200101T000000Z", headers=self.bearer)
        self.assertEqual((response.status, await response.json()), (404, {"error": "not found"}))

    async def test_unknown_routes_and_methods_are_json_too(self):
        self.assertEqual(
            await self.get_json("/api/nothing", headers=self.bearer, expect=404), {"error": "not found"}
        )
        response = await self.client.post("/api/status", headers=self.bearer)
        self.assertEqual(response.status, 405)
        self.assertEqual(await response.json(), {"error": "method not allowed"})

    async def test_video_supports_range_requests(self):
        clip = self.add_clip(0)
        data = (self.config.clips_dir / f"{clip['id']}.mp4").read_bytes()
        url = f"/api/clips/{clip['id']}/video"

        whole = await self.client.get(url, params={"t": self.token()})
        self.assertEqual(whole.status, 200)
        self.assertEqual(whole.headers["Content-Type"], "video/mp4")
        self.assertEqual(whole.headers["Accept-Ranges"], "bytes")
        self.assertEqual(await whole.read(), data)

        part = await self.client.get(url, params={"t": self.token()}, headers={"Range": "bytes=10-99"})
        self.assertEqual(part.status, 206)
        self.assertEqual(part.headers["Content-Range"], f"bytes 10-99/{len(data)}")
        self.assertEqual(part.headers["Content-Type"], "video/mp4")
        self.assertEqual(await part.read(), data[10:100])

        tail = await self.client.get(url, params={"t": self.token()}, headers={"Range": "bytes=100-"})
        self.assertEqual((tail.status, await tail.read()), (206, data[100:]))

        bad = await self.client.get(
            url, params={"t": self.token()}, headers={"Range": f"bytes={len(data) + 10}-"}
        )
        self.assertEqual(bad.status, 416)

    async def test_download_is_an_attachment_with_an_mp4_name(self):
        clip = self.add_clip(0, label="cooper's hawk", labels={"cooper's hawk": 0.98})
        response = await self.client.get(
            f"/api/clips/{clip['id']}/download", params={"t": self.token()}
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["Content-Type"], "video/mp4")
        self.assertEqual(
            response.headers["Content-Disposition"],
            f'attachment; filename="owl-cooper-s-hawk-{clip["id"]}.mp4"',
        )

    async def test_download_filenames_cannot_break_out_of_the_header(self):
        clip = self.add_clip(0, label='odd "name"\r\nX-Evil: 1 é', labels={"x": 1.0})
        response = await self.client.get(f"/api/clips/{clip['id']}/download", headers=self.bearer)
        disposition = response.headers["Content-Disposition"]
        self.assertEqual(disposition, f'attachment; filename="owl-odd-name-x-evil-1-{clip["id"]}.mp4"')
        self.assertNotIn("X-Evil", response.headers)

    async def test_thumbnail_is_a_jpeg(self):
        clip = self.add_clip(0)
        response = await self.client.get(f"/api/clips/{clip['id']}/thumb", params={"t": self.token()})
        self.assertEqual((response.status, response.headers["Content-Type"]), (200, "image/jpeg"))

    async def test_delete_removes_the_video_thumbnail_and_metadata(self):
        clip = self.add_clip(0)
        other = self.add_clip(60)
        response = await self.client.delete(f"/api/clips/{clip['id']}", headers=self.bearer)
        self.assertEqual((response.status, await response.json()), (200, {"deleted": clip["id"]}))
        self.assertEqual([p.name for p in self.config.clips_dir.glob(f"{clip['id']}*")], [])
        for suffix in ("", "/video", "/download", "/thumb"):
            await self.get_json(f"/api/clips/{clip['id']}{suffix}", headers=self.bearer, expect=404)
        listed = await self.get_json("/api/clips", headers=self.bearer)
        self.assertEqual([c["id"] for c in listed["clips"]], [other["id"]])
        self.assertEqual((await self.get_json("/api/status", headers=self.bearer))["clip_count"], 1)

    async def test_a_media_token_may_delete(self):
        # The website spec lets a token call DELETE (the site itself always sends the secret).
        clip = self.add_clip(0)
        response = await self.client.delete(f"/api/clips/{clip['id']}", params={"t": self.token()})
        self.assertEqual(response.status, 200)


class CorsTest(ApiTestCase):
    ORIGIN: ClassVar[dict] = {"Origin": "https://owl.example.app"}

    async def test_any_origin_is_allowed_on_every_kind_of_response(self):
        clip = self.add_clip(0)
        cases = [
            ("/api/health", {}), ("/api/status", self.bearer), ("/api/clips", {}),   # 401
            ("/api/clips/20200101T000000Z", self.bearer),                              # 404
            (f"/api/clips/{clip['id']}/video", self.bearer),
            (f"/api/clips/{clip['id']}/thumb", self.bearer),
        ]
        for path, headers in cases:
            with self.subTest(path=path):
                response = await self.client.get(path, headers={**self.ORIGIN, **headers})
                self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")
                self.assertIn("Content-Range", response.headers["Access-Control-Expose-Headers"])

    async def test_preflight(self):
        response = await self.client.options(
            "/api/clips", headers={**self.ORIGIN, "Access-Control-Request-Method": "GET",
                                   "Access-Control-Request-Headers": "authorization, range"},
        )
        self.assertEqual(response.status, 204)
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")
        self.assertIn("Range", response.headers["Access-Control-Allow-Headers"])
        self.assertIn("Authorization", response.headers["Access-Control-Allow-Headers"])


class OriginListTest(ApiTestCase):
    def config_overrides(self) -> dict:
        return {"cors_origins": ("https://owl.example.app",)}

    async def test_only_listed_origins_are_echoed(self):
        listed = await self.client.get("/api/health", headers={"Origin": "https://owl.example.app"})
        self.assertEqual(listed.headers["Access-Control-Allow-Origin"], "https://owl.example.app")
        self.assertEqual(listed.headers["Vary"], "Origin")
        other = await self.client.get("/api/health", headers={"Origin": "https://evil.example"})
        self.assertNotIn("Access-Control-Allow-Origin", other.headers)


class LiveTest(ApiTestCase):
    ORIGIN: ClassVar[dict] = {"Origin": "https://owl.example.app"}

    def url(self, path: str = "index.m3u8", token: str | None = None) -> str:
        return f"/live/{token or self.token()}/{path}"

    async def test_the_playlist_is_proxied_from_mediamtx(self):
        response = await self.client.get(self.url(), headers=self.ORIGIN)
        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["Content-Type"], "application/vnd.apple.mpegurl")
        self.assertEqual(await response.read(), PLAYLIST)
        self.assertEqual(self.upstream_requests, [("index.m3u8", "")])

    async def test_streamed_responses_carry_the_cors_headers(self):
        # The headers of a streamed response are sent as soon as it starts, so these have to be
        # set before that; hls.js cannot read the playlist or the segments without them.
        for path in ("index.m3u8", "main.m3u8", "seg0.mp4"):
            with self.subTest(path=path):
                response = await self.client.get(self.url(path), headers=self.ORIGIN)
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")

    async def test_playlists_are_never_cached_and_query_strings_pass_through_untouched(self):
        response = await self.client.get(self.url("main.m3u8") + "?_HLS_msn=12&_HLS_part=3&_HLS_skip=YES")
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(self.upstream_requests[-1][0], "main.m3u8")
        self.assertEqual(
            parse_qs(self.upstream_requests[-1][1]),
            {"_HLS_msn": ["12"], "_HLS_part": ["3"], "_HLS_skip": ["YES"]},
        )

    async def test_segments_are_proxied_as_is(self):
        response = await self.client.get(self.url("seg1.mp4"))
        self.assertEqual((response.status, await response.read()), (200, SEGMENT))

    async def test_the_token_is_checked_on_every_request_under_live(self):
        for path in ("index.m3u8", "main.m3u8", "seg1.mp4", "part2.mp4"):
            for token in (auth.make_token(SECRET, -1), "garbage", auth.make_token("another-secret" * 4)):
                with self.subTest(path=path, token=token[:12]):
                    response = await self.client.get(self.url(path, token))
                    self.assertEqual(response.status, 401)
                    self.assertEqual(await response.json(), {"error": "unauthorized"})
        self.assertEqual(self.upstream_requests, [])

    async def test_the_secret_also_opens_the_live_route(self):
        response = await self.client.get("/live/anything/index.m3u8", headers=self.bearer)
        self.assertEqual(response.status, 200)

    async def test_502_json_when_the_camera_is_offline(self):
        self.set_vision(age=60)
        response = await self.client.get(self.url(), headers=self.ORIGIN)
        self.assertEqual(response.status, 502)
        self.assertEqual(await response.json(), {"error": "camera offline"})
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")
        self.assertEqual(self.upstream_requests, [])

        self.config.status_path.unlink()
        self.assertEqual((await self.client.get(self.url())).status, 502)

        self.set_vision()  # the camera is back: no restart needed
        self.assertEqual((await self.client.get(self.url())).status, 200)

    async def test_502_json_when_mediamtx_has_no_stream(self):
        self.upstream_status["index.m3u8"] = 404  # what MediaMTX says while nothing is publishing
        response = await self.client.get(self.url(), headers=self.ORIGIN)
        self.assertEqual(response.status, 502)
        self.assertIn("error", await response.json())
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")
        del self.upstream_status["index.m3u8"]
        self.assertEqual((await self.client.get(self.url())).status, 200)

    async def test_a_missing_segment_is_an_ordinary_404(self):
        self.upstream_status["seg9.mp4"] = 404
        self.assertEqual((await self.client.get(self.url("seg9.mp4"))).status, 404)

    async def test_502_json_when_mediamtx_errors_or_is_unreachable(self):
        self.upstream_status["index.m3u8"] = 500
        self.assertEqual((await self.client.get(self.url())).status, 502)
        await self.upstream.close()
        response = await self.client.get(self.url(), headers=self.ORIGIN)
        self.assertEqual(response.status, 502)
        self.assertEqual(await response.json(), {"error": "stream unavailable"})
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")

    async def test_path_tricks_do_not_reach_other_mediamtx_paths(self):
        response = await self.client.get(f"/live/{self.token()}/..%2F..%2Fapi%2Fsecret")
        self.assertEqual(response.status, 404)
        self.assertEqual(self.upstream_requests, [])


if __name__ == "__main__":
    unittest.main()
