"""The camera service's safety nets (watchdog, disk warnings), notifications and the species list."""

import http.server
import tempfile
import threading
import time
import unittest
from collections import namedtuple
from pathlib import Path
from typing import ClassVar
from unittest import mock

import numpy as np

from owl import clips, species
from owl.classify import COCO_SCIENTIFIC, HintClassifier, square_crop
from owl.notify import Notifier
from owl.vision import DiskGuard, Watchdog

from .support import clip_id_at, make_config, write_clip

REPO = Path(__file__).resolve().parent.parent
Usage = namedtuple("Usage", "total used free")


class WatchdogTest(unittest.TestCase):
    def test_a_loop_that_keeps_beating_is_left_alone(self):
        stalled = threading.Event()
        dog = Watchdog(0.4, on_stall=stalled.set)
        dog.start()
        self.addCleanup(dog.stop)
        for _ in range(12):
            dog.beat()
            time.sleep(0.1)
        self.assertFalse(stalled.is_set())

    def test_a_stalled_loop_is_reported(self):
        stalled = threading.Event()
        dog = Watchdog(0.3, on_stall=stalled.set)
        dog.start()
        self.addCleanup(dog.stop)
        self.assertTrue(stalled.wait(3), "the watchdog never fired")

    def test_a_stopped_watchdog_stays_quiet(self):
        stalled = threading.Event()
        dog = Watchdog(0.2, on_stall=stalled.set)
        dog.start()
        dog.stop()
        self.assertFalse(stalled.wait(0.8))

    def test_by_default_a_stall_ends_the_process_with_a_failure_code(self):
        with mock.patch("owl.vision.os._exit") as exit_:
            Watchdog._exit()
        exit_.assert_called_once_with(70)


class FakeNotifier:
    def __init__(self):
        self.sent = []

    def send(self, title, message, snapshot=None, **kwargs):
        self.sent.append((title, message, kwargs))


class DiskGuardTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.config = make_config(Path(tmp.name), min_free_gb=5, low_disk_percent=10)
        self.config.clips_dir.mkdir(parents=True)
        self.notifier = FakeNotifier()
        self.guard = DiskGuard(self.config, self.notifier)

    def run_with_free(self, free_gb: float, total_gb: float = 100):
        with mock.patch.object(clips.shutil, "disk_usage", return_value=Usage(total_gb * 1e9, 0, free_gb * 1e9)):
            self.guard.run()

    def test_a_comfortable_disk_is_silent(self):
        self.run_with_free(50)
        self.assertEqual(self.notifier.sent, [])

    def test_below_10_percent_the_owner_is_warned_once_a_day(self):
        self.run_with_free(8)
        self.run_with_free(7)
        (title, message, _), = self.notifier.sent
        self.assertEqual(title, "Clip disk is almost full")
        self.assertIn("8%", message)
        self.assertIn("deleted early", message)
        with mock.patch("owl.vision.time.monotonic", return_value=time.monotonic() + 25 * 3600):
            self.run_with_free(7)
        self.assertEqual(len(self.notifier.sent), 2)

    def test_clips_deleted_early_are_announced(self):
        for i, age in enumerate((10, 9, 8)):
            started_at = time.time() - age * 86400
            write_clip(self.config.clips_dir, clip_id_at(started_at), started_at=started_at)
        def usage(path):  # 2 GB free with three clips, and each deleted clip frees 2 GB more
            left = len(list(self.config.clips_dir.glob("*.json")))
            return Usage(100e9, 0, (2 + 2 * (3 - left)) * 1e9)

        with mock.patch.object(clips.shutil, "disk_usage", side_effect=usage):
            self.guard.run()
        self.assertEqual(len(clips.list_clips(self.config.clips_dir)), 1)
        titles = [title for title, *_ in self.notifier.sent]
        self.assertIn("Clip disk is full", titles)
        message = next(m for t, m, _ in self.notifier.sent if t == "Clip disk is full")
        self.assertIn("2 oldest clips", message)
        self.assertIn("5 GB", message)


class NtfyHandler(http.server.BaseHTTPRequestHandler):
    received: ClassVar[list] = []

    def do_PUT(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).received.append((self.path, dict(self.headers), body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


class NotifierTest(unittest.TestCase):
    def setUp(self):
        NtfyHandler.received = []
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), NtfyHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def wait_for(self, count: int):
        deadline = time.time() + 5
        while len(NtfyHandler.received) < count and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(len(NtfyHandler.received), count)

    def test_an_alert_with_a_snapshot_opens_the_site_when_tapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "20261007T215601Z.jpg"
            snapshot.write_bytes(b"\xff\xd8 jpeg bytes")
            Notifier(self.url, "owl-topic", "", "https://owl.example.app").send(
                "Red fox spotted", "Red fox in the backyard (99% confident).", snapshot
            )
            self.wait_for(1)
        path, headers, body = NtfyHandler.received[0]
        self.assertEqual(path, "/owl-topic")
        self.assertEqual(headers["Click"], "https://owl.example.app")
        self.assertEqual(headers["Title"], "Red fox spotted")
        self.assertEqual(headers["Message"], "Red fox in the backyard (99% confident).")
        self.assertEqual(headers["Filename"], "20261007T215601Z.jpg")
        self.assertEqual(headers["Priority"], "high")
        self.assertEqual(body, b"\xff\xd8 jpeg bytes")
        self.assertNotIn("Authorization", headers)

    def test_a_plain_alert_and_an_access_token(self):
        Notifier(self.url, "owl-topic", "tk_secret", "").send(
            "Clip disk is full", "Free space fell.", None, tags="warning", priority="default"
        )
        self.wait_for(1)
        _, headers, body = NtfyHandler.received[0]
        self.assertEqual(body, b"Free space fell.")
        self.assertEqual((headers["Tags"], headers["Priority"]), ("warning", "default"))
        self.assertEqual(headers["Authorization"], "Bearer tk_secret")
        self.assertNotIn("Click", headers)

    def test_without_a_topic_nothing_is_sent(self):
        Notifier(self.url, "", "", "").send("x", "y")
        time.sleep(0.3)
        self.assertEqual(NtfyHandler.received, [])

    def test_an_unreachable_server_does_not_raise(self):
        self.server.shutdown()
        Notifier(self.url, "owl-topic", "", "").send("x", "y")
        time.sleep(0.3)


class SpeciesTest(unittest.TestCase):
    def test_the_shipped_list_is_valid_lower_case_and_complete(self):
        listed = species.load(REPO / "species.txt")
        self.assertGreater(len(listed), 50)
        for item in listed:
            self.assertEqual(item.common, item.common.lower())
            self.assertTrue(item.scientific)
            self.assertIn(item.category, ("mammal", "bird", "reptile", "amphibian", "pet"))

    def test_bad_lines_are_reported_with_their_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "species.txt"
            path.write_text("# comment\nmammal | red fox | Vulpes vulpes\nfish | trout | Salmo\n")
            with self.assertRaisesRegex(ValueError, r"species.txt:3"):
                species.load(path)
            path.write_text("mammal | Red Fox | Vulpes vulpes\nmammal | red fox | Vulpes vulpes\n")
            with self.assertRaisesRegex(ValueError, "listed twice"):
                species.load(path)

    def test_names_are_stored_lower_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "species.txt"
            path.write_text("bird | Northern Cardinal | Cardinalis cardinalis\n")
            self.assertEqual(species.load(path)[0].common, "northern cardinal")


class ClassifierHelpersTest(unittest.TestCase):
    def test_the_fallback_classifier_still_gives_scientific_names(self):
        classifier = HintClassifier(0.5)
        for label, scientific in COCO_SCIENTIFIC.items():
            prediction, = classifier.classify(np.zeros((8, 8, 3), np.uint8), (label, 0.9))
            self.assertEqual((prediction.label, prediction.scientific), (label, scientific))
        self.assertEqual(classifier.classify(np.zeros((8, 8, 3), np.uint8), ("dog", 0.9))[0].category, "pet")
        self.assertEqual(classifier.classify(np.zeros((8, 8, 3), np.uint8), ("bird", 0.9))[0].category, "bird")
        self.assertEqual(classifier.classify(np.zeros((8, 8, 3), np.uint8), None)[0].category, "none")

    def test_crops_are_square_and_stay_inside_the_frame(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        for box in ((0, 0, 50, 50), (1200, 650, 1280, 720), (500, 300, 900, 600), (0, 0, 1280, 720)):
            crop = square_crop(frame, box)
            self.assertEqual(crop.shape[0], crop.shape[1], box)
            self.assertLessEqual(crop.shape[0], 720)


if __name__ == "__main__":
    unittest.main()
