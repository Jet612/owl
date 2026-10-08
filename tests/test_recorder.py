"""What the recorder stores about a sighting, against the spec's table of clip shapes."""

import json
import tempfile
import time
import unittest
from pathlib import Path

import av
import cv2
import numpy as np

from owl import clips
from owl.classify import Prediction, Result
from owl.recorder import (
    MP4_OPTIONS,
    THUMB_WIDTH,
    UNIDENTIFIED,
    Recorder,
    Session,
    Sighting,
    _save_jpeg,
)

from .support import make_config, make_mp4, top_level_boxes

BOX = (400, 200, 700, 500)
CLIP_ID = "20261007T215601Z"


class FakeCircular:
    def __init__(self):
        self.opened = []

    def open_output(self, output):
        self.opened.append(output)

    def close_output(self):
        pass


class FakeNotifier:
    def __init__(self):
        self.sent = []

    def send(self, title, message, snapshot=None, **kwargs):
        self.sent.append((title, message, snapshot))


def frame() -> np.ndarray:
    """A smooth 1280x720 test picture, so its JPEG is a realistic size."""
    x = np.linspace(0, 255, 1280, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, 720, dtype=np.uint8)[:, None]
    return np.stack([np.broadcast_to(x, (720, 1280)), np.broadcast_to(y, (720, 1280)),
                     np.full((720, 1280), 90, dtype=np.uint8)], axis=-1).copy()


class RecorderTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.config = make_config(Path(tmp.name))
        self.config.clips_dir.mkdir(parents=True)
        self.circular, self.notifier = FakeCircular(), FakeNotifier()
        self.recorder = Recorder(self.config, self.circular, self.notifier, worker=None)

    def begin(self, now: float | None = None) -> Session:
        """Start a recording the way the camera loop does, with a real (tiny) file behind it."""
        self.now = now or time.time()
        self.recorder.start(self.now, frame())
        self.session = self.recorder._session
        make_mp4(self.config.clips_dir / f"{self.session.clip_id}.mp4.part", options=MP4_OPTIONS)
        return self.session

    def see(self, label, score, category, scientific="", votes=2):
        """Record a confirmed sighting the way the classifier's frames would."""
        self.session.best[label] = Sighting(score, frame(), BOX)
        self.session.votes[label] = votes
        self.session.confirmed[label] = Prediction(label, score, category, scientific)

    def finish(self) -> dict | None:
        self.recorder._finish(self.now + 25, "activity ended")
        meta = self.config.clips_dir / f"{self.session.clip_id}.json"
        return json.loads(meta.read_text()) if meta.exists() else None


class ClipShapeTest(RecorderTestCase):
    """One test per row of the spec's table of sightings."""

    def test_an_animal(self):
        self.begin()
        self.see("red fox", 0.99, "mammal", "Vulpes vulpes")
        meta = self.finish()
        self.assertEqual(
            {k: meta[k] for k in ("label", "scientific", "category", "identified", "labels",
                                   "label_categories", "categories")},
            {"label": "red fox", "scientific": "Vulpes vulpes", "category": "mammal",
             "identified": True, "labels": {"red fox": 0.99},
             "label_categories": {"red fox": "mammal"}, "categories": ["mammal"]},
        )

    def test_a_person(self):
        self.begin()
        self.see("person", 0.97, "person")
        meta = self.finish()
        self.assertEqual(
            {k: meta[k] for k in ("label", "scientific", "category", "identified", "labels",
                                   "label_categories", "categories")},
            {"label": "person", "scientific": "", "category": "person", "identified": True,
             "labels": {"person": 0.97}, "label_categories": {"person": "person"},
             "categories": ["person"]},
        )

    def test_a_person_and_a_fox_make_one_clip_and_the_animal_wins(self):
        self.begin()
        self.see("person", 0.97, "person")
        self.see("red fox", 0.99, "mammal", "Vulpes vulpes")
        meta = self.finish()
        self.assertEqual(meta["label"], "red fox")
        self.assertEqual(meta["scientific"], "Vulpes vulpes")
        self.assertEqual(meta["category"], "mammal")
        self.assertEqual(meta["labels"], {"red fox": 0.99, "person": 0.97})
        self.assertEqual(meta["label_categories"], {"red fox": "mammal", "person": "person"})
        self.assertEqual(meta["categories"], ["mammal", "person"])
        self.assertEqual(len(list(self.config.clips_dir.glob("*.json"))), 1)

    def test_an_animal_that_could_not_be_named(self):
        session = self.begin()
        session.guess_frames = 5
        session.guesses = {"red fox": 0.62, "coyote": 0.48}
        meta = self.finish()
        self.assertEqual(meta["label"], UNIDENTIFIED)
        self.assertEqual(meta["label"], "unidentified animal")
        self.assertEqual(
            {k: meta[k] for k in ("scientific", "category", "identified", "categories")},
            {"scientific": "", "category": "unknown", "identified": False, "categories": ["unknown"]},
        )
        self.assertEqual(meta["labels"], {"red fox": 0.62, "coyote": 0.48})
        self.assertEqual(set(meta["label_categories"].values()), {"unknown"})

    def test_an_animal_only_the_hailo_detector_noticed_is_kept_as_unidentified(self):
        session = self.begin()
        session.detector = {"dog": 0.55}
        meta = self.finish()
        self.assertEqual((meta["label"], meta["identified"], meta["labels"]), (UNIDENTIFIED, False, {"dog": 0.55}))

    def test_nothing_real_means_no_clip(self):
        session = self.begin()
        session.negatives = 6
        self.assertIsNone(self.finish())
        self.assertEqual(list(self.config.clips_dir.iterdir()), [])
        self.assertGreater(self.recorder.quiet_until, self.now)

    def test_unidentified_clips_can_be_switched_off(self):
        config = make_config(self.config.data_dir, unidentified_days=0)
        self.recorder = Recorder(config, self.circular, self.notifier, worker=None)
        session = self.begin()
        session.guess_frames = 5
        session.guesses = {"red fox": 0.6}
        self.assertIsNone(self.finish())


class ClipFieldsTest(RecorderTestCase):
    def test_times_duration_size_and_expiry(self):
        self.begin()
        self.see("red fox", 0.99, "mammal", "Vulpes vulpes")
        meta = self.finish()
        self.assertEqual(meta["id"], self.session.clip_id)
        self.assertEqual(meta["size_bytes"], (self.config.clips_dir / f"{meta['id']}.mp4").stat().st_size)
        self.assertAlmostEqual(meta["duration"], 2.0, delta=0.3)
        self.assertAlmostEqual(meta["ended_at"] - meta["started_at"], meta["duration"], delta=0.1)
        self.assertEqual(meta["ended_at"], round(self.now + 25 - self.config.pre_roll, 3))

    def test_identified_clips_expire_30_days_after_they_start(self):
        self.begin()
        self.see("red fox", 0.99, "mammal", "Vulpes vulpes")
        meta = self.finish()
        self.assertEqual(meta["expires_at"], round(meta["started_at"] + 30 * 86400, 3))

    def test_unidentified_clips_expire_3_days_after_they_start(self):
        session = self.begin()
        session.guess_frames, session.guesses = 5, {"red fox": 0.6}
        meta = self.finish()
        self.assertEqual(meta["expires_at"], round(meta["started_at"] + 3 * 86400, 3))

    def test_only_the_documented_fields_are_stored(self):
        self.begin()
        self.see("red fox", 0.99, "mammal", "Vulpes vulpes")
        self.assertEqual(set(self.finish()), {
            "id", "started_at", "ended_at", "expires_at", "duration", "label", "scientific",
            "category", "identified", "labels", "label_categories", "categories", "size_bytes",
        })

    def test_the_clip_is_listed_only_after_it_has_finished(self):
        self.begin()
        self.see("red fox", 0.99, "mammal", "Vulpes vulpes")
        self.assertEqual(clips.list_clips(self.config.clips_dir), [])
        self.finish()
        self.assertEqual(len(clips.list_clips(self.config.clips_dir)), 1)

    def test_a_clip_leaves_its_video_thumbnail_and_metadata_and_no_partial_file(self):
        self.begin()
        self.see("red fox", 0.99, "mammal", "Vulpes vulpes")
        self.finish()
        self.assertEqual(
            sorted(p.suffix for p in self.config.clips_dir.iterdir()), [".jpg", ".json", ".mp4"]
        )

    def test_the_recording_is_started_as_an_mp4_with_the_index_up_front(self):
        self.begin()
        output, = self.circular.opened
        self.assertEqual(output._format, "mp4")
        self.assertEqual(output._options, {"movflags": "+faststart"})
        self.assertTrue(output._output_name.endswith(f"{self.session.clip_id}.mp4.part"))

    def test_the_stored_video_has_its_index_up_front_and_plays(self):
        self.begin()
        self.see("red fox", 0.99, "mammal", "Vulpes vulpes")
        self.finish()
        video = self.config.clips_dir / f"{self.session.clip_id}.mp4"
        boxes = top_level_boxes(video)
        self.assertLess(boxes.index("moov"), boxes.index("mdat"))
        with av.open(str(video)) as container:
            self.assertEqual(container.streams.video[0].codec_context.name, "mpeg4")


class StatusTest(RecorderTestCase):
    def test_recording_and_identifying_then_current_clip(self):
        self.assertEqual(self.recorder.status(), {
            "recording": False, "current_clip": None, "last_detection": None, "identifying": False,
        })
        self.begin()
        self.assertEqual(
            self.recorder.status(),
            {"recording": True, "current_clip": None, "last_detection": None, "identifying": True},
        )
        self.recorder._confirm(self.session, Prediction("red fox", 0.99, "mammal", "Vulpes vulpes"))
        status = self.recorder.status()
        self.assertEqual((status["recording"], status["identifying"]), (True, False))
        self.assertEqual(status["current_clip"], self.session.clip_id)
        self.assertEqual(status["last_detection"]["label"], "red fox")
        self.assertEqual(status["last_detection"]["score"], 0.99)
        self.assertAlmostEqual(status["last_detection"]["at"], time.time(), delta=5)
        self.see("red fox", 0.99, "mammal", "Vulpes vulpes")
        self.finish()
        self.assertEqual(self.recorder.status()["recording"], False)
        self.assertIsNone(self.recorder.status()["current_clip"])


class IdentificationTest(RecorderTestCase):
    def result(self, label, score, category="mammal", scientific=""):
        return Result(
            self.session.id, [Prediction(label, score, category, scientific)],
            self.config.species_min_score, frame(), BOX,
        )

    def test_a_species_needs_two_clear_frames(self):
        self.begin()
        self.recorder.handle(self.result("red fox", 0.97, scientific="Vulpes vulpes"))
        self.assertEqual(self.session.confirmed, {})
        self.assertEqual(self.notifier.sent, [])
        self.recorder.handle(self.result("red fox", 0.98, scientific="Vulpes vulpes"))
        self.assertIn("red fox", self.session.confirmed)
        self.assertEqual(len(self.notifier.sent), 1)

    def test_unsure_frames_only_count_as_guesses(self):
        self.begin()
        for _ in range(5):
            self.recorder.handle(self.result("red fox", 0.7))
        self.assertEqual(self.session.confirmed, {})
        self.assertEqual(self.session.guess_frames, 5)
        self.assertEqual(self.session.guesses, {"red fox": 0.7})

    def test_frames_of_nothing_outvote_a_few_lucky_frames(self):
        self.begin()
        for _ in range(4):
            self.recorder.handle(self.result("empty scene", 0.9, category="none"))
        self.recorder.handle(self.result("red fox", 0.97))
        self.recorder.handle(self.result("red fox", 0.97))
        self.assertEqual(self.session.confirmed, {})

    def test_the_classifier_never_confirms_a_person(self):
        self.begin()
        for _ in range(4):
            self.recorder.handle(self.result("person", 0.99, category="person"))
        self.assertEqual(self.session.confirmed, {})

    def test_a_person_is_confirmed_by_the_detector_alone(self):
        class Grab:
            def rgb(self):
                return frame()

        self.begin()
        self.recorder.observe_person(0.93, BOX, Grab())
        self.assertEqual(self.session.confirmed["person"].category, "person")
        self.assertEqual(self.notifier.sent[0][0], "Person spotted")

    def test_the_notification_names_the_animal_and_carries_the_snapshot(self):
        self.begin()
        self.recorder._confirm(self.session, Prediction("red fox", 0.99, "mammal", "Vulpes vulpes"))
        (title, message, snapshot), = self.notifier.sent
        self.assertEqual(title, "Red fox spotted")
        self.assertIn("99%", message)
        self.assertEqual(snapshot, self.config.clips_dir / f"{self.session.clip_id}.jpg")

    def test_one_notification_per_species_in_the_cooldown_and_muted_species_stay_quiet(self):
        config = make_config(self.config.data_dir, notify_mute=("eastern gray squirrel",))
        self.recorder = Recorder(config, self.circular, self.notifier, worker=None)
        self.begin()
        for label in ("red fox", "red fox", "eastern gray squirrel", "white-tailed deer"):
            self.recorder._notify(self.session, Prediction(label, 0.99, "mammal"))
        self.assertEqual([t for t, *_ in self.notifier.sent], ["Red fox spotted", "White-tailed deer spotted"])


class ThumbnailTest(unittest.TestCase):
    def test_snapshots_are_small_with_the_box_and_label_drawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.jpg"
            _save_jpeg(path, frame(), BOX, "red fox 99%")
            image = cv2.imread(str(path))
            self.assertEqual((image.shape[1], image.shape[0]), (THUMB_WIDTH, 360))
            self.assertLess(path.stat().st_size, 60_000)
            # The box edge, scaled by 640/1280, is green.
            x0, y0, *_ = (v // 2 for v in BOX)
            self.assertGreater(int(image[y0 + 40, x0 - 1 : x0 + 2, 1].max()), 200)
            self.assertLess(int(image[y0 + 40, x0 - 1 : x0 + 2, 2].min()), 60)

    def test_a_snapshot_without_a_box_is_just_the_scaled_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.jpg"
            _save_jpeg(path, frame())
            self.assertEqual(cv2.imread(str(path)).shape[:2], (360, 640))

    def test_a_small_frame_is_not_enlarged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.jpg"
            _save_jpeg(path, frame()[:240, :320])
            self.assertEqual(cv2.imread(str(path)).shape[:2], (240, 320))


class Mp4IndexTest(unittest.TestCase):
    def test_faststart_puts_the_index_before_the_video_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain, fast = Path(tmp) / "plain.mp4", Path(tmp) / "fast.mp4"
            make_mp4(plain)
            make_mp4(fast, options=MP4_OPTIONS)
            plain_boxes, fast_boxes = top_level_boxes(plain), top_level_boxes(fast)
            self.assertGreater(plain_boxes.index("moov"), plain_boxes.index("mdat"))
            self.assertLess(fast_boxes.index("moov"), fast_boxes.index("mdat"))


if __name__ == "__main__":
    unittest.main()
