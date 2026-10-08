"""The clip library on disk: ordering, retention, early deletion and leftovers."""

import os
import tempfile
import time
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

from owl import clips

from .support import clip_id_at, write_clip

DAY = 86400
Usage = namedtuple("Usage", "total used free")


class ClipsTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def add(self, age_days: float, **kwargs) -> dict:
        started_at = time.time() - age_days * DAY
        return write_clip(self.dir, clip_id_at(started_at), started_at=started_at, **kwargs)

    def ids(self) -> list[str]:
        return [c["id"] for c in clips.list_clips(self.dir)]

    def age_file(self, name: str, seconds: float) -> None:
        old = time.time() - seconds
        os.utime(self.dir / name, (old, old))


class ListTest(ClipsTestCase):
    def test_newest_first_with_same_second_clips_in_creation_order(self):
        base = clip_id_at(1_791_400_000)
        for suffix, offset in (("", 0), ("-1", 0.5), ("-2", 0.9)):
            write_clip(self.dir, base + suffix, started_at=1_791_400_000 + offset)
        write_clip(self.dir, clip_id_at(1_791_399_000), started_at=1_791_399_000)
        self.assertEqual(self.ids()[:3], [base + "-2", base + "-1", base])

    def test_a_clip_is_listed_only_once_its_metadata_exists(self):
        (self.dir / "20261007T000000Z.mp4.part").write_bytes(b"recording")
        (self.dir / "20261007T000000Z.jpg").write_bytes(b"placeholder")
        self.assertEqual(self.ids(), [])

    def test_unreadable_and_malformed_metadata_is_skipped(self):
        good = self.add(1)
        (self.dir / "20200101T000000Z.json").write_text("{ nope")
        (self.dir / "20200102T000000Z.json").write_text("[]")
        (self.dir / "20200103T000000Z.json").write_text('{"id": "other", "started_at": 1}')
        (self.dir / "20200104T000000Z.json").write_text('{"id": "20200104T000000Z"}')
        self.assertEqual(self.ids(), [good["id"]])


class DeleteTest(ClipsTestCase):
    def test_removes_all_three_files_and_reports_whether_anything_was_there(self):
        clip = self.add(1)
        self.assertTrue(clips.delete_clip(self.dir, clip["id"]))
        self.assertEqual(list(self.dir.iterdir()), [])
        self.assertFalse(clips.delete_clip(self.dir, clip["id"]))

    def test_ids_that_are_not_clip_ids_are_refused(self):
        (self.dir.parent / "victim.json").write_text("{}")
        self.addCleanup((self.dir.parent / "victim.json").unlink)
        for bad in ("../victim", "", "a/b", "20261007T215601Z/../x", "*"):
            self.assertFalse(clips.delete_clip(self.dir, bad), bad)
        self.assertTrue((self.dir.parent / "victim.json").exists())


class RetentionTest(ClipsTestCase):
    def prune(self, retention_days=30):
        with mock.patch.object(clips.shutil, "disk_usage", return_value=Usage(100e9, 1e9, 99e9)):
            return clips.prune(self.dir, retention_days, min_free_gb=5)

    def test_clips_older_than_30_days_by_started_at_are_deleted(self):
        old, new = self.add(31), self.add(29)
        self.assertEqual(self.prune(), [])
        self.assertEqual(self.ids(), [new["id"]])
        self.assertEqual([p.suffix for p in sorted(self.dir.glob(f"{old['id']}*"))], [])

    def test_the_clip_count_drops_to_match(self):
        for age in (35, 33, 10, 1):
            self.add(age)
        self.prune()
        self.assertEqual(len(list(self.dir.glob("*.json"))), 2)

    def test_a_clip_says_when_it_expires(self):
        now = time.time()
        short = self.add(5, expires_at=now - 10)    # an unidentified clip kept for 3 days
        long = self.add(4, expires_at=now + 100)
        self.prune()
        self.assertEqual(self.ids(), [long["id"]])
        self.assertFalse((self.dir / f"{short['id']}.mp4").exists())

    def test_retention_days_applies_to_clips_without_an_expiry(self):
        self.add(10)
        self.prune(retention_days=7)
        self.assertEqual(self.ids(), [])


class EarlyDeletionTest(ClipsTestCase):
    def test_oldest_clips_go_first_when_the_disk_is_short_of_space(self):
        made = [self.add(age) for age in (9, 8, 7, 6, 5, 4, 3, 2, 1, 0.5)]

        def usage(path):  # every clip on disk takes 1 GB of a 20 GB disk
            return Usage(20e9, 0, 20e9 - 1e9 * len(list(self.dir.glob("*.json"))))

        with mock.patch.object(clips.shutil, "disk_usage", side_effect=usage):
            deleted = clips.prune(self.dir, 30, min_free_gb=12)  # 10 GB free now; needs 12

        self.assertEqual([c["id"] for c in deleted], [made[0]["id"], made[1]["id"]])
        self.assertEqual(self.ids(), [c["id"] for c in made[2:]][::-1])

    def test_nothing_is_deleted_early_when_there_is_room(self):
        self.add(1)
        with mock.patch.object(clips.shutil, "disk_usage", return_value=Usage(100e9, 0, 50e9)):
            self.assertEqual(clips.prune(self.dir, 30, min_free_gb=5), [])
        self.assertEqual(len(self.ids()), 1)


class LeftoversTest(ClipsTestCase):
    def prune(self):
        with mock.patch.object(clips.shutil, "disk_usage", return_value=Usage(100e9, 0, 99e9)):
            clips.prune(self.dir, 30, min_free_gb=5)

    def test_files_of_an_interrupted_recording_are_removed_once_stale(self):
        for name in ("a.mp4.part", "20261001T000000Z.jpg", "20261001T000000Z.json.tmp", "x.mp4"):
            (self.dir / name).write_bytes(b"x")
            self.age_file(name, 2 * 3600)
        self.prune()
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_a_recording_in_progress_is_left_alone(self):
        for name in ("20261007T215601Z.mp4.part", "20261007T215601Z.jpg"):
            (self.dir / name).write_bytes(b"x")
        self.prune()
        self.assertEqual(len(list(self.dir.iterdir())), 2)

    def test_the_files_of_a_finished_clip_are_never_treated_as_leftovers(self):
        clip = self.add(1)
        for suffix in (".mp4", ".jpg", ".json"):
            self.age_file(clip["id"] + suffix, 5 * 3600)
        self.prune()
        self.assertEqual(self.ids(), [clip["id"]])
        self.assertEqual(len(list(self.dir.iterdir())), 3)


class LowDiskTest(ClipsTestCase):
    def warning(self, free_gb: float, total_gb: float = 100, percent: float = 10):
        with mock.patch.object(clips.shutil, "disk_usage", return_value=Usage(total_gb * 1e9, 0, free_gb * 1e9)):
            return clips.low_disk_warning(self.dir, percent)

    def test_warns_below_the_threshold_only(self):
        self.assertIsNone(self.warning(free_gb=10))
        self.assertIsNone(self.warning(free_gb=50))
        message = self.warning(free_gb=8)
        self.assertIn("8%", message)
        self.assertIn("8.0 GB of 100 GB", message)


if __name__ == "__main__":
    unittest.main()
