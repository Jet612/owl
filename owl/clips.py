"""On-disk clip library: <id>.mp4, <id>.jpg and <id>.json per clip, all in one directory."""

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path

LOG = logging.getLogger(__name__)

# Clip ids are UTC timestamps, so sorting them sorts clips by start time.
CLIP_ID = re.compile(r"^\d{8}T\d{6}Z(-\d+)?$")

# Files without a finished clip around them (a recording the power cut interrupted) are removed
# once they are this old. A recording in progress is far younger than this.
STALE_AFTER = 3600


def new_clip_id(clips_dir: Path, started_at: float) -> str:
    base = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(started_at))
    clip_id, n = base, 1
    while (clips_dir / f"{clip_id}.json").exists() or (clips_dir / f"{clip_id}.mp4.part").exists():
        clip_id, n = f"{base}-{n}", n + 1
    return clip_id


def write_json(path: Path, data: dict) -> None:
    """Write atomically so readers never see a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def list_clips(clips_dir: Path) -> list[dict]:
    """Finished clips, newest first. A clip is finished once its .json exists."""
    clips = []
    # Sort by id, not file name: "<id>-1.json" sorts before "<id>.json" but is the newer clip.
    for meta_path in sorted(clips_dir.glob("*.json"), key=lambda path: path.stem, reverse=True):
        try:
            clip = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            LOG.warning("Skipping unreadable clip metadata %s", meta_path)
            continue
        if not isinstance(clip, dict) or clip.get("id") != meta_path.stem \
                or not isinstance(clip.get("started_at"), (int, float)):
            LOG.warning("Skipping malformed clip metadata %s", meta_path)
            continue
        clips.append(clip)
    return clips


def clip_files(clips_dir: Path, clip_id: str) -> list[Path]:
    return [clips_dir / f"{clip_id}{ext}" for ext in (".json", ".mp4", ".jpg")]


def delete_clip(clips_dir: Path, clip_id: str) -> bool:
    """Delete a clip's metadata, video and thumbnail together. False if there was nothing to delete."""
    if not CLIP_ID.match(clip_id):
        return False
    found = False
    for path in clip_files(clips_dir, clip_id):
        try:
            path.unlink()
            found = True
        except FileNotFoundError:
            pass
    return found


def low_disk_warning(clips_dir: Path, percent: float) -> str | None:
    """A sentence for the owner if the disk holding the clips has less than `percent` free."""
    usage = shutil.disk_usage(clips_dir)
    free_percent = usage.free * 100 / usage.total
    if free_percent >= percent:
        return None
    return (
        f"Only {free_percent:.0f}% of the clip disk is free "
        f"({usage.free / 1e9:.1f} GB of {usage.total / 1e9:.0f} GB)."
    )


def prune(clips_dir: Path, retention_days: int, min_free_gb: float) -> list[dict]:
    """Delete expired clips, then the oldest clips while the disk is short of space.

    Returns the clips deleted early for lack of space, so the owner can be told about them.
    """
    now = time.time()
    clips = sorted(list_clips(clips_dir), key=lambda c: c["started_at"])
    remaining = []
    for clip in clips:
        # Each clip says when it expires; older clips without that use the default retention.
        if now > clip.get("expires_at", clip["started_at"] + retention_days * 86400):
            LOG.info("Deleting clip %s (expired)", clip["id"])
            delete_clip(clips_dir, clip["id"])
        else:
            remaining.append(clip)
    early = []
    while remaining and shutil.disk_usage(clips_dir).free < min_free_gb * 1e9:
        clip = remaining.pop(0)
        LOG.warning("Low disk space, deleting oldest clip %s", clip["id"])
        delete_clip(clips_dir, clip["id"])
        early.append(clip)

    _remove_stale_files(clips_dir, now)
    return early


def _remove_stale_files(clips_dir: Path, now: float) -> None:
    """Remove what a crash or power loss leaves behind: unfinished recordings and orphaned files."""
    for path in clips_dir.iterdir():
        if path.suffix == ".json":
            continue
        try:
            if path.stat().st_mtime >= now - STALE_AFTER:
                continue
            # A video or thumbnail is only kept with the .json that makes it a finished clip.
            if path.suffix in (".mp4", ".jpg") and (clips_dir / f"{path.stem}.json").exists():
                continue
            if path.suffix in (".mp4", ".jpg", ".part", ".tmp"):
                LOG.info("Removing stale file %s", path.name)
                path.unlink(missing_ok=True)
        except OSError as exc:
            LOG.warning("Could not clean up %s: %s", path.name, exc)
