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
    for meta_path in sorted(clips_dir.glob("*.json"), reverse=True):
        try:
            clips.append(json.loads(meta_path.read_text()))
        except (OSError, ValueError):
            LOG.warning("Skipping unreadable clip metadata %s", meta_path)
    return clips


def clip_files(clips_dir: Path, clip_id: str) -> list[Path]:
    return [clips_dir / f"{clip_id}{ext}" for ext in (".json", ".mp4", ".jpg")]


def delete_clip(clips_dir: Path, clip_id: str) -> bool:
    found = False
    for path in clip_files(clips_dir, clip_id):
        try:
            path.unlink()
            found = True
        except FileNotFoundError:
            pass
    return found


def prune(clips_dir: Path, retention_days: int, min_free_gb: float) -> None:
    """Delete expired clips, then the oldest clips while the disk is short of space."""
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
    while remaining and shutil.disk_usage(clips_dir).free < min_free_gb * 1e9:
        clip = remaining.pop(0)
        LOG.warning("Low disk space, deleting oldest clip %s", clip["id"])
        delete_clip(clips_dir, clip["id"])

    # Recordings interrupted by a crash or power loss leave .part files behind.
    for part in clips_dir.glob("*.part"):
        if part.stat().st_mtime < now - 3600:
            part.unlink(missing_ok=True)
