"""Helpers shared by the tests: a throwaway config, fake clips and a tiny real MP4."""

import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from owl import clips
from owl.config import Config

SECRET = "s3cret-" + "x" * 40


def make_config(data_dir: Path, **overrides) -> Config:
    return Config(data_dir=data_dir, api_secret=SECRET, ntfy_topic="", **overrides)


def clip_id_at(started_at: float) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(started_at))


def write_clip(
    clips_dir: Path,
    clip_id: str,
    *,
    started_at: float,
    label: str = "red fox",
    scientific: str = "Vulpes vulpes",
    category: str = "mammal",
    identified: bool = True,
    labels: dict[str, float] | None = None,
    label_categories: dict[str, str] | None = None,
    video: bytes | None = None,
    expires_at: float | None = None,
) -> dict:
    """Write a finished clip (.json, .mp4 and .jpg) shaped like the ones the recorder makes."""
    labels = labels if labels is not None else {label: 0.99}
    label_categories = label_categories if label_categories is not None else {
        name: category for name in labels
    }
    video = video if video is not None else clip_id.encode() * 50
    clips_dir.mkdir(parents=True, exist_ok=True)
    (clips_dir / f"{clip_id}.mp4").write_bytes(video)
    (clips_dir / f"{clip_id}.jpg").write_bytes(b"\xff\xd8\xff\xe0 fake jpeg")
    meta = {
        "id": clip_id,
        "started_at": started_at,
        "ended_at": started_at + 10,
        "duration": 10.0,
        "label": label,
        "scientific": scientific,
        "category": category,
        "identified": identified,
        "labels": labels,
        "label_categories": label_categories,
        "categories": sorted(set(label_categories.values())),
        "size_bytes": len(video),
    }
    if expires_at is not None:
        meta["expires_at"] = expires_at
    clips.write_json(clips_dir / f"{clip_id}.json", meta)
    return meta


def make_mp4(path: Path, seconds: int = 2, fps: int = 10, options: dict | None = None) -> None:
    """A real, playable MP4 (MPEG-4 video; the test machine's ffmpeg may lack an H.264 encoder)."""
    with av.open(str(path), "w", format="mp4", options=options) as container:
        stream = container.add_stream("mpeg4", rate=Fraction(fps))
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(seconds * fps):
            image = np.full((48, 64, 3), (i * 7) % 256, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def top_level_boxes(path: Path) -> list[str]:
    """The names of an MP4's top-level boxes, in file order."""
    data = path.read_bytes()
    names, offset = [], 0
    while offset + 8 <= len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        names.append(data[offset + 4 : offset + 8].decode("latin-1"))
        if size < 8:
            break
        offset += size
    return names
