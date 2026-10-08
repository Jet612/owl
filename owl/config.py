"""Settings shared by the vision and API services, read from OWL_* environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in _str(name, default).split(",") if item.strip())


@dataclass(frozen=True)
class Config:
    data_dir: Path = field(default_factory=lambda: Path(_str("OWL_DATA_DIR", "/var/lib/owl")))
    log_level: str = field(default_factory=lambda: _str("OWL_LOG_LEVEL", "INFO"))

    # Camera and encoding
    rtsp_url: str = field(default_factory=lambda: _str("OWL_RTSP_URL", "rtsp://127.0.0.1:8554/owl"))
    width: int = field(default_factory=lambda: _int("OWL_VIDEO_WIDTH", 1280))
    height: int = field(default_factory=lambda: _int("OWL_VIDEO_HEIGHT", 720))
    fps: int = field(default_factory=lambda: _int("OWL_VIDEO_FPS", 30))
    bitrate: int = field(default_factory=lambda: _int("OWL_VIDEO_BITRATE", 2_500_000))
    hflip: bool = field(default_factory=lambda: _bool("OWL_HFLIP", False))
    vflip: bool = field(default_factory=lambda: _bool("OWL_VFLIP", False))

    # Triggers. A recording starts when the Hailo detector sees a person or an animal-like
    # object (COCO classes), or when something moves. Deer, foxes and the like aren't COCO
    # classes; they are usually caught by movement and then named by the classifier.
    model_path: str = field(
        default_factory=lambda: _str("OWL_MODEL", "/usr/share/hailo-models/yolov8s_h8l.hef")
    )
    animals: tuple[str, ...] = field(
        default_factory=lambda: _list("OWL_ANIMALS", "bird,cat,dog,horse,sheep,cow,bear")
    )
    animal_min_score: float = field(default_factory=lambda: _float("OWL_ANIMAL_MIN_SCORE", 0.35))
    detect_people: bool = field(default_factory=lambda: _bool("OWL_DETECT_PEOPLE", True))
    person_min_score: float = field(default_factory=lambda: _float("OWL_PERSON_MIN_SCORE", 0.5))
    detect_fps: float = field(default_factory=lambda: _float("OWL_DETECT_FPS", 10))
    # The detector must fire in this many of the last `confirm_window` frames.
    confirm_hits: int = field(default_factory=lambda: _int("OWL_CONFIRM_HITS", 3))
    confirm_window: int = field(default_factory=lambda: _int("OWL_CONFIRM_WINDOW", 5))
    motion: bool = field(default_factory=lambda: _bool("OWL_MOTION", True))
    # Smallest moving blob, as a fraction of the frame. Lower it to catch smaller or farther animals.
    motion_min_area: float = field(default_factory=lambda: _float("OWL_MOTION_MIN_AREA", 0.0015))
    motion_hits: int = field(default_factory=lambda: _int("OWL_MOTION_HITS", 4))
    motion_window: int = field(default_factory=lambda: _int("OWL_MOTION_WINDOW", 6))

    # Species identification. "bioclip" names the species; "none" trusts the Hailo labels.
    classifier: str = field(default_factory=lambda: _str("OWL_CLASSIFIER", "bioclip").lower())
    species_file: str = field(
        default_factory=lambda: _str("OWL_SPECIES_FILE", "/etc/owl/species.txt")
    )
    classify_threads: int = field(default_factory=lambda: _int("OWL_CLASSIFY_THREADS", 3))
    # A frame counts for a species at this confidence, and the species is confirmed at this many
    # frames. The classifier is right about 97% of the time at 0.95 and under 40% below 0.9.
    species_min_score: float = field(default_factory=lambda: _float("OWL_SPECIES_MIN_SCORE", 0.95))
    species_confirm_frames: int = field(
        default_factory=lambda: _int("OWL_SPECIES_CONFIRM_FRAMES", 2)
    )
    # Once something is confirmed, look again only this often (seconds), to catch a second animal.
    confirmed_interval: float = field(default_factory=lambda: _float("OWL_CONFIRMED_INTERVAL", 5))
    # Keep a clip nothing could be named in if the classifier thought it saw an animal (at least
    # this confident) in this many frames, for a few days. 0 days throws such clips away.
    unidentified_min_frames: int = field(
        default_factory=lambda: _int("OWL_UNIDENTIFIED_MIN_FRAMES", 3)
    )
    unidentified_min_score: float = field(
        default_factory=lambda: _float("OWL_UNIDENTIFIED_MIN_SCORE", 0.5)
    )
    unidentified_days: int = field(default_factory=lambda: _int("OWL_UNIDENTIFIED_DAYS", 3))
    # Give up on a trigger after this many "nothing there" frames in a row.
    abort_negatives: int = field(default_factory=lambda: _int("OWL_ABORT_NEGATIVES", 4))
    # After a false alarm, ignore movement-only triggers for this long (seconds).
    quiet_after_false: float = field(default_factory=lambda: _float("OWL_QUIET_AFTER_FALSE", 30))

    # Recording
    pre_roll: float = field(default_factory=lambda: _float("OWL_PRE_ROLL", 5))
    post_roll: float = field(default_factory=lambda: _float("OWL_POST_ROLL", 10))
    max_clip: float = field(default_factory=lambda: _float("OWL_MAX_CLIP", 120))
    retention_days: int = field(default_factory=lambda: _int("OWL_RETENTION_DAYS", 30))
    min_free_gb: float = field(default_factory=lambda: _float("OWL_MIN_FREE_GB", 5))

    # Notifications
    ntfy_server: str = field(default_factory=lambda: _str("OWL_NTFY_SERVER", "https://ntfy.sh"))
    ntfy_topic: str = field(default_factory=lambda: _str("OWL_NTFY_TOPIC", ""))
    ntfy_token: str = field(default_factory=lambda: _str("OWL_NTFY_TOKEN", ""))
    # At most one notification per species (or "person") in this many seconds.
    notify_cooldown: float = field(default_factory=lambda: _float("OWL_NOTIFY_COOLDOWN", 300))
    # Species to record without notifying, e.g. "eastern gray squirrel,person".
    notify_mute: tuple[str, ...] = field(default_factory=lambda: _list("OWL_NOTIFY_MUTE", ""))
    site_url: str = field(default_factory=lambda: _str("OWL_SITE_URL", ""))

    # API
    api_host: str = field(default_factory=lambda: _str("OWL_API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _int("OWL_API_PORT", 8080))
    api_secret: str = field(default_factory=lambda: _str("OWL_API_SECRET", ""))
    hls_url: str = field(default_factory=lambda: _str("OWL_HLS_URL", "http://127.0.0.1:8888/owl"))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: _list("OWL_CORS_ORIGINS", "*"))

    @property
    def clips_dir(self) -> Path:
        return self.data_dir / "clips"

    @property
    def status_path(self) -> Path:
        return self.data_dir / "status.json"
