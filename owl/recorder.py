"""Sightings: one session per visit, from the first trigger to the end of the post-roll.

A session starts recording right away, so the clip includes the seconds before the
trigger. While it records, crops of what moved are sent to the species classifier.
When a species has been seen clearly in enough frames it is confirmed, which sends
the phone notification. If the session ends and nothing real was ever identified,
the clip is deleted.
"""

import logging
import time
from dataclasses import dataclass, field

import av
import cv2
import numpy as np
from picamera2.outputs import CircularOutput2, PyavOutput

from . import clips
from .classify import ClassifierWorker, Job, Prediction, Result
from .config import Config
from .notify import Notifier

LOG = logging.getLogger("owl.recorder")

UNIDENTIFIED = "unidentified animal"
# While nothing is confirmed, classify as fast as the classifier can keep up, but not faster than
# this; if it still can't tell after FAST_RESULTS frames, ease off to SLOW_INTERVAL seconds, so a
# windy day of ambiguous triggers doesn't keep the CPU busy.
MIN_SUBMIT_INTERVAL = 0.25
FAST_RESULTS = 6
SLOW_INTERVAL = 3.0
BOX_COLOUR = (0, 220, 0)

Box = tuple[int, int, int, int]


@dataclass
class Candidate:
    """Something the camera may be looking at, in analysis-frame pixels."""

    box: Box
    hint: tuple[str, float] | None = None  # COCO label and score if the Hailo detector found it


@dataclass
class Sighting:
    """The clearest frame of one label so far."""

    score: float
    frame: np.ndarray  # RGB, full frame
    box: Box  # full-frame pixels


@dataclass
class Session:
    id: int
    clip_id: str
    started_at: float  # includes the pre-roll
    last_activity: float
    last_submit: float = 0.0
    negatives: int = 0  # "nothing there" results in a row
    negative_frames: int = 0  # ...and in total
    results: int = 0  # classifier results received
    guess_frames: int = 0  # results that named an animal, but not clearly enough
    votes: dict[str, int] = field(default_factory=dict)
    best: dict[str, Sighting] = field(default_factory=dict)
    guesses: dict[str, float] = field(default_factory=dict)
    confirmed: dict[str, Prediction] = field(default_factory=dict)
    detector: dict[str, float] = field(default_factory=dict)  # animals the Hailo detector held on to
    snapshot_label: str | None = None
    snapshot_score: float = 0.0


def _save_jpeg(path, rgb: np.ndarray, box: Box | None = None, caption: str = "") -> None:
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if box:
        x0, y0, x1, y1 = box
        cv2.rectangle(image, (x0, y0), (x1, y1), BOX_COLOUR, 2)
        if caption:
            cv2.putText(
                image, caption, (x0 + 4, max(y0 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, BOX_COLOUR, 2,
            )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        tmp = path.with_suffix(".jpg.tmp")
        tmp.write_bytes(encoded.tobytes())
        tmp.replace(path)


class Recorder:
    def __init__(
        self, config: Config, circular: CircularOutput2, notifier: Notifier, worker: ClassifierWorker
    ):
        self._config = config
        self._circular = circular
        self._notifier = notifier
        self._worker = worker
        self._session: Session | None = None
        self._next_id = 1
        self._last_notified: dict[str, float] = {}
        self.quiet_until = 0.0  # movement-only triggers are ignored until then
        self.last_detection: dict | None = None

    @property
    def active(self) -> bool:
        return self._session is not None

    @property
    def confirmed_clip(self) -> str | None:
        """The clip being recorded, once it is known to hold something real."""
        session = self._session
        return session.clip_id if session and session.confirmed else None

    # -- driven by the camera loop -------------------------------------------------------

    def start(self, now: float, frame: np.ndarray) -> None:
        clips_dir = self._config.clips_dir
        clip_id = clips.new_clip_id(clips_dir, now)
        self._circular.open_output(PyavOutput(
            str(clips_dir / f"{clip_id}.mp4.part"), format="mp4", options={"movflags": "+faststart"}
        ))
        self._session = Session(
            self._next_id, clip_id, started_at=now - self._config.pre_roll, last_activity=now
        )
        self._next_id += 1
        # A placeholder thumbnail, replaced by the best frame once something is identified.
        _save_jpeg(clips_dir / f"{clip_id}.jpg", frame)
        LOG.info("Trigger: recording clip %s", clip_id)

    def touch(self, now: float) -> None:
        self._session.last_activity = now

    def note_detector(self, label: str, score: float) -> None:
        """The Hailo detector has held on to an animal-like object."""
        detector = self._session.detector
        detector[label] = max(detector.get(label, 0.0), score)

    def observe_person(self, score: float, box: Box, grab) -> None:
        """The Hailo detector has held on to a person. It decides these, not the classifier."""
        session = self._session
        sighting = session.best.get("person")
        if sighting is None or score > sighting.score + 0.05:
            session.best["person"] = Sighting(score, grab.rgb(), box)
        if "person" not in session.confirmed:
            self._confirm(session, Prediction("person", score, "person"))
        else:
            self._write_snapshot(session, "person")

    def submit(self, now: float, candidates: list[Candidate], grab, scale: float) -> None:
        """Send the most promising candidate to the classifier, if it is free."""
        session = self._session
        if not candidates or not self._worker.can_submit():
            return
        if session.confirmed:
            interval = self._config.confirmed_interval
        else:
            interval = MIN_SUBMIT_INTERVAL if session.results < FAST_RESULTS else SLOW_INTERVAL
        if now - session.last_submit < interval:
            return
        best = max(candidates, key=lambda c: (
            c.hint is not None, c.hint[1] if c.hint else 0.0,
            (c.box[2] - c.box[0]) * (c.box[3] - c.box[1]),
        ))
        box = tuple(int(v * scale) for v in best.box)
        if self._worker.submit(Job(session.id, grab.rgb(), box, best.hint)):
            session.last_submit = now

    def handle(self, result: Result) -> None:
        session = self._session
        if session is None or result.session_id != session.id or not result.predictions:
            return
        session.results += 1
        top = result.predictions[0]
        if top.category == "person":
            # The classifier is unreliable on people; only the Hailo detector confirms them.
            return
        if top.category == "none" or top.score < self._config.unidentified_min_score:
            # Either nothing is there, or the classifier has no real idea: both mean nothing clear.
            session.negatives += 1
            session.negative_frames += 1
            return
        session.negatives = 0
        known = session.best.get(top.label)
        if known is None or top.score > known.score:
            session.best[top.label] = Sighting(top.score, result.frame, result.box)
        if top.score < result.min_score:
            session.guess_frames += 1
            session.guesses[top.label] = max(session.guesses.get(top.label, 0.0), top.score)
            return
        session.votes[top.label] = session.votes.get(top.label, 0) + 1
        # Confirmed only if it was seen clearly in several frames, and in more of them than the
        # classifier saw nothing at all, so a patch of leaves that looks like an animal in
        # an odd frame or two doesn't count.
        votes = session.votes[top.label]
        if votes >= self._config.species_confirm_frames and votes > session.negative_frames \
                and top.label not in session.confirmed:
            self._confirm(session, top)
        elif top.label in session.confirmed:
            self._write_snapshot(session, top.label)

    def tick(self, now: float) -> None:
        """End the session when it has run its course."""
        session = self._session
        if session is None:
            return
        config = self._config
        if now - session.started_at > config.max_clip:
            self._finish(now, "reached the maximum clip length")
        elif (not session.confirmed and not session.guess_frames and not session.detector
              and session.negatives >= config.abort_negatives):
            self._finish(now, "nothing there")
        # The circular buffer holds back `pre_roll` seconds of video, so the clip has to
        # stay open that much longer to end `post_roll` seconds after the last movement.
        elif now - session.last_activity > config.post_roll + config.pre_roll:
            self._finish(now, "activity ended")

    def close(self) -> None:
        if self._session:
            self._finish(time.time(), "shutting down")

    def status(self) -> dict:
        session = self._session
        return {
            "recording": session is not None,
            "current_clip": self.confirmed_clip,
            "last_detection": self.last_detection,
            "identifying": session is not None and not session.confirmed,
        }

    # -- identification ------------------------------------------------------------------

    def _confirm(self, session: Session, pred: Prediction) -> None:
        session.confirmed[pred.label] = pred
        self.last_detection = {"label": pred.label, "score": round(pred.score, 3), "at": time.time()}
        LOG.info("Identified %s (%.0f%%)", pred.label, pred.score * 100)
        self._write_snapshot(session, pred.label)
        self._notify(session, pred)

    def _write_snapshot(self, session: Session, label: str) -> None:
        sighting = session.best.get(label)
        if sighting is None:
            return
        if label == session.snapshot_label and sighting.score <= session.snapshot_score + 0.01:
            return
        session.snapshot_label, session.snapshot_score = label, sighting.score
        _save_jpeg(
            self._config.clips_dir / f"{session.clip_id}.jpg", sighting.frame, sighting.box,
            f"{label} {sighting.score:.0%}",
        )

    def _notify(self, session: Session, pred: Prediction) -> None:
        config = self._config
        now = time.time()
        if pred.label in config.notify_mute:
            return
        if now - self._last_notified.get(pred.label, 0.0) < config.notify_cooldown:
            return
        self._last_notified[pred.label] = now
        name = pred.label.capitalize()
        self._notifier.send(
            f"{name} spotted",
            f"{name} in the backyard ({pred.score:.0%} confident). Recording now.",
            config.clips_dir / f"{session.clip_id}.jpg",
        )

    # -- finishing -----------------------------------------------------------------------

    def _decide(self, session: Session) -> dict | None:
        """What to store about the clip, or None if there was nothing worth keeping."""
        if session.confirmed:
            wildlife = sorted(
                (label for label, pred in session.confirmed.items() if pred.category != "person"),
                key=lambda label: (session.votes.get(label, 0), session.best[label].score),
                reverse=True,
            )
            ranked = wildlife + (["person"] if "person" in session.confirmed else [])
            label = ranked[0]
            confirmed = session.confirmed[label]
            return {
                "label": label,
                "scientific": confirmed.scientific,
                "category": confirmed.category,
                "identified": True,
                "labels": {name: round(session.best[name].score, 3) for name in ranked},
                "label_categories": {name: session.confirmed[name].category for name in ranked},
            }
        keep_unidentified = self._config.unidentified_days > 0
        if keep_unidentified and (
            session.guess_frames >= self._config.unidentified_min_frames or session.detector
        ):
            if session.guesses:
                labels = dict(sorted(session.guesses.items(), key=lambda kv: -kv[1])[:3])
            else:
                labels = dict(session.detector)
            return {
                "label": UNIDENTIFIED,
                "scientific": "",
                "category": "unknown",
                "identified": False,
                "labels": {name: round(score, 3) for name, score in labels.items()},
                "label_categories": {name: "unknown" for name in labels},
            }
        return None

    def _finish(self, now: float, reason: str) -> None:
        session, self._session = self._session, None
        self._circular.close_output()
        clips_dir = self._config.clips_dir
        part = clips_dir / f"{session.clip_id}.mp4.part"
        video = clips_dir / f"{session.clip_id}.mp4"

        outcome = self._decide(session)
        if outcome is None or not part.exists() or part.stat().st_size == 0:
            LOG.info("Discarding clip %s (%s, nothing identified)", session.clip_id, reason)
            part.unlink(missing_ok=True)
            clips.delete_clip(clips_dir, session.clip_id)
            self.quiet_until = now + self._config.quiet_after_false
            return

        if outcome["identified"]:
            self._write_snapshot(session, outcome["label"])
        elif session.best:
            label = max(session.best, key=lambda name: session.best[name].score)
            sighting = session.best[label]
            _save_jpeg(
                clips_dir / f"{session.clip_id}.jpg", sighting.frame, sighting.box,
                f"{label}? {sighting.score:.0%}",
            )
        part.rename(video)
        # Clips start on a keyframe, so trust the file's own duration over the wall clock.
        ended_at = now - self._config.pre_roll
        with av.open(str(video)) as container:
            duration = container.duration / av.time_base if container.duration else 0.0
        started_at = ended_at - duration if duration else session.started_at
        days = self._config.retention_days if outcome["identified"] else self._config.unidentified_days
        meta = {
            "id": session.clip_id,
            "started_at": round(started_at, 3),
            "ended_at": round(ended_at, 3),
            "expires_at": round(ended_at + days * 86400, 3),
            "duration": round(ended_at - started_at, 1),
            **outcome,
            "categories": sorted(set(outcome["label_categories"].values())),
            "size_bytes": video.stat().st_size,
        }
        clips.write_json(clips_dir / f"{session.clip_id}.json", meta)
        LOG.info("Saved clip %s: %s, %.0fs (%s)", session.clip_id, meta["label"], meta["duration"], reason)
