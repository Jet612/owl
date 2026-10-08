"""Camera service: live stream to MediaMTX, trigger detection, clip recording and notifications.

One H.264 encoder feeds two outputs: the RTSP stream that MediaMTX serves, and a
circular buffer that holds the last few seconds so clips start before the animal
was first seen. Triggers are the Hailo detector (people and animal-like objects)
and plain movement; the species classifier then says what it was (recorder.py).
"""

import collections
import logging
import signal
import threading
import time

import cv2
from libcamera import Transform
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput2, PyavOutput

from . import clips
from .classify import ClassifierWorker
from .config import Config
from .detect import Detector
from .motion import MotionDetector
from .notify import Notifier
from .recorder import Candidate, Recorder

LOG = logging.getLogger("owl.vision")

STATUS_INTERVAL = 2.0
PRUNE_INTERVAL = 3600.0
RTSP_RETRY = 5.0


class LiveStream:
    """Keeps a PyavOutput publishing to MediaMTX, reconnecting whenever MediaMTX goes away."""

    def __init__(self, url: str, encoder: H264Encoder):
        self._url = url
        self._encoder = encoder
        self._output: PyavOutput | None = None
        self._failed = threading.Event()
        self._next_attempt = 0.0

    @property
    def connected(self) -> bool:
        return self._output is not None

    def maintain(self, circular: CircularOutput2) -> None:
        if self._failed.is_set():
            LOG.warning("Lost connection to MediaMTX; reconnecting")
            self._failed.clear()
            self._output = None
            self._encoder.output = [circular]
            self._next_attempt = time.monotonic() + RTSP_RETRY
        if self._output or time.monotonic() < self._next_attempt:
            return
        output = PyavOutput(self._url, format="rtsp")
        output.error_callback = lambda exc: self._failed.set()
        try:
            output.start()
        except Exception as exc:  # noqa: BLE001 - libav raises many types; any failure means retry
            LOG.warning("Cannot publish to %s (%s); retrying in %.0fs", self._url, exc, RTSP_RETRY)
            self._next_attempt = time.monotonic() + RTSP_RETRY
            return
        self._output = output
        # Assigning a new output to a running encoder sends it the stream headers.
        self._encoder.output = [circular, output]
        LOG.info("Publishing live stream to %s", self._url)


class Grab:
    """One camera request, converting the full-size frame to RGB only if someone needs it."""

    def __init__(self, request):
        self._request = request
        self._rgb = None

    def rgb(self):
        if self._rgb is None:
            self._rgb = cv2.cvtColor(self._request.make_array("main"), cv2.COLOR_YUV420p2RGB)
        return self._rgb


def run(config: Config) -> None:
    config.clips_dir.mkdir(parents=True, exist_ok=True)
    clips.prune(config.clips_dir, config.retention_days, config.min_free_gb)

    labels = config.animals + (("person",) if config.detect_people else ())
    detector = Detector(
        config.model_path, labels, min(config.animal_min_score, config.person_min_score)
    )
    size = detector.input_size
    lores = (size, size * config.height // config.width // 2 * 2)
    scale = config.width / size
    motion = MotionDetector(config.motion_min_area) if config.motion else None

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_video_configuration(
            main={"size": (config.width, config.height), "format": "YUV420"},
            # BGR888 is RGB byte order in memory, which is what the models want.
            lores={"size": lores, "format": "BGR888"},
            transform=Transform(hflip=config.hflip, vflip=config.vflip),
            controls={"FrameRate": config.fps},
        )
    )
    encoder = H264Encoder(bitrate=config.bitrate, iperiod=config.fps, framerate=config.fps)
    circular = CircularOutput2(buffer_duration_ms=int(config.pre_roll * 1000))
    live = LiveStream(config.rtsp_url, encoder)
    worker = ClassifierWorker(config)
    recorder = Recorder(
        config, circular,
        Notifier(config.ntfy_server, config.ntfy_topic, config.ntfy_token, config.site_url),
        worker,
    )

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    worker.start()
    picam2.start_recording(encoder, circular)
    LOG.info(
        "Camera running at %dx%d %dfps; triggers: %s%s",
        config.width, config.height, config.fps, ", ".join(labels),
        " and movement" if motion else "",
    )

    animal_hits = collections.deque(maxlen=config.confirm_window)
    person_hits = collections.deque(maxlen=config.confirm_window)
    motion_hits = collections.deque(maxlen=config.motion_window)
    detect_interval = 1.0 / config.detect_fps
    next_detect = next_status = 0.0
    next_prune = time.monotonic() + PRUNE_INTERVAL
    try:
        while not stop.is_set():
            live.maintain(circular)
            mono = time.monotonic()
            if mono < next_detect:
                time.sleep(min(next_detect - mono, 0.05))
                continue
            next_detect = mono + detect_interval

            request = picam2.capture_request()
            try:
                grab = Grab(request)
                now = time.time()
                frame = request.make_array("lores")
                detections = detector.detect(frame)
                animals = [
                    d for d in detections
                    if d.label != "person" and d.score >= config.animal_min_score
                ]
                people = [
                    d for d in detections
                    if d.label == "person" and d.score >= config.person_min_score
                ]
                moving = motion.update(frame) if motion else []
                animal_hits.append(bool(animals))
                person_hits.append(bool(people))
                motion_hits.append(bool(moving))
                # A detector has to fire in a few frames of a short window, to ignore one-off glitches.
                animal_active = sum(animal_hits) >= config.confirm_hits
                person_active = sum(person_hits) >= config.confirm_hits
                motion_active = sum(motion_hits) >= config.motion_hits

                candidates = [Candidate(d.box, (d.label, d.score)) for d in animals]
                candidates += [Candidate(box) for box in moving]

                if recorder.active:
                    if candidates or people:
                        recorder.touch(now)
                elif animal_active or person_active or (motion_active and now >= recorder.quiet_until):
                    recorder.start(now, grab.rgb())

                if recorder.active:
                    if animal_active:
                        for d in animals:
                            recorder.note_detector(d.label, d.score)
                    if person_active and people:
                        top = max(people, key=lambda d: d.score)
                        recorder.observe_person(
                            top.score, tuple(int(v * scale) for v in top.box), grab
                        )
                    recorder.submit(now, candidates, grab, scale)
            finally:
                request.release()

            for result in worker.poll():
                recorder.handle(result)
            recorder.tick(time.time())

            if mono >= next_status:
                next_status = mono + STATUS_INTERVAL
                clips.write_json(config.status_path, {
                    "updated_at": time.time(),
                    "streaming": live.connected,
                    "classifier": worker.state,
                    **recorder.status(),
                })
            if mono >= next_prune:
                next_prune = mono + PRUNE_INTERVAL
                clips.prune(config.clips_dir, config.retention_days, config.min_free_gb)
    finally:
        LOG.info("Stopping")
        recorder.close()
        worker.stop()
        picam2.stop_recording()
        detector.close()
        config.status_path.unlink(missing_ok=True)


def main() -> None:
    config = Config()
    logging.basicConfig(level=config.log_level, format="%(levelname)s %(name)s: %(message)s")
    run(config)


if __name__ == "__main__":
    main()
