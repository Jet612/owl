"""Species identification.

BioCLIP (https://huggingface.co/imageomics/bioclip), a CLIP model trained on
biodiversity photos, scores a crop of the frame against the species list plus a
few "not wildlife" choices. It runs on the Pi's CPU, about one frame a second,
on its own thread so the camera loop never waits for it.

Run `python -m owl.classify --download` once to fetch the model weights.
"""

import hashlib
import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config

LOG = logging.getLogger(__name__)

MODEL_ID = "hf-hub:imageomics/bioclip"

# What a frame can show besides wildlife: (label, category, prompts). Each is its own choice for
# the classifier, and when one of them wins the frame counts as "nothing there". Plenty of
# specific ones matter: with only a vague "empty yard" option, bark and leaf litter get
# mistaken for snakes and rodents. "person" is not decided here (see recorder.py).
EXTRAS = (
    ("person", "person", (
        "a photo of a person.",
        "a photo of a person walking outdoors.",
    )),
    ("empty scene", "none", ("a photo of an empty backyard with grass, leaves, branches and shadows.",)),
    ("night yard", "none", ("a grayscale infrared photo of an empty yard at night.",)),
    ("bark", "none", ("a photo of tree bark, branches and twigs.",)),
    ("fence", "none", ("a photo of a wooden fence and a garden.",)),
    ("ground", "none", ("a photo of grass, soil and fallen leaves on the ground.",)),
    ("foliage", "none", ("a photo of shrubs, flowers and green foliage.",)),
    ("snow", "none", ("a photo of snow on the ground.",)),
    ("blur", "none", ("a blurry, out of focus photo.",)),
    ("furniture", "none", ("a photo of garden furniture, a grill or flower pots.",)),
    ("insect", "none", ("a close-up photo of an insect or spider on a camera lens.",)),
    ("object", "none", ("a photo of a car, a chair, a garden hose or a trash can.",)),
)

# Templates for species; {sci} and {common} come from species.txt.
SPECIES_PROMPTS = (
    "a photo of {sci}, {common}.",
    "a trail camera photo of a {common}, {sci}, in a backyard.",
)

COCO_CATEGORY = {"bird": "bird", "cat": "pet", "dog": "pet"}  # everything else counts as a mammal
COCO_SCIENTIFIC = {
    "bird": "Aves", "cat": "Felis catus", "dog": "Canis lupus familiaris",
    "horse": "Equus ferus caballus", "sheep": "Ovis aries", "cow": "Bos taurus", "bear": "Ursidae",
}


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float
    category: str  # mammal, bird, reptile, amphibian, pet, person, or none for "not wildlife"
    scientific: str = ""


@dataclass
class Job:
    session_id: int
    frame: np.ndarray  # RGB, full frame
    box: tuple[int, int, int, int]  # where in the frame the movement or detection is
    hint: tuple[str, float] | None  # COCO label and score if the Hailo detector found it


@dataclass
class Result:
    session_id: int
    predictions: list[Prediction]  # best first
    min_score: float  # confidence a prediction needs to count
    frame: np.ndarray
    box: tuple[int, int, int, int]


def square_crop(frame: np.ndarray, box: tuple[int, int, int, int], pad: float = 1.3,
                min_side: int = 160) -> np.ndarray:
    """A square view of `frame` centred on `box` with some context around it.

    The model squares its input by cutting the sides off, so the crop is made square here
    instead, keeping the whole animal in view.
    """
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = box
    side = int(min(max(max(x1 - x0, y1 - y0) * pad, min_side), height, width))
    left = int(min(max((x0 + x1) / 2 - side / 2, 0), width - side))
    top = int(min(max((y0 + y1) / 2 - side / 2, 0), height - side))
    return frame[top : top + side, left : left + side]


class SpeciesClassifier:
    def __init__(self, species_file: Path, threads: int, min_score: float):
        # Heavy imports stay here so a service that doesn't classify never loads them.
        import open_clip
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        from . import species as species_list

        torch.set_num_threads(threads)
        species = species_list.load(species_file)
        classes = [
            (s.common, s.category, s.scientific,
             [p.format(sci=s.scientific, common=s.common) for p in SPECIES_PROMPTS])
            for s in species
        ] + [(label, category, "", list(prompts)) for label, category, prompts in EXTRAS]
        self._classes = [Prediction(label, 0.0, category, sci) for label, category, sci, _ in classes]

        model, _, self._preprocess = open_clip.create_model_and_transforms(MODEL_ID)
        model.eval()

        # Encoding the text takes about 0.15 s per prompt, so keep the result until the
        # species list or the prompts change.
        key = hashlib.sha256(
            json.dumps([MODEL_ID, [prompts for *_, prompts in classes]]).encode()
        ).hexdigest()[:16]
        home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
        cache = home / "owl" / f"text-{key}.pt"
        if cache.exists():
            self._text = torch.load(cache, weights_only=True)
        else:
            LOG.info("Encoding the species list (about a minute, first run only)")
            tokenizer = open_clip.get_tokenizer(MODEL_ID)
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                # PyTorch's fused CPU attention returns NaN for this text model's mask;
                # the plain math version is fine.
                features = []
                for _, _, _, prompts in classes:
                    encoded = model.encode_text(tokenizer(prompts))
                    encoded = encoded / encoded.norm(dim=-1, keepdim=True)
                    mean = encoded.mean(dim=0)
                    features.append(mean / mean.norm())
            self._text = torch.stack(features)
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self._text, cache)
            except OSError as exc:
                LOG.warning("Could not cache the encoded species list (%s)", exc)
        # The text half of the model is no longer needed.
        self._visual = model.visual
        self._scale = model.logit_scale.exp().item()
        self._torch = torch
        self.min_score = min_score
        LOG.info("Species classifier ready: %d species, %d threads", len(species), threads)

    def classify(self, rgb: np.ndarray, hint: tuple[str, float] | None = None) -> list[Prediction]:
        from PIL import Image

        torch = self._torch
        image = self._preprocess(Image.fromarray(rgb)).unsqueeze(0)
        with torch.inference_mode():
            features = self._visual(image)
            features = features / features.norm(dim=-1, keepdim=True)
            probabilities = (self._scale * features @ self._text.T).softmax(dim=-1)[0]
        if torch.isnan(probabilities).any():
            raise RuntimeError("classifier produced NaN scores")
        top = probabilities.topk(3)
        return [
            Prediction(c.label, float(score), c.category, c.scientific)
            for score, c in ((s, self._classes[i]) for s, i in zip(top.values, top.indices))
        ]


class HintClassifier:
    """No species model: trusts the Hailo detector's COCO label, as the camera did originally."""

    def __init__(self, min_score: float):
        self.min_score = min_score

    def classify(self, rgb: np.ndarray, hint: tuple[str, float] | None = None) -> list[Prediction]:
        if hint is None:
            return [Prediction("empty scene", 1.0, "none")]
        label, score = hint
        return [Prediction(
            label, score, COCO_CATEGORY.get(label, "mammal"), COCO_SCIENTIFIC.get(label, "")
        )]


class ClassifierWorker:
    """Runs a classifier on its own thread. Submit jobs, poll for results.

    Holds one job at a time: when the classifier is busy, new frames are skipped, so
    results are always about recent frames.
    """

    def __init__(self, config: Config):
        self._config = config
        self._jobs: queue.Queue[Job | None] = queue.Queue(maxsize=1)
        self._results: queue.SimpleQueue[Result] = queue.SimpleQueue()
        self._working = False
        self.state = "loading"  # loading, ready, or fallback (Hailo labels only)
        self._thread = threading.Thread(target=self._run, name="classifier", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass

    def can_submit(self) -> bool:
        return self.state != "loading" and not self._working and self._jobs.empty()

    def submit(self, job: Job) -> bool:
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            return False
        return True

    def poll(self) -> list[Result]:
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def _build(self):
        config = self._config
        if config.classifier == "bioclip":
            try:
                return SpeciesClassifier(
                    Path(config.species_file), config.classify_threads, config.species_min_score
                ), "ready"
            except Exception:  # noqa: BLE001 - any failure here must not stop the camera
                LOG.exception(
                    "Species identification is unavailable; falling back to the Hailo detector's labels"
                )
        return HintClassifier(config.animal_min_score + 0.15), "fallback"

    def _run(self) -> None:
        classifier, self.state = self._build()
        while (job := self._jobs.get()) is not None:
            self._working = True
            started = time.monotonic()
            try:
                predictions = classifier.classify(square_crop(job.frame, job.box), job.hint)
            except Exception:  # noqa: BLE001 - a bad frame must not kill the worker
                LOG.exception("Classification failed")
                predictions = []
            LOG.debug(
                "Classified in %.2fs: %s", time.monotonic() - started,
                ", ".join(f"{p.label} {p.score:.2f}" for p in predictions),
            )
            self._results.put(Result(
                job.session_id, predictions, classifier.min_score, job.frame, job.box
            ))
            self._working = False


def main() -> None:
    """`python -m owl.classify --download` fetches the model; `<image>...` classifies photos."""
    from PIL import Image

    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    config = Config()
    args = sys.argv[1:]
    if not args:
        raise SystemExit("usage: python -m owl.classify --download | <image>...")
    species_file = Path(config.species_file)
    if not species_file.exists():
        species_file = Path(__file__).resolve().parent.parent / "species.txt"
    classifier = SpeciesClassifier(species_file, config.classify_threads, config.species_min_score)
    if args == ["--download"]:
        return
    for path in args:
        image = np.array(Image.open(path).convert("RGB"))
        height, width = image.shape[:2]
        started = time.monotonic()
        top = classifier.classify(square_crop(image, (0, 0, width, height), pad=1.0))
        print(f"{path}: " + ", ".join(f"{p.label} {p.score:.2f}" for p in top)
              + f" ({time.monotonic() - started:.2f}s)")


if __name__ == "__main__":
    main()
