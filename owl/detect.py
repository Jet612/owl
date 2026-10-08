"""YOLOv8 object detection on the Hailo AI HAT."""

from dataclasses import dataclass

import numpy as np
from picamera2.devices import Hailo

# Class order baked into the Hailo model zoo's COCO YOLOv8 models.
COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    # Box in pixels of the analysed frame: x0, y0, x1, y1.
    box: tuple[int, int, int, int]


class Detector:
    """Runs the model on frames that are wider than tall by letterboxing them into the square input."""

    def __init__(self, model_path: str, labels: tuple[str, ...], min_score: float):
        unknown = set(labels) - set(COCO_LABELS)
        if unknown:
            raise ValueError(f"Not COCO labels: {', '.join(sorted(unknown))}")
        self._hailo = Hailo(model_path)
        self._size, width, _ = self._hailo.get_input_shape()
        if self._size != width:
            raise ValueError("Expected a square model input")
        self._wanted = {COCO_LABELS.index(label) for label in labels}
        self._min_score = min_score
        self._input = np.zeros((self._size, self._size, 3), dtype=np.uint8)

    @property
    def input_size(self) -> int:
        return self._size

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """`frame` is RGB, `input_size` pixels wide and no taller than that."""
        height, width, _ = frame.shape
        self._input[:height, :width] = frame
        results = self._hailo.run(self._input)
        found = []
        for class_id in self._wanted:
            for y0, x0, y1, x1, score in results[class_id]:
                if score < self._min_score:
                    continue
                box = (
                    int(x0 * self._size),
                    int(y0 * self._size),
                    int(x1 * self._size),
                    min(int(y1 * self._size), height),
                )
                found.append(Detection(COCO_LABELS[class_id], float(score), box))
        return found

    def close(self) -> None:
        self._hailo.close()
