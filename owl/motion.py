"""Motion detection on the low-resolution camera frames.

Unlike the Hailo detector, which only knows the 80 COCO classes, this notices
anything that moves, so a deer or a fox sets off a recording even though no
model has a class for them. The species classifier then decides what it was.
"""

import cv2
import numpy as np

Box = tuple[int, int, int, int]


class MotionDetector:
    WIDTH = 320  # frames are shrunk to this width before analysis
    WARMUP = 50  # frames spent learning the background after start-up
    MAX_BOXES = 3
    # A change covering this much of the frame is a light change (a cloud, the camera
    # adjusting its exposure), not an animal. The background model re-learns quickly.
    LIGHT_CHANGE = 0.35

    def __init__(self, min_area: float):
        """`min_area`: smallest moving blob that counts, as a fraction of the frame."""
        self._min_area = min_area
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=36, detectShadows=True
        )
        self._open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self._frames = 0
        self._fast_frames = 0

    def update(self, rgb: np.ndarray) -> list[Box]:
        """Feed the next frame; returns boxes (x0, y0, x1, y1) of moving blobs, biggest first."""
        height, width = rgb.shape[:2]
        small = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(
            small, (self.WIDTH, height * self.WIDTH // width), interpolation=cv2.INTER_AREA
        )
        small = cv2.GaussianBlur(small, (5, 5), 0)

        rate = 0.05 if self._fast_frames else -1
        self._fast_frames = max(self._fast_frames - 1, 0)
        mask = self._subtractor.apply(small, learningRate=rate)
        self._frames += 1
        if self._frames < self.WARMUP:
            return []

        # The subtractor marks shadows as 127; only the 255 pixels are real movement.
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(mask) > self.LIGHT_CHANGE * mask.size:
            self._fast_frames = 20
            return []

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open)
        # Merge the pieces of one animal, which the subtractor tends to split up.
        merged = cv2.dilate(mask, self._merge)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Judge size by the moving pixels themselves: the merge step fattens every blob.
        min_pixels = max(8, int(self._min_area * mask.size * 0.5))
        scale = width / self.WIDTH
        found = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            pixels = cv2.countNonZero(mask[y : y + h, x : x + w])
            if pixels >= min_pixels:
                box = (int(x * scale), int(y * scale), int((x + w) * scale), int((y + h) * scale))
                found.append((pixels, box))
        found.sort(reverse=True)
        return [box for _, box in found[: self.MAX_BOXES]]
