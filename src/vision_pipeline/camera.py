# vision_pipeline/camera.py
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class Webcam:
    """Small helper around cv2.VideoCapture with predictable resolution."""

    def __init__(
        self,
        index: int = 0,
        width: Optional[int] = 1280,
        height: Optional[int] = 720,
        fps: Optional[int] = 30,
    ) -> None:
        self.index = index
        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open webcam index {index}")
        self._configure(width, height, fps)

    def _configure(self, width: Optional[int], height: Optional[int], fps: Optional[int]) -> None:
        if width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps is not None:
            self._capture.set(cv2.CAP_PROP_FPS, float(fps))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Return the latest frame.
        Returns:
            (success flag, frame or None if failure).
        """
        success, frame = self._capture.read()
        if not success:
            return False, None
        return True, frame

    def release(self) -> None:
        """Release the camera resource if it is still open."""
        if self._capture.isOpened():
            self._capture.release()

    def __enter__(self) -> "Webcam":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()
