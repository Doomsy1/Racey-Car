# vision_pipeline/visualize.py
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config

Color = Tuple[int, int, int]
Point = Tuple[float, float]

COLOR_LEFT: Color = (255, 0, 0)
COLOR_RIGHT: Color = (0, 0, 255)
COLOR_MID: Color = (0, 255, 255)
COLOR_ROI: Color = (0, 255, 0)


def _to_polyline(points: Sequence[Point]) -> Optional[np.ndarray]:
    """
    Convert float points to a cv2 polyline array, skipping invalid points.
    Filtering here prevents NaN/inf coordinates (from noisy detections) from
    crashing OpenCV drawing calls.
    """
    if not points:
        return None
    sanitized: list[tuple[int, int]] = []
    for x, y in points:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        sanitized.append((int(round(x)), int(round(y))))
    if not sanitized:
        return None
    pts = np.asarray(sanitized, dtype=np.int32)
    return pts.reshape((-1, 1, 2))


def draw_polyline(frame: np.ndarray, points: Sequence[Point], color: Color, thickness: int = 2) -> None:
    """Draw a polyline or fallback to points if insufficient samples."""
    poly = _to_polyline(points)
    if poly is None:
        return
    if len(poly) == 1:
        cv2.circle(frame, tuple(poly[0, 0]), radius=4, color=color, thickness=-1)
        return
    cv2.polylines(frame, [poly], False, color, thickness, lineType=cv2.LINE_AA)


def draw_roi(frame: np.ndarray, roi_rect: Tuple[int, int, int, int], color: Color = COLOR_ROI) -> None:
    """Visualize the active ROI rectangle on the main frame."""
    x, y, w, h = roi_rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def draw_text(frame: np.ndarray, text: str, success: bool) -> None:
    """Overlay status text with green/red tint based on success flag."""
    color = (0, 255, 0) if success else (0, 0, 255)
    cv2.putText(
        frame,
        text,
        config.STATUS_TEXT_POS,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.STATUS_TEXT_SCALE,
        color,
        config.STATUS_TEXT_THICKNESS,
        lineType=cv2.LINE_AA,
    )


def show_mask(mask: np.ndarray, window_name: str = config.MASK_WINDOW_NAME) -> None:
    """Display the binary mask used for debugging."""
    cv2.imshow(window_name, mask)
