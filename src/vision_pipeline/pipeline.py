# vision_pipeline/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from . import config

Point = Tuple[float, float]
PointList = List[Point]


@dataclass
class FrameResult:
    success: bool
    roi: np.ndarray
    binary_mask: np.ndarray
    left_border: PointList
    right_border: PointList
    midline: PointList
    roi_rect: Tuple[int, int, int, int]


class VisionPipeline:
    """Core classical OpenCV pipeline for detecting tape borders + midline."""

    def __init__(self) -> None:
        self._clahe = (
            cv2.createCLAHE(
                clipLimit=config.PREPROCESS.clahe_clip_limit,
                tileGridSize=config.PREPROCESS.clahe_tile_grid,
            )
            if config.PREPROCESS.enable_clahe
            else None
        )

    def process_frame(self, frame: np.ndarray) -> Dict[str, object]:
        """Full processing pipeline for a single BGR frame."""
        roi, roi_rect, roi_mask = self._extract_roi(frame)
        if roi.size == 0:
            return self._format_result(False, roi, roi, [], [], [], roi_rect)

        gray = self._preprocess(roi)
        mask = self._segment_lanes(gray, roi_mask)

        left_roi, right_roi = self._extract_borders(mask)
        left_roi = self._smooth_points(left_roi)
        right_roi = self._smooth_points(right_roi)

        midline_roi = self._compute_midline(left_roi, right_roi)
        success = len(midline_roi) >= config.SCANLINES.min_midline_points

        left_border = self._roi_points_to_frame(left_roi, roi_rect)
        right_border = self._roi_points_to_frame(right_roi, roi_rect)
        midline = self._roi_points_to_frame(midline_roi, roi_rect)

        return self._format_result(success, roi, mask, left_border, right_border, midline, roi_rect)

    def _extract_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int], np.ndarray]:
        h, w = frame.shape[:2]
        full_frame_roi = config.ROI.force_full_frame or config.FRAME_WIDTH is None or config.FRAME_HEIGHT is None
        if full_frame_roi:
            top, bottom, left, right = 0, h, 0, w
        else:
            top = int(h * config.ROI.top_ratio)
            bottom = int(h * config.ROI.bottom_ratio)
            left = int(w * config.ROI.left_ratio)
            right = int(w * config.ROI.right_ratio)

        top = max(0, min(top, h - 1))
        bottom = max(top + 1, min(bottom, h))
        left = max(0, min(left, w - 1))
        right = max(left + 1, min(right, w))

        roi = frame[top:bottom, left:right].copy()
        roi_rect = (left, top, right - left, bottom - top)

        mask = np.ones(roi.shape[:2], dtype=np.uint8) * 255
        if config.ROI.use_trapezoid_mask and not full_frame_roi:
            mask = np.zeros_like(mask)
            roi_h, roi_w = mask.shape
            shrink = int(roi_w * config.ROI.trapezoid_top_shrink * 0.5)
            shrink = max(0, min(shrink, roi_w // 2))
            pts = np.array(
                [
                    [shrink, 0],
                    [roi_w - shrink, 0],
                    [roi_w - 1, roi_h - 1],
                    [0, roi_h - 1],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [pts], 255)
        return roi, roi_rect, mask

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        if roi.ndim == 2 or (roi.ndim == 3 and roi.shape[2] == 1):
            gray = roi if roi.ndim == 2 else roi[:, :, 0]
        elif roi.ndim == 3 and roi.shape[2] == 4:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(
            gray,
            config.PREPROCESS.gaussian_kernel,
            config.PREPROCESS.gaussian_sigma,
        )
        if self._clahe is not None:
            blurred = self._clahe.apply(blurred)
        return blurred

    def _segment_lanes(self, gray: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        if config.THRESHOLD.use_adaptive:
            mask = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                config.THRESHOLD.adaptive_block_size,
                config.THRESHOLD.adaptive_c,
            )
        else:
            _, mask = cv2.threshold(
                gray,
                config.THRESHOLD.dark_threshold,
                255,
                cv2.THRESH_BINARY_INV,
            )

        mask = cv2.bitwise_and(mask, roi_mask)

        if config.MORPHOLOGY.open_iterations > 0:
            open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPHOLOGY.open_kernel)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, open_kernel, iterations=config.MORPHOLOGY.open_iterations
            )
        if config.MORPHOLOGY.close_iterations > 0:
            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPHOLOGY.close_kernel)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, close_kernel, iterations=config.MORPHOLOGY.close_iterations
            )
        if config.THIN_LINE.enable and config.THIN_LINE.iterations > 0:
            thin_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.THIN_LINE.kernel)
            mask = cv2.dilate(mask, thin_kernel, iterations=config.THIN_LINE.iterations)
        return mask

    def _extract_borders(self, mask: np.ndarray) -> Tuple[PointList, PointList]:
        height, _ = mask.shape
        start_y = int(height * config.SCANLINES.start_ratio)
        end_y = int(height * config.SCANLINES.end_ratio)
        start_y = max(0, min(start_y, height - 1))
        end_y = max(start_y + 1, min(end_y, height))
        step = max(1, config.SCANLINES.step_pixels)

        left_points: PointList = []
        right_points: PointList = []

        for y in range(start_y, end_y, step):
            row = mask[y, :]
            cols = np.flatnonzero(row > 0)
            if cols.size < config.SCANLINES.min_blob_width:
                continue
            clusters = self._cluster_columns(cols)
            clusters = [cluster for cluster in clusters if cluster.size >= config.SCANLINES.min_blob_width]
            if len(clusters) < 2:
                continue
            left_cluster = min(clusters, key=lambda c: c[0])
            right_cluster = max(clusters, key=lambda c: c[-1])
            if right_cluster[0] - left_cluster[-1] < config.SCANLINES.min_border_separation:
                continue
            left_points.append((float(left_cluster.mean()), float(y)))
            right_points.append((float(right_cluster.mean()), float(y)))

        return left_points, right_points

    def _cluster_columns(self, cols: np.ndarray) -> List[np.ndarray]:
        if cols.size == 0:
            return []
        gaps = np.where(np.diff(cols) > config.SCANLINES.max_gap)[0]
        clusters = np.split(cols, gaps + 1)
        return [cluster for cluster in clusters if cluster.size > 0]

    def _smooth_points(self, points: PointList) -> PointList:
        window = config.SCANLINES.smoothing_window
        if len(points) < 3 or window <= 1:
            return points
        xs = np.array([p[0] for p in points], dtype=np.float32)
        kernel = np.ones(window, dtype=np.float32) / window
        pad = window // 2
        padded = np.pad(xs, (pad, pad), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return [(float(smoothed[i]), points[i][1]) for i in range(len(points))]

    def _compute_midline(self, left: PointList, right: PointList) -> PointList:
        midline: PointList = []
        for l_pt, r_pt in zip(left, right):
            mid_x = 0.5 * (l_pt[0] + r_pt[0])
            mid_y = l_pt[1]  # y indices match by construction
            midline.append((mid_x, mid_y))
        return self._smooth_points(midline)

    @staticmethod
    def _roi_points_to_frame(points: PointList, roi_rect: Tuple[int, int, int, int]) -> PointList:
        x_off, y_off, _, _ = roi_rect
        return [(x + x_off, y + y_off) for x, y in points]

    @staticmethod
    def _format_result(
        success: bool,
        roi: np.ndarray,
        mask: np.ndarray,
        left_border: PointList,
        right_border: PointList,
        midline: PointList,
        roi_rect: Tuple[int, int, int, int],
    ) -> Dict[str, object]:
        return {
            "success": success,
            "roi": roi,
            "binary_mask": mask,
            "left_border": left_border,
            "right_border": right_border,
            "midline": midline,
            "roi_rect": roi_rect,
        }


def test_single_image(path: str) -> None:
    """
    Quick offline test helper.
    Loads an image, runs the pipeline, and shows frame + mask for tuning.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {path}")
    pipeline = VisionPipeline()
    result = pipeline.process_frame(image)

    from . import visualize

    frame_copy = image.copy()
    visualize.draw_roi(frame_copy, result["roi_rect"])
    visualize.draw_polyline(frame_copy, result["left_border"], visualize.COLOR_LEFT, thickness=3)
    visualize.draw_polyline(frame_copy, result["right_border"], visualize.COLOR_RIGHT, thickness=3)
    visualize.draw_polyline(frame_copy, result["midline"], visualize.COLOR_MID, thickness=3)
    visualize.draw_text(frame_copy, f"Midline pts: {len(result['midline'])}", result["success"])

    cv2.imshow(config.MAIN_WINDOW_NAME, frame_copy)
    if result["binary_mask"] is not None:
        visualize.show_mask(result["binary_mask"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
