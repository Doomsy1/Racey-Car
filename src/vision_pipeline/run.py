# vision_pipeline/main.py
from __future__ import annotations

import argparse
import sys
from typing import Optional

import cv2
import numpy as np

from . import config, visualize
from .camera import Webcam
from .pipeline import VisionPipeline


_PIPELINE: Optional[VisionPipeline] = None


def _get_pipeline() -> VisionPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = VisionPipeline()
    return _PIPELINE


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """Normalize incoming frames so downstream code always receives BGR."""
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL racecar vision pipeline")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Display the binary mask window for debugging",
    )
    return parser.parse_args()


def main(use_webcam: bool = True, frame=None) -> Optional[np.ndarray]:
    pipeline = _get_pipeline()

    if use_webcam:
        args = parse_args()
        try:
            with Webcam(
                index=args.camera_index,
                width=config.FRAME_WIDTH,
                height=config.FRAME_HEIGHT,
                fps=config.CAMERA_FPS,
            ) as camera:
                while True:
                    ok, frame = camera.read()
                    if not ok or frame is None:
                        print("Failed to read frame from webcam; retrying...", file=sys.stderr)
                        continue

                    result = pipeline.process_frame(frame)
                    visualize.draw_roi(frame, result["roi_rect"])
                    visualize.draw_polyline(frame, result["left_border"], visualize.COLOR_LEFT, thickness=3)
                    visualize.draw_polyline(frame, result["right_border"], visualize.COLOR_RIGHT, thickness=3)
                    visualize.draw_polyline(frame, result["midline"], visualize.COLOR_MID, thickness=3)
                    visualize.draw_text(
                        frame,
                        f"midline pts: {len(result['midline'])}",
                        result["success"],
                    )

                    cv2.imshow(config.MAIN_WINDOW_NAME, frame)
                    if args.show_mask and result["binary_mask"] is not None:
                        visualize.show_mask(result["binary_mask"])

                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
        finally:
            cv2.destroyAllWindows()
    else:
        if frame is None:
            print("No frame provided for processing.", file=sys.stderr)
            return None

        frame = _ensure_bgr(frame)
        result = pipeline.process_frame(frame)
        visualize.draw_roi(frame, result["roi_rect"])
        visualize.draw_polyline(frame, result["left_border"], visualize.COLOR_LEFT, thickness=3)
        visualize.draw_polyline(frame, result["right_border"], visualize.COLOR_RIGHT, thickness=3)
        visualize.draw_polyline(frame, result["midline"], visualize.COLOR_MID, thickness=3)
        visualize.draw_text(
            frame,
            f"midline pts: {len(result['midline'])}",
            result["success"],
        )
        if True and result["binary_mask"] is not None:
            visualize.show_mask(result["binary_mask"])

        return frame



if __name__ == "__main__":
    main()
