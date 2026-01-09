# vision_pipeline/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


FRAME_WIDTH: Optional[int] = None#1280
FRAME_HEIGHT: Optional[int] = None#720
CAMERA_FPS: Optional[int] = 30


@dataclass(frozen=True)
class RoiConfig:
    """Relative ROI definition (fractions of full-frame width/height)."""

    top_ratio: float = 0.45
    bottom_ratio: float = 1.0
    left_ratio: float = 0.05
    right_ratio: float = 0.95
    use_trapezoid_mask: bool = True
    trapezoid_top_shrink: float = 0.35  # fraction of ROI width trimmed at the top
    force_full_frame: bool = False


ROI = RoiConfig()


@dataclass(frozen=True)
class PreprocessConfig:
    """Noise reduction and brightness normalization parameters."""

    gaussian_kernel: Tuple[int, int] = (5, 5)
    gaussian_sigma: float = 0.0
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)


PREPROCESS = PreprocessConfig()


@dataclass(frozen=True)
class ThresholdConfig:
    """Binary segmentation parameters for dark tape on a light floor."""

    use_adaptive: bool = False
    dark_threshold: int = 80  # pixels darker than this are considered tape
    adaptive_block_size: int = 31
    adaptive_c: int = 5


THRESHOLD = ThresholdConfig()


@dataclass(frozen=True)
class MorphologyConfig:
    """Morphological cleanup to remove speckles and close gaps."""

    open_kernel: Tuple[int, int] = (3, 3)
    close_kernel: Tuple[int, int] = (7, 7)
    open_iterations: int = 0
    close_iterations: int = 2


MORPHOLOGY = MorphologyConfig()


@dataclass(frozen=True)
class ThinLineBoostConfig:
    """Optional dilation to keep very narrow lines visible."""

    enable: bool = True
    kernel: Tuple[int, int] = (3, 3)
    iterations: int = 1


THIN_LINE = ThinLineBoostConfig()


@dataclass(frozen=True)
class ScanlineConfig:
    """Scanline sampling + smoothing for border detection."""

    start_ratio: float = 0.1
    end_ratio: float = 0.95
    step_pixels: int = 12
    min_blob_width: int = 4
    max_gap: int = 8
    smoothing_window: int = 5
    min_midline_points: int = 6
    min_border_separation: int = 30  # reject detections that collapse together


SCANLINES = ScanlineConfig()


MASK_WINDOW_NAME: str = "Binary Mask"
MAIN_WINDOW_NAME: str = "RL Racecar Vision"
STATUS_TEXT_POS: Tuple[int, int] = (20, 30)
STATUS_TEXT_SCALE: float = 0.7
STATUS_TEXT_THICKNESS: int = 2
