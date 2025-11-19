"""
Perspective transform utilities for converting the forward-facing race
camera feed into a bird's-eye/projection-aligned view of the track.
"""

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'models', 'track_config.yaml')
)


@dataclass(frozen=True)
class BirdsEyeSpec:
    """Configuration describing the size of the bird's-eye patch."""

    world_x_limits: Tuple[float, float]
    world_y_limits: Tuple[float, float]
    meters_per_pixel: float

    def __post_init__(self) -> None:
        if self.world_x_limits[1] <= self.world_x_limits[0]:
            raise ValueError("world_x_limits must have max > min")
        if self.world_y_limits[1] <= self.world_y_limits[0]:
            raise ValueError("world_y_limits must have max > min")
        if self.meters_per_pixel <= 0:
            raise ValueError("meters_per_pixel must be positive")

    @property
    def width_pixels(self) -> int:
        span = self.world_y_limits[1] - self.world_y_limits[0]
        return max(1, int(round(span / self.meters_per_pixel)))

    @property
    def height_pixels(self) -> int:
        span = self.world_x_limits[1] - self.world_x_limits[0]
        return max(1, int(round(span / self.meters_per_pixel)))


class BirdsEyeTransformer:
    """
    Compute and apply a perspective transform from the forward-facing camera
    to a top-down bird's-eye view of the ground plane.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        spec: Optional[BirdsEyeSpec] = None,
    ) -> None:
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._config = self._load_config(self.config_path)
        self._camera_cfg = self._config['camera']
        self._track_cfg = self._config.get('track', {})

        self.width = int(self._camera_cfg['resolution']['width'])
        self.height = int(self._camera_cfg['resolution']['height'])

        # Camera intrinsics derived from vertical FOV.
        fov_rad = np.deg2rad(float(self._camera_cfg['fov']))
        if fov_rad <= 0:
            raise ValueError("Camera FOV must be positive in track_config.yaml")
        self.fy = 0.5 * self.height / np.tan(fov_rad / 2.0)
        self.fx = self.fy  # Square pixels assumption.
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        self.camera_position = np.array(
            [
                float(self._camera_cfg['position_offset']['x']),
                float(self._camera_cfg['position_offset']['y']),
                float(self._camera_cfg['position_offset']['z']),
            ],
            dtype=np.float32,
        )
        self.pitch_degrees = float(self._camera_cfg['pitch'])

        self.rotation = self._build_rotation_matrix(self.pitch_degrees)

        # Default bird's-eye spec spans the outer track radius region.
        if spec is None:
            outer_radius = float(self._track_cfg.get('outer_radius', 3.0))
            x_limits = (0.0, max(outer_radius * 2.0, 0.5))
            y_limits = (-outer_radius, outer_radius)
            spec = BirdsEyeSpec(
                world_x_limits=x_limits,
                world_y_limits=y_limits,
                meters_per_pixel=0.01,
            )
        self.spec = spec

        self.output_size = (self.spec.width_pixels, self.spec.height_pixels)
        self.homography = self._compute_homography()

    @staticmethod
    def _load_config(path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"track_config.yaml not found: {path}")
        with open(path, 'r') as fh:
            config = yaml.safe_load(fh) or {}
        if 'camera' not in config:
            raise ValueError("track_config.yaml missing 'camera' section")
        return config

    @staticmethod
    def _build_rotation_matrix(pitch_degrees: float) -> np.ndarray:
        """
        Construct a rotation matrix that maps world coordinates (X forward,
        Y left, Z up) into camera coordinates (X right, Y down, Z forward).
        """
        pitch = np.deg2rad(pitch_degrees)
        cos_p = np.cos(pitch)
        sin_p = np.sin(pitch)

        # Rotate around the car's left axis (world Y). Negative pitch tilts downward.
        rot_pitch = np.array(
            [
                [cos_p, 0.0, -sin_p],
                [0.0, 1.0, 0.0],
                [sin_p, 0.0, cos_p],
            ],
            dtype=np.float32,
        )

        forward = rot_pitch @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
        up = rot_pitch @ np.array([0.0, 0.0, 1.0], dtype=np.float32)

        forward /= np.linalg.norm(forward)
        up /= np.linalg.norm(up)

        left = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = -left
        down = -up

        # Rows correspond to camera X(right), Y(down), Z(forward).
        rotation = np.vstack([right, down, forward])
        return rotation

    def _world_to_camera(self, point: Sequence[float]) -> np.ndarray:
        point_vec = np.asarray(point, dtype=np.float32)
        delta = point_vec - self.camera_position
        return self.rotation @ delta

    def _project_point(self, point: Sequence[float]) -> np.ndarray:
        cam_coords = self._world_to_camera(point)
        if cam_coords[2] <= 1e-6:
            raise ValueError(
                f"Point {point} projects behind the camera; adjust bird's-eye bounds."
            )
        normalized = cam_coords[:2] / cam_coords[2]
        u = self.fx * normalized[0] + self.cx
        v = self.fy * normalized[1] + self.cy
        return np.array([u, v], dtype=np.float32)

    def project_world_points(
        self, points: Iterable[Sequence[float]]
    ) -> np.ndarray:
        """Project arbitrary 3D points (in meters) into image coordinates."""
        projected = [self._project_point(pt) for pt in points]
        return np.asarray(projected, dtype=np.float32)

    def _compute_homography(self) -> np.ndarray:
        x_min, x_max = self.spec.world_x_limits
        y_min, y_max = self.spec.world_y_limits
        world_points = [
            (x_min, y_max, 0.0),  # near left
            (x_min, y_min, 0.0),  # near right
            (x_max, y_min, 0.0),  # far right
            (x_max, y_max, 0.0),  # far left
        ]
        image_points = self.project_world_points(world_points)

        dst_points = np.array(
            [
                [0.0, self.spec.height_pixels - 1.0],  # bottom-left
                [self.spec.width_pixels - 1.0, self.spec.height_pixels - 1.0],
                [self.spec.width_pixels - 1.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )

        homography = cv2.getPerspectiveTransform(image_points, dst_points)
        return homography

    def warp(self, image: np.ndarray, flags: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Apply the bird's-eye homography to an input image.

        Args:
            image: numpy array shaped (H, W) or (H, W, C) matching camera resolution.
            flags: OpenCV interpolation flag.

        Returns:
            Bird's-eye image with shape (output_height, output_width[, C]).
        """
        if image.ndim not in (2, 3):
            raise ValueError("Input image must be grayscale or color array.")
        if image.shape[0] != self.height or image.shape[1] != self.width:
            raise ValueError(
                f"Input image shape {image.shape[:2]} does not match camera "
                f"resolution {(self.height, self.width)}"
            )

        warped = cv2.warpPerspective(
            image,
            self.homography,
            self.output_size,
            flags=flags,
        )
        return warped


def create_default_transformer() -> BirdsEyeTransformer:
    """
    Helper for constructing a BirdsEyeTransformer with repository defaults.
    """
    return BirdsEyeTransformer()
