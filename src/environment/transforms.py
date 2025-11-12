import os

import cv2
import numpy as np
import yaml


class BirdEyeTransform:
    """Inverse perspective mapping to a bird's‑eye view."""

    def __init__(self, config_path):
        """
        Initialize bird's-eye transform with configuration.

        Args:
            config_path: Path to track_config.yaml
        """
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config or 'camera' not in config:
            raise ValueError(f"Invalid configuration file: {config_path}")

        # Camera parameters
        camera_config = config['camera']
        self.img_width = camera_config['resolution']['width']
        self.img_height = camera_config['resolution']['height']
        self.fov_deg = camera_config['fov']
        self.fov_rad = np.radians(self.fov_deg)

        # Camera position and orientation
        self.camera_height = config['spawn']['position'][2] + camera_config['position_offset']['z']
        self.camera_x_offset = camera_config['position_offset']['x']
        self.pitch_deg = camera_config['pitch']
        self.pitch_rad = np.radians(self.pitch_deg)

        # Bird's-eye view parameters
        transform_config = camera_config.get('bird_eye_transform', {})
        self.pixels_per_meter = transform_config.get('pixels_per_meter', 100.0)

        # Camera intrinsics
        # PyBullet's computeProjectionMatrixFOV takes a VERTICAL FOV. Respect that here.
        # Compute fy from vertical FOV, then derive horizontal FOV and fx using aspect.
        self.aspect = self.img_width / self.img_height
        self.fy = (self.img_height / 2.0) / np.tan(self.fov_rad / 2.0)
        self.hfov_rad = 2.0 * np.arctan(np.tan(self.fov_rad / 2.0) * self.aspect)
        self.fx = (self.img_width / 2.0) / np.tan(self.hfov_rad / 2.0)
        # Principal point at pixel center
        self.cx = (self.img_width - 1) / 2.0
        self.cy = (self.img_height - 1) / 2.0

        # Precompute camera position and basis (pitch is fixed)
        self.cam_pos = np.array([self.camera_x_offset, 0.0, self.camera_height], dtype=float)
        cos_pitch = np.cos(self.pitch_rad)
        sin_pitch = np.sin(self.pitch_rad)
        car_forward = np.array([1.0, 0.0, 0.0], dtype=float)
        car_left = np.array([0.0, 1.0, 0.0], dtype=float)
        car_up = np.array([0.0, 0.0, 1.0], dtype=float)
        self.cam_forward = cos_pitch * car_forward + sin_pitch * car_up
        self.cam_up = cos_pitch * car_up - sin_pitch * car_forward
        self.cam_right = -car_left
        # Normalize to be safe
        self.cam_forward = self.cam_forward / np.linalg.norm(self.cam_forward)
        self.cam_up = self.cam_up / np.linalg.norm(self.cam_up)
        self.cam_right = self.cam_right / np.linalg.norm(self.cam_right)

        # Pre-calculate output bounds by projecting camera view to ground
        self._calculate_output_bounds()

    def apply(self, image):
        """Warp camera image to bird's‑eye view."""
        # Create meshgrid for output image pixels
        out_y, out_x = np.meshgrid(np.arange(self.output_height),
                                     np.arange(self.output_width), indexing='ij')

        # Output X → ground_y (left), Output Y (top) → ground_x (farther)
        # Use pixel centers and map image-left to car-left.
        ground_y = self.ground_y_max - ((out_x + 0.5) / self.pixels_per_meter)
        ground_x = self.ground_x_max - ((out_y + 0.5) / self.pixels_per_meter)

        # Back-project ground coordinates to camera image coordinates
        img_x, img_y = self._ground_to_image(ground_x, ground_y)

        # Sample from source image using cv2.remap
        img_x_32f = img_x.astype(np.float32)
        img_y_32f = img_y.astype(np.float32)

        bird_eye = cv2.remap(image, img_x_32f, img_y_32f, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=128)

        return bird_eye

    def _calculate_output_bounds(self):
        """Project camera frustum edges to ground and set BEV output size."""
        edge_points = []

        # Top edge (may not hit ground with steep pitch)
        for u in np.linspace(0, self.img_width-1, 20):
            edge_points.append([u, 0])

        # Bottom edge (definitely hits ground)
        for u in np.linspace(0, self.img_width-1, 20):
            edge_points.append([u, self.img_height-1])

        # Left edge
        for v in np.linspace(0, self.img_height-1, 20):
            edge_points.append([0, v])

        # Right edge
        for v in np.linspace(0, self.img_height-1, 20):
            edge_points.append([self.img_width-1, v])

        edge_points = np.array(edge_points)

        ground_x, ground_y = self._image_to_ground(edge_points[:, 0], edge_points[:, 1])

        valid = np.isfinite(ground_x) & np.isfinite(ground_y)
        ground_x = ground_x[valid]
        ground_y = ground_y[valid]

        if len(ground_x) == 0:
            self.ground_x_min, self.ground_x_max = -3.0, 3.0
            self.ground_y_min, self.ground_y_max = 0.0, 6.0
        else:
            margin = 0.1  # meters
            self.ground_x_min = np.min(ground_x) - margin
            self.ground_x_max = np.max(ground_x) + margin
            self.ground_y_min = np.min(ground_y) - margin
            self.ground_y_max = np.max(ground_y) + margin

        # Calculate output dimensions
        # Width spans lateral (ground_y), height spans forward (ground_x)
        width_meters = self.ground_y_max - self.ground_y_min
        height_meters = self.ground_x_max - self.ground_x_min
        self.output_width = max(1, int(np.round(width_meters * self.pixels_per_meter)))
        self.output_height = max(1, int(np.round(height_meters * self.pixels_per_meter)))

    def _image_to_ground(self, img_x, img_y):
        """Project pixel coordinates to ground coordinates (meters)."""
        # Convert pixel to normalized image coordinates
        norm_u = (img_x - self.cx) / self.fx
        norm_v = (img_y - self.cy) / self.fy

        # Ray in camera space (forward + u*right - v*up)
        norm_u_col = norm_u[..., np.newaxis] if np.ndim(norm_u) > 0 else norm_u
        norm_v_col = norm_v[..., np.newaxis] if np.ndim(norm_v) > 0 else norm_v
        ray_in_camera_view = self.cam_forward + norm_u_col * self.cam_right + norm_v_col * (-self.cam_up)

        # Normalize
        ray_in_camera_view = ray_in_camera_view / np.linalg.norm(ray_in_camera_view, axis=-1, keepdims=True)

        t = -self.cam_pos[2] / ray_in_camera_view[..., 2]

        if ray_in_camera_view.ndim == 1:
            ground_x = self.cam_pos[0] + t * ray_in_camera_view[0]
            ground_y = self.cam_pos[1] + t * ray_in_camera_view[1]
        else:
            t_col = t[..., np.newaxis]
            ground_point = self.cam_pos + t_col * ray_in_camera_view
            ground_x = ground_point[..., 0]
            ground_y = ground_point[..., 1]

        return ground_x, ground_y

    def _ground_to_image(self, ground_x, ground_y):
        """Back-project ground coordinates to pixel coordinates."""
        # Vector from camera to ground point (in car frame)
        # Ensure ground_x and ground_y are broadcast to same shape
        vec_x = ground_x - self.cam_pos[0]
        vec_y = ground_y - self.cam_pos[1]
        vec_z = np.full_like(vec_x, -self.cam_pos[2])

        vec = np.stack([vec_x, vec_y, vec_z], axis=-1)

        # Project vector onto camera axes to get camera-space coordinates
        depth = np.sum(vec * self.cam_forward, axis=-1)
        cam_u = np.sum(vec * self.cam_right, axis=-1)
        cam_v = np.sum(vec * (-self.cam_up), axis=-1)

        # Normalized image coordinates
        norm_u = cam_u / depth
        norm_v = cam_v / depth

        # Convert to pixel coordinates
        img_x = self.cx + self.fx * norm_u
        img_y = self.cy + self.fy * norm_v

        return img_x, img_y
