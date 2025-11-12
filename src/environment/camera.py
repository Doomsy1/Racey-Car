"""
Module for segmentation mask-based track line extraction.
"""

import os

import numpy as np
import pybullet as p
import yaml


class RaceCamera:

    def __init__(self, config_path, track_ids, physics_client):
        """
        Initialize camera with track IDs for segmentation filtering.

        Args:
            config_path: Path to track_config.yaml
            track_ids: Tuple of (inner_track_ids, outer_track_ids) from Track
            physics_client: PyBullet physics client ID
        """
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config or 'camera' not in config:
            raise ValueError(f"Invalid configuration file: {config_path}")

        self.width = config['camera']['resolution']['width']
        self.height = config['camera']['resolution']['height']
        self.fov = config['camera']['fov']
        self.near = config['camera']['near_plane']
        self.far = config['camera']['far_plane']

        # Camera position offset relative to car (in car's local frame)
        self.offset_x = config['camera']['position_offset']['x']
        self.offset_y = config['camera']['position_offset']['y']
        self.offset_z = config['camera']['position_offset']['z']

        # Camera pitch (tilt angle in degrees)
        self.pitch_degrees = config['camera']['pitch']
        self.pitch_radians = np.radians(self.pitch_degrees)

        self.physics_client = physics_client

        inner_ids, outer_ids = track_ids
        self.track_ids_set = set(inner_ids + outer_ids)

        # Compute projection matrix (constant)
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.width / self.height,
            nearVal=self.near,
            farVal=self.far
        )

    def capture_frame(self, car_id):
        """
        Capture camera frame with track lines extracted via segmentation mask.

        Args:
            car_id: PyBullet body ID of the car

        Returns:
            numpy array (height, width) with binary mask:
            - 255 (white) for background
            - 0 (black) for track lines
        """
        # Get car position and orientation
        car_pos, car_orn = p.getBasePositionAndOrientation(
            car_id,
            physicsClientId=self.physics_client
        )

        # Compute view matrix based on car pose
        view_matrix = self._compute_view_matrix(car_pos, car_orn)

        # Get camera image with segmentation mask
        # Try hardware OpenGL first, fallback to default renderer if unavailable
        try:
            img_data = p.getCameraImage(
                width=self.width,
                height=self.height,
                viewMatrix=view_matrix,
                projectionMatrix=self.projection_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                physicsClientId=self.physics_client
            )
        except Exception:
            # Fallback to default renderer if hardware OpenGL fails
            img_data = p.getCameraImage(
                width=self.width,
                height=self.height,
                viewMatrix=view_matrix,
                projectionMatrix=self.projection_matrix,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                physicsClientId=self.physics_client
            )

        # Extract segmentation mask (5th element of return tuple)
        # With ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, low 24 bits are objectUniqueId,
        # high bits encode linkIndex via right shift 24. We only need objectUniqueId here.
        seg_raw = np.asarray(img_data[4], dtype=np.int32).reshape((self.height, self.width))
        object_ids = seg_raw & ((1 << 24) - 1)

        # Filter segmentation mask to extract track pixels (objectUniqueId match)
        track_mask = np.isin(object_ids, list(self.track_ids_set))

        # Create output image: white background (255), black track lines (0)
        output_image = np.ones((self.height, self.width), dtype=np.uint8) * 255
        output_image[track_mask] = 0

        return output_image

    def _compute_view_matrix(self, car_pos, car_orn):
        """
        Compute camera view matrix from car position and orientation.

        The camera is offset from the car in the car's local coordinate frame
        and pitched down to look at the track.

        Args:
            car_pos: Car position [x, y, z]
            car_orn: Car orientation as quaternion [x, y, z, w]

        Returns:
            OpenGL view matrix (list of 16 floats)
        """
        # Convert car orientation to rotation matrix
        rot_matrix = np.array(p.getMatrixFromQuaternion(car_orn)).reshape((3, 3))

        # Extract car's local axes
        car_forward = rot_matrix[:, 0]  # X-axis in car's frame
        car_left = rot_matrix[:, 1]     # Y-axis in car's frame
        car_up = rot_matrix[:, 2]       # Z-axis in car's frame

        # Calculate camera position in world frame
        # Start from car position and add offset in car's local frame
        camera_pos = np.array(car_pos) + \
                     self.offset_x * car_forward + \
                     self.offset_y * car_left + \
                     self.offset_z * car_up

        # Calculate camera forward direction with pitch
        # Pitch rotation is around the car's lateral (Y) axis
        # Positive pitch looks up, negative pitch looks down
        cos_pitch = np.cos(self.pitch_radians)
        sin_pitch = np.sin(self.pitch_radians)

        # Rotate car's forward and up vectors by pitch angle around car's left axis.
        # This keeps the camera basis orthonormal and the horizon stable.
        camera_forward = (cos_pitch * car_forward) + (sin_pitch * car_up)
        camera_forward = camera_forward / np.linalg.norm(camera_forward)
        camera_up_vec = (cos_pitch * car_up) - (sin_pitch * car_forward)
        camera_up_vec = camera_up_vec / np.linalg.norm(camera_up_vec)

        # Camera target is a point along the forward direction
        camera_target = camera_pos + camera_forward

        # Camera up vector aligned with pitched camera
        camera_up = camera_up_vec

        # Compute view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos.tolist(),
            cameraTargetPosition=camera_target.tolist(),
            cameraUpVector=camera_up.tolist()
        )

        return view_matrix
