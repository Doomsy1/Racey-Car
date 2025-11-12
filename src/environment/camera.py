"""Race camera that outputs a binary frame from segmentation masks."""

import os

import numpy as np
import pybullet as p
import yaml


class RaceCamera:

    def __init__(self, config_path, track_ids, physics_client):
        """Initialize camera and segmentation filter."""
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

        # Camera position offset relative to car (local frame)
        self.offset_x = config['camera']['position_offset']['x']
        self.offset_y = config['camera']['position_offset']['y']
        self.offset_z = config['camera']['position_offset']['z']

        # Camera pitch (degrees)
        self.pitch_degrees = config['camera']['pitch']
        self.pitch_radians = np.radians(self.pitch_degrees)

        self.physics_client = physics_client

        inner_ids, outer_ids = track_ids
        self.track_ids_set = set(inner_ids + outer_ids)

        # Projection matrix
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.width / self.height,
            nearVal=self.near,
            farVal=self.far
        )

    def capture_frame(self, car_id):
        """Return a binary image where track objects are black on white."""
        # Car pose
        car_pos, car_orn = p.getBasePositionAndOrientation(
            car_id,
            physicsClientId=self.physics_client
        )

        # View matrix from car pose
        view_matrix = self._compute_view_matrix(car_pos, car_orn)

        # Segmentation mask image (prefer hardware OpenGL)
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
            img_data = p.getCameraImage(
                width=self.width,
                height=self.height,
                viewMatrix=view_matrix,
                projectionMatrix=self.projection_matrix,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                physicsClientId=self.physics_client
            )

        # Extract objectUniqueId from segmentation buffer
        seg_raw = np.asarray(img_data[4], dtype=np.int32).reshape((self.height, self.width))
        object_ids = seg_raw & ((1 << 24) - 1)

        track_mask = np.isin(object_ids, list(self.track_ids_set))

        output_image = np.ones((self.height, self.width), dtype=np.uint8) * 255
        output_image[track_mask] = 0

        return output_image

    def _compute_view_matrix(self, car_pos, car_orn):
        """Compute OpenGL view matrix for the offset, pitched camera."""
        rot_matrix = np.array(p.getMatrixFromQuaternion(car_orn)).reshape((3, 3))

        car_forward = rot_matrix[:, 0]
        car_left = rot_matrix[:, 1]
        car_up = rot_matrix[:, 2]

        camera_pos = np.array(car_pos) + self.offset_x * car_forward + self.offset_y * car_left + self.offset_z * car_up

        cos_pitch = np.cos(self.pitch_radians)
        sin_pitch = np.sin(self.pitch_radians)

        camera_forward = (cos_pitch * car_forward) + (sin_pitch * car_up)
        camera_forward = camera_forward / np.linalg.norm(camera_forward)
        camera_up_vec = (cos_pitch * car_up) - (sin_pitch * car_forward)
        camera_up_vec = camera_up_vec / np.linalg.norm(camera_up_vec)

        camera_target = camera_pos + camera_forward
        camera_up = camera_up_vec

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos.tolist(),
            cameraTargetPosition=camera_target.tolist(),
            cameraUpVector=camera_up.tolist()
        )

        return view_matrix
