import os

import numpy as np
import pybullet as p
import yaml


class Track:
    """Build a circular track from cylinder segments."""

    def __init__(self, config_path):
        """Initialize from config and precompute geometry."""
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config or 'track' not in config:
            raise ValueError(f"Invalid configuration file: {config_path}")

        self.inner_radius = config['track']['inner_radius']
        self.outer_radius = config['track']['outer_radius']
        self.num_segments = config['track']['num_segments']
        self.line_radius = config['track']['line_radius']
        self.line_height = config['track']['line_height']

        self.inner_track_ids = []
        self.outer_track_ids = []

        self.inner_points = self._generate_circle_points(self.inner_radius)
        self.outer_points = self._generate_circle_points(self.outer_radius)

    def _generate_circle_points(self, radius):
        """Return (N,3) points on a circle of given radius at fixed height."""
        angles = np.linspace(0, 2 * np.pi, self.num_segments, endpoint=False)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        z = np.full_like(x, self.line_height)

        return np.column_stack([x, y, z])

    def spawn_in_pybullet(self, physics_client):
        """Create cylinders in the physics world."""
        self.inner_track_ids = self._create_track_cylinders(
            self.inner_points,
            physics_client,
            color=[1, 1, 1, 1]  # White
        )

        self.outer_track_ids = self._create_track_cylinders(
            self.outer_points,
            physics_client,
            color=[1, 1, 1, 1]  # White
        )

    def _create_track_cylinders(self, points, physics_client, color):
        """Create cylinder bodies connecting successive points; return their IDs."""
        body_ids = []

        for i in range(len(points)):
            start_point = points[i]
            end_point = points[(i + 1) % len(points)]

            segment_vector = end_point - start_point
            segment_length = np.linalg.norm(segment_vector)
            segment_direction = segment_vector / segment_length
            midpoint = (start_point + end_point) / 2.0

            quaternion = self._get_rotation_quaternion(segment_direction)

            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=self.line_radius,
                height=segment_length,
                physicsClientId=physics_client
            )

            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=self.line_radius,
                length=segment_length,
                rgbaColor=color,
                physicsClientId=physics_client
            )

            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=midpoint,
                baseOrientation=quaternion,
                physicsClientId=physics_client
            )

            body_ids.append(body_id)

        return body_ids

    def _get_rotation_quaternion(self, target_direction):
        """Quaternion rotating cylinder Z-axis to target direction."""
        z_axis = np.array([0, 0, 1])

        dot = np.dot(z_axis, target_direction)
        if np.abs(dot - 1.0) < 1e-6:
            return [0, 0, 0, 1]

        if np.abs(dot + 1.0) < 1e-6:
            return [1, 0, 0, 0]

        rotation_axis = np.cross(z_axis, target_direction)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        angle = np.arccos(np.clip(dot, -1.0, 1.0))

        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)
        cos_half = np.cos(half_angle)

        quaternion = [
            rotation_axis[0] * sin_half,
            rotation_axis[1] * sin_half,
            rotation_axis[2] * sin_half,
            cos_half
        ]

        return quaternion

    def get_track_ids(self):
        """Return (inner_ids, outer_ids) for spawned cylinders."""
        return (self.inner_track_ids, self.outer_track_ids)
