import os

import numpy as np
import pybullet as p
import yaml


class Track:
    """
    Generates a circular race track using PyBullet cylinder primitives.

    The track consists of two concentric circles (inner and outer boundaries),
    each represented as a series of connected line segments. Each segment is
    a thin cylinder positioned and oriented between consecutive points.
    """

    def __init__(self, config_path):
        """
        Initialize track with configuration.

        Args:
            config_path: Path to track_config.yaml
        """
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

        # Generate track geometry
        self.inner_points = self._generate_circle_points(self.inner_radius)
        self.outer_points = self._generate_circle_points(self.outer_radius)

    def _generate_circle_points(self, radius):
        """
        Generate 3D points for a circular track.

        Args:
            radius: Circle radius in meters

        Returns:
            numpy array of shape (num_segments, 3) with [x, y, z] coordinates
        """
        angles = np.linspace(0, 2 * np.pi, self.num_segments, endpoint=False)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        z = np.full_like(x, self.line_height)  # All points at same height

        return np.column_stack([x, y, z])

    def spawn_in_pybullet(self, physics_client):
        """
        Create track cylinders in PyBullet simulation.

        Args:
            physics_client: PyBullet physics client ID
        """
        # Create inner track cylinders
        self.inner_track_ids = self._create_track_cylinders(
            self.inner_points,
            physics_client,
            color=[1, 1, 1, 1]  # White
        )

        # Create outer track cylinders
        self.outer_track_ids = self._create_track_cylinders(
            self.outer_points,
            physics_client,
            color=[1, 1, 1, 1]  # White
        )

    def _create_track_cylinders(self, points, physics_client, color):
        """
        Create cylinder bodies for each track segment.

        Args:
            points: numpy array of 3D points
            physics_client: PyBullet physics client ID
            color: RGBA color for cylinders

        Returns:
            List of PyBullet body IDs
        """
        body_ids = []

        for i in range(len(points)):
            # Get start and end points (wrap around for last segment)
            start_point = points[i]
            end_point = points[(i + 1) % len(points)]

            # Calculate segment properties
            segment_vector = end_point - start_point
            segment_length = np.linalg.norm(segment_vector)
            segment_direction = segment_vector / segment_length
            midpoint = (start_point + end_point) / 2.0

            # Calculate rotation to align cylinder with segment
            # Cylinder's default axis is Z, we need to align it with segment direction
            quaternion = self._get_rotation_quaternion(segment_direction)

            # Create cylinder collision shape
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=self.line_radius,
                height=segment_length,
                physicsClientId=physics_client
            )

            # Create visual shape
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=self.line_radius,
                length=segment_length,
                rgbaColor=color,
                physicsClientId=physics_client
            )

            # Create multi-body with both collision and visual
            body_id = p.createMultiBody(
                baseMass=0,  # Static object
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=midpoint,
                baseOrientation=quaternion,
                physicsClientId=physics_client
            )

            body_ids.append(body_id)

        return body_ids

    def _get_rotation_quaternion(self, target_direction):
        """
        Calculate quaternion to rotate cylinder's Z-axis to align with target direction.

        Rotation algorithm:
        1. Find rotation axis (cross product of Z-axis and target)
        2. Find rotation angle (arc cosine of dot product)
        3. Convert axis-angle to quaternion

        Args:
            target_direction: numpy array of normalized direction vector

        Returns:
            List [x, y, z, w] representing quaternion
        """
        # Cylinder's default orientation (Z-axis)
        z_axis = np.array([0, 0, 1])

        # Handle case where vectors are already aligned
        dot = np.dot(z_axis, target_direction)
        if np.abs(dot - 1.0) < 1e-6:
            return [0, 0, 0, 1]  # No rotation needed

        # Handle case where vectors are opposite
        if np.abs(dot + 1.0) < 1e-6:
            # Rotate 180 degrees around any perpendicular axis
            return [1, 0, 0, 0]

        # Calculate rotation axis (perpendicular to both vectors)
        rotation_axis = np.cross(z_axis, target_direction)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        # Calculate rotation angle
        angle = np.arccos(np.clip(dot, -1.0, 1.0))

        # Convert axis-angle to quaternion
        # q = [sin(angle/2) * axis, cos(angle/2)]
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
        """
        Get PyBullet body IDs for track cylinders.

        Returns:
            Tuple of (inner_track_ids, outer_track_ids) where each is a list of ints
        """
        return (self.inner_track_ids, self.outer_track_ids)
