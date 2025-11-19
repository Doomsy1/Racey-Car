import os

import numpy as np
import pybullet as p
import yaml


class Track:
    """Build a seeded procedural track from cylinder segments."""

    def __init__(self, config_path, seed=None):
        """Initialize from config and precompute geometry."""
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config or 'track' not in config:
            raise ValueError(f"Invalid configuration file: {config_path}")

        track_cfg = config['track']
        self.inner_radius = track_cfg['inner_radius']
        self.outer_radius = track_cfg['outer_radius']
        self.num_segments = track_cfg['num_segments']
        self.line_radius = track_cfg['line_radius']
        self.line_height = track_cfg['line_height']
        self.seed = seed if seed is not None else track_cfg.get('seed')
        self.radius_jitter = track_cfg.get('radius_jitter')
        self.num_features = max(0, int(track_cfg.get('num_features', 8)))
        self.straight_feature_ratio = float(track_cfg.get('straight_feature_ratio', 0.4))
        self.angle_warp_strength = float(track_cfg.get('angle_warp_strength', 0.3))
        self.angle_warp_harmonics = int(track_cfg.get('angle_warp_harmonics', 3))
        self.high_freq_scale = float(track_cfg.get('high_freq_scale', 0.25))
        self.num_chicanes = max(0, int(track_cfg.get('num_chicanes', 3)))
        self.oval_ratio = max(1.0, float(track_cfg.get('oval_ratio', 1.35)))
        self.oval_rotation = float(track_cfg.get('oval_rotation', 0.0))

        feature_width_range = track_cfg.get('feature_width_range', [0.25, 0.7])
        if not feature_width_range or len(feature_width_range) != 2:
            raise ValueError("feature_width_range must be a list of two values [min, max]")
        self.feature_width_range = (
            float(feature_width_range[0]),
            float(feature_width_range[1])
        )

        chicane_spacing = track_cfg.get('chicane_spacing', [0.06, 0.14])
        if not chicane_spacing or len(chicane_spacing) != 2:
            raise ValueError("chicane_spacing must be [min_fraction, max_fraction]")
        chicane_min = float(chicane_spacing[0])
        chicane_max = float(chicane_spacing[1])
        if chicane_min < 0 or chicane_max <= chicane_min:
            raise ValueError("chicane_spacing fractions must satisfy 0 <= min < max")
        circumference = 2 * np.pi
        self.chicane_spacing = (
            chicane_min * circumference,
            chicane_max * circumference
        )

        self.inner_track_ids = []
        self.outer_track_ids = []

        self.track_width = self.outer_radius - self.inner_radius
        if self.track_width <= 0:
            raise ValueError("outer_radius must be larger than inner_radius")
        self.half_width = self.track_width / 2.0
        self.base_radius = (self.inner_radius + self.outer_radius) / 2.0
        if self.radius_jitter is None:
            self.radius_jitter = max(self.base_radius * 0.25, self.half_width * 0.75)

        self._rng = np.random.default_rng(self.seed)

        centerline = self._generate_centerline_points()
        self.inner_points = self._offset_curve(centerline, -self.half_width)
        self.outer_points = self._offset_curve(centerline, self.half_width)

    def _generate_centerline_points(self):
        """Create a complex but non-self-intersecting closed curve for the centerline."""
        base_angles = np.linspace(0, 2 * np.pi, self.num_segments, endpoint=False)
        warped_angles = self._warp_angles(base_angles)
        base_points = self._build_base_oval(warped_angles)
        outward = self._compute_outward_vectors(base_points)
        variation = self._compose_variation_profile(warped_angles)
        variation = self._limit_variation(variation)
        points = base_points + outward * variation[:, None]
        points = self._enforce_no_self_intersection(points, base_points, outward, variation)
        z = np.full((points.shape[0], 1), self.line_height)
        return np.hstack([points, z])

    def _warp_angles(self, base_angles):
        """Stretch the parameterization to carve out straights and packed turn complexes."""
        weights = np.ones_like(base_angles)
        harmonics = max(1, self.angle_warp_harmonics)
        strength = max(0.0, self.angle_warp_strength)
        if strength > 0:
            for harmonic in range(1, harmonics + 1):
                amplitude = strength * self._rng.uniform(0.5, 1.0) / harmonic
                phase = self._rng.uniform(0, 2 * np.pi)
                weights += amplitude * np.sin(harmonic * base_angles + phase)
        weights = np.clip(weights, 0.15, None)
        cumulative = np.cumsum(weights)
        cumulative *= (2 * np.pi / cumulative[-1])
        return cumulative

    def _build_base_oval(self, angles):
        """Return points on a rotated oval (ellipse) to give an overall racetrack feel."""
        major = self.base_radius * self.oval_ratio
        minor = self.base_radius / self.oval_ratio
        x = major * np.cos(angles)
        y = minor * np.sin(angles)
        if abs(self.oval_rotation) > 1e-6:
            rot = np.array([
                [np.cos(self.oval_rotation), -np.sin(self.oval_rotation)],
                [np.sin(self.oval_rotation), np.cos(self.oval_rotation)]
            ])
            xy = np.column_stack([x, y]) @ rot.T
        else:
            xy = np.column_stack([x, y])
        return xy

    def _compute_outward_vectors(self, points):
        """Approximate outward directions by normalizing vectors from the origin."""
        vectors = np.asarray(points, dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normals = np.zeros_like(vectors)
        valid = norms.squeeze() > 1e-6
        normals[valid] = vectors[valid] / norms[valid]
        normals[~valid] = np.array([1.0, 0.0])
        return normals

    def _compose_variation_profile(self, angles):
        """Blend feature, chicane, and ripple profiles."""
        if not self.radius_jitter or self.radius_jitter <= 0:
            return np.zeros_like(angles)

        profile = np.zeros_like(angles, dtype=float)
        profile += self._apply_feature_field(angles)
        profile += self._apply_high_freq_variation(angles)
        profile -= np.mean(profile)
        return profile

    def _limit_variation(self, variation):
        """Ensure offsets stay within configured jitter and track width."""
        max_offset = self.radius_jitter
        max_allowed = self.base_radius - self.half_width * 1.1
        if max_allowed <= 0:
            return np.zeros_like(variation)
        limit = min(max_offset, max_allowed)
        peak = np.max(np.abs(variation))
        if peak <= 1e-6:
            return variation
        scale = min(1.0, limit / peak)
        return variation * scale

    def _enforce_no_self_intersection(self, points, base_points, outward, variation):
        """Scale variation until the polygon no longer self-intersects."""
        if not self._has_self_intersections(points):
            return points

        scaled_variation = variation.copy()
        for _ in range(8):
            scaled_variation *= 0.8
            candidate = base_points + outward * scaled_variation[:, None]
            if not self._has_self_intersections(candidate):
                return candidate

        # Fall back to a clean oval if we cannot resolve intersections.
        return base_points

    def _has_self_intersections(self, points):
        pts = np.asarray(points, dtype=float)
        n = len(pts)
        if n < 4:
            return False
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            for j in range(i + 1, n):
                if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                    continue
                q1 = pts[j]
                q2 = pts[(j + 1) % n]
                if self._segments_intersect(p1, p2, q1, q2):
                    return True
        return False

    def _segments_intersect(self, p1, p2, q1, q2):
        """Return True if the closed segments intersect (excluding shared endpoints)."""
        o1 = self._orientation(p1, p2, q1)
        o2 = self._orientation(p1, p2, q2)
        o3 = self._orientation(q1, q2, p1)
        o4 = self._orientation(q1, q2, p2)

        if o1 * o2 < 0 and o3 * o4 < 0:
            return True

        eps = 1e-6
        if abs(o1) < eps and self._on_segment(p1, p2, q1):
            return True
        if abs(o2) < eps and self._on_segment(p1, p2, q2):
            return True
        if abs(o3) < eps and self._on_segment(q1, q2, p1):
            return True
        if abs(o4) < eps and self._on_segment(q1, q2, p2):
            return True
        return False

    def _orientation(self, a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def _on_segment(self, a, b, point):
        return (
            min(a[0], b[0]) - 1e-6 <= point[0] <= max(a[0], b[0]) + 1e-6 and
            min(a[1], b[1]) - 1e-6 <= point[1] <= max(a[1], b[1]) + 1e-6
        )

    def _apply_feature_field(self, angles):
        """Use gaussian bumps/dips to introduce straights, hairpins, and flowing bends."""
        if self.num_features <= 0 or self.radius_jitter <= 0:
            return np.zeros_like(angles)

        contributions = np.zeros_like(angles, dtype=float)
        width_min = max(0.05, min(self.feature_width_range))
        width_max = max(width_min + 1e-3, max(self.feature_width_range))
        straight_ratio = np.clip(self.straight_feature_ratio, 0.0, 1.0)

        for _ in range(self.num_features):
            center = self._rng.uniform(0, 2 * np.pi)
            width = self._rng.uniform(width_min, width_max)
            if self._rng.random() < straight_ratio:
                amplitude = self._rng.uniform(0.5, 1.0) * self.radius_jitter
            else:
                amplitude = -self._rng.uniform(0.4, 0.9) * self.radius_jitter
            contributions += amplitude * self._gaussian_profile(angles, center, width)

        contributions += self._build_chicanes(angles)
        return contributions

    def _build_chicanes(self, angles):
        """Create paired bumps/dips to mimic S-bends."""
        if self.num_chicanes <= 0 or self.radius_jitter <= 0:
            return np.zeros_like(angles)

        contributions = np.zeros_like(angles, dtype=float)
        spacing_min = max(0.01, min(self.chicane_spacing))
        spacing_max = max(spacing_min + 1e-3, max(self.chicane_spacing))

        for _ in range(self.num_chicanes):
            center = self._rng.uniform(0, 2 * np.pi)
            offset = self._rng.uniform(spacing_min, spacing_max)
            width = max(0.05, offset * 0.6)
            amplitude = self._rng.uniform(0.35, 0.85) * self.radius_jitter
            contributions += amplitude * self._gaussian_profile(
                angles,
                center - offset / 2.0,
                width
            )
            contributions -= amplitude * self._gaussian_profile(
                angles,
                center + offset / 2.0,
                width
            )
        return contributions

    def _apply_high_freq_variation(self, angles):
        """Add smaller undulations to keep turns from feeling perfectly symmetric."""
        strength = max(0.0, self.high_freq_scale) * self.radius_jitter
        if strength <= 0:
            return np.zeros_like(angles)

        contributions = np.zeros_like(angles, dtype=float)
        for harmonic in range(2, 6):
            amplitude = strength * self._rng.uniform(0.4, 1.0) / harmonic
            phase = self._rng.uniform(0, 2 * np.pi)
            contributions += amplitude * np.sin(harmonic * angles + phase)
        return contributions

    def _gaussian_profile(self, angles, center, width):
        diff = self._wrap_angle_difference(angles, center)
        sigma = max(width, 1e-3)
        return np.exp(-0.5 * (diff / sigma) ** 2)

    def _wrap_angle_difference(self, angles, center):
        diff = angles - center
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return diff

    def _offset_curve(self, centerline_points, offset):
        """Offset the centerline along its normals to build inner/outer boundaries."""
        centerline_xy = centerline_points[:, :2]
        tangents = self._compute_tangents(centerline_xy)
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        offsets = centerline_xy + normals * offset
        z = np.full((centerline_xy.shape[0], 1), self.line_height)

        return np.hstack([offsets, z])

    def _compute_tangents(self, points):
        """Return unit tangents for a closed polyline."""
        pts = np.asarray(points, dtype=float)
        tangents = np.zeros_like(pts)
        for i in range(len(pts)):
            prev_point = pts[i - 1]
            next_point = pts[(i + 1) % len(pts)]
            vec = next_point - prev_point
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                tangents[i] = np.array([1.0, 0.0])
            else:
                tangents[i] = vec / norm
        return tangents

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
            if segment_length < 1e-6:
                continue
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
