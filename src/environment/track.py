import hashlib
import json
import os
import pickle
from pathlib import Path
import tempfile

import numpy as np
import pybullet as p
import yaml


class Track:
    """Build a seeded procedural track from cylinder segments."""

    _GEOMETRY_CACHE_VERSION = 1
    _GEOMETRY_CACHE_ENV = "RACEY_TRACK_CACHE_DIR"

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
        self.track_mode = str(track_cfg.get("track_mode", "oval")).lower()
        self.figure8_x_scale = float(track_cfg.get("figure8_x_scale", 1.0))
        self.figure8_y_scale = float(track_cfg.get("figure8_y_scale", 0.55))
        self.figure8_split = float(track_cfg.get("figure8_split", 0.18))
        self.figure8_smooth_window = max(1, int(track_cfg.get("figure8_smooth_window", 9)))
        self.control_points = int(track_cfg.get('control_points', max(12, self.num_segments // 3)))
        base_step = (2 * np.pi) / max(self.control_points, 3)
        self.theta_min = float(track_cfg.get('theta_min', base_step * 0.5))
        self.theta_max = float(track_cfg.get('theta_max', base_step * 1.5))
        self.curvature_min = float(track_cfg.get('curvature_min', 0.05))
        self.curvature_max = float(track_cfg.get('curvature_max', 0.7))
        self.curvature_margin = float(track_cfg.get('curvature_margin', 0.05))
        self.curvature_noise_scale = float(track_cfg.get('curvature_noise_scale', 0.35))
        self.spline_samples_per_seg = int(track_cfg.get('spline_samples_per_seg', max(6, self.num_segments // max(self.control_points, 1))))
        self.curvature_relax = float(track_cfg.get('curvature_relax', 2.5))
        self.curvature_tolerance = float(track_cfg.get('curvature_tolerance', 0.05))
        self.affine_min = float(track_cfg.get('affine_min', 0.8))
        self.affine_max = float(track_cfg.get('affine_max', 1.2))
        self.affine_shear = float(track_cfg.get('affine_shear', 0.3))
        self.drift_amp_frac = float(track_cfg.get('drift_amp_frac', 0.12))
        self.radial_mod_amp = float(track_cfg.get('radial_mod_amp', 0.25))
        self.radial_mod_amp2 = float(track_cfg.get('radial_mod_amp2', 0.15))
        self.straight_count = int(track_cfg.get('straight_count', 2))
        self.straight_min_frac = float(track_cfg.get('straight_min_frac', 0.05))
        self.straight_max_frac = float(track_cfg.get('straight_max_frac', 0.12))
        self.straight_epsilon = float(track_cfg.get('straight_epsilon', 0.03))
        self.corner_min_spacing_frac = float(track_cfg.get('corner_min_spacing_frac', 0.08))
        self.corner_kappa_diff = float(track_cfg.get('corner_kappa_diff', 0.05))
        self.hairpin_kappa = float(track_cfg.get('hairpin_kappa', 0.4))
        self.hairpin_window_frac = float(track_cfg.get('hairpin_window_frac', 0.06))
        self.boundary_min_gap = float(track_cfg.get('boundary_min_gap', self.line_radius * 4.0))
        self.enforce_corner_spacing = bool(track_cfg.get('enforce_corner_spacing', True))
        self.enforce_hairpin_sign = bool(track_cfg.get('enforce_hairpin_sign', True))
        self.validate_overlap = bool(track_cfg.get('validate_overlap', True))
        self.boundary_width_min_frac = float(track_cfg.get('boundary_width_min_frac', 0.6))
        self.direction_change_max_deg = float(track_cfg.get('direction_change_max_deg', 160.0))
        self.direction_change_window_frac = float(track_cfg.get('direction_change_window_frac', 0.15))

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

        geometry = self._load_cached_geometry()
        if geometry is None:
            centerline = self._generate_centerline_points()
            geometry = self._build_geometry(centerline)
            self._store_cached_geometry(geometry)
        self._apply_geometry(geometry)

    def _effective_geometry_config(self):
        return {
            "seed": None if self.seed is None else int(self.seed),
            "track_mode": self.track_mode,
            "inner_radius": float(self.inner_radius),
            "outer_radius": float(self.outer_radius),
            "num_segments": int(self.num_segments),
            "line_radius": float(self.line_radius),
            "line_height": float(self.line_height),
            "radius_jitter": float(self.radius_jitter),
            "num_features": int(self.num_features),
            "straight_feature_ratio": float(self.straight_feature_ratio),
            "angle_warp_strength": float(self.angle_warp_strength),
            "angle_warp_harmonics": int(self.angle_warp_harmonics),
            "high_freq_scale": float(self.high_freq_scale),
            "num_chicanes": int(self.num_chicanes),
            "oval_ratio": float(self.oval_ratio),
            "oval_rotation": float(self.oval_rotation),
            "figure8_x_scale": float(self.figure8_x_scale),
            "figure8_y_scale": float(self.figure8_y_scale),
            "figure8_split": float(self.figure8_split),
            "figure8_smooth_window": int(self.figure8_smooth_window),
            "control_points": int(self.control_points),
            "theta_min": float(self.theta_min),
            "theta_max": float(self.theta_max),
            "curvature_min": float(self.curvature_min),
            "curvature_max": float(self.curvature_max),
            "curvature_margin": float(self.curvature_margin),
            "curvature_noise_scale": float(self.curvature_noise_scale),
            "spline_samples_per_seg": int(self.spline_samples_per_seg),
            "curvature_relax": float(self.curvature_relax),
            "curvature_tolerance": float(self.curvature_tolerance),
            "affine_min": float(self.affine_min),
            "affine_max": float(self.affine_max),
            "affine_shear": float(self.affine_shear),
            "drift_amp_frac": float(self.drift_amp_frac),
            "radial_mod_amp": float(self.radial_mod_amp),
            "radial_mod_amp2": float(self.radial_mod_amp2),
            "straight_count": int(self.straight_count),
            "straight_min_frac": float(self.straight_min_frac),
            "straight_max_frac": float(self.straight_max_frac),
            "straight_epsilon": float(self.straight_epsilon),
            "corner_min_spacing_frac": float(self.corner_min_spacing_frac),
            "corner_kappa_diff": float(self.corner_kappa_diff),
            "hairpin_kappa": float(self.hairpin_kappa),
            "hairpin_window_frac": float(self.hairpin_window_frac),
            "boundary_min_gap": float(self.boundary_min_gap),
            "enforce_corner_spacing": bool(self.enforce_corner_spacing),
            "enforce_hairpin_sign": bool(self.enforce_hairpin_sign),
            "validate_overlap": bool(self.validate_overlap),
            "boundary_width_min_frac": float(self.boundary_width_min_frac),
            "direction_change_max_deg": float(self.direction_change_max_deg),
            "direction_change_window_frac": float(self.direction_change_window_frac),
            "feature_width_range": [float(v) for v in self.feature_width_range],
            "chicane_spacing": [float(v) for v in self.chicane_spacing],
        }

    def _geometry_cache_key(self):
        payload = {
            "version": self._GEOMETRY_CACHE_VERSION,
            "geometry": self._effective_geometry_config(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _geometry_cache_path(self):
        root = os.environ.get(self._GEOMETRY_CACHE_ENV)
        if not root:
            root = os.path.join(tempfile.gettempdir(), "racey-car-track-cache")
        return Path(root) / f"{self._geometry_cache_key()}.pkl"

    def _build_geometry(self, centerline):
        centerline_arr = np.asarray(centerline, dtype=float)
        centerline_xy = centerline_arr[:, :2]
        return {
            "centerline_points": centerline_arr,
            "centerline_tangents": self._compute_tangents(centerline_xy),
            "centerline_arc_lengths": self._compute_arc_lengths(centerline_xy),
            "total_length": float(self._compute_total_length(centerline_xy)),
            "centerline_curvatures": self._polyline_curvature(centerline_xy),
            "inner_points": self._offset_curve(centerline_arr, -self.half_width),
            "outer_points": self._offset_curve(centerline_arr, self.half_width),
        }

    def _apply_geometry(self, geometry):
        self.centerline_points = np.array(geometry["centerline_points"], dtype=float, copy=True)
        self.centerline_tangents = np.array(geometry["centerline_tangents"], dtype=float, copy=True)
        self.centerline_arc_lengths = np.array(geometry["centerline_arc_lengths"], dtype=float, copy=True)
        self.centerline_curvatures = np.array(geometry["centerline_curvatures"], dtype=float, copy=True)
        self.inner_points = np.array(geometry["inner_points"], dtype=float, copy=True)
        self.outer_points = np.array(geometry["outer_points"], dtype=float, copy=True)
        self.total_length = float(geometry["total_length"])

    def _validate_geometry(self, geometry):
        required = {
            "centerline_points",
            "centerline_tangents",
            "centerline_arc_lengths",
            "centerline_curvatures",
            "inner_points",
            "outer_points",
            "total_length",
        }
        if not required.issubset(geometry):
            raise ValueError("geometry cache payload is incomplete")
        if len(np.asarray(geometry["centerline_points"])) < 3:
            raise ValueError("geometry cache payload is degenerate")

    def _load_cached_geometry(self):
        path = self._geometry_cache_path()
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("version") != self._GEOMETRY_CACHE_VERSION:
                return None
            geometry = payload.get("geometry")
            if not isinstance(geometry, dict):
                return None
            self._validate_geometry(geometry)
            return geometry
        except Exception:
            return None

    def _store_cached_geometry(self, geometry):
        try:
            self._validate_geometry(geometry)
            path = self._geometry_cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=f"{path.stem}-", suffix=".tmp", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    pickle.dump(
                        {"version": self._GEOMETRY_CACHE_VERSION, "geometry": geometry},
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                os.replace(tmp_path, path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception:
            return

    def _generate_centerline_points(self):
        """Create a complex but non-self-intersecting closed curve for the centerline."""
        if self.track_mode == "figure8":
            return self._generate_figure8_centerline()
        if self.track_mode == "oval":
            return self._generate_oval_centerline()
        attempts = 60
        coarse_count = max(self.num_segments, min(max(self.num_segments * 2, 96), 240))
        dense_count = max(self.num_segments * 4, 400)
        last_candidate = None
        reject_counts = {
            "control_self_intersect": 0,
            "spline_fail": 0,
            "curvature": 0,
            "boundary_intersect": 0,
            "boundary_gap": 0,
        }
        for _ in range(attempts):
            control_points, angles = self._generate_control_points()
            if self._has_self_intersections(control_points):
                reject_counts["control_self_intersect"] += 1
                continue
            spline_points, curvatures = self._sample_catmull_rom(control_points)
            if spline_points is None or curvatures is None:
                reject_counts["spline_fail"] += 1
                continue
            spline_points = self._apply_center_drift(spline_points)
            straight_intervals = self._build_straight_intervals(spline_points)
            straight_mask = self._mask_from_intervals(
                self._arc_length_param(spline_points), straight_intervals
            )
            spline_points = self._straighten_points(spline_points, straight_mask)
            curvature_points = self._resample_closed_polyline(
                spline_points, max(self.num_segments, len(spline_points))
            )
            curvatures = self._polyline_curvature(curvature_points)
            curvatures = self._smooth_signal(curvatures, window=9)
            curvature_mask = self._mask_from_intervals(
                self._arc_length_param(curvature_points), straight_intervals
            )
            ok, reason, stats = self._curvature_ok(curvature_points, curvatures, curvature_mask)
            if not ok:
                reject_counts["curvature"] += 1
                if reason:
                    reject_counts[f"curvature_{reason}"] = reject_counts.get(f"curvature_{reason}", 0) + 1
                last_curv_stats = stats
                continue
            resampled = self._resample_closed_polyline(spline_points, self.num_segments)
            last_candidate = resampled
            coarse = self._resample_closed_polyline(spline_points, coarse_count)
            if self._candidate_has_intersections(coarse, self.half_width):
                reject_counts["boundary_intersect"] += 1
                continue
            if not self._boundaries_far_enough(coarse, self.half_width):
                reject_counts["boundary_gap"] += 1
                continue
            dense = self._resample_closed_polyline(spline_points, dense_count)
            if self._candidate_has_intersections(dense, self.half_width):
                reject_counts["boundary_intersect"] += 1
                continue
            if not self._boundaries_far_enough(dense, self.half_width):
                reject_counts["boundary_gap"] += 1
                continue
        # Centerline clearance check removed; rely on dense boundary checks + post-build overlap validation.
            z = np.full((resampled.shape[0], 1), self.line_height)
            return np.hstack([resampled, z])

        if last_candidate is not None:
            z = np.full((last_candidate.shape[0], 1), self.line_height)
            return np.hstack([last_candidate, z])

        if "last_curv_stats" in locals() and last_curv_stats:
            print(f"[track] last curvature stats: {last_curv_stats}")
        print(f"[track] rejected all candidates: {reject_counts}")
        # Fall back to a clean oval if we cannot resolve intersections.
        return self._generate_oval_centerline()

    def _generate_oval_centerline(self):
        base_angles = np.linspace(0, 2 * np.pi, self.num_segments, endpoint=False)
        base_points = self._build_base_oval(base_angles)
        z = np.full((base_points.shape[0], 1), self.line_height)
        return np.hstack([base_points, z])

    def _generate_figure8_centerline(self):
        """Build a smooth, non-overlapping figure-8-style centerline."""
        base_angles = np.linspace(0, 2 * np.pi, self.num_segments, endpoint=False)
        x_amp = max(0.5, self.figure8_x_scale) * self.base_radius
        y_amp_base = max(0.2, self.figure8_y_scale) * self.base_radius
        split_base = max(0.12, self.figure8_split) * self.base_radius

        # Retry with progressively wider waist separation until boundaries are valid.
        for attempt in range(10):
            split = split_base * (1.0 + 0.3 * attempt)
            y_amp = y_amp_base * max(0.65, 1.0 - 0.04 * attempt)

            x = x_amp * np.sin(base_angles)
            # cos(theta) separates the two passes at x~=0, avoiding true self-intersection.
            y = y_amp * np.sin(2.0 * base_angles) + split * np.cos(base_angles)
            points_xy = np.column_stack([x, y])
            points_xy = self._smooth_closed_polyline(points_xy, self.figure8_smooth_window)

            if self._candidate_has_intersections(points_xy, self.half_width):
                continue
            if not self._boundaries_far_enough(points_xy, self.half_width):
                continue

            z = np.full((points_xy.shape[0], 1), self.line_height)
            return np.hstack([points_xy, z])

        # If all figure-8 attempts fail geometry checks, fall back to a safe oval.
        return self._generate_oval_centerline()

    def _generate_control_points(self):
        """Generate control points with multi-scale angular spacing."""
        count = max(8, self.control_points)
        deltas = self._rng.uniform(self.theta_min, self.theta_max, size=count)
        deltas *= (2 * np.pi) / np.sum(deltas)
        angles = np.cumsum(np.concatenate(([0.0], deltas[:-1])))
        eps = self._rng.uniform(-self.radius_jitter, self.radius_jitter, size=count)
        radii = self.base_radius + eps
        phase1 = self._rng.uniform(0, 2 * np.pi)
        phase2 = self._rng.uniform(0, 2 * np.pi)
        radii *= (
            1.0
            + self.radial_mod_amp * np.sin(2 * angles + phase1)
            + self.radial_mod_amp2 * np.sin(3 * angles + phase2)
        )
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        points = np.column_stack([xs, ys])
        points = self._apply_affine_warp(points)
        return points, angles

    def _apply_affine_warp(self, points):
        a = self._rng.uniform(self.affine_min, self.affine_max)
        d = self._rng.uniform(self.affine_min, self.affine_max)
        b = self._rng.uniform(-self.affine_shear, self.affine_shear)
        c = self._rng.uniform(-self.affine_shear, self.affine_shear)
        mat = np.array([[a, b], [c, d]], dtype=float)
        return points @ mat.T

    def _catmull_rom(self, p0, p1, p2, p3, t):
        t2 = t * t
        t3 = t2 * t
        return 0.5 * (
            (2 * p1)
            + (-p0 + p2) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )

    def _catmull_rom_derivatives(self, p0, p1, p2, p3, t):
        t2 = t * t
        dp = 0.5 * (
            (-p0 + p2)
            + 2 * (2 * p0 - 5 * p1 + 4 * p2 - p3) * t
            + 3 * (-p0 + 3 * p1 - 3 * p2 + p3) * t2
        )
        ddp = 0.5 * (
            2 * (2 * p0 - 5 * p1 + 4 * p2 - p3)
            + 6 * (-p0 + 3 * p1 - 3 * p2 + p3) * t
        )
        return dp, ddp

    def _sample_catmull_rom(self, control_points):
        n = len(control_points)
        if n < 4:
            return None, None
        samples = []
        curvatures = []
        for i in range(n):
            p0 = control_points[(i - 1) % n]
            p1 = control_points[i % n]
            p2 = control_points[(i + 1) % n]
            p3 = control_points[(i + 2) % n]
            for j in range(self.spline_samples_per_seg):
                t = j / self.spline_samples_per_seg
                pt = self._catmull_rom(p0, p1, p2, p3, t)
                dp, ddp = self._catmull_rom_derivatives(p0, p1, p2, p3, t)
                speed = np.linalg.norm(dp)
                if speed < 1e-8:
                    curvature = 0.0
                else:
                    curvature = abs(dp[0] * ddp[1] - dp[1] * ddp[0]) / (speed ** 3)
                samples.append(pt)
                curvatures.append(curvature)
        return np.array(samples, dtype=float), np.array(curvatures, dtype=float)

    def _curvature_ok(self, points, curvatures, straight_mask=None):
        if len(points) < 3:
            return False, "too_few_points", None
        lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        total = float(np.sum(lengths))
        if total <= 1e-6:
            return False, "degenerate_length", None
        s = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
        phases = self._rng.uniform(0, 2 * np.pi, size=2)
        g = 0.5 + self.curvature_noise_scale * (
            0.6 * np.sin(2 * np.pi * s / total + phases[0])
            + 0.4 * np.sin(4 * np.pi * s / total + phases[1])
        )
        g = np.clip(g, 0.0, 1.0)
        k_target = self.curvature_min + (self.curvature_max - self.curvature_min) * g
        if straight_mask is None:
            straight_mask = self._mask_from_intervals(s, self._build_straight_intervals(points))
        straight_limit = np.where(straight_mask, k_target, self.straight_epsilon)
        limit = (straight_limit + self.curvature_margin) * self.curvature_relax
        over = curvatures > limit
        over_frac = float(np.mean(over))
        stats = {
            "over_frac": over_frac,
            "tol": float(self.curvature_tolerance),
            "max_curv": float(np.max(curvatures)) if len(curvatures) else 0.0,
            "max_limit": float(np.max(limit)) if len(limit) else 0.0,
            "straight_frac": float(np.mean(~straight_mask)) if len(straight_mask) else 0.0,
            "n": int(len(curvatures)),
            "total_length": float(total),
        }
        if over_frac > self.curvature_tolerance:
            return False, "over_limit", stats
        if self.enforce_corner_spacing and not self._corner_spacing_ok(s, curvatures, total):
            return False, "corner_spacing", stats
        if self.enforce_hairpin_sign and not self._hairpin_sign_ok(points, curvatures):
            return False, "hairpin", stats
        if not self._direction_change_ok(points, total):
            return False, "direction_change", stats
        return True, None, stats

    def _direction_change_ok(self, points, total_length):
        if len(points) < 4 or total_length <= 1e-6:
            return True
        window = max(1e-6, self.direction_change_window_frac * total_length)
        max_change = np.deg2rad(self.direction_change_max_deg)
        pts = np.asarray(points, dtype=float)
        segs = np.roll(pts, -1, axis=0) - pts
        headings = np.arctan2(segs[:, 1], segs[:, 0])
        headings = np.unwrap(headings)
        s = self._arc_length_param(pts)
        n = len(pts)
        j = 0
        for i in range(n):
            if j < i:
                j = i
            while j < n and (s[j] - s[i]) < window:
                j += 1
            if j >= n:
                break
            delta = abs(headings[j] - headings[i])
            if delta > max_change:
                return False
        return True

    def _build_straight_intervals(self, points):
        intervals = []
        if self.straight_count <= 0 or len(points) < 3:
            return intervals
        total = float(np.sum(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)))
        if total <= 1e-6:
            return intervals
        for _ in range(self.straight_count):
            length = self._rng.uniform(self.straight_min_frac, self.straight_max_frac) * total
            start = self._rng.uniform(0.0, total)
            end = (start + length) % total
            intervals.append((start, end))
        return intervals

    def _mask_from_intervals(self, s, intervals):
        mask = np.ones_like(s, dtype=bool)
        if not intervals:
            return mask
        total = s[-1] if len(s) else 0.0
        if total <= 1e-6:
            return mask
        for start, end in intervals:
            if start < end:
                mask[(s >= start) & (s <= end)] = False
            else:
                mask[(s >= start) | (s <= end)] = False
        return mask

    def _arc_length_param(self, points):
        if len(points) < 2:
            return np.array([0.0], dtype=float)
        lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        return np.concatenate(([0.0], np.cumsum(lengths[:-1])))

    def _corner_spacing_ok(self, s, curvatures, total):
        if len(curvatures) < 3:
            return True
        peaks = []
        for i in range(1, len(curvatures) - 1):
            if curvatures[i] > curvatures[i - 1] and curvatures[i] > curvatures[i + 1]:
                peaks.append((s[i], curvatures[i]))
        if len(peaks) < 2:
            return True
        peaks.sort(key=lambda x: x[0])
        min_spacing = self.corner_min_spacing_frac * total
        for i in range(len(peaks)):
            s0, k0 = peaks[i]
            s1, k1 = peaks[(i + 1) % len(peaks)]
            delta = (s1 - s0) if i + 1 < len(peaks) else (total - s0 + s1)
            if delta < min_spacing:
                return False
            if abs(k1 - k0) < self.corner_kappa_diff:
                return False
        return True

    def _hairpin_sign_ok(self, points, curvatures):
        if len(curvatures) < 3 or len(points) < 3:
            return True
        # Signed curvature from local turning direction.
        signs = np.zeros(len(curvatures), dtype=float)
        pts = np.asarray(points, dtype=float)
        for i in range(1, len(pts) - 1):
            v1 = pts[i] - pts[i - 1]
            v2 = pts[i + 1] - pts[i]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            signs[i] = np.sign(cross)
        window = max(2, int(len(curvatures) * self.hairpin_window_frac))
        for i, k in enumerate(curvatures):
            if k < self.hairpin_kappa:
                continue
            start = max(0, i - window)
            end = min(len(curvatures), i + window)
            window_signs = signs[start:end]
            if np.any(window_signs < 0) and np.any(window_signs > 0):
                return False
        return True

    def _resample_closed_polyline(self, points, num_samples):
        pts = np.asarray(points, dtype=float)
        n = len(pts)
        if n == 0:
            return pts
        segs = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        total = float(np.sum(segs))
        if total <= 1e-6:
            return np.tile(pts[0], (num_samples, 1))
        cum = np.concatenate(([0.0], np.cumsum(segs)))
        samples = []
        for i in range(num_samples):
            target = (i / num_samples) * total
            idx = int(np.searchsorted(cum, target, side="right") - 1)
            idx = max(0, min(idx, n - 1))
            seg_len = segs[idx]
            if seg_len <= 1e-8:
                samples.append(pts[idx])
                continue
            t = (target - cum[idx]) / seg_len
            p0 = pts[idx]
            p1 = pts[(idx + 1) % n]
            samples.append(p0 + t * (p1 - p0))
        return np.array(samples, dtype=float)

    def _apply_center_drift(self, points):
        if len(points) < 3:
            return points
        lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        total = float(np.sum(lengths))
        if total <= 1e-6:
            return points
        s = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
        amp = self.base_radius * self.drift_amp_frac
        phase = self._rng.uniform(0, 2 * np.pi, size=2)
        dx = amp * np.sin(2 * np.pi * s / total + phase[0])
        dy = amp * np.sin(2 * np.pi * s / total + phase[1])
        drift = np.column_stack([dx, dy])
        return points + drift

    @staticmethod
    def _smooth_signal(values, window=5):
        vals = np.asarray(values, dtype=float)
        if window <= 1 or len(vals) < window:
            return vals
        half = window // 2
        padded = np.pad(vals, (half, half), mode="wrap")
        kernel = np.ones(window, dtype=float) / window
        return np.convolve(padded, kernel, mode="valid")

    @classmethod
    def _smooth_closed_polyline(cls, points, window=5):
        pts = np.asarray(points, dtype=float)
        if window <= 1 or len(pts) < window:
            return pts
        xs = cls._smooth_signal(pts[:, 0], window=window)
        ys = cls._smooth_signal(pts[:, 1], window=window)
        return np.column_stack([xs, ys])

    def _straighten_points(self, points, straight_mask=None):
        if self.straight_count <= 0 or len(points) < 4:
            return points
        pts = np.asarray(points, dtype=float).copy()
        lengths = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        total = float(np.sum(lengths))
        if total <= 1e-6:
            return pts
        if straight_mask is None:
            s = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
            mask = self._mask_from_intervals(s, self._build_straight_intervals(points))
        else:
            mask = straight_mask
        if np.all(mask):
            return pts

        false_idx = np.where(~mask)[0]
        if len(false_idx) == 0:
            return pts

        runs = []
        start = false_idx[0]
        prev = false_idx[0]
        for idx in false_idx[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                runs.append((start, prev))
                start = idx
                prev = idx
        runs.append((start, prev))

        # Merge wrap-around run.
        if (not mask[0]) and (not mask[-1]) and len(runs) > 1:
            first = runs[0]
            last = runs[-1]
            runs = runs[1:-1]
            runs.insert(0, (last[0], first[1] + len(pts)))

        for start, end in runs:
            if end < start:
                continue
            idxs = list(range(start, end + 1))
            wrapped = [i % len(pts) for i in idxs]
            p0 = pts[wrapped[0]]
            p1 = pts[wrapped[-1]]
            count = len(wrapped)
            for i, idx in enumerate(wrapped):
                t = i / max(1, count - 1)
                pts[idx] = p0 + t * (p1 - p0)
        return pts

    def _polyline_curvature(self, points):
        pts = np.asarray(points, dtype=float)
        n = len(pts)
        if n < 3:
            return np.zeros(n, dtype=float)
        curv = np.zeros(n, dtype=float)
        for i in range(n):
            p0 = pts[(i - 1) % n]
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            a = np.linalg.norm(p1 - p0)
            b = np.linalg.norm(p2 - p1)
            c = np.linalg.norm(p2 - p0)
            if a < 1e-8 or b < 1e-8 or c < 1e-8:
                curv[i] = 0.0
                continue
            cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
            curv[i] = 2.0 * abs(cross) / (a * b * c)
        return curv

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

    def _enforce_no_self_intersection(self, points, base_points, outward, variation, half_width):
        """Scale variation until centerline and track borders no longer self-intersect."""
        if not self._candidate_has_intersections(points, half_width):
            return points

        scaled_variation = variation.copy()
        for _ in range(8):
            scaled_variation *= 0.8
            candidate = base_points + outward * scaled_variation[:, None]
            if not self._candidate_has_intersections(candidate, half_width):
                return candidate

        # Fall back to a clean oval if we cannot resolve intersections.
        return base_points

    def _candidate_has_intersections(self, points, half_width):
        if self._has_self_intersections(points[:, :2]):
            return True
        inner = self._offset_curve(points, -half_width)
        outer = self._offset_curve(points, half_width)
        if self._has_self_intersections(inner[:, :2]):
            return True
        if self._has_self_intersections(outer[:, :2]):
            return True
        return False

    def _boundaries_far_enough(self, points, half_width):
        inner = self._offset_curve(points, -half_width)[:, :2]
        outer = self._offset_curve(points, half_width)[:, :2]
        # Ensure the offset width does not collapse.
        widths = np.linalg.norm(outer - inner, axis=1)
        min_width = max(1e-6, self.track_width * self.boundary_width_min_frac)
        if np.min(widths) < min_width:
            return False
        diffs = inner[:, None, :] - outer[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        return bool(np.min(dists) >= self.boundary_min_gap)


    def _has_self_intersections(self, points):
        pts = np.asarray(points, dtype=float)
        n = len(pts)
        if n < 4:
            return False
        mins, maxs = self._segment_bounds(pts)
        for i in range(n):
            j_start = i + 2
            j_stop = n if i > 0 else n - 1
            if j_start >= j_stop:
                continue
            overlaps = self._bbox_overlap_mask(mins[i], maxs[i], mins[j_start:j_stop], maxs[j_start:j_stop])
            if not np.any(overlaps):
                continue
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            candidate_indices = np.nonzero(overlaps)[0] + j_start
            for j in candidate_indices:
                q1 = pts[j]
                q2 = pts[(j + 1) % n]
                if self._segments_intersect(p1, p2, q1, q2):
                    return True
        return False

    def _has_self_intersections_exact(self, points):
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

    def _segment_bounds(self, points):
        pts = np.asarray(points, dtype=float)
        nxt = np.roll(pts, -1, axis=0)
        return np.minimum(pts, nxt), np.maximum(pts, nxt)

    def _bbox_overlap_mask(self, min_a, max_a, mins_b, maxs_b, eps=1e-6):
        return (
            (mins_b[:, 0] <= (max_a[0] + eps))
            & (maxs_b[:, 0] >= (min_a[0] - eps))
            & (mins_b[:, 1] <= (max_a[1] + eps))
            & (maxs_b[:, 1] >= (min_a[1] - eps))
        )

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

    def _compute_arc_lengths(self, points_xy):
        """Cumulative arc lengths of a closed polyline."""
        pts = np.asarray(points_xy, dtype=float)
        n = len(pts)
        arc = [0.0]
        total = 0.0
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            seg_len = float(np.linalg.norm(p1 - p0))
            total += seg_len
            arc.append(total)
        return np.array(arc[:-1], dtype=float)

    def _compute_total_length(self, points_xy):
        pts = np.asarray(points_xy, dtype=float)
        total = 0.0
        for i in range(len(pts)):
            p0 = pts[i]
            p1 = pts[(i + 1) % len(pts)]
            total += float(np.linalg.norm(p1 - p0))
        return total

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
        if self.validate_overlap:
            self._assert_no_overlap(physics_client)

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

    def _assert_no_overlap(self, physics_client):
        """Post-build overlap check using AABBs of non-adjacent segments."""
        self._assert_boundary_no_overlap(self.inner_track_ids, physics_client, "inner")
        self._assert_boundary_no_overlap(self.outer_track_ids, physics_client, "outer")

    def _assert_boundary_no_overlap(self, body_ids, physics_client, label):
        n = len(body_ids)
        if n < 4:
            return
        aabbs = [p.getAABB(bid, physicsClientId=physics_client) for bid in body_ids]
        for i in range(n):
            a_min, a_max = aabbs[i]
            for j in range(i + 1, n):
                if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                    continue
                b_min, b_max = aabbs[j]
                if self._aabb_overlap(a_min, a_max, b_min, b_max):
                    raise RuntimeError(
                        f"Track {label} boundary overlap detected between segments {i} and {j}."
                    )

    @staticmethod
    def _aabb_overlap(a_min, a_max, b_min, b_max):
        return (
            a_min[0] <= b_max[0]
            and a_max[0] >= b_min[0]
            and a_min[1] <= b_max[1]
            and a_max[1] >= b_min[1]
            and a_min[2] <= b_max[2]
            and a_max[2] >= b_min[2]
        )

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

    def get_centerline_points(self):
        """Return cached centerline points."""
        return self.centerline_points

    def get_centerline_tangents(self):
        """Return cached tangents along the centerline."""
        return self.centerline_tangents

    def get_centerline_arc_lengths(self):
        """Return cached cumulative arc lengths for the centerline."""
        return self.centerline_arc_lengths

    def project_onto_centerline(self, point_xy):
        """
        Project a 2D point to the closest point on the discrete centerline.

        Returns:
            s: arc-length position along the loop (meters)
            lateral_error: signed lateral offset from centerline (meters, left-positive)
            tangent: unit vector along the centerline at the projection
        """
        pts = self.centerline_points[:, :2]  # (N, 2)
        n = len(pts)

        # Build segment vectors: p0[i] = pts[i], p1[i] = pts[(i+1) % n]
        p0 = pts                              # (N, 2)
        p1 = pts[np.arange(1, n + 1) % n]    # (N, 2)
        segs = p1 - p0                        # (N, 2)

        # Squared lengths of each segment; mask degenerate ones
        seg_len_sq = (segs * segs).sum(axis=1)          # (N,)
        valid = seg_len_sq >= 1e-12                      # (N,) bool

        # Clamp parameter t to [0, 1] for each segment simultaneously
        diff_p0 = point_xy - p0                          # (N, 2)
        dot_num = (diff_p0 * segs).sum(axis=1)           # (N,)
        # Avoid division by zero on degenerate segments (they will be masked out)
        safe_len_sq = np.where(valid, seg_len_sq, 1.0)
        t = np.clip(dot_num / safe_len_sq, 0.0, 1.0)    # (N,)

        # Closest point on each segment and squared distance to it
        proj = p0 + t[:, np.newaxis] * segs              # (N, 2)
        diff = point_xy - proj                            # (N, 2)
        dist_sq = (diff * diff).sum(axis=1)              # (N,)

        # Ignore degenerate segments when picking the winner
        dist_sq_masked = np.where(valid, dist_sq, np.inf)
        best_i = int(np.argmin(dist_sq_masked))

        seg_len = np.sqrt(seg_len_sq[best_i])
        seg_best = segs[best_i]
        t_best = t[best_i]
        diff_best = diff[best_i]

        best_s = self.centerline_arc_lengths[best_i] + t_best * seg_len
        # Left-positive normal: rotate seg by +90 degrees => (-seg_y, seg_x)
        normal = np.array([-seg_best[1], seg_best[0]])
        normal_norm = np.linalg.norm(normal) or 1.0
        best_lateral = float(np.dot(diff_best, normal / normal_norm))
        best_tangent = seg_best / (seg_len or 1.0)

        return float(best_s % self.total_length), best_lateral, best_tangent

    def get_tangent_at_s(self, s_value):
        """Return unit tangent at arc-length position s_value."""
        if self.centerline_points is None or len(self.centerline_points) == 0:
            return np.array([1.0, 0.0], dtype=float)
        pts = self.centerline_points[:, :2]
        s_mod = float(s_value % self.total_length)
        # Find segment containing s_mod.
        idx = int(np.searchsorted(self.centerline_arc_lengths, s_mod, side="right") - 1)
        idx = max(0, min(idx, len(pts) - 1))
        next_idx = (idx + 1) % len(pts)
        seg = pts[next_idx] - pts[idx]
        seg_norm = np.linalg.norm(seg)
        if seg_norm < 1e-8:
            return np.array([1.0, 0.0], dtype=float)
        return seg / seg_norm

    def get_curvature_at_s(self, s_value):
        """Return curvature at arc-length position s_value."""
        if self.centerline_curvatures is None or len(self.centerline_curvatures) == 0:
            return 0.0
        s_mod = float(s_value % self.total_length)
        idx = int(np.searchsorted(self.centerline_arc_lengths, s_mod, side="right") - 1)
        idx = max(0, min(idx, len(self.centerline_curvatures) - 1))
        return float(self.centerline_curvatures[idx])

    def get_lookahead_curvature(self, s_value, lookahead_meters):
        """Return the max curvature over [s_value, s_value + lookahead_meters]."""
        if self.centerline_curvatures is None or len(self.centerline_curvatures) == 0:
            return 0.0
        n = len(self.centerline_arc_lengths)
        s_start = float(s_value % self.total_length)
        s_end = s_start + lookahead_meters
        idx_start = int(np.searchsorted(self.centerline_arc_lengths, s_start, side="right") - 1)
        idx_start = max(0, min(idx_start, n - 1))
        if s_end <= self.total_length:
            idx_end = int(np.searchsorted(self.centerline_arc_lengths, s_end, side="right") - 1)
            idx_end = max(0, min(idx_end, n - 1))
            if idx_end >= idx_start:
                return float(np.max(self.centerline_curvatures[idx_start:idx_end + 1]))
            else:
                return float(self.centerline_curvatures[idx_start])
        else:
            # Wraps around the loop
            max_first = float(np.max(self.centerline_curvatures[idx_start:]))
            s_end_wrapped = s_end - self.total_length
            idx_end = int(np.searchsorted(self.centerline_arc_lengths, s_end_wrapped, side="right") - 1)
            idx_end = max(0, min(idx_end, n - 1))
            max_second = float(np.max(self.centerline_curvatures[:idx_end + 1]))
            return max(max_first, max_second)

    def get_spawn_pose(self, index=0, height=None):
        """Return a position/quaternion aligned with the centerline."""
        if self.centerline_points is None or self.centerline_tangents is None:
            raise RuntimeError("Centerline data not initialized.")
        idx = index % len(self.centerline_points)
        pos = np.array(self.centerline_points[idx], dtype=float)
        if height is not None:
            pos[2] = height
        tangent = self.centerline_tangents[idx]
        yaw = np.arctan2(tangent[1], tangent[0])
        quat = p.getQuaternionFromEuler([0.0, 0.0, float(yaw)])
        return pos, quat
