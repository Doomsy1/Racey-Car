import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

from environment.track import Track


class TrackGenerationCacheTests(unittest.TestCase):
    def _write_config(self, directory: str, *, num_segments: int = 32) -> str:
        config = {
            "track": {
                "track_mode": "random",
                "inner_radius": 2.5,
                "outer_radius": 3.25,
                "num_segments": num_segments,
                "line_radius": 0.0125,
                "line_height": 0.0025,
                "seed": 1,
                "radius_jitter": 0.8,
                "num_features": 4,
                "straight_feature_ratio": 0.2,
                "feature_width_range": [0.2, 0.4],
                "angle_warp_strength": 0.2,
                "angle_warp_harmonics": 2,
                "high_freq_scale": 0.1,
                "num_chicanes": 1,
                "chicane_spacing": [0.08, 0.12],
                "oval_ratio": 1.1,
                "control_points": 10,
                "theta_min": 0.2,
                "theta_max": 0.8,
                "curvature_min": 0.01,
                "curvature_max": 1.2,
                "curvature_margin": 0.2,
                "curvature_relax": 3.0,
                "curvature_tolerance": 0.5,
                "curvature_noise_scale": 0.15,
                "spline_samples_per_seg": 3,
                "straight_count": 1,
                "straight_min_frac": 0.05,
                "straight_max_frac": 0.08,
                "straight_epsilon": 0.2,
                "corner_min_spacing_frac": 0.02,
                "corner_kappa_diff": 0.0,
                "hairpin_kappa": 1.5,
                "hairpin_window_frac": 0.08,
                "boundary_min_gap": 0.02,
                "boundary_width_min_frac": 0.4,
                "direction_change_max_deg": 220.0,
                "direction_change_window_frac": 0.1,
                "validate_overlap": False,
                "enforce_corner_spacing": False,
                "enforce_hairpin_sign": False,
                "oval_rotation": 0.0,
            }
        }
        path = os.path.join(directory, f"track_{num_segments}.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle)
        return path

    def test_track_reuses_cached_geometry_for_same_seed_and_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(tmpdir, num_segments=36)
            cache_dir = os.path.join(tmpdir, "cache")

            with mock.patch.dict(os.environ, {"RACEY_TRACK_CACHE_DIR": cache_dir}, clear=False):
                original = Track(config_path, seed=7)

                with mock.patch.object(
                    Track,
                    "_generate_centerline_points",
                    side_effect=AssertionError("expected cached geometry"),
                ):
                    cached = Track(config_path, seed=7)

            self.assertTrue(np.array_equal(cached.centerline_points, original.centerline_points))
            self.assertTrue(np.array_equal(cached.inner_points, original.inner_points))
            self.assertTrue(np.array_equal(cached.outer_points, original.outer_points))
            self.assertAlmostEqual(cached.total_length, original.total_length)

    def test_track_cache_invalidates_when_geometry_config_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            config_path = self._write_config(tmpdir, num_segments=32)

            with mock.patch.dict(os.environ, {"RACEY_TRACK_CACHE_DIR": cache_dir}, clear=False):
                Track(config_path, seed=3)

                changed_path = self._write_config(tmpdir, num_segments=48)
                with mock.patch.object(
                    Track,
                    "_generate_centerline_points",
                    side_effect=RuntimeError("cache miss expected"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "cache miss expected"):
                        Track(changed_path, seed=3)

    def test_corrupt_cache_falls_back_to_regeneration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            config_path = self._write_config(tmpdir, num_segments=40)

            with mock.patch.dict(os.environ, {"RACEY_TRACK_CACHE_DIR": cache_dir}, clear=False):
                original = Track(config_path, seed=11)
                cache_path = original._geometry_cache_path()
                with open(cache_path, "wb") as handle:
                    handle.write(b"not a valid cache payload")

                regenerated = Track(config_path, seed=11)

            self.assertEqual(regenerated.centerline_points.shape, (40, 3))
            self.assertTrue(np.isfinite(regenerated.centerline_points).all())
            self.assertTrue(np.isfinite(regenerated.inner_points).all())
            self.assertTrue(np.isfinite(regenerated.outer_points).all())

    def test_broad_phase_intersection_check_matches_exact_check(self):
        track = Track.__new__(Track)

        shapes = [
            np.array([[0.0, 0.0], [1.0, 0.0], [1.2, 1.0], [0.0, 1.0]], dtype=float),
            np.array([[0.0, 0.0], [1.5, 1.5], [0.0, 1.5], [1.5, 0.0]], dtype=float),
            np.array([[0.0, 0.0], [2.0, 0.0], [2.2, 0.8], [1.3, 1.7], [0.2, 1.5]], dtype=float),
        ]

        for points in shapes:
            with self.subTest(points=points.tolist()):
                exact = Track._has_self_intersections_exact(track, points)
                broad = Track._has_self_intersections(track, points)
                self.assertEqual(broad, exact)


if __name__ == "__main__":
    unittest.main()
