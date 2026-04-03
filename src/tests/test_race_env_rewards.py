import unittest

import numpy as np

from environment.race_env import RaceCarEnv


class RaceEnvRewardTests(unittest.TestCase):
    def test_gate_positions_skip_start_finish_line(self):
        gates = RaceCarEnv._build_gate_positions(track_length=12.0, gate_count=4)

        np.testing.assert_allclose(gates, np.array([2.4, 4.8, 7.2, 9.6]))

    def _make_env(self) -> RaceCarEnv:
        env = RaceCarEnv.__new__(RaceCarEnv)
        env.progress_credit_multiplier = 1.25
        env.progress_credit_slack = 0.05
        env.start_line_point = np.zeros(2, dtype=float)
        env.start_line_normal = np.array([1.0, 0.0], dtype=float)
        env.start_zone_radius = 1.0
        env.start_line_rearm_distance = 1.0
        env.start_line_armed = True
        env.lap_started = False
        env.launched_this_lap = False
        env.lap_time = 0.0
        env.distance_travelled = 0.0
        env.lap_progress_s = 0.0
        env.validated_lap_progress_s = 0.0
        env.launch_gate_progress = 2.0
        env.start_zone_stall_penalty = 0.2
        env.start_zone_spin_penalty = 0.3
        env.start_zone_min_step_distance = 0.25
        env.max_angular_velocity = 5.0
        env.track_length = 10.0
        env.min_lap_time = 5.0
        env.min_lap_distance = 4.0
        env.min_lap_progress_ratio = 0.95
        env.prev_start_line_value = -0.1
        return env

    def test_clip_progress_delta_caps_credit_to_real_displacement(self):
        env = self._make_env()

        clipped = env._clip_progress_delta(5.0, step_distance=0.1)

        self.assertAlmostEqual(clipped, 0.175)

    def test_speed_reward_scale_requires_forward_aligned_progress(self):
        env = self._make_env()

        self.assertEqual(env._speed_reward_scale(0.0, 1.0, 1.0), 0.0)
        self.assertEqual(env._speed_reward_scale(0.5, 1.0, -0.5), 0.0)
        self.assertAlmostEqual(env._speed_reward_scale(0.5, 1.0, 0.8), 0.4)

    def test_start_zone_caps_positive_reward_until_line_rearms(self):
        env = self._make_env()
        env.start_line_armed = False

        capped = env._cap_start_zone_reward(1.5, np.array([0.2, 0.0]))
        uncapped = env._cap_start_zone_reward(1.5, np.array([0.2, 2.0]))

        self.assertEqual(capped, 0.0)
        self.assertEqual(uncapped, 1.5)

    def test_start_line_crossing_requires_leaving_line_zone_before_rearming(self):
        env = self._make_env()

        self.assertFalse(env._update_lap_state(np.array([0.1, 0.0]), prog_delta=0.2))
        self.assertTrue(env.lap_started)
        self.assertFalse(env.start_line_armed)

        env.lap_time = 10.0
        env.distance_travelled = 10.0
        env.lap_progress_s = 10.0
        env.validated_lap_progress_s = 8.5
        self.assertFalse(env._update_lap_state(np.array([-0.1, 0.0]), prog_delta=0.0))
        self.assertFalse(env._update_lap_state(np.array([0.1, 0.0]), prog_delta=0.0))

        self.assertFalse(env._update_lap_state(np.array([-0.1, 1.5]), prog_delta=0.0))
        self.assertTrue(env.start_line_armed)
        env.launched_this_lap = True
        self.assertTrue(env._update_lap_state(np.array([0.1, 1.5]), prog_delta=0.0))

    def test_launch_gate_arms_reward_only_after_enough_forward_progress(self):
        env = self._make_env()
        env.lap_started = True

        self.assertFalse(env._maybe_launch_reward())
        env.lap_progress_s = 2.1
        self.assertTrue(env._maybe_launch_reward())
        self.assertTrue(env.launched_this_lap)

    def test_lap_completion_requires_launch_gate(self):
        env = self._make_env()
        env.lap_started = True
        env.start_line_armed = True
        env.lap_time = 10.0
        env.distance_travelled = 10.0
        env.lap_progress_s = 10.0
        env.validated_lap_progress_s = 8.5

        self.assertFalse(env._update_lap_state(np.array([0.1, 1.5]), prog_delta=0.0))

        env.prev_start_line_value = -0.1
        env.launched_this_lap = True
        self.assertTrue(env._update_lap_state(np.array([0.1, 1.5]), prog_delta=0.0))

    def test_start_zone_penalty_hits_spin_without_displacement(self):
        env = self._make_env()

        penalty = env._start_zone_penalty(step_distance=0.0, yaw_rate=5.0)
        moving_penalty = env._start_zone_penalty(step_distance=0.4, yaw_rate=0.0)

        self.assertGreater(penalty, 0.4)
        self.assertLess(moving_penalty, penalty)


if __name__ == "__main__":
    unittest.main()
