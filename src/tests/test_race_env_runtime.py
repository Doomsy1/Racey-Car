import unittest
from unittest import mock

import numpy as np

from environment.race_env import RaceCarEnv


class RaceEnvRuntimeTests(unittest.TestCase):
    def test_connect_if_needed_sets_search_path_on_connected_client(self):
        env = RaceCarEnv.__new__(RaceCarEnv)
        env.physics_client = None
        env.gui = False

        with mock.patch("environment.race_env.p.connect", return_value=17) as connect_mock:
            with mock.patch("environment.race_env.p.setAdditionalSearchPath") as search_path_mock:
                with mock.patch("environment.race_env.p.setRealTimeSimulation") as realtime_mock:
                    env._connect_if_needed()

        connect_mock.assert_called_once()
        search_path_mock.assert_called_once()
        self.assertEqual(search_path_mock.call_args.kwargs["physicsClientId"], 17)
        realtime_mock.assert_called_once_with(0, physicsClientId=17)

    def test_reset_simulation_world_loads_plane_on_env_client(self):
        env = RaceCarEnv.__new__(RaceCarEnv)
        env.physics_client = 23
        env.cache_track = False
        env._track_spawned = False
        env.gravity = 0.0
        env.time_step = 0.02
        env.config_path = "models/track_config.yaml"
        env.track = mock.Mock()
        env.track.get_track_ids.return_value = ([1], [2])
        env.spawn_position = np.array([0.0, 0.0, 0.0], dtype=float)
        env._spawn_car = mock.Mock()
        env._spawn_start_line_marker = mock.Mock()

        with mock.patch("environment.race_env.RaceCamera", return_value=mock.Mock()):
            with mock.patch("environment.race_env.p.resetSimulation") as reset_mock:
                with mock.patch("environment.race_env.p.setGravity") as gravity_mock:
                    with mock.patch("environment.race_env.p.setTimeStep") as timestep_mock:
                        with mock.patch("environment.race_env.p.loadURDF") as load_urdf_mock:
                            env._reset_simulation_world()

        reset_mock.assert_called_once_with(physicsClientId=23)
        gravity_mock.assert_called_once_with(0, 0, env.gravity, physicsClientId=23)
        timestep_mock.assert_called_once_with(env.time_step, physicsClientId=23)
        load_urdf_mock.assert_called_once_with("plane.urdf", physicsClientId=23)


if __name__ == "__main__":
    unittest.main()
