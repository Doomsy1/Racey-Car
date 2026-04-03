import os
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import yaml
from gymnasium import spaces

from environment.camera import RaceCamera
from environment.track import Track


class RaceCarEnv(gym.Env):
    """Gymnasium environment with top-down FOV occupancy observations."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config_path: Optional[str] = None,
        gui: bool = False,
        max_episode_duration: Optional[float] = 120.0,
        min_lap_time: float = 5.0,
        min_lap_distance: float = 4.0,
        min_lap_progress_ratio: float = 0.95,
        observation_scale: float = 0.25,
        terminate_off_track: bool = False,
        reward_speed_scale: float = 3.0,
        reward_progress_weight: float = 1.0,
        reward_speed_weight: float = 0.1,
        off_track_penalty: float = -1.0,
        off_track_distance_scale: float = 0.0,
        allow_reverse: bool = False,
        throttle_bias: float = 0.0,
        capture_camera: bool = False,
        cache_track: bool = False,
        track_seed: Optional[int] = None,
        gate_count: int = 40,
        gate_reward: float = 1.0,
        lap_bonus: float = 5.0,
        steer_rate_penalty: float = 0.02,
        steer_speed_scale: float = 1.0,
        gate_heading_reward: float = 0.0,
        gate_heading_deg: float = 15.0,
        vehicle_mass: float = 1.0,
        weight_shift_grip_loss: float = 0.25,
        weight_shift_slip_drag: float = 1.5,
        wheelbase_m: float = 0.3,
        axle_track_m: float = 0.2,
        cg_height_m: float = 0.05,
        front_weight_bias: float = 0.5,
        load_sensitivity: float = 0.15,
        random_spawn: bool = False,
        lookahead_distance: float = 1.5,
        ray_count: int = 7,
        max_consecutive_off_track_steps: int = 20,
        stall_penalty: float = 0.1,
    ):
        super().__init__()
        self.gui = gui
        self.render_mode = "rgb_array"

        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models", "track_config.yaml"
        )
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        if not self.config:
            raise ValueError(
                f"Configuration file is empty or invalid: {self.config_path}"
            )

        # Physics / car tuning
        self.time_step = float(self.config["physics"]["time_step"])
        self.gravity = float(self.config["physics"]["gravity"])
        car_cfg = self.config["physics"]["car"]
        self.max_linear_velocity = float(car_cfg["max_linear_velocity"])
        self.max_angular_velocity = float(car_cfg["max_angular_velocity"])
        self.max_longitudinal_accel = float(
            car_cfg.get("max_linear_accel", self.max_linear_velocity * 2.0)
        )
        self.max_brake_accel = float(
            car_cfg.get("max_brake_accel", self.max_longitudinal_accel)
        )
        self.linear_drag = float(car_cfg.get("linear_drag", 0.2))
        self.aero_drag = float(car_cfg.get("aero_drag", 0.05))
        self.cornering_grip_scale = float(car_cfg.get("cornering_grip_scale", 0.6))
        self.friction_coefficient = float(car_cfg.get("friction_coefficient", 1.1))
        self.spawn_position = np.array(self.config["spawn"]["position"], dtype=float)
        self.spawn_orientation = tuple(self.config["spawn"]["orientation"])
        self.spawn_centerline_index = int(
            self.config["spawn"].get("centerline_index", 0)
        )

        self.max_episode_duration = max_episode_duration
        self.max_episode_steps = (
            int(np.ceil(max_episode_duration / self.time_step))
            if max_episode_duration is not None
            else None
        )
        self.min_lap_time = min_lap_time
        self.min_lap_distance = min_lap_distance
        if not (0.0 < min_lap_progress_ratio <= 1.0):
            raise ValueError("min_lap_progress_ratio must be in the range (0, 1].")
        self.min_lap_progress_ratio = float(min_lap_progress_ratio)
        self.terminate_off_track = bool(terminate_off_track)
        self.speed_reward_scale = float(reward_speed_scale)
        self.reward_progress_weight = float(reward_progress_weight)
        self.reward_speed_weight = float(reward_speed_weight)
        self.off_track_penalty = float(off_track_penalty)
        self.off_track_distance_scale = float(off_track_distance_scale)
        self.allow_reverse = bool(allow_reverse)
        self.throttle_bias = float(throttle_bias)
        self.capture_camera = bool(capture_camera)
        self.cache_track = bool(cache_track)
        self.track_seed = track_seed
        self.gate_count = max(0, int(gate_count))
        self.gate_reward = float(gate_reward)
        self.lap_bonus = float(lap_bonus)
        self.gate_heading_reward = float(gate_heading_reward)
        self.gate_heading_deg = float(gate_heading_deg)
        self.steer_rate_penalty = float(steer_rate_penalty)
        self.steer_speed_scale = float(steer_speed_scale)
        self.vehicle_mass = max(1e-6, float(vehicle_mass))
        self.weight_shift_grip_loss = float(weight_shift_grip_loss)
        self.weight_shift_slip_drag = float(weight_shift_slip_drag)
        self.wheelbase_m = max(1e-3, float(wheelbase_m))
        self.axle_track_m = max(1e-3, float(axle_track_m))
        self.cg_height_m = max(1e-3, float(cg_height_m))
        self.front_weight_bias = float(np.clip(front_weight_bias, 0.3, 0.7))
        self.load_sensitivity = float(load_sensitivity)
        self.random_spawn = bool(random_spawn)
        self.lookahead_distance = float(lookahead_distance)
        self.ray_count = max(1, int(ray_count))
        self.max_consecutive_off_track_steps = max(1, int(max_consecutive_off_track_steps))
        self.stall_penalty = float(stall_penalty)
        self.curvature_speed_lookahead = 3.0

        # Track helper
        self.track = Track(self.config_path, seed=self.track_seed)
        self.centerline_points = self.track.centerline_points
        self.inner_polygon = self.track.inner_points[:, :2]
        self.outer_polygon = self.track.outer_points[:, :2]
        self.track_width = self.track.outer_radius - self.track.inner_radius
        self.track_margin = max(0.05, self.track_width * 0.1)
        self.occupancy_blend_width = max(0.05, self.track_width * 0.5)
        self.progress_credit_multiplier = 1.25
        self.progress_credit_slack = max(0.05, self.track_width * 0.05)
        self.start_line_rearm_distance = max(1.0, self.track_width * 0.75)
        self.start_zone_radius = max(1.5, self.track_width * 1.5)
        self.launch_gate_progress = max(self.track_width * 3.0, self.max_linear_velocity * self.time_step * 12.0)
        self.start_zone_min_step_distance = max(0.05, self.max_linear_velocity * self.time_step * 0.25)
        self.start_zone_stall_penalty = 0.2
        self.start_zone_spin_penalty = 0.3
        self._force_centerline_spawn()
        self._ensure_valid_spawn_pose()
        self.track_length = float(self.track.total_length)
        self.gate_positions = self._build_gate_positions(self.track_length, self.gate_count)

        # Camera config (used for rendering + FOV geometry)
        camera_cfg = self.config["camera"]
        self.camera_width = int(camera_cfg["resolution"]["width"])
        self.camera_height = int(camera_cfg["resolution"]["height"])
        self.observation_scale = float(observation_scale)
        # Build a square, driver-centric occupancy map size from observation_scale.
        # SB3 NatureCNN needs >= 36 pixels per side (8x8/4, 4x4/2, 3x3/1 conv stack).
        self.fov_grid_size = int(np.clip(round(256 * self.observation_scale), 36, 128))
        self.fov_degrees = float(camera_cfg.get("fov", 90.0))
        self.fov_range = float(
            max(
                self.track_width * 2.0,
                min(12.0, float(camera_cfg.get("far_plane", 25.0)) * 0.4),
            )
        )
        self.rear_fov_fraction = 0.2
        self.fov_rear_range = self.fov_range * self.rear_fov_fraction
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(2, self.fov_grid_size, self.fov_grid_size),
            dtype=np.uint8,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # Action: [target_speed_norm, steer]

        self.physics_client: Optional[int] = None
        self.car_id: Optional[int] = None
        self.camera: Optional[RaceCamera] = None
        self.start_line_id: Optional[int] = None
        self.latest_frame: Optional[np.ndarray] = None
        self._track_spawned = False
        self.track_ids_set: set[int] = set()
        self.wheel_link_names = [
            "front_left_wheel",
            "front_right_wheel",
            "rear_left_wheel",
            "rear_right_wheel",
        ]
        # Populated after loading the URDF.
        self.wheel_link_indices: list[int] = []

        self.lap_started = False
        self.lap_time = 0.0
        self.distance_travelled = 0.0
        self.lap_progress_s = 0.0
        self.validated_lap_progress_s = 0.0
        self.launched_this_lap = False
        self.laps_completed = 0
        self.elapsed_steps = 0
        self.prev_start_line_value = -0.1
        self.start_line_normal = self._calc_start_line_normal()
        self.start_line_point = self.spawn_position[:2].copy()
        self.last_position_xy = self.spawn_position[:2].copy()
        self.np_random = np.random.default_rng()
        self.current_linear_velocity = 0.0
        self.prev_progress_s = 0.0
        self.next_gate_index = 0
        self.prev_steer = 0.0
        self.consecutive_off_track_steps = 0
        self.start_line_armed = True

    # ------------------------------------------------------------------ Gym API
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        if self.random_spawn:
            self._sample_random_spawn_pose()

        self._connect_if_needed()
        self._reset_simulation_world()
        self._reset_start_line_pose()

        self.prev_progress_s, self.last_lateral_error, self.last_tangent_vec = self._track_progress(self.spawn_position[:2])
        self.next_gate_index = 0
        self.prev_steer = 0.0
        observation = self._get_observation()
        info = {"lap_time": self.lap_time, "event": "reset"}
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.car_id is None:
            raise RuntimeError("Environment must be reset before calling step().")

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.throttle_bias != 0.0:
            action = action.copy()
            action[0] = np.clip(
                action[0] + self.throttle_bias,
                self.action_space.low[0],
                self.action_space.high[0],
            )

        steer_rate_cost = 0.0
        if self.steer_rate_penalty > 0.0:
            steer_rate_cost = self.steer_rate_penalty * abs(
                float(action[1]) - self.prev_steer
            )
        self.prev_steer = float(action[1])

        self._apply_action(action)
        p.stepSimulation(physicsClientId=self.physics_client)
        self.elapsed_steps += 1

        observation = self._get_observation()
        car_pos, _ = p.getBasePositionAndOrientation(
            self.car_id, physicsClientId=self.physics_client
        )
        car_xy = np.array(car_pos[:2], dtype=float)

        terminated = False
        truncated = False
        info: Dict[str, Any] = {"event": "running", "lap_time": self.lap_time}
        reward = 0.0
        info["speed"] = abs(self.current_linear_velocity)
        if steer_rate_cost:
            reward -= steer_rate_cost

        progress_s, lateral_error, tangent_vec = self._track_progress(car_xy)
        self.last_lateral_error = lateral_error
        self.last_tangent_vec = tangent_vec
        raw_prog_delta = self._progress_delta(progress_s, self.prev_progress_s)
        step_distance = float(np.linalg.norm(car_xy - self.last_position_xy))
        prog_delta = self._clip_progress_delta(raw_prog_delta, step_distance)
        # Track boundary check (use margin to avoid flicker at the edges).
        off_track_boundary = self.track_width * 0.5 + self.track_margin
        off_track = abs(lateral_error) > off_track_boundary
        if off_track:
            self.consecutive_off_track_steps += 1
            # Graduated penalty: scales with distance beyond the track edge so
            # the agent gets a reward gradient pointing back toward the track.
            overshoot = abs(lateral_error) - off_track_boundary
            scale = 1.0 + self.off_track_distance_scale * overshoot / max(self.track_width, 1e-6)
            reward = self.off_track_penalty * scale
            info["event"] = "off_track"
            if self.terminate_off_track:
                terminated = True
            elif self.consecutive_off_track_steps >= self.max_consecutive_off_track_steps:
                terminated = True
                info["event"] = "consecutive_off_track"
        else:
            self.consecutive_off_track_steps = 0
            self.prev_progress_s = progress_s

            lap_complete = self._update_lap_state(car_xy, prog_delta)
            if self.lap_started:
                self.lap_time += self.time_step
                self.distance_travelled += step_distance
                self._maybe_launch_reward()
                if self.launched_this_lap and self.gate_positions.size > 0:
                    while (
                        self.next_gate_index < len(self.gate_positions)
                        and self.lap_progress_s
                        >= self.gate_positions[self.next_gate_index]
                    ):
                        reward += self.gate_reward
                        self.next_gate_index += 1

            # Dense reward: centerline progress + curvature-aware speed bonus.
            max_step_progress = max(self.max_linear_velocity * self.time_step, 1e-6)
            progress_reward = float(np.clip(prog_delta / max_step_progress, -1.0, 1.0))

            # Compute target speed based on upcoming curvature.
            lookahead_curv = self.track.get_lookahead_curvature(
                progress_s, self.curvature_speed_lookahead
            )
            curvature_max = self.track.curvature_max
            target_speed = self.max_linear_velocity * (
                1.0 - lookahead_curv / max(curvature_max, 1e-6)
            )
            target_speed = float(np.clip(
                target_speed,
                0.2 * self.max_linear_velocity,
                self.max_linear_velocity,
            ))
            current_speed = abs(self.current_linear_velocity)
            speed_reward = 1.0 - abs(current_speed - target_speed) / max(
                self.max_linear_velocity, 1e-6
            )
            heading_alignment = max(0.0, self._heading_alignment_reward(tangent_vec))
            speed_scale = self._speed_reward_scale(
                prog_delta, max_step_progress, heading_alignment
            )

            reward += (
                self.reward_progress_weight * progress_reward
                + self.reward_speed_weight * float(speed_reward) * speed_scale
            )
            if self.lap_started and not lap_complete:
                reward = self._cap_start_zone_reward(reward, car_xy, launched=self.launched_this_lap)
                if not self.launched_this_lap and self._in_start_zone(car_xy):
                    _, ang_vel = p.getBaseVelocity(self.car_id, physicsClientId=self.physics_client)
                    reward -= self._start_zone_penalty(step_distance, float(ang_vel[2]))

            if lap_complete:
                # Do NOT terminate — ending the episode on lap completion lets the
                # agent reward-hack by farming per-step progress for the full episode
                # instead of completing laps (one lap at max speed earns ~232 reward
                # in ~175 steps; circles for 6000 steps earn ~1200 with no lap).
                # Instead: give a discrete bonus, reset lap counters, and continue.
                # The next start-line crossing (after prev_start_line_value goes
                # negative again) will naturally start the next lap.
                reward += self.lap_bonus
                self.laps_completed += 1
                info["event"] = "lap_complete"
                info["lap_time"] = self.lap_time
                info["laps_completed"] = self.laps_completed
                self.lap_started = False
                self.lap_time = 0.0
                self.distance_travelled = 0.0
                self.lap_progress_s = 0.0
                self.validated_lap_progress_s = 0.0
                self.launched_this_lap = False
                self.next_gate_index = 0
                # prev_start_line_value intentionally NOT reset here — it stays
                # positive so crossing is only re-detected after the car passes
                # back below zero (i.e., after a full forward approach next lap).

        self.last_position_xy = car_xy

        if self.stall_penalty > 0.0:
            speed_frac = abs(self.current_linear_velocity) / max(self.max_linear_velocity, 1e-6)
            reward -= self.stall_penalty * (1.0 - min(speed_frac, 1.0))

        if (
            not terminated
            and self.max_episode_steps is not None
            and self.elapsed_steps >= self.max_episode_steps
        ):
            truncated = True
            info["event"] = "timeout"

        reward = float(np.clip(reward, -1000.0, 1000.0))
        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if (
            self.latest_frame is None
            and self.camera is not None
            and self.car_id is not None
        ):
            self.latest_frame = self.camera.capture_frame(self.car_id)
        return self.latest_frame

    def close(self) -> None:
        if self.physics_client is not None:
            try:
                p.disconnect(physicsClientId=self.physics_client)
            finally:
                self.physics_client = None
                self.car_id = None

    # ---------------------------------------------------------------- Utilities
    def _connect_if_needed(self) -> None:
        if self.physics_client is not None:
            return
        mode = p.GUI if self.gui else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        p.setRealTimeSimulation(0, physicsClientId=self.physics_client)

    def _reset_simulation_world(self) -> None:
        assert self.physics_client is not None
        if not self.cache_track or not self._track_spawned:
            p.resetSimulation(physicsClientId=self.physics_client)
            p.setGravity(0, 0, self.gravity, physicsClientId=self.physics_client)
            p.setTimeStep(self.time_step, physicsClientId=self.physics_client)

            p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
            self.track.spawn_in_pybullet(self.physics_client)
            track_ids = self.track.get_track_ids()
            self.camera = RaceCamera(self.config_path, track_ids, self.physics_client)
            inner_ids, outer_ids = track_ids
            self.track_ids_set = set(inner_ids + outer_ids)

            self._spawn_car()
            self._spawn_start_line_marker()
            self._track_spawned = True
        else:
            if self.car_id is None:
                self._spawn_car()
            else:
                self._reset_car_pose()

        self.lap_started = False
        self.lap_time = 0.0
        self.distance_travelled = 0.0
        self.lap_progress_s = 0.0
        self.validated_lap_progress_s = 0.0
        self.launched_this_lap = False
        self.laps_completed = 0
        self.elapsed_steps = 0
        self.prev_start_line_value = -0.1
        self.last_position_xy = self.spawn_position[:2].copy()
        self.current_linear_velocity = 0.0
        self.consecutive_off_track_steps = 0
        self.start_line_armed = True
        self.latest_frame = None

    def _spawn_car(self) -> None:
        assert self.physics_client is not None
        car_urdf_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models", "car.urdf"
        )
        if not os.path.exists(car_urdf_path):
            raise FileNotFoundError(f"Car URDF file not found: {car_urdf_path}")

        self.car_id = p.loadURDF(
            car_urdf_path,
            basePosition=self.spawn_position.tolist(),
            baseOrientation=self.spawn_orientation,
            physicsClientId=self.physics_client,
        )
        self._cache_wheel_links()
        self._reset_car_pose()

    def _reset_car_pose(self) -> None:
        assert self.car_id is not None and self.physics_client is not None
        p.resetBasePositionAndOrientation(
            self.car_id,
            self.spawn_position.tolist(),
            self.spawn_orientation,
            physicsClientId=self.physics_client,
        )
        p.resetBaseVelocity(
            self.car_id,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            physicsClientId=self.physics_client,
        )

    def _spawn_start_line_marker(self) -> None:
        assert self.physics_client is not None
        line_half_extents = [
            0.02,
            max(self.track_width * 0.5, 0.1),
            0.001,
        ]
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=line_half_extents,
            physicsClientId=self.physics_client,
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=line_half_extents,
            rgbaColor=[1.0, 0.2, 0.2, 1.0],
            physicsClientId=self.physics_client,
        )

        start_pos = self.spawn_position.copy()
        start_pos[2] = self.track.line_height * 1.5

        self.start_line_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=start_pos.tolist(),
            baseOrientation=self.spawn_orientation,
            physicsClientId=self.physics_client,
        )
        self._reset_start_line_pose()

    def _reset_start_line_pose(self) -> None:
        if self.start_line_id is None or self.physics_client is None:
            return
        self.start_line_normal = self._calc_start_line_normal()
        self.start_line_point = self.spawn_position[:2].copy()
        start_pos = self.spawn_position.copy()
        start_pos[2] = self.track.line_height * 1.5
        p.resetBasePositionAndOrientation(
            self.start_line_id,
            start_pos.tolist(),
            self.spawn_orientation,
            physicsClientId=self.physics_client,
        )

    def _sample_random_spawn_pose(self) -> None:
        """Randomize spawn along the centerline for training diversity."""
        if self.centerline_points is None or len(self.centerline_points) == 0:
            return
        idx = int(self.np_random.integers(0, len(self.centerline_points)))
        pos, quat = self.track.get_spawn_pose(
            index=idx, height=float(self.spawn_position[2])
        )
        self.spawn_position = pos
        self.spawn_orientation = tuple(quat)

    def _apply_action(self, action: np.ndarray) -> float:
        assert self.car_id is not None and self.physics_client is not None
        throttle = float(action[0])
        steer = float(action[1])

        g = 9.81
        speed = self.current_linear_velocity

        # Interpret throttle as a target speed command (normalized).
        max_accel = self.max_longitudinal_accel / self.vehicle_mass
        max_brake = self.max_brake_accel / self.vehicle_mass
        target_speed = throttle * self.max_linear_velocity
        if not self.allow_reverse and target_speed < 0.0:
            target_speed = 0.0
        speed_error = target_speed - speed
        desired_accel = speed_error / max(self.time_step, 1e-6)
        a_long_cmd = np.clip(desired_accel, -max_brake, max_accel)

        # Drag components (linear + quadratic) oppose motion.
        drag_accel = np.sign(speed) * (
            self.linear_drag * abs(speed) + self.aero_drag * speed * speed
        )
        speed_ratio = min(1.0, abs(speed) / max(self.max_linear_velocity, 1e-6))
        transfer_long = abs(a_long_cmd) / max(max(max_accel, max_brake), 1e-6)
        transfer_lat = abs(steer) * speed_ratio
        combined_transfer = transfer_long + transfer_lat
        slip_excess = max(0.0, combined_transfer - 1.0)
        slip_drag = slip_excess * self.weight_shift_slip_drag * g
        a_long_net = a_long_cmd - drag_accel - np.sign(speed) * slip_drag

        # Friction-circle: limit combined longitudinal + lateral acceleration.
        transfer_ratio = min(1.0, combined_transfer / 2.0)
        effective_mu = self.friction_coefficient * (
            1.0 - self.weight_shift_grip_loss * transfer_ratio
        )
        effective_mu = max(0.1 * self.friction_coefficient, effective_mu)
        base_friction_limit = effective_mu * g
        a_long_clipped = np.clip(a_long_net, -base_friction_limit, base_friction_limit)

        steering_scale = 1.0 / (1.0 + self.steer_speed_scale * speed_ratio)
        desired_angular_velocity = steer * self.max_angular_velocity * steering_scale

        # Estimate per-wheel loads for load-sensitive grip.
        total_weight = self.vehicle_mass * g
        static_front = total_weight * self.front_weight_bias
        static_rear = total_weight - static_front
        long_transfer = (self.vehicle_mass * a_long_clipped * self.cg_height_m) / self.wheelbase_m
        front_load = static_front + long_transfer
        rear_load = static_rear - long_transfer
        lat_transfer = (
            self.vehicle_mass
            * (speed * desired_angular_velocity)
            * self.cg_height_m
            / self.axle_track_m
        )
        front_left = front_load * 0.5 + lat_transfer * 0.5
        front_right = front_load * 0.5 - lat_transfer * 0.5
        rear_left = rear_load * 0.5 + lat_transfer * 0.5
        rear_right = rear_load * 0.5 - lat_transfer * 0.5
        loads = np.array([front_left, front_right, rear_left, rear_right], dtype=float)
        min_load = 0.05 * total_weight / 4.0
        loads = np.maximum(loads, min_load)
        avg_load = total_weight / 4.0
        load_ratio = loads / max(avg_load, 1e-6)
        mu_per_wheel = self.friction_coefficient * (
            1.0 - self.load_sensitivity * (load_ratio - 1.0)
        )
        mu_per_wheel = np.clip(mu_per_wheel, 0.1 * self.friction_coefficient, None)
        friction_limit = float(np.sum(mu_per_wheel * loads) / self.vehicle_mass)
        a_long_clipped = np.clip(a_long_net, -friction_limit, friction_limit)

        # Determine max allowable yaw rate given remaining lateral budget.
        if speed != 0.0:
            remaining_lat = max(0.0, friction_limit**2 - a_long_clipped**2) ** 0.5
            max_yaw_from_lat = remaining_lat / abs(speed)
            angular_velocity = np.clip(
                desired_angular_velocity, -max_yaw_from_lat, max_yaw_from_lat
            )
        else:
            angular_velocity = desired_angular_velocity

        # Integrate speed with clipped acceleration.
        self.current_linear_velocity = speed + a_long_clipped * self.time_step
        self.current_linear_velocity = np.clip(
            self.current_linear_velocity,
            -self.max_linear_velocity,
            self.max_linear_velocity,
        )
        if not self.allow_reverse and self.current_linear_velocity < 0.0:
            self.current_linear_velocity = 0.0

        car_pos, car_orn = p.getBasePositionAndOrientation(
            self.car_id, physicsClientId=self.physics_client
        )
        rot_matrix = np.array(p.getMatrixFromQuaternion(car_orn)).reshape((3, 3))
        car_forward = rot_matrix[:, 0]
        linear_velocity_world = self.current_linear_velocity * car_forward

        p.resetBaseVelocity(
            self.car_id,
            [linear_velocity_world[0], linear_velocity_world[1], 0.0],
            [0.0, 0.0, angular_velocity],
            physicsClientId=self.physics_client,
        )

        # Ensure the car stays glued to the track plane.
        fixed_height = self.spawn_position[2]
        if abs(car_pos[2] - fixed_height) > 1e-3:
            p.resetBasePositionAndOrientation(
                self.car_id,
                [car_pos[0], car_pos[1], fixed_height],
                car_orn,
                physicsClientId=self.physics_client,
            )

        return throttle

    def _get_observation(self) -> np.ndarray:
        assert self.car_id is not None and self.physics_client is not None
        # Capture only if explicitly requested for rendering.
        if self.capture_camera and self.camera is not None:
            self.latest_frame = self.camera.capture_frame(self.car_id)
        return self._build_driver_fov_observation()

    def _build_driver_fov_observation(self) -> np.ndarray:
        """Build 2-channel observation: occupancy + (speed-bar, prev-steer)."""
        assert self.car_id is not None and self.physics_client is not None
        n = self.fov_grid_size
        if n <= 1:
            return np.zeros((2, n, n), dtype=np.uint8)

        car_pos, car_orn = p.getBasePositionAndOrientation(
            self.car_id, physicsClientId=self.physics_client
        )
        car_xy = np.array(car_pos[:2], dtype=float)
        rot_matrix = np.array(p.getMatrixFromQuaternion(car_orn)).reshape((3, 3))
        forward = rot_matrix[:2, 0]
        norm = float(np.linalg.norm(forward))
        if norm <= 1e-8:
            forward = np.array([1.0, 0.0], dtype=float)
        else:
            forward = forward / norm
        right = np.array([forward[1], -forward[0]], dtype=float)

        rows = np.arange(n, dtype=float)
        cols = np.arange(n, dtype=float)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")

        forward_extent = self.fov_range
        rear_extent = self.fov_rear_range
        forward_dist = ((n - 1 - rr) / max(1.0, n - 1)) * (
            forward_extent + rear_extent
        ) - rear_extent
        tan_half = np.tan(np.deg2rad(self.fov_degrees) * 0.5)
        lateral_span_max = forward_extent * tan_half
        lateral_dist = ((cc / max(1.0, n - 1)) * 2.0 - 1.0) * lateral_span_max

        valid = np.abs(lateral_dist) <= (lateral_span_max + 1e-6)
        if not np.any(valid):
            return np.zeros((2, n, n), dtype=np.uint8)

        world_xy = (
            car_xy[None, :]
            + forward_dist[..., None] * forward[None, None, :]
            + lateral_dist[..., None] * right[None, None, :]
        )
        valid_points = world_xy[valid]
        on_track_values = self._points_track_occupancy_values(valid_points)

        grid = np.zeros((n, n), dtype=np.uint8)
        grid[valid] = on_track_values

        # Encode second channel as two spatial bars so the CNN sees
        # the car's current speed and recent steering action:
        #
        # Top half:    speed bar (left-anchored horizontal fill)
        #              wider = faster, empty = stopped
        #
        # Bottom half: previous steer needle (center-anchored)
        #              right-fill = last steer was rightward (+)
        #              left-fill  = last steer was leftward  (-)
        #
        # Stacked across 4 frame-stack steps this gives the policy its recent
        # speed history and action sequence without any physics internals.
        dynamics = np.zeros((n, n), dtype=np.uint8)
        split = n // 2
        center_col = n // 2

        speed_norm = abs(self.current_linear_velocity) / max(
            self.max_linear_velocity, 1e-6
        )
        speed_fill = int(round(min(speed_norm, 1.0) * n))
        if speed_fill > 0:
            dynamics[:split, :speed_fill] = 255

        steer_norm = float(np.clip(self.prev_steer, -1.0, 1.0))
        steer_fill = int(round(abs(steer_norm) * center_col))
        if steer_fill > 0:
            if steer_norm >= 0:
                dynamics[split:, center_col : min(n, center_col + steer_fill)] = 255
            else:
                dynamics[split:, max(0, center_col - steer_fill) : center_col] = 255

        return np.stack([grid, dynamics], axis=0)

    def _compute_lookahead_distance(self) -> float:
        """Cast rays within camera FOV to estimate clear distance to track walls."""
        if self.physics_client is None or self.car_id is None or not self.track_ids_set:
            return self.lookahead_distance
        car_pos, car_orn = p.getBasePositionAndOrientation(
            self.car_id, physicsClientId=self.physics_client
        )
        rot_matrix = np.array(p.getMatrixFromQuaternion(car_orn)).reshape((3, 3))
        forward = rot_matrix[:, 0]
        up = rot_matrix[:, 2]
        origin = np.array(car_pos, dtype=float) + up * 0.05

        fov_rad = np.deg2rad(self.camera.fov)
        if self.ray_count == 1:
            angles = [0.0]
        else:
            angles = np.linspace(-fov_rad * 0.5, fov_rad * 0.5, self.ray_count)

        max_dist = float(self.camera.far)
        ray_from = []
        ray_to = []
        perp = np.array([-forward[1], forward[0], 0.0])
        for ang in angles:
            ca = np.cos(ang)
            sa = np.sin(ang)
            dir_vec = ca * forward + sa * perp
            dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)
            ray_from.append(origin.tolist())
            ray_to.append((origin + dir_vec * max_dist).tolist())

        results = p.rayTestBatch(ray_from, ray_to, physicsClientId=self.physics_client)
        distances = []
        for hit in results:
            hit_id = hit[0]
            hit_frac = hit[2]
            if hit_id in self.track_ids_set and hit_frac >= 0.0:
                distances.append(hit_frac * max_dist)
            else:
                distances.append(max_dist)

        if not distances:
            return self.lookahead_distance
        center_idx = len(distances) // 2
        window = distances[max(0, center_idx - 1) : min(len(distances), center_idx + 2)]
        return float(np.clip(np.mean(window), 0.1, max_dist))

    def _force_centerline_spawn(self) -> None:
        """Set spawn to the track centerline to guarantee a valid start."""
        center_pos, center_quat = self.track.get_spawn_pose(
            index=self.spawn_centerline_index, height=float(self.spawn_position[2])
        )
        self.spawn_position = center_pos
        self.spawn_orientation = tuple(center_quat)

    def _ensure_valid_spawn_pose(self) -> None:
        """Ensure spawn pose lies on the procedurally generated track."""
        if self._is_within_track(self.spawn_position[:2]):
            return
        fallback_pos, fallback_quat = self.track.get_spawn_pose(
            height=float(self.spawn_position[2])
        )
        self.spawn_position = fallback_pos
        self.spawn_orientation = tuple(fallback_quat)

    def _is_within_track(self, point_xy: np.ndarray) -> bool:
        inside_outer = self._point_in_polygon(point_xy, self.outer_polygon)
        inside_inner = self._point_in_polygon(point_xy, self.inner_polygon)
        if inside_outer and not inside_inner:
            return True

        # Allow a small margin near edges to avoid numeric flicker pushing the car off-track.
        dist_outer = self._distance_to_polyline(point_xy, self.outer_polygon)
        dist_inner = self._distance_to_polyline(point_xy, self.inner_polygon)

        near_outer = dist_outer <= self.track_margin
        near_inner = dist_inner <= self.track_margin

        return (near_outer and not inside_inner) or (inside_outer and near_inner)

    def _update_lap_state(self, car_xy: np.ndarray, prog_delta: float) -> bool:
        was_lap_started = self.lap_started
        if (
            not self.start_line_armed
            and np.linalg.norm(car_xy - self.start_line_point) >= self.start_line_rearm_distance
        ):
            self.start_line_armed = True
        line_value = self._start_line_value(car_xy)
        crossed = self.start_line_armed and line_value >= 0.0 and self.prev_start_line_value < 0.0
        self.prev_start_line_value = line_value
        if crossed:
            self.start_line_armed = False

        if crossed and not was_lap_started:
            # First time leaving the grid starts the lap timer.
            self.lap_started = True
            self.lap_time = 0.0
            self.distance_travelled = 0.0
            self.lap_progress_s = 0.0
            self.validated_lap_progress_s = 0.0
            self.launched_this_lap = False

        if self.lap_started:
            self.lap_progress_s += prog_delta
            if self.launched_this_lap:
                self.validated_lap_progress_s += prog_delta

        if crossed and was_lap_started:
            min_progress = self.track_length * self.min_lap_progress_ratio
            min_validated_progress = max(0.0, min_progress - self.launch_gate_progress)
            if (
                self.launched_this_lap
                and self.lap_time >= self.min_lap_time
                and self.distance_travelled >= self.min_lap_distance
                and self.lap_progress_s >= min_progress
                and self.validated_lap_progress_s >= min_validated_progress
            ):
                return True
        return False

    def _clip_progress_delta(self, prog_delta: float, step_distance: float) -> float:
        """Cap projected centerline progress by physically plausible displacement."""
        max_credit = self.progress_credit_slack + self.progress_credit_multiplier * max(step_distance, 0.0)
        return float(np.clip(prog_delta, -max_credit, max_credit))

    def _cap_start_zone_reward(self, reward: float, car_xy: np.ndarray, launched: bool = False) -> float:
        """Prevent line camping from staying net-positive before the line is re-armed."""
        if (
            (not self.start_line_armed or not launched)
            and self._in_start_zone(car_xy)
        ):
            return min(float(reward), 0.0)
        return float(reward)

    def _in_start_zone(self, car_xy: np.ndarray) -> bool:
        return bool(np.linalg.norm(car_xy - self.start_line_point) < self.start_zone_radius)

    def _maybe_launch_reward(self) -> bool:
        """Arm positive dense reward only after the car has clearly left the line and moved forward."""
        if not self.lap_started or self.launched_this_lap:
            return self.launched_this_lap
        if self.lap_progress_s < self.launch_gate_progress:
            return False
        self.launched_this_lap = True
        return True

    def _start_zone_penalty(self, step_distance: float, yaw_rate: float) -> float:
        """Penalize spinning or stalling near the start line before the launch gate."""
        stall_frac = 1.0 - np.clip(step_distance / max(self.start_zone_min_step_distance, 1e-6), 0.0, 1.0)
        yaw_frac = np.clip(abs(yaw_rate) / max(self.max_angular_velocity, 1e-6), 0.0, 1.0)
        penalty = self.start_zone_stall_penalty * stall_frac + self.start_zone_spin_penalty * yaw_frac
        return float(penalty)

    @staticmethod
    def _build_gate_positions(track_length: float, gate_count: int) -> np.ndarray:
        """Place gates strictly inside the lap so the start/finish line is never rewarded."""
        if gate_count <= 0 or track_length <= 0.0:
            return np.array([], dtype=float)
        spacing = track_length / (gate_count + 1)
        return np.arange(1, gate_count + 1, dtype=float) * spacing

    @staticmethod
    def _speed_reward_scale(
        prog_delta: float, max_step_progress: float, heading_alignment: float,
    ) -> float:
        """Reward speed only when it accompanies forward, aligned motion."""
        forward_scale = np.clip(max(0.0, prog_delta) / max(max_step_progress, 1e-6), 0.0, 1.0)
        alignment_scale = np.clip(max(0.0, heading_alignment), 0.0, 1.0)
        return float(forward_scale * alignment_scale)

    def _cache_wheel_links(self) -> None:
        assert self.car_id is not None and self.physics_client is not None
        # Map URDF link names to joint indices for fast wheel position lookups.
        name_to_index: dict[str, int] = {}
        for joint_idx in range(
            p.getNumJoints(self.car_id, physicsClientId=self.physics_client)
        ):
            info = p.getJointInfo(
                self.car_id, joint_idx, physicsClientId=self.physics_client
            )
            link_name = info[12].decode("utf-8")
            if link_name in self.wheel_link_names:
                name_to_index[link_name] = joint_idx
        self.wheel_link_indices = [
            name_to_index[name]
            for name in self.wheel_link_names
            if name in name_to_index
        ]

    def _any_wheel_on_track(self) -> bool:
        assert self.car_id is not None and self.physics_client is not None
        if not self.wheel_link_indices:
            # Fallback to body position if wheel links are unavailable.
            car_pos, _ = p.getBasePositionAndOrientation(
                self.car_id, physicsClientId=self.physics_client
            )
            return self._is_within_track(np.array(car_pos[:2], dtype=float))

        for link_idx in self.wheel_link_indices:
            link_state = p.getLinkState(
                self.car_id,
                link_idx,
                computeForwardKinematics=True,
                physicsClientId=self.physics_client,
            )
            link_pos = link_state[0]
            if self._is_within_track(np.array(link_pos[:2], dtype=float)):
                return True
        return False

    def _start_line_value(self, car_xy: np.ndarray) -> float:
        rel = car_xy - self.start_line_point
        return float(np.dot(rel, self.start_line_normal))

    def _calc_start_line_normal(self) -> np.ndarray:
        rot_matrix = np.array(
            p.getMatrixFromQuaternion(self.spawn_orientation)
        ).reshape((3, 3))
        forward = rot_matrix[:, 0][:2]
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            return np.array([1.0, 0.0], dtype=float)
        return forward / norm

    def _track_progress(self, car_xy: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """Return (arc-length progress, lateral error, tangent) along the centerline."""
        return self.track.project_onto_centerline(car_xy)

    def _progress_delta(self, current_s: float, prev_s: float) -> float:
        """Compute wrapped progress change along the loop."""
        delta = current_s - prev_s
        if delta < -0.5 * self.track_length:
            delta += self.track_length
        elif delta > 0.5 * self.track_length:
            delta -= self.track_length
        return delta

    def _heading_alignment_reward(self, tangent_vec: np.ndarray) -> float:
        """Return cosine alignment between velocity direction and track tangent."""
        if self.car_id is None:
            return 0.0
        lin_vel, _ = p.getBaseVelocity(self.car_id, physicsClientId=self.physics_client)
        vel_xy = np.array(lin_vel[:2], dtype=float)
        speed = np.linalg.norm(vel_xy)
        if speed < 1e-6:
            return 0.0
        vel_dir = vel_xy / speed
        tangent = tangent_vec / (np.linalg.norm(tangent_vec) + 1e-8)
        alignment = float(np.dot(vel_dir, tangent))
        return alignment

    @staticmethod
    def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        x, y = point
        inside = False
        n = len(polygon)
        px1, py1 = polygon[0]
        for i in range(n + 1):
            px2, py2 = polygon[i % n]
            if min(py1, py2) < y <= max(py1, py2) and x <= max(px1, px2):
                if py1 != py2:
                    xinters = (y - py1) * (px2 - px1) / (py2 - py1 + 1e-12) + px1
                else:
                    xinters = px1
                if px1 == px2 or x <= xinters:
                    inside = not inside
            px1, py1 = px2, py2
        return inside

    @staticmethod
    def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """Vectorized point-in-polygon check using ray casting."""
        pts = np.asarray(points, dtype=float)
        poly = np.asarray(polygon, dtype=float)
        if pts.size == 0:
            return np.zeros((0,), dtype=bool)
        px = pts[:, 0][:, None]
        py = pts[:, 1][:, None]
        x1 = poly[:, 0][None, :]
        y1 = poly[:, 1][None, :]
        x2 = np.roll(poly[:, 0], -1)[None, :]
        y2 = np.roll(poly[:, 1], -1)[None, :]

        y_cross = (y1 > py) != (y2 > py)
        x_inter = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1
        x_cross = px < x_inter
        inside = np.sum(y_cross & x_cross, axis=1) % 2 == 1
        return inside

    def _points_within_track(self, points: np.ndarray) -> np.ndarray:
        inside_outer = self._points_in_polygon(points, self.outer_polygon)
        inside_inner = self._points_in_polygon(points, self.inner_polygon)
        return inside_outer & (~inside_inner)

    def _points_within_track_with_margin(self, points: np.ndarray) -> np.ndarray:
        inside_outer = self._points_in_polygon(points, self.outer_polygon)
        inside_inner = self._points_in_polygon(points, self.inner_polygon)
        strict = inside_outer & (~inside_inner)
        if points.size == 0:
            return strict
        dist_outer = self._distance_to_polyline_batch(points, self.outer_polygon)
        dist_inner = self._distance_to_polyline_batch(points, self.inner_polygon)
        near_outer = dist_outer <= self.track_margin
        near_inner = dist_inner <= self.track_margin
        margin_ok = (near_outer & (~inside_inner)) | (inside_outer & near_inner)
        return strict | margin_ok

    def _points_track_occupancy_values(self, points: np.ndarray) -> np.ndarray:
        """Soft occupancy values to reduce boundary flicker in the grid."""
        if points.size == 0:
            return np.zeros((0,), dtype=np.uint8)
        on_track = self._points_within_track_with_margin(points)
        values = np.zeros(points.shape[0], dtype=np.uint8)
        if not np.any(on_track):
            return values
        track_points = points[on_track]
        dist_outer = self._distance_to_polyline_batch(track_points, self.outer_polygon)
        dist_inner = self._distance_to_polyline_batch(track_points, self.inner_polygon)
        edge_dist = np.minimum(dist_outer, dist_inner)
        depth = np.clip(edge_dist / max(self.occupancy_blend_width, 1e-6), 0.0, 1.0)
        values[on_track] = np.round(depth * 255.0).astype(np.uint8)
        return values

    @staticmethod
    def _distance_to_polyline(point: np.ndarray, polygon: np.ndarray) -> float:
        """Return minimum distance from point to a closed polyline."""
        min_dist = float("inf")
        n = len(polygon)
        px, py = point
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            seg_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
            if seg_len_sq < 1e-12:
                dist_sq = (px - x1) ** 2 + (py - y1) ** 2
            else:
                t = max(
                    0.0,
                    min(
                        1.0,
                        ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / seg_len_sq,
                    ),
                )
                proj_x = x1 + t * (x2 - x1)
                proj_y = y1 + t * (y2 - y1)
                dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
            if dist_sq < min_dist:
                min_dist = dist_sq
        return float(np.sqrt(min_dist))

    @staticmethod
    def _distance_to_polyline_batch(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """Return minimum distances from many points to a closed polyline."""
        pts = np.asarray(points, dtype=float)
        poly = np.asarray(polygon, dtype=float)
        if pts.size == 0:
            return np.zeros((0,), dtype=float)
        min_sq = np.full(pts.shape[0], np.inf, dtype=float)
        n = poly.shape[0]
        px = pts[:, 0]
        py = pts[:, 1]
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-12:
                dist_sq = (px - x1) ** 2 + (py - y1) ** 2
            else:
                t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
                t = np.clip(t, 0.0, 1.0)
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
            min_sq = np.minimum(min_sq, dist_sq)
        return np.sqrt(min_sq)
