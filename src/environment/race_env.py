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
    """Gymnasium environment with handcrafted state features and optional camera render."""

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
        reward_speed_scale: float = 10.0,
        off_track_distance_scale: float = 5.0,
        allow_reverse: bool = False,
        throttle_bias: float = 0.0,
        capture_camera: bool = False,
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

        # Track helper
        self.track = Track(self.config_path)
        self.inner_polygon = self.track.inner_points[:, :2]
        self.outer_polygon = self.track.outer_points[:, :2]
        self.track_width = self.track.outer_radius - self.track.inner_radius
        self._force_centerline_spawn()
        self._ensure_valid_spawn_pose()
        self.track_length = float(self.track.total_length)

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
        self.off_track_distance_scale = float(off_track_distance_scale)
        self.allow_reverse = bool(allow_reverse)
        self.throttle_bias = float(throttle_bias)
        self.capture_camera = bool(capture_camera)

        # Camera config (still used for rendering; observations are handcrafted features)
        camera_cfg = self.config["camera"]
        self.camera_width = int(camera_cfg["resolution"]["width"])
        self.camera_height = int(camera_cfg["resolution"]["height"])
        self.observation_scale = float(observation_scale)
        self.obs_width = max(1, int(round(self.camera_width * self.observation_scale)))
        self.obs_height = max(
            1, int(round(self.camera_height * self.observation_scale))
        )
        # Feature vector: [speed_norm, lateral_error_norm, heading_align, on_track, yaw_rate_norm]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.physics_client: Optional[int] = None
        self.car_id: Optional[int] = None
        self.camera: Optional[RaceCamera] = None
        self.start_line_id: Optional[int] = None
        self.latest_frame: Optional[np.ndarray] = None
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
        self.elapsed_steps = 0
        self.prev_start_line_value = -0.1
        self.start_line_normal = self._calc_start_line_normal()
        self.start_line_point = self.spawn_position[:2].copy()
        self.last_position_xy = self.spawn_position[:2].copy()
        self.np_random = np.random.default_rng()
        self.current_linear_velocity = 0.0
        self.prev_progress_s = 0.0

    # ------------------------------------------------------------------ Gym API
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self._connect_if_needed()
        self._reset_simulation_world()

        self.prev_progress_s, _, _ = self._track_progress(self.spawn_position[:2])
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

        self._apply_action(action)
        p.stepSimulation(physicsClientId=self.physics_client)
        self.elapsed_steps += 1

        observation = self._get_observation()
        car_pos, car_orn = p.getBasePositionAndOrientation(
            self.car_id, physicsClientId=self.physics_client
        )
        car_xy = np.array(car_pos[:2], dtype=float)

        terminated = False
        truncated = False
        info: Dict[str, Any] = {"event": "running", "lap_time": self.lap_time}
        reward = 0.0
        info["speed"] = abs(self.current_linear_velocity)

        progress_s, lateral_error, tangent_vec = self._track_progress(car_xy)
        prog_delta = self._progress_delta(progress_s, self.prev_progress_s)
        # Track boundary check (use margin to avoid flicker at the edges).
        off_track_margin = max(0.05, self.track_width * 0.1)
        off_track_boundary = self.track_width * 0.5 + off_track_margin
        off_track = abs(lateral_error) > off_track_boundary
        if off_track:
            # Penalize distance beyond the boundary to encourage returning.
            excess_dist = abs(lateral_error) - off_track_boundary
            scaled_excess = max(0.0, excess_dist) * self.off_track_distance_scale
            capped_excess = min(scaled_excess, 50.0)
            distance_penalty = capped_excess * np.log1p(capped_excess)
            reward -= distance_penalty
            info["event"] = "off_track"
            if self.terminate_off_track:
                terminated = True
        else:
            self.prev_progress_s = progress_s

            lap_complete = self._update_lap_state(car_xy, prog_delta)
            if self.lap_started:
                self.lap_time += self.time_step
                step_distance = np.linalg.norm(car_xy - self.last_position_xy)
                self.distance_travelled += step_distance

            # Speed incentive: scale by alignment to avoid rewarding speed when misaligned.
            speed_ratio = abs(self.current_linear_velocity) / max(
                self.max_linear_velocity, 1e-6
            )
            alignment = self._heading_alignment_reward(tangent_vec)
            alignment_weight = 0.5 * (alignment + 1.0)
            scaled_speed = np.tanh(speed_ratio) * self.speed_reward_scale
            speed_term = max(0.0, scaled_speed)
            # n log n growth to avoid exponential blow-ups.
            reward += (speed_term * np.log1p(speed_term)) * alignment_weight

            if lap_complete:
                terminated = True
                info["event"] = "lap_complete"
                info["lap_time"] = self.lap_time

        self.last_position_xy = car_xy

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
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(0, physicsClientId=self.physics_client)

    def _reset_simulation_world(self) -> None:
        assert self.physics_client is not None
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.physics_client)
        p.setTimeStep(self.time_step, physicsClientId=self.physics_client)

        p.loadURDF("plane.urdf")
        self.track.spawn_in_pybullet(self.physics_client)
        track_ids = self.track.get_track_ids()
        self.camera = RaceCamera(self.config_path, track_ids, self.physics_client)

        self._spawn_car()
        self._spawn_start_line_marker()

        self.lap_started = False
        self.lap_time = 0.0
        self.distance_travelled = 0.0
        self.lap_progress_s = 0.0
        self.elapsed_steps = 0
        self.prev_start_line_value = -0.1
        self.last_position_xy = self.spawn_position[:2].copy()
        self.current_linear_velocity = 0.0

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

    def _apply_action(self, action: np.ndarray) -> float:
        assert self.car_id is not None and self.physics_client is not None
        throttle = float(action[0])
        steer = float(action[1])
        if not self.allow_reverse and throttle < 0.0:
            throttle = 0.0

        g = 9.81
        speed = self.current_linear_velocity

        # Longitudinal acceleration command with distinct accel/brake limits.
        a_long_cmd = throttle * self.max_longitudinal_accel
        if a_long_cmd < 0:
            a_long_cmd = np.clip(a_long_cmd, -self.max_brake_accel, 0.0)
        else:
            a_long_cmd = np.clip(a_long_cmd, 0.0, self.max_longitudinal_accel)

        # Drag components (linear + quadratic) oppose motion.
        drag_accel = np.sign(speed) * (
            self.linear_drag * abs(speed) + self.aero_drag * speed * speed
        )
        a_long_net = a_long_cmd - drag_accel

        # Friction-circle: limit combined longitudinal + lateral acceleration.
        friction_limit = self.friction_coefficient * g
        a_long_clipped = np.clip(a_long_net, -friction_limit, friction_limit)

        desired_angular_velocity = steer * self.max_angular_velocity

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
        assert self.camera is not None and self.car_id is not None
        # Capture only if explicitly requested for rendering.
        if self.capture_camera:
            self.latest_frame = self.camera.capture_frame(self.car_id)

        # Handcrafted feature vector for MLPPolicy (based on current pose/velocity)
        car_pos, car_orn = p.getBasePositionAndOrientation(
            self.car_id, physicsClientId=self.physics_client
        )
        car_xy = np.array(car_pos[:2], dtype=float)
        progress_s, lateral_error, tangent_vec = self._track_progress(car_xy)

        lin_vel, ang_vel = p.getBaseVelocity(
            self.car_id, physicsClientId=self.physics_client
        )
        vel_xy = np.array(lin_vel[:2], dtype=float)
        speed = float(np.linalg.norm(vel_xy))
        speed_norm = speed / max(self.max_linear_velocity, 1e-6)
        lateral_norm = lateral_error / max(self.track_width * 0.5, 1e-6)
        heading_align = self._heading_alignment_reward(tangent_vec)
        # On-track flag: 1 if any wheel is on the track surface.
        on_track = 1.0 if self._any_wheel_on_track() else 0.0
        yaw_rate_norm = float(ang_vel[2]) / max(self.max_angular_velocity, 1e-6)

        speed_norm = float(np.clip(speed_norm, 0.0, 1.0))
        lateral_norm = float(np.clip(lateral_norm, -1.0, 1.0))
        heading_align = float(np.clip(heading_align, -1.0, 1.0))
        yaw_rate_norm = float(np.clip(yaw_rate_norm, -1.0, 1.0))

        features = np.array(
            [speed_norm, lateral_norm, heading_align, on_track, yaw_rate_norm],
            dtype=np.float32,
        )
        return features

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
        margin = max(0.05, self.track_width * 0.1)
        dist_outer = self._distance_to_polyline(point_xy, self.outer_polygon)
        dist_inner = self._distance_to_polyline(point_xy, self.inner_polygon)

        near_outer = dist_outer <= margin
        near_inner = dist_inner <= margin

        return (near_outer and not inside_inner) or (inside_outer and near_inner)

    def _update_lap_state(self, car_xy: np.ndarray, prog_delta: float) -> bool:
        was_lap_started = self.lap_started
        line_value = self._start_line_value(car_xy)
        crossed = line_value >= 0.0 and self.prev_start_line_value < 0.0
        self.prev_start_line_value = line_value

        if crossed and not was_lap_started:
            # First time leaving the grid starts the lap timer.
            self.lap_started = True
            self.lap_time = 0.0
            self.distance_travelled = 0.0
            self.lap_progress_s = 0.0

        if self.lap_started:
            self.lap_progress_s += prog_delta

        if crossed and was_lap_started:
            min_progress = self.track_length * self.min_lap_progress_ratio
            if (
                self.lap_time >= self.min_lap_time
                and self.distance_travelled >= self.min_lap_distance
                and self.lap_progress_s >= min_progress
            ):
                return True
        return False

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
