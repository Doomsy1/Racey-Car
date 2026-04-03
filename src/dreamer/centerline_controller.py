"""PD controller that follows the track centerline for expert demonstration collection."""

import numpy as np
import pybullet as p


class CenterlineController:
    """Produces [throttle, steer] actions to follow the track centerline.

    Uses the env's track.project_onto_centerline() to get lateral error and tangent,
    then applies a PD controller on lateral error for steering and a fixed throttle.

    The controller reads car state directly from the PyBullet physics client
    attached to each environment instance.
    """

    def __init__(
        self,
        kp: float = 2.0,
        kd: float = 0.5,
        base_throttle: float = 0.5,
        throttle_steer_reduction: float = 0.3,
    ):
        self.kp = kp
        self.kd = kd
        self.base_throttle = base_throttle
        self.throttle_steer_reduction = throttle_steer_reduction
        # Per-env previous lateral error for derivative term
        self._prev_lateral: dict[int, float] = {}

    def reset(self, env_id: int = 0) -> None:
        self._prev_lateral.pop(env_id, None)

    def act(self, env: "RaceCarEnv", env_id: int = 0) -> np.ndarray:
        """Compute a centerline-following action for the given env.

        Args:
            env: A RaceCarEnv instance with an active physics client.
            env_id: Stream identifier for tracking per-env derivative state.

        Returns:
            np.ndarray of shape (2,) with [throttle, steer], clipped to action_space.
        """
        car_pos, car_orn = p.getBasePositionAndOrientation(
            env.car_id, physicsClientId=env.physics_client
        )
        car_xy = np.array(car_pos[:2], dtype=float)

        # Get lateral error and tangent — use cached values from env.step() if available
        # to avoid duplicating the expensive projection computation.
        if hasattr(env, 'last_lateral_error') and env.last_lateral_error is not None:
            lateral_error = env.last_lateral_error
            tangent = env.last_tangent_vec
        else:
            _progress_s, lateral_error, tangent = env.track.project_onto_centerline(car_xy)

        # Compute car heading from quaternion
        rot_matrix = np.array(p.getMatrixFromQuaternion(car_orn)).reshape((3, 3))
        car_forward = rot_matrix[:2, 0]
        car_forward_norm = np.linalg.norm(car_forward)
        if car_forward_norm > 1e-6:
            car_forward = car_forward / car_forward_norm

        # Heading error: cross product of tangent and car_forward gives sin(angle)
        # Positive cross → car is pointing right of tangent → need to steer left
        heading_error = float(tangent[0] * car_forward[1] - tangent[1] * car_forward[0])

        # Derivative of lateral error
        prev = self._prev_lateral.get(env_id, lateral_error)
        d_lateral = lateral_error - prev
        self._prev_lateral[env_id] = lateral_error

        # PD steering: negative lateral_error means car is right of center,
        # positive means left. Steer to reduce error.
        # Also incorporate heading error for smoother tracking.
        steer = -self.kp * lateral_error - self.kd * d_lateral - 1.0 * heading_error
        steer = float(np.clip(steer, -1.0, 1.0))

        # Reduce throttle in proportion to steering magnitude for smoother cornering
        throttle = self.base_throttle * (1.0 - self.throttle_steer_reduction * abs(steer))
        throttle = float(np.clip(throttle, 0.0, 1.0))

        return np.array([throttle, steer], dtype=np.float32)
