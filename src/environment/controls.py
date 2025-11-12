import os

import numpy as np
import pybullet as p
import yaml


class TankDriveController:
    """Tank-style drive with Z locking."""

    def __init__(self, config_path, car_id, physics_client):
        """Initialize controller from config."""
        self.car_id = car_id
        self.physics_client = physics_client

        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            raise ValueError(f"Configuration file is empty or invalid: {config_path}")

        # Get velocity limits
        self.max_linear_velocity = float(config['physics']['car']['max_linear_velocity'])
        self.max_angular_velocity = float(config['physics']['car']['max_angular_velocity'])

        # Get fixed height from spawn position
        self.fixed_height = config['spawn']['position'][2]

    def update(self, keys):
        """Update car velocity from keyboard state."""
        keys = keys or {}
        up_pressed = p.B3G_UP_ARROW in keys and (keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN)
        down_pressed = p.B3G_DOWN_ARROW in keys and (keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN)
        left_pressed = p.B3G_LEFT_ARROW in keys and (keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN)
        right_pressed = p.B3G_RIGHT_ARROW in keys and (keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN)

        # Calculate velocity commands (-1 to +1)
        forward_input = (1.0 if up_pressed else 0.0) + (-1.0 if down_pressed else 0.0)
        turn_input = (1.0 if left_pressed else 0.0) + (-1.0 if right_pressed else 0.0)

        # Scale to actual velocities
        linear_velocity = forward_input * self.max_linear_velocity
        angular_velocity = turn_input * self.max_angular_velocity

        car_pos, car_orn = p.getBasePositionAndOrientation(
            self.car_id,
            physicsClientId=self.physics_client
        )

        if abs(car_pos[2] - self.fixed_height) > 0.001:
            p.resetBasePositionAndOrientation(
                self.car_id,
                [car_pos[0], car_pos[1], self.fixed_height],
                car_orn,
                physicsClientId=self.physics_client
            )

        rot_matrix = np.array(p.getMatrixFromQuaternion(car_orn)).reshape((3, 3))
        car_forward = rot_matrix[:, 0]  # X-axis in car's local frame

        linear_velocity_world = linear_velocity * car_forward

        p.resetBaseVelocity(
            objectUniqueId=self.car_id,
            linearVelocity=[linear_velocity_world[0], linear_velocity_world[1], 0.0],
            angularVelocity=[0.0, 0.0, angular_velocity],
            physicsClientId=self.physics_client
        )
