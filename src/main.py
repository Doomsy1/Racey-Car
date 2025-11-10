"""
Main entry point for Racey-Car simulator.

This script initializes the PyBullet simulation, creates the track and camera,
and displays the camera output.
"""

from typing import Dict, Optional

import pybullet as p
import pybullet_data
import cv2
import math
import yaml
import os
import time

from environment.track import Track
from environment.camera import RaceCamera


class RaceSimulator:
    """Main simulation controller."""
    
    # Control Defaults (can be overridden in track_config.yaml)
    DEFAULT_TURN_GAIN = 0.8
    DEFAULT_TURN_GAIN_SPEED_SCALE = 0.6
    DEFAULT_MAX_TURN_RATE_CHANGE = 3.0
    DEFAULT_PREVENT_INNER_WHEEL_REVERSE = True
    DEFAULT_INNER_WHEEL_PROTECT_THRESHOLD = 0.25
    DEFAULT_MIN_FORWARD_WHEEL_FRACTION = 0.2
    DEFAULT_BASE_LINEAR_DAMPING = 0.06
    DEFAULT_BASE_ANGULAR_DAMPING = 0.18
    DEFAULT_MAX_WHEEL_ACCEL = 600.0

    def __init__(self):
        """Initialize the simulation."""
        # Paths
        self.config_path = os.path.join(os.path.dirname(__file__), 'models', 'track_config.yaml')
        self.car_urdf_path = os.path.join(os.path.dirname(__file__), 'models', 'car.urdf')

        # Load configuration
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        if not self.config:
            raise ValueError(f"Configuration file is empty or invalid: {self.config_path}")

        # Initialize PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        gravity = self.config['physics']['gravity']
        p.setGravity(0, 0, gravity)
        p.setTimeStep(self.config['physics']['time_step'])
        # Improve contact stability
        p.setPhysicsEngineParameter(
            numSolverIterations=50,
            erp=0.2,
            contactERP=0.2,
            frictionERP=0.2,
        )

        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Load car
        if not os.path.exists(self.car_urdf_path):
            raise FileNotFoundError(f"Car URDF file not found: {self.car_urdf_path}")
        spawn_pos = self.config['spawn']['position']
        spawn_orn = self.config['spawn']['orientation']
        self.car_id = p.loadURDF(
            self.car_urdf_path,
            basePosition=spawn_pos,
            baseOrientation=spawn_orn
        )
        if self.car_id < 0:
            raise RuntimeError(f"Failed to load car URDF: {self.car_urdf_path}")

        # Get wheel joint indices for tank drive control
        self.wheel_joints = []
        num_joints = p.getNumJoints(self.car_id, physicsClientId=self.physics_client)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.car_id, i, physicsClientId=self.physics_client)
            joint_name = joint_info[1].decode('utf-8')
            if 'wheel' in joint_name.lower():
                self.wheel_joints.append(i)
        
        if not self.wheel_joints:
            raise RuntimeError("No wheel joints found in car URDF. Check car model.")
        
        # Separate left and right wheels for tank drive
        # Left wheels: front_left, rear_left
        # Right wheels: front_right, rear_right
        self.left_wheel_joints = []
        self.right_wheel_joints = []
        for i in self.wheel_joints:
            joint_info = p.getJointInfo(self.car_id, i, physicsClientId=self.physics_client)
            joint_name = joint_info[1].decode('utf-8').lower()
            if 'left' in joint_name:
                self.left_wheel_joints.append(i)
            elif 'right' in joint_name:
                self.right_wheel_joints.append(i)
        
        if not self.left_wheel_joints or not self.right_wheel_joints:
            raise RuntimeError("Could not identify left/right wheel pairs. Check car model joint names.")
        
        # Configure dynamics for stability (friction/damping)
        # Ground plane friction
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0, spinningFriction=0.0,
                         rollingFriction=0.0, restitution=0.0, physicsClientId=self.physics_client)
        # Base link damping to reduce jitter
        base_linear_damping = float(self.config['physics'].get('base_linear_damping', self.DEFAULT_BASE_LINEAR_DAMPING))
        base_angular_damping = float(self.config['physics'].get('base_angular_damping', self.DEFAULT_BASE_ANGULAR_DAMPING))
        p.changeDynamics(self.car_id, -1, linearDamping=base_linear_damping, angularDamping=base_angular_damping, restitution=0.0,
                         physicsClientId=self.physics_client)
        # Wheel contact/friction
        for joint_idx in self.wheel_joints:
            p.changeDynamics(self.car_id, joint_idx, lateralFriction=0.8, restitution=0.0,
                             rollingFriction=0.0008, spinningFriction=0.0005,
                             physicsClientId=self.physics_client)

        # Create track
        self.track = Track(self.config_path)
        self.track.spawn_in_pybullet(self.physics_client)

        # Create camera (needs track IDs for segmentation)
        track_ids = self.track.get_track_ids()
        self.camera = RaceCamera(self.config_path, track_ids, self.physics_client)

        # Running flag
        self.running = True

        # Previous wheel targets for slew limiting
        self.prev_left_velocity = 0.0
        self.prev_right_velocity = 0.0
        self.prev_turn_cmd = 0.0

        # Runtime polish: ensure explicit non-realtime stepping and declutter GUI
        p.setRealTimeSimulation(0, physicsClientId=self.physics_client)
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physics_client)
        except Exception:
            # Some builds may not support this flag; ignore if unavailable
            pass

        # Initialize OpenCV window (lazy init to avoid headless issues)
        try:
            cv2.namedWindow('Race Camera', cv2.WINDOW_NORMAL)
        except Exception:
            pass
        
        print("Simulation ready. Arrow keys: drive | Q: quit")


    def _handle_tank_drive(self, keys: Optional[Dict[int, int]]) -> None:
        """
        Handle tank drive control using arrow keys.
        
        Tank drive:
        - Up arrow: both sides forward
        - Down arrow: both sides backward
        - Left arrow: left backward, right forward (rotate left)
        - Right arrow: right backward, left forward (rotate right)
        
        Args:
            keys: Dictionary from p.getKeyboardEvents() (may be None or empty)
        """
        # Essential config (must be in YAML)
        max_wheel_velocity = float(self.config['physics']['car'].get('max_wheel_velocity', 15.0))
        wheel_max_force = float(self.config['physics']['car'].get('wheel_max_force', 0.05))
        in_place_turn_velocity = float(self.config['physics']['car'].get('in_place_turn_velocity', max_wheel_velocity * 0.5))
        
        # Optional config (defaults defined in class constants)
        max_wheel_accel = float(self.config['physics']['car'].get('max_wheel_accel', self.DEFAULT_MAX_WHEEL_ACCEL))
        dt = float(self.config['physics'].get('time_step', 0.03333333333333333))
        
        # Determine forward/turn from keys (supports empty keys)
        keys = keys or {}
        up_pressed = p.B3G_UP_ARROW in keys and (keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN)
        down_pressed = p.B3G_DOWN_ARROW in keys and (keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN)
        left_pressed = p.B3G_LEFT_ARROW in keys and (keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN)
        right_pressed = p.B3G_RIGHT_ARROW in keys and (keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN)

        forward_input = (1.0 if up_pressed else 0.0) + (-1.0 if down_pressed else 0.0)
        turn_input = (1.0 if right_pressed else 0.0) + (-1.0 if left_pressed else 0.0)

        # Input shaping (squared inputs preserve sign, smooth near zero)
        def shape(x: float) -> float:
            return math.copysign(x * x, x)

        forward = shape(forward_input)
        turn_gain = float(self.config['physics']['car'].get('turn_gain', self.DEFAULT_TURN_GAIN))
        # Speed-dependent turn gain scaling to reduce twitch at higher forward speeds
        turn_gain_speed_scale = float(self.config['physics']['car'].get('turn_gain_speed_scale', self.DEFAULT_TURN_GAIN_SPEED_SCALE))
        effective_turn_gain = turn_gain * max(0.0, 1.0 - turn_gain_speed_scale * abs(forward))
        raw_turn = shape(turn_input) * effective_turn_gain

        # Slew-limit the turn command itself (unitless -1..1) for smoother steering transitions
        max_turn_rate_change = float(self.config['physics']['car'].get('max_turn_rate_change', self.DEFAULT_MAX_TURN_RATE_CHANGE))
        allowed_turn_delta = max_turn_rate_change * dt
        turn = self.prev_turn_cmd + max(-allowed_turn_delta, min(allowed_turn_delta, raw_turn - self.prev_turn_cmd))
        self.prev_turn_cmd = turn

        # Arcade/tank mixing
        left_cmd = max(-1.0, min(1.0, forward + turn))
        right_cmd = max(-1.0, min(1.0, forward - turn))

        # In-place spin has its own limited velocity
        if abs(forward) < 1e-6 and abs(turn) > 0.0:
            spin = math.copysign(in_place_turn_velocity, turn)
            target_left = spin
            target_right = -spin
        else:
            # Prevent inner wheel from reversing when moving forward/back for more comfortable arcs
            prevent_inner_reverse = bool(self.config['physics']['car'].get('prevent_inner_wheel_reverse', self.DEFAULT_PREVENT_INNER_WHEEL_REVERSE))
            protect_threshold = float(self.config['physics']['car'].get('inner_wheel_protect_threshold', self.DEFAULT_INNER_WHEEL_PROTECT_THRESHOLD))
            min_forward_frac = float(self.config['physics']['car'].get('min_forward_wheel_fraction', self.DEFAULT_MIN_FORWARD_WHEEL_FRACTION))

            if prevent_inner_reverse and abs(forward) >= protect_threshold:
                if forward > 0.0:
                    min_wheel = min_forward_frac * forward
                    left_cmd = max(left_cmd, min_wheel)
                    right_cmd = max(right_cmd, min_wheel)
                elif forward < 0.0:
                    max_wheel = min_forward_frac * forward  # negative
                    left_cmd = min(left_cmd, max_wheel)
                    right_cmd = min(right_cmd, max_wheel)

            target_left = max(-1.0, min(1.0, left_cmd)) * max_wheel_velocity
            target_right = max(-1.0, min(1.0, right_cmd)) * max_wheel_velocity

        # Slew rate limiting to avoid step changes
        allowed_delta = max_wheel_accel * dt
        new_left = self.prev_left_velocity + max(-allowed_delta, min(allowed_delta, target_left - self.prev_left_velocity))
        new_right = self.prev_right_velocity + max(-allowed_delta, min(allowed_delta, target_right - self.prev_right_velocity))

        self.prev_left_velocity = new_left
        self.prev_right_velocity = new_right
        
        # Apply velocity control with limited force (acts like torque cap)
        for joint_idx in self.left_wheel_joints:
            p.setJointMotorControl2(self.car_id, joint_idx, p.VELOCITY_CONTROL,
                                    targetVelocity=new_left, force=wheel_max_force,
                                    physicsClientId=self.physics_client)
        for joint_idx in self.right_wheel_joints:
            p.setJointMotorControl2(self.car_id, joint_idx, p.VELOCITY_CONTROL,
                                    targetVelocity=new_right, force=wheel_max_force,
                                    physicsClientId=self.physics_client)

    def run(self) -> None:
        """
        Main simulation loop.

        Handles physics stepping, camera capture, and display.
        """
        while self.running:
            # Check for quit key from PyBullet window
            keys = p.getKeyboardEvents(physicsClientId=self.physics_client)
            if keys:
                if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                    self.running = False
                    break
            
            # Handle tank drive with arrow keys (always call, even if keys is None)
            self._handle_tank_drive(keys if keys else {})

            # Step physics
            p.stepSimulation(physicsClientId=self.physics_client)

            # Capture and display camera frame
            frame = self.camera.capture_frame(self.car_id)
            try:
                cv2.imshow('Race Camera', frame)
                # Check for quit key from OpenCV window
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    self.running = False
                    break
            except Exception:
                pass

            # Small sleep to control simulation speed
            time.sleep(self.config['physics']['time_step'])

        # Cleanup
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        p.disconnect(physicsClientId=self.physics_client)


def main():
    """Entry point."""
    try:
        simulator = RaceSimulator()
        simulator.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
