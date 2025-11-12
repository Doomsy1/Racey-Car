import pybullet as p
import pybullet_data
import cv2
import numpy as np
import yaml
import os
import time

from environment.track import Track
from environment.camera import RaceCamera


class RaceSimulator:
    def __init__(self):
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

        # Store spawn height for Z position locking
        self.fixed_height = spawn_pos[2]

        # Create track
        self.track = Track(self.config_path)
        self.track.spawn_in_pybullet(self.physics_client)

        track_ids = self.track.get_track_ids()
        self.camera = RaceCamera(self.config_path, track_ids, self.physics_client)

        # Running flag
        self.running = True

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


    def _handle_tank_drive(self, keys):
        # Get velocity limits from config
        max_linear_velocity = float(self.config['physics']['car']['max_linear_velocity'])
        max_angular_velocity = float(self.config['physics']['car']['max_angular_velocity'])

        # Read key states (support empty keys dict)
        keys = keys or {}
        up_pressed = p.B3G_UP_ARROW in keys and (keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN)
        down_pressed = p.B3G_DOWN_ARROW in keys and (keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN)
        left_pressed = p.B3G_LEFT_ARROW in keys and (keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN)
        right_pressed = p.B3G_RIGHT_ARROW in keys and (keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN)

        # Calculate velocity commands (-1 to +1)
        forward_input = (1.0 if up_pressed else 0.0) + (-1.0 if down_pressed else 0.0)
        turn_input = (1.0 if left_pressed else 0.0) + (-1.0 if right_pressed else 0.0)

        # Scale to actual velocities
        linear_velocity = forward_input * max_linear_velocity
        angular_velocity = turn_input * max_angular_velocity

        # Get car's current position and orientation
        car_pos, car_orn = p.getBasePositionAndOrientation(
            self.car_id,
            physicsClientId=self.physics_client
        )

        # Lock Z position to fixed height
        if abs(car_pos[2] - self.fixed_height) > 0.001:
            p.resetBasePositionAndOrientation(
                self.car_id,
                [car_pos[0], car_pos[1], self.fixed_height],
                car_orn,
                physicsClientId=self.physics_client
            )

        # Convert car orientation to rotation matrix
        rot_matrix = np.array(p.getMatrixFromQuaternion(car_orn)).reshape((3, 3))
        car_forward = rot_matrix[:, 0]  # X-axis in car's local frame

        # Compute linear velocity in world frame
        linear_velocity_world = linear_velocity * car_forward

        # Set base velocity (linear in world frame, angular around world Z-axis)
        p.resetBaseVelocity(
            objectUniqueId=self.car_id,
            linearVelocity=[linear_velocity_world[0], linear_velocity_world[1], 0.0],
            angularVelocity=[0.0, 0.0, angular_velocity],
            physicsClientId=self.physics_client
        )

    def run(self):
        while self.running:
            # Check for quit key from PyBullet window
            keys = p.getKeyboardEvents(physicsClientId=self.physics_client)
            if keys:
                if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                    self.running = False
                    break
            
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
