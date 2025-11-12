import pybullet as p
import pybullet_data
import cv2
import yaml
import os
import time

from environment.track import Track
from environment.camera import RaceCamera
from environment.controls import TankDriveController
from environment.transforms import BirdEyeTransform


class RaceSimulator:
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(__file__), 'models', 'track_config.yaml')
        self.car_urdf_path = os.path.join(os.path.dirname(__file__), 'models', 'car.urdf')

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        if not self.config:
            raise ValueError(f"Configuration file is empty or invalid: {self.config_path}")

        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        gravity = self.config['physics']['gravity']
        p.setGravity(0, 0, gravity)
        p.setTimeStep(self.config['physics']['time_step'])

        self.plane_id = p.loadURDF("plane.urdf")

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

        self.track = Track(self.config_path)
        self.track.spawn_in_pybullet(self.physics_client)

        track_ids = self.track.get_track_ids()
        self.camera = RaceCamera(self.config_path, track_ids, self.physics_client)

        self.controller = TankDriveController(self.config_path, self.car_id, self.physics_client)

        self.bird_eye_transform = BirdEyeTransform(self.config_path)

        self.running = True

        self.bird_eye_view = False

        # Non-realtime stepping and reduced GUI clutter
        p.setRealTimeSimulation(0, physicsClientId=self.physics_client)
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physics_client)
        except Exception:
            pass

        try:
            cv2.namedWindow('Race Camera', cv2.WINDOW_NORMAL)
        except Exception:
            pass
        
        print("Simulation ready. Arrow keys: drive | S: toggle view | Q: quit")


    def run(self):
        while self.running:
            keys = p.getKeyboardEvents(physicsClientId=self.physics_client)
            if keys:
                if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                    self.running = False
                    break

            self.controller.update(keys if keys else {})

            p.stepSimulation(physicsClientId=self.physics_client)

            frame = self.camera.capture_frame(self.car_id)

            if self.bird_eye_view:
                frame = self.bird_eye_transform.apply(frame)

            try:
                cv2.imshow('Race Camera', frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    self.running = False
                    break
                elif key in (ord('s'), ord('S')):
                    self.bird_eye_view = not self.bird_eye_view
                    view_mode = "bird's-eye" if self.bird_eye_view else "camera"
                    print(f"Switched to {view_mode} view")
            except Exception:
                pass

            time.sleep(self.config['physics']['time_step'])

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
