import pybullet as p
import pybullet_data
import yaml
import os
import time
import numpy as np

from environment.track import Track
from environment.controls import TankDriveController
from environment.friction_obstacles import spawn_friction_obstacles, spawn_track_friction_surface


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

        # # Spawn friction pads with different colors (COMMENTED OUT)
        # self.friction_obstacle_ids = []
        # friction_obstacles_config = self.config.get('friction_obstacles', {})
        # if friction_obstacles_config.get('enabled', False):
        #     self.friction_obstacle_ids = spawn_friction_obstacles(
        #         self.track,
        #         self.physics_client,
        #         self.car_id,
        #         friction_obstacles_config
        #     )
        #     print(f"Spawned {len(self.friction_obstacle_ids)} friction pads with different colors")

        # Spawn track-wide friction surface
        self.track_friction_surface_ids = []
        track_friction_config = self.config.get('track_friction_surface', {})
        if track_friction_config.get('enabled', False):
            self.track_friction_surface_ids = spawn_track_friction_surface(
                self.track,
                self.physics_client,
                self.car_id,
                track_friction_config
            )
            friction_coeff = track_friction_config.get('friction_coefficient', 3)
            print(f"Spawned track-wide friction surface with {len(self.track_friction_surface_ids)} segments (friction: {friction_coeff})")

        self.controller = TankDriveController(self.config_path, self.car_id, self.physics_client)

        self.running = True
        self.bird_eye_view = False  # Toggle for bird's-eye camera

        # Non-realtime stepping
        p.setRealTimeSimulation(0, physicsClientId=self.physics_client)
        
        # Enable GUI for free camera movement (camera starts in normal/free mode)
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=self.physics_client)
            # Camera starts in normal/free mode - user can scroll/rotate with mouse
            # Press 'S' to toggle to bird's-eye view
        except Exception:
            pass
        
        print("Simulation ready. Arrow keys: drive | S: toggle bird's-eye view | Q: quit")
        print("Use mouse to scroll/rotate camera in PyBullet window")


    def _update_camera_view(self):
        """Update camera to bird's-eye view above car."""
        car_pos, car_orn = p.getBasePositionAndOrientation(
            self.car_id,
            physicsClientId=self.physics_client
        )
        
        # Bird's-eye view: camera above car looking down
        camera_height = 5.0  # Height above car
        camera_pos = [car_pos[0], car_pos[1], car_pos[2] + camera_height]
        target_pos = [car_pos[0], car_pos[1], car_pos[2]]
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 1, 0]
        )
        
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_height,
            cameraYaw=0,
            cameraPitch=-90,  # Look straight down
            cameraTargetPosition=car_pos,
            physicsClientId=self.physics_client
        )

    def run(self):
        while self.running:
            keys = p.getKeyboardEvents(physicsClientId=self.physics_client)
            if keys:
                if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                    self.running = False
                    break
                elif ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
                    self.bird_eye_view = not self.bird_eye_view
                    if self.bird_eye_view:
                        self._update_camera_view()
                        print("Switched to bird's-eye view (press S again to free camera)")
                    else:
                        print("Switched to free camera mode (use mouse to scroll/rotate)")

            # Update controller with keyboard input
            self.controller.update(keys if keys else {})

            # Update bird's-eye camera to follow car if enabled
            if self.bird_eye_view:
                self._update_camera_view()

            p.stepSimulation(physicsClientId=self.physics_client)

            time.sleep(self.config['physics']['time_step'])

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
