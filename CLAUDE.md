# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyBullet-based race track simulator with camera line extraction for autonomous vehicle testing. The camera captures track boundaries using PyBullet's segmentation masks (not color filtering), outputting clean binary images of track lines for computer vision algorithms.

## Running the Simulation

```bash
# Install dependencies
uv pip install -r requirements.txt
# or with pip
pip install -r requirements.txt

# Run the simulation
cd src
python main.py
```

**Controls**:
- Arrow Keys: Tank drive control
  - `↑`: Forward
  - `↓`: Backward
  - `←`: Rotate left
  - `→`: Rotate right
  - Combinations: Forward/Backward + Left/Right for turning while moving
- `S`: Toggle bird's-eye view
- `Q`: Quit

## Architecture

### Critical Design Pattern: Segmentation Mask-Based Line Extraction

The track line extraction relies on PyBullet's segmentation masks, NOT color filtering. This is a key architectural constraint:

1. **Track Construction** ([track.py](src/environment/track.py)): The track is built from cylinder primitives, where each cylinder segment has a unique PyBullet body ID
2. **Camera Capture** ([camera.py](src/environment/camera.py)): Uses PyBullet's `ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX` flag to capture body IDs in the segmentation buffer
3. **Binary Output**: The camera filters pixels by checking if their body ID matches any track segment ID, creating a black-on-white binary image

### Core Components

- **RaceSimulator** ([main.py](src/main.py)): Main simulation loop, coordinates all components, handles keyboard input
- **Track** ([track.py](src/environment/track.py)): Generates circular track geometry from cylinder segments based on YAML config
- **RaceCamera** ([camera.py](src/environment/camera.py)): Captures segmentation mask images and converts them to binary track line images
- **TankDriveController** ([controls.py](src/environment/controls.py)): Implements tank-style driving with Z-axis height locking
- **BirdEyeTransform** ([transforms.py](src/environment/transforms.py)): Inverse perspective mapping to convert camera view to top-down bird's-eye view

### Configuration

All simulation parameters are centralized in [src/models/track_config.yaml](src/models/track_config.yaml):
- Track geometry (inner/outer radius, segment count)
- Camera parameters (resolution, FOV, pitch, position offset)
- Physics settings (time step, gravity, velocity limits)
- Spawn position and orientation
- Bird's-eye transform parameters

### Physics Model

- Non-realtime physics stepping for deterministic simulation
- Gravity disabled (`gravity: 0.0`) for simplified 2D-like motion
- Tank drive with velocity-based control (no force-based actuation)
- Z-axis height is locked to prevent vertical drift

### Bird's-Eye View Transform

The bird's-eye transform ([transforms.py](src/environment/transforms.py)) performs inverse perspective mapping:
- Projects camera frustum edges to the ground plane to determine BEV bounds
- Uses camera intrinsics (computed from FOV and resolution) for pixel-to-ray unprojection
- Back-projects ground coordinates to image coordinates using `cv2.remap`
- Configurable scale via `pixels_per_meter` in YAML
