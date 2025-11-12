PyBullet-based race track simulator with segmentation mask-based camera line extraction testing.

## Installation

```bash
pip install -r requirements.txt
# or: uv pip install -r requirements.txt
```

## Running

```bash
cd src
python main.py
```

## Controls

-   **Arrow Keys**: Tank drive (↑↓ forward/backward, ←→ rotate)
-   **S**: Toggle bird's‑eye view
-   **Q**: Quit

## Architecture

Uses PyBullet's segmentation masks (not color filtering) for track detection. Each track segment is a cylinder primitive with a unique body ID that appears in the camera's segmentation mask.

## Configuration

Edit `src/models/track_config.yaml` to adjust track geometry, camera parameters, physics settings, and car controls.
