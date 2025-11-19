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

### Track generation

-   `track.seed` (optional) fixes the random generator so the same racetrack is rebuilt each run. Omit the value for a fresh layout every launch.
-   `oval_ratio`/`oval_rotation` set the base oval footprint, `radius_jitter` controls how far sections can push inward/outward, `num_features`/`straight_feature_ratio`/`feature_width_range` control the number and style of large-scale straights vs. tight bends, while `angle_warp_strength`, `high_freq_scale`, and `num_chicanes`/`chicane_spacing` (fractions of a lap) add long straights, ripples, and S-bends; `inner_radius`/`outer_radius` still define the overall width and the generator automatically dampens offsets to avoid self-intersections.
