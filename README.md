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

### Vision pipeline tuning

-   Set `vision_pipeline.config.FRAME_WIDTH`/`FRAME_HEIGHT` to `None` when you want to use the camera's native resolution. The pipeline will automatically expand the ROI to the full frame when both values are unspecified.
-   Adjust `vision_pipeline.config.ROI.force_full_frame` if you always want the full view, regardless of the configured resolution.
-   Narrow black lines can be emphasized by tweaking `vision_pipeline.config.THIN_LINE` (post-threshold dilation) or lowering `vision_pipeline.config.SCANLINES.min_blob_width`. Morphological opening is disabled by default so thin tape is not eroded away.
-   For a minimal usage guide and the black/white mask entrypoint, see `src/vision_pipeline/README.md`.
