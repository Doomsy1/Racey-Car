# Vision Pipeline (Quick Use)

## Entrypoint (black/white result)

Call:

`vision_pipeline.run.main(use_webcam=False, frame=<numpy_frame>, output_mode="mask")`

Example:

```python
import cv2
from vision_pipeline import run

frame = cv2.imread("input.png")  # grayscale/BGR/BGRA all supported
mask_bgr = run.main(use_webcam=False, frame=frame, output_mode="mask")
```

## What you get

- Return type: `numpy.ndarray`
- Shape: same height/width as input, 3 channels (`H x W x 3`)
- Pixel values:
  - `0` (black) inside the detected lane region (between left/right borders)
  - `255` (white) outside that region
- If borders cannot be formed for a frame, result is fully white.

## Other mode

Use `output_mode="overlay"` to get the visual centerline/border debug image instead of the mask.

## Local webcam test

Run from `src/`:
`python -m vision_pipeline.run --camera-index 0 --show-mask`
