import cv2
import numpy as np


def contour_to_points(contour):
    return contour.squeeze().astype(np.float64)


def order_contour_by_y(points):
    pts = points.copy()
    start_idx = np.argmin(pts[:, 1])
    
    ordered = [pts[start_idx]]
    remaining = list(range(len(pts)))
    remaining.remove(start_idx)
    
    while remaining:
        last = ordered[-1]
        dists = [np.linalg.norm(pts[i] - last) for i in remaining]
        nearest = remaining[np.argmin(dists)]
        ordered.append(pts[nearest])
        remaining.remove(nearest)
    
    return np.array(ordered)


def resample_contour_open(points, num_points):
    if len(points) < 2:
        return points
    
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = cumlen[-1]
    
    if total_len == 0:
        return points[:num_points]
    
    target_lengths = np.linspace(0, total_len, num_points)
    
    resampled = np.zeros((num_points, 2))
    for i, t in enumerate(target_lengths):
        idx = np.searchsorted(cumlen, t) - 1
        idx = np.clip(idx, 0, len(points) - 2)
        seg_len = seg_lengths[idx]
        frac = (t - cumlen[idx]) / seg_len if seg_len > 1e-10 else 0
        resampled[i] = points[idx] + frac * (points[idx + 1] - points[idx])
    
    return resampled


def align_open_contours(ref_pts, target_pts):
    dist_fwd = np.sum(np.linalg.norm(ref_pts - target_pts, axis=1))
    dist_rev = np.sum(np.linalg.norm(ref_pts - target_pts[::-1], axis=1))
    return target_pts[::-1] if dist_rev < dist_fwd else target_pts


def extract_track_borders(image_path, num_points=500, poly_degree=4, visualize=True):
    img = cv2.imread(image_path)
    if img is None:
        print("Could not load image.")
        return None

    height, width = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    inverted = cv2.bitwise_not(blurred)
    _, binary_mask = cv2.threshold(inverted, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.erode(binary_mask, kernel, iterations=3)
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=3)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    track_contours = [c for c in contours if cv2.contourArea(c) >= 500]
    
    print(f"Found {len(track_contours)} track contours")

    if len(track_contours) < 2:
        print("Need at least 2 contours")
        return None

    track_contours = sorted(track_contours, key=cv2.contourArea, reverse=True)[:2]

    # Convert, order, resample
    inner_pts = contour_to_points(track_contours[0])
    outer_pts = contour_to_points(track_contours[1])
    
    inner_ordered = order_contour_by_y(inner_pts)
    outer_ordered = order_contour_by_y(outer_pts)
    
    inner_resampled = resample_contour_open(inner_ordered, num_points)
    outer_resampled = resample_contour_open(outer_ordered, num_points)
    
    outer_aligned = align_open_contours(inner_resampled, outer_resampled)
    
    # Get raw centerline points by averaging
    raw_centerline = (inner_resampled + outer_aligned) / 2
    
    # Fit polynomial: x = f(y)
    ys = raw_centerline[:, 1]
    xs = raw_centerline[:, 0]
    
    coeffs = np.polyfit(ys, xs, poly_degree)
    poly = np.poly1d(coeffs)
    
    print(f"Polynomial coefficients (degree {poly_degree}): {coeffs}")
    
    # Generate centerline ONLY within the y-range where we have data
    y_min = int(np.min(ys))
    y_max = int(np.max(ys))
    
    print(f"Centerline valid y-range: {y_min} to {y_max}")
    
    y_vals = np.arange(y_min, y_max + 1)
    x_vals = poly(y_vals)
    x_vals = np.clip(x_vals, 0, width - 1)
    
    centerline = np.column_stack([x_vals, y_vals])
    
    # Create mask
    centerline_mask = np.zeros((height, width), dtype=np.uint8)
    pts = centerline.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(centerline_mask, [pts], isClosed=False, color=255, thickness=1)
    
    # Track widths from raw centerline
    track_widths = np.linalg.norm(outer_aligned - inner_resampled, axis=1)

    result_img = img.copy()
    
    if visualize:
        # Draw the polynomial centerline (red)
        cv2.polylines(result_img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

        cv2.imshow("Binary Mask", binary_mask)
        cv2.imshow("Centerline Mask", centerline_mask)
        cv2.imshow("Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        'centerline': centerline,
        'centerline_mask': centerline_mask,
        'poly_coeffs': coeffs,
        'track_widths': track_widths,
        'image': result_img
    }


if __name__ == "__main__":
    result = extract_track_borders("sample_images/sample_image1.png", num_points=500, poly_degree=4)
    
    # if result is not None:
    #     np.savez('track_data.npz',
    #              centerline=result['centerline'],
    #              centerline_mask=result['centerline_mask'],
    #              poly_coeffs=result['poly_coeffs'],
    #              track_widths=result['track_widths'])
        
    #     cv2.imwrite('centerline_mask.png', result['centerline_mask'])
        
    #     print("\nSaved: track_data.npz, centerline_mask.png")
    #     print(f"\nTo get x for any y:  x = np.polyval(poly_coeffs, y)")