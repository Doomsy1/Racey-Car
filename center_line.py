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


def extrapolate_contour_to_edges(points, height, num_fit_points=20):
    """
    Extrapolate a contour to reach y=0 and y=height-1.
    Uses a linear fit on the endpoints to extend smoothly.
    """
    pts = points.copy()
    
    # Sort by y to ensure ordered from top to bottom
    pts = pts[pts[:, 1].argsort()]
    
    y_min, y_max = pts[0, 1], pts[-1, 1]
    
    extended = [pts]
    
    # Extrapolate toward top (y=0) if needed
    if y_min > 0:
        # Fit line to first N points
        top_pts = pts[:num_fit_points]
        coeffs_top = np.polyfit(top_pts[:, 1], top_pts[:, 0], 1)  # x = f(y)
        
        new_ys = np.arange(0, int(y_min))
        new_xs = np.polyval(coeffs_top, new_ys)
        top_extension = np.column_stack([new_xs, new_ys])
        extended.insert(0, top_extension)
    
    # Extrapolate toward bottom (y=height-1) if needed
    if y_max < height - 1:
        # Fit line to last N points
        bottom_pts = pts[-num_fit_points:]
        coeffs_bot = np.polyfit(bottom_pts[:, 1], bottom_pts[:, 0], 1)  # x = f(y)
        
        new_ys = np.arange(int(y_max) + 1, height)
        new_xs = np.polyval(coeffs_bot, new_ys)
        bottom_extension = np.column_stack([new_xs, new_ys])
        extended.append(bottom_extension)
    
    return np.vstack(extended)


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
    
    # Extrapolate contours to image edges BEFORE resampling
    inner_extended = extrapolate_contour_to_edges(inner_ordered, height, num_fit_points=20)
    outer_extended = extrapolate_contour_to_edges(outer_ordered, height, num_fit_points=20)
    
    inner_resampled = resample_contour_open(inner_extended, num_points)
    outer_resampled = resample_contour_open(outer_extended, num_points)
    
    outer_aligned = align_open_contours(inner_resampled, outer_resampled)
    
    # Get raw centerline points by averaging
    raw_centerline = (inner_resampled + outer_aligned) / 2
    
    # Fit polynomial: x = f(y)
    ys = raw_centerline[:, 1]
    xs = raw_centerline[:, 0]
    
    coeffs = np.polyfit(ys, xs, poly_degree)
    poly = np.poly1d(coeffs)
    
    print(f"Polynomial coefficients (degree {poly_degree}): {coeffs}")
    
    # Generate centerline for full image height (now safe because we extrapolated)
    y_vals = np.arange(0, height)
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
        # Draw resampled borders
        for pt in inner_resampled:
            cv2.circle(result_img, tuple(pt.astype(int)), 2, (255, 0, 0), -1)
        for pt in outer_aligned:
            cv2.circle(result_img, tuple(pt.astype(int)), 2, (0, 255, 0), -1)
        
        # Draw correspondence lines between borders
        for i in range(0, len(inner_resampled), 25):
            p1 = tuple(inner_resampled[i].astype(int))
            p2 = tuple(outer_aligned[i].astype(int))
            cv2.line(result_img, p1, p2, (128, 128, 128), 1)
        
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
    result = extract_track_borders("sample_images/sample_image.png", num_points=500, poly_degree=4)
    
    # if result is not None:
    #     np.savez('track_data.npz',
    #              centerline=result['centerline'],
    #              centerline_mask=result['centerline_mask'],
    #              poly_coeffs=result['poly_coeffs'],
    #              track_widths=result['track_widths'])
        
    #     cv2.imwrite('centerline_mask.png', result['centerline_mask'])
        
    #     print("\nSaved: track_data.npz, centerline_mask.png")
    #     print(f"\nTo get x for any y:  x = np.polyval(poly_coeffs, y)")