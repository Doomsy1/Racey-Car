import cv2
import numpy as np

def contour_to_points(contour):
    # Converts an OpenCV contour object to a list of (x, y) points
    # OpenCV contour objects have shape (n, 1, 2), the 1 is useless
    # Convert pixel values to NumPy float64 for precise calcs
    return contour.squeeze().astype(np.float64)

def order_contour_by_y(points):
    """
    Rearrange contour points to create a path starting from the topmost point
    and connecting to nearest neighbors. This creates a smooth traversal from
    top to bottom of the track.
    
    Steps:
    1. Find the point with smallest y-value (topmost)
    2. Start our ordered list with that point
    3. Repeatedly find the closest remaining point and add it to our path
    4. Continue until all points are in the ordered path
    """
    pts = points.copy()
    # Find index of topmost point (smallest y coordinate)
    start_idx = np.argmin(pts[:, 1])
    
    # Start building our ordered path
    ordered = [pts[start_idx]]
    remaining = list(range(len(pts)))
    remaining.remove(start_idx)
    
    # Keep adding the nearest point until we've used all points
    while remaining:
        last = ordered[-1]
        # Calculate distance from last point to all remaining points
        dists = [np.linalg.norm(pts[i] - last) for i in remaining]
        # Find which remaining point is closest
        nearest = remaining[np.argmin(dists)]
        ordered.append(pts[nearest])
        remaining.remove(nearest)
    
    return np.array(ordered)


def resample_contour_open(points, num_points):
    """
    Redistribute points evenly along the contour path.
    Instead of having random spacing between points, this creates evenly-spaced
    points along the entire length of the track border.
    
    Think of it like: if you had beads on a string bunched up randomly, this
    spaces them out evenly along the string.
    
    Steps:
    1. Calculate the distance between each consecutive point
    2. Find the total length of the path
    3. Create evenly-spaced target positions along that length
    4. For each target position, interpolate between existing points
    """
    if len(points) < 2:
        return points
    
    # Calculate the distance between each pair of consecutive points
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    
    # Build cumulative distances (how far along the path each point is)
    cumlen = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = cumlen[-1]
    
    if total_len == 0:
        return points[:num_points]
    
    # Create evenly-spaced target positions along the total length
    target_lengths = np.linspace(0, total_len, num_points)
    
    # For each target position, find where it falls and interpolate
    resampled = np.zeros((num_points, 2))
    for i, t in enumerate(target_lengths):
        # Find which segment this target position falls in
        idx = np.searchsorted(cumlen, t) - 1
        idx = np.clip(idx, 0, len(points) - 2)
        
        seg_len = seg_lengths[idx]
        # Calculate how far along this segment we are (0 to 1)
        frac = (t - cumlen[idx]) / seg_len if seg_len > 1e-10 else 0
        # Linear interpolation between the two segment endpoints
        resampled[i] = points[idx] + frac * (points[idx + 1] - points[idx])
    
    return resampled


def align_open_contours(ref_pts, target_pts):
    """
    Make sure two contours are going in the same direction.
    Since we want to match up inner and outer track borders, they need to
    start from the same end. This checks both directions and picks the better match.
    
    Compares the total distance between matched points when going forward vs backward,
    and flips the target if backward matching is better.
    """
    # Sum up distances when points are matched in forward order
    dist_fwd = np.sum(np.linalg.norm(ref_pts - target_pts, axis=1))
    # Sum up distances when target is reversed
    dist_rev = np.sum(np.linalg.norm(ref_pts - target_pts[::-1], axis=1))
    # Return reversed version if it matches better, otherwise return as-is
    return target_pts[::-1] if dist_rev < dist_fwd else target_pts


def extract_track_borders(image_path, num_points=500, poly_degree=4, visualize=True):
    """
    Main function: Extract the centerline of a race track from an image.
    
    Process overview:
    1. Load and preprocess the image to isolate the track
    2. Find the inner and outer borders of the track
    3. Order and align these borders so they correspond
    4. Calculate the centerline by averaging the borders
    5. Fit a smooth polynomial curve through the centerline
    6. Return the centerline and related data
    """
    
    # Step 1: Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Could not load image.")
        return None
    height, width = img.shape[:2]

    # Step 2: Image processing, convert to binary mask
    # Convert to grayscale for simpler processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Invert so track becomes white, background becomes black
        # We do this because OpenCV works better on bright targets
    inverted = cv2.bitwise_not(blurred)
    # Threshold to create pure black and white
    _, binary_mask = cv2.threshold(inverted, 200, 255, cv2.THRESH_BINARY)

    # Step 3: Clean up the binary mask
    # Use morphological operations (erosion then dilation) to remove small noise and fill small gaps
    kernel = np.ones((5, 5), np.uint8) # Kernel is the window size for the erosion/dilation operations
    binary_mask = cv2.erode(binary_mask, kernel, iterations=3) 
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=3) 

    # Step 4: Find the track borders
    # Find all white shapes in the image
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Keep only contours that are large enough to be track borders
    track_contours = [c for c in contours if cv2.contourArea(c) >= 500]
    
    print(f"Found {len(track_contours)} track contours")

    if len(track_contours) < 2:
        print("Need at least 2 contours")
        return None

    # Keep the two largest contours
    track_contours = sorted(track_contours, key=cv2.contourArea, reverse=True)[:2]

    # Step 6: Process the OpenCV Contour objects
    # Convert from OpenCV format to simple point arrays
    inner_pts = contour_to_points(track_contours[0])
    outer_pts = contour_to_points(track_contours[1])
    
    # Order points to create a smooth path from top to bottom
    inner_ordered = order_contour_by_y(inner_pts)
    outer_ordered = order_contour_by_y(outer_pts)
    
    # Redistribute points evenly along each border
    inner_resampled = resample_contour_open(inner_ordered, num_points)
    outer_resampled = resample_contour_open(outer_ordered, num_points)
    
    # Make sure both borders go in the same direction
    outer_aligned = align_open_contours(inner_resampled, outer_resampled)
    
    # Step 7: Calculate the centerline
    # Get the middle point between inner and outer at each position
    raw_centerline = (inner_resampled + outer_aligned) / 2
    
    # Step 8: Fit a smooth polynomial curve
    ys = raw_centerline[:, 1]
    xs = raw_centerline[:, 0]
    
    # Find polynomial coefficients that best fit our centerline points
    coeffs = np.polyfit(ys, xs, poly_degree)
    poly = np.poly1d(coeffs)
    
    print(f"Polynomial coefficients (degree {poly_degree}): {coeffs}")
    
    # Only generate centerline where we actually have data
        # Interpolation where we dont have points didnt work very well, look into this later
    y_min = int(np.min(ys))
    y_max = int(np.max(ys))
    
    print(f"Centerline valid y-range: {y_min} to {y_max}")
    
    # Create centerline points for every y-value in our range
    y_vals = np.arange(y_min, y_max + 1)
    x_vals = poly(y_vals)
    # Make sure x values stay within image bounds
    x_vals = np.clip(x_vals, 0, width - 1)
    
    # Combine x and y into (x, y) coordinate pairs
    centerline = np.column_stack([x_vals, y_vals])
    
    # Step 9: Create a mask image with just the centerline
    centerline_mask = np.zeros((height, width), dtype=np.uint8)
    pts = centerline.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(centerline_mask, [pts], isClosed=False, color=255, thickness=1)
    
    # Step 10: Calculate track width at each point
    # Distance between inner and outer borders tells us how wide the track is
    track_widths = np.linalg.norm(outer_aligned - inner_resampled, axis=1)

    # Step 11: Create visualization
    result_img = img.copy()
    
    if visualize:
        # Draw the polynomial centerline in red on the original image
        cv2.polylines(result_img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
        cv2.imshow("Binary Mask", binary_mask)
        cv2.imshow("Centerline Mask", centerline_mask)
        cv2.imshow("Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        'centerline': centerline,            # Array of (x,y) points along center
        'centerline_mask': centerline_mask,  # Binary image with centerline drawn
        'poly_coeffs': coeffs,               # Polynomial coefficients for the curve
        'track_widths': track_widths,        # How wide the track is at each point
        'image': result_img                  # Visualization with centerline drawn
    }


if __name__ == "__main__":
    # Run the extraction on a sample image
    result = extract_track_borders("sample_images/sample_image0.png", num_points=500, poly_degree=4)