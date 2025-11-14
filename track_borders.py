import cv2
import numpy as np

img = cv2.imread("sample_image.png")
dimensions = img.shape

height = dimensions[0]
width = dimensions[1]

print(f"Image Dimensions: Height={height}, Width={width}")  


# ----------------------------------------------------------
#  Extract Track Borders
# ----------------------------------------------------------
def extract_track_borders(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    inverted = cv2.bitwise_not(blurred)
    thresh_value = 200
    _, binary_mask = cv2.threshold(inverted, thresh_value, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = img.copy()
    print(f"Found {len(contours)} potential track border segments.")

    min_contour_area = 500
    track_line_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_contour_area:
            cv2.drawContours(result_img, [contour], -1, (0, 255, 0), 2)
            track_line_contours.append(contour)

    # --- Compute center line ---
    center_line = extract_center_line(track_line_contours, height, width)

    # --- Fit polynomial to smooth/complete the center line ---
    smooth_center = fit_centerline_polynomial(center_line, height)

    # --- Get center line points ---
    center_line = extract_center_line(track_line_contours, height, width)

    # --- Draw the center line in RED on the result image ---
    for (cx, cy) in center_line:
        cv2.circle(result_img, (cx, cy), 2, (0, 0, 255), -1)

    # --- Draw the smoothed center line in RED ---
    for (cx, cy) in smooth_center:
        if 0 <= cx < width:
            cv2.circle(result_img, (cx, cy), 2, (0, 0, 255), -1)

    # --- Show results ---
    cv2.imshow("1. Original Image", img)
    cv2.imshow("2. Binary Mask (Isolated Borders)", binary_mask)
    cv2.imshow("3. Track Borders + Center Line", result_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return track_line_contours

# ----------------------------------------------------------
#  Extract Center Line
# ----------------------------------------------------------
def extract_center_line(contours, height, width):
    if len(contours) != 2:
        print("Error: Need exactly 2 contours to extract center line.")
        return []

    left_contour = contours[0]
    right_contour = contours[1]

    left_x_by_y = [[] for _ in range(height)]
    right_x_by_y = [[] for _ in range(height)]

    # Left contour
    for point in left_contour:
        x = point[0][0]
        y = point[0][1]
        if 0 <= y < height:
            left_x_by_y[y].append(x)

    # Right contour
    for point in right_contour:
        x = point[0][0]
        y = point[0][1]
        if 0 <= y < height:
            right_x_by_y[y].append(x)

    left_y_by_x = [[] for _ in range(width)]
    right_y_by_x = [[] for _ in range(width)]

    # Left contour
    for point in left_contour:
        x = point[0][0]
        y = point[0][1]
        if 0 <= x < width:
            left_y_by_x[x].append(y)

    # Right contour
    for point in right_contour:
        x = point[0][0]
        y = point[0][1]
        if 0 <= x < width:
            right_y_by_x[x].append(y)

    center_line = []

    # Compute center points
    for y in range(height):
        if left_x_by_y[y] and right_x_by_y[y]:
            lm = np.mean(left_x_by_y[y])
            rm = np.mean(right_x_by_y[y])
            cx = int((lm + rm) / 2)
            center_line.append((cx, y))
    # for x in range(width):
    #     if left_y_by_x[x] and right_y_by_x[x]:
    #         lt = np.min(left_y_by_x[x])
    #         rt = np.min(right_y_by_x[x])
    #         cy = int((lt + rt) / 2)
    #         center_line.append((x, cy))

    print("Extracted center line points:", len(center_line))

    return center_line



# ----------------------------------------------------------
#  Fit Center Line Polynomial
# ----------------------------------------------------------

def fit_centerline_polynomial(center_line, height):
    if len(center_line) < 3:
        print("Not enough points for polynomial fitting.")
        return center_line

    pts = np.array(center_line)
    ys = pts[:, 1]
    xs = pts[:, 0]

    # Fit a 2nd-degree polynomial
    coeffs = np.polyfit(ys, xs, deg=2)
    poly_fn = np.poly1d(coeffs)

    smoothed = []
    for y in range(height):
        cx = int(poly_fn(y))
        smoothed.append((cx, y))

    return smoothed

track_borders = extract_track_borders('sample_image.png')
