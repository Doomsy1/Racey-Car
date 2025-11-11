import cv2
import numpy as np

def extract_track_borders(image_path):
    img = cv2.imread(image_path) # imread loads image file into a numpy array
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None

    # Resize image for standardization and speed up processing, keep it original size for now
    "img = cv2.resize(img, (640, 480))"

    # --- Pre-processing & Noise Reduction ---
    
    # Convert image to grayscale, makes processing simpler, reduces noise from color variations
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to smooth out small noise
        # Gaussian Blur averages pixel values with neighbors, with closer pixels having more influence
        # Kernel size: (5, 5) indicates the window size for the blur
        # Standard deviation: 0 lets OpenCV compute it based on kernel size
            # This SD determines the weighting of neighboring pixels
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- Isolate the Track Borders (Black Lines) ---

    # 4. Color/Intensity Thresholding
    # We want to find the black lines, which have a very low brightness (intensity).
        # For other track colours, we might need to adjust this step
    # OpenCV thresholding works best when the target is bright (white), so invert the colours first
    
    # Invert the image 
    inverted = cv2.bitwise_not(blurred)

    # Set a threshold to isolate the brightest parts (which were the original black lines).
    # We are setting a high minimum brightness (200 out of 255). 
    # Any pixel brighter than `thresh_value` in the *inverted* image is turned pure white (255).
    # Everything else is turned black (0). 
    # This creates a clean binary mask.
    thresh_value = 200
    _, binary_mask = cv2.threshold(inverted, thresh_value, 255, cv2.THRESH_BINARY)
    
    # Erode and Dilate to reduce noise
    kernel = np.ones((3,3), np.uint8) # Kernel is the window size (3x3) for the erode/dilate operations
    # Erode looks at the white pixels, then if any of its neighbors are black, it turns that pixel black too
    binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
    # Dilate looks at the black pixels, then if any of its neighbors are white, it turns that pixel white too
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    # Erode followed by Dilate reduces small noises while keeping the main structures intact.
    
    # --- Contour and Line Extraction ---
    
    # 5. Find Contours
    # RETR_EXTERNAL retrieves only the outermost contours, ignoring any holes
    # CHAIN_APPROX_SIMPLE turns smooth curves into straight line segments improving efficiency
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 6. Filter and Draw Contours (This is optional, just to visualize the results)
    # We will draw the detected borders on a copy of the original image
    result_img = img.copy()
    
    print(f"Found {len(contours)} potential track border segments.")
    
    # Loop through detected contours and filter them by size
    min_contour_area = 500 # Adjust this to filter out small noise blobs
    
    # Store the significant line contours
    track_line_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_contour_area:
            # Draw the significant contour in green
            cv2.drawContours(result_img, [contour], -1, (0, 255, 0), 2)
            track_line_contours.append(contour)
            
    # --- Display Results ---
    cv2.imshow("1. Original Image", img)
    cv2.imshow("2. Binary Mask (Isolated Borders)", binary_mask)
    cv2.imshow("3. Detected Borders", result_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return track_line_contours  # returns a binary mask of the detected track borders
                                # binary mask is a numpy array with white pixels where borders are detected, black otherwise

track_borders = extract_track_borders('sample_image.png')