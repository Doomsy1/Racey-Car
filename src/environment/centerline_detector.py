"""Centerline detection from binary segmentation mask images."""

import numpy as np


def detect_centerline(binary_image, centerline_row_fraction=0.6):
    """
    Detect track centerline from binary camera image.
    
    Args:
        binary_image: Grayscale binary image (black track, white background)
        centerline_row_fraction: Fraction of image height to sample centerline (0.0-1.0)
        
    Returns:
        tuple: (centerline_x, centerline_points, left_boundary, right_boundary)
            - centerline_x: X-coordinate of centerline at sampling row (pixels)
            - centerline_points: List of (x, y) tuples for centerline visualization
            - left_boundary: X-coordinate of left boundary at sampling row
            - right_boundary: X-coordinate of right boundary at sampling row
    """
    height, width = binary_image.shape
    
    # Sample bottom portion of image (e.g., bottom 60%)
    start_row = int(height * (1.0 - centerline_row_fraction))
    end_row = height
    
    # Find centerline at the sampling row (middle of bottom region)
    sampling_row = int((start_row + end_row) / 2)
    
    # Find left and right boundaries at sampling row
    row_data = binary_image[sampling_row, :]
    
    # Find leftmost black pixel (track boundary)
    left_boundary = None
    for x in range(width):
        if row_data[x] < 128:  # Black pixel (track)
            left_boundary = x
            break
    
    # Find rightmost black pixel (track boundary)
    right_boundary = None
    for x in range(width - 1, -1, -1):
        if row_data[x] < 128:  # Black pixel (track)
            right_boundary = x
            break
    
    # Compute centerline
    if left_boundary is not None and right_boundary is not None:
        centerline_x = (left_boundary + right_boundary) / 2.0
    else:
        # Fallback: use image center if boundaries not found
        centerline_x = width / 2.0
        left_boundary = 0
        right_boundary = width - 1
    
    # Generate centerline points for visualization (scan multiple rows)
    centerline_points = []
    for y in range(start_row, end_row, max(1, (end_row - start_row) // 20)):
        row_data = binary_image[y, :]
        
        # Find boundaries at this row
        left = None
        right = None
        for x in range(width):
            if row_data[x] < 128:
                if left is None:
                    left = x
                right = x
        
        if left is not None and right is not None:
            center_x = (left + right) / 2.0
            centerline_points.append((int(center_x), y))
    
    return centerline_x, centerline_points, left_boundary, right_boundary

