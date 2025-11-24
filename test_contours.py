import cv2
import numpy as np

def visualize_all_contours(image_path, threshold_val=200, min_area=0):
    img = cv2.imread(image_path)
    if img is None:
        print("Could not load image.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    inverted = cv2.bitwise_not(blurred)
    _, binary_mask = cv2.threshold(inverted, threshold_val, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.erode(binary_mask, kernel, iterations=3)
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=3)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    result_img = img.copy()
    
    print(f"Found {len(contours)} contours:\n")
    
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < min_area:
            continue
            
        # Random color for each contour
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        
        # Draw contour
        cv2.drawContours(result_img, [c], -1, color, 2)
        
        # Label with index and area
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(result_img, f"{i}: {int(area)}", (cx, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        print(f"Contour {i}: area = {int(area)}")
    
    cv2.imshow("Binary Mask", binary_mask)
    cv2.imshow("All Contours", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_all_contours("sample_image4.png", threshold_val=200, min_area=10000)