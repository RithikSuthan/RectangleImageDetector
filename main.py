import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    return edges


def detect_rectangles_with_hough(image):
    edges = detect_edges(image)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image


def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect corners using Harris Corner Detection
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)

    # Mark detected corners in the image
    image[corners > 0.01 * corners.max()] = [0, 0, 255]

    return image, corners


def find_rectangles(corners, image):
    rects = []

    # Threshold the corners to get a binary image
    _, binary = cv2.threshold(corners, 0.01 * corners.max(), 255, cv2.THRESH_BINARY)
    binary = np.uint8(binary)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour with accuracy proportional to the perimeter
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the approximated contour has 4 points and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            rects.append(approx)
            # Draw the rectangle using bounding box for better visualization
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image, rects

def check_intensity_contrast(image, rects):
    final_rects = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for rect in rects:
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [rect], -1, 255, -1)

        # Extract the inside of the rectangle
        inside_rect = cv2.bitwise_and(gray, gray, mask=mask)

        # Create the inverse mask to extract the area outside the rectangle
        inverse_mask = cv2.bitwise_not(mask)
        outside_rect = cv2.bitwise_and(gray, gray, mask=inverse_mask)

        # Compute SSIM between the inside and outside regions
        ssim_index = ssim(inside_rect, outside_rect, full=False)

        # Compute Histogram Comparison
        hist_inside = cv2.calcHist([inside_rect], [0], mask, [256], [0, 256])
        hist_outside = cv2.calcHist([outside_rect], [0], inverse_mask, [256], [0, 256])
        hist_comparison = cv2.compareHist(hist_inside, hist_outside, cv2.HISTCMP_CORREL)

        # Compute Gradient Magnitude Difference
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_inside = cv2.bitwise_and(grad_mag, grad_mag, mask=mask)
        grad_outside = cv2.bitwise_and(grad_mag, grad_mag, mask=inverse_mask)
        grad_diff = np.abs(np.mean(grad_inside) - np.mean(grad_outside))

        # Combine SSIM, Histogram Comparison, and Gradient Difference
        if ssim_index < 0.8 or hist_comparison < 0.7 or grad_diff > 10:  # Adjust thresholds as needed
            final_rects.append(rect)
            # Draw the rectangle using bounding box for better visualization
            x, y, w, h = cv2.boundingRect(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return image, final_rects
def detect_rectangles(image_path):
    image = cv2.imread(image_path)

    # Step 1: Detect corners
    image_with_corners, corners = detect_corners(image.copy())

    # Step 2: Extract candidate rectangles using contours
    image_with_rectangles, rects = find_rectangles(corners, image.copy())

    # Step 3: Improve rectangle detection with Hough Transform
    image_with_hough_lines = detect_rectangles_with_hough(image.copy())

    # Step 4: Check intensity contrast
    final_image, final_rects = check_intensity_contrast(image_with_hough_lines.copy(), rects)

    # Display the results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Corners Detected')
    plt.imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title('Rectangles Detected with Hough Lines')
    plt.imshow(cv2.cvtColor(image_with_hough_lines, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 3)
    plt.title('Final Rectangles with Intensity Check')
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

    plt.show()


# Example usage
detect_rectangles('noisy.JPG')

