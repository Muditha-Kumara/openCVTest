import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_lines_pro(image_path):
    # 1. Load and Smooth
    img = cv2.imread(image_path)
    if img is None:
        return

    # Bilateral filtering removes turf noise while keeping line edges sharp
    smooth = cv2.bilateralFilter(img, 9, 75, 75)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)

    # 2. Advanced Color Masks
    # White: high brightness, low saturation
    white_mask = cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 40, 255]))
    # Yellow: specific hue range
    yellow_mask = cv2.inRange(hsv, np.array([18, 90, 90]), np.array([32, 255, 255]))
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # 3. Apply Trapezoidal ROI (Ignore trees/sky)
    height, width = combined_mask.shape
    roi_mask = np.zeros_like(combined_mask)
    # Define points that cover only the ground area
    pts = np.array(
        [
            [
                (0, height),
                (int(width * 0.2), int(height * 0.45)),
                (int(width * 0.8), int(height * 0.45)),
                (width, height),
            ]
        ],
        np.int32,
    )
    cv2.fillPoly(roi_mask, pts, 255)
    masked_img = cv2.bitwise_and(combined_mask, roi_mask)

    # 4. Morphological Cleaning
    kernel = np.ones((5, 5), np.uint8)
    # Closing fills small gaps within the detected lines
    clean_mask = cv2.morphologyEx(masked_img, cv2.MORPH_CLOSE, kernel)

    # 5. High-Precision Hough Line Detection
    # Increase minLineLength to filter out short noisy artifacts
    lines = cv2.HoughLinesP(
        clean_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=150,
        maxLineGap=40,
    )

    # 6. Draw Results
    output = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw thick green lines for visibility
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 6, cv2.LINE_AA)

    return output


# Run it
result = detect_lines_pro("20251206_115810.jpg")
plt.figure(figsize=(12, 8))
plt.imshow(result)
plt.title("Optimized Field Line Detection")
plt.axis("off")
plt.show()
