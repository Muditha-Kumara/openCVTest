import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_field_lines_refined(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return

    # 1. Pre-processing
    # Bilateral filter is better than Gaussian for this; it smooths the turf
    # but keeps the sharp edges of the white/yellow lines.
    smooth = cv2.bilateralFilter(img, 9, 75, 75)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)

    # 2. Refined Color Masks (Minor loosening to catch faint parts)
    # White: Lowered brightness threshold slightly
    white_mask = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([180, 45, 255]))
    # Yellow: Widened hue range slightly
    yellow_mask = cv2.inRange(hsv, np.array([15, 70, 70]), np.array([35, 255, 255]))
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # 3. ROI (Region of Interest)
    height, width = mask.shape
    roi_mask = np.zeros_like(mask)
    polygon = np.array(
        [
            [
                (0, height),
                (0, int(height * 0.42)),
                (width, int(height * 0.42)),
                (width, height),
            ]
        ],
        np.int32,
    )
    cv2.fillPoly(roi_mask, polygon, 255)
    masked_data = cv2.bitwise_and(mask, roi_mask)

    # 4. Stronger Morphological "Closing"
    # Using a larger 7x7 kernel to bridge bigger gaps in the lines
    kernel = np.ones((7, 7), np.uint8)
    clean_mask = cv2.morphologyEx(masked_data, cv2.MORPH_CLOSE, kernel)

    # 5. Optimized Hough Lines
    # Increased maxLineGap to 100 to jump across missing line segments
    # Lowered minLineLength slightly to catch distant lines
    lines = cv2.HoughLinesP(
        clean_mask, 1, np.pi / 180, threshold=40, minLineLength=80, maxLineGap=100
    )

    # 6. Draw results
    output = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Anti-aliased line for smoother visual result
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 5, cv2.LINE_AA)

    return output


# Run it
result = detect_field_lines_refined("20251206_115810.jpg")
plt.figure(figsize=(12, 8))
plt.imshow(result)
plt.title("Refined Field Line Detection")
plt.axis("off")
plt.show()
