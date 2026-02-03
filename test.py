import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_field_lines_clean(image_path):
    # 1. Load and Blur
    img = cv2.imread(image_path)
    if img is None:
        return
    # Gaussian Blur reduces high-frequency turf noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 2. Refined Color Masks
    # Tightened yellow to avoid dry grass; white kept bright
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([20, 80, 80]), np.array([40, 255, 255]))
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # 3. Region of Interest (ROI) - Crucial for removing background noise
    height, width = mask.shape
    roi_mask = np.zeros_like(mask)
    # Define a trapezoid covering only the turf area
    polygon = np.array(
        [
            [
                (0, height),
                (0, int(height * 0.45)),
                (width, int(height * 0.45)),
                (width, height),
            ]
        ],
        np.int32,
    )
    cv2.fillPoly(roi_mask, polygon, 255)
    masked_data = cv2.bitwise_and(mask, roi_mask)

    # 4. Morphological "Closing" to connect dashed lines
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(masked_data, cv2.MORPH_CLOSE, kernel)

    # 5. Hough Lines with strict length requirements
    # Increase minLineLength to ignore small noise dots
    lines = cv2.HoughLinesP(
        clean_mask, 1, np.pi / 180, threshold=50, minLineLength=150, maxLineGap=40
    )

    # 6. Draw results
    output = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 8)

    return output


# Run it
result = detect_field_lines_clean("20251206_115810.jpg")
plt.figure(figsize=(12, 8))
plt.imshow(result)
plt.title("Cleaned Field Line Detection")
plt.axis("off")
plt.show()
