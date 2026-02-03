import cv2
import numpy as np
import itertools


# 1. MATHEMATICAL REPRESENTATION [cite: 28, 29]
def line_to_slope_intercept(a, b, c):
    """
    Converts general form ax + by + c = 0 to y = kx + b0
    Based on teacher's code in Screenshot 18-41-52
    """
    if b == 0:
        return None  # Vertical line
    k = -a / b
    b0 = -c / b
    return k, b0


def line_intersections(lines):
    """
    Computes analytical intersection points for all line pairs [cite: 31, 32]
    """
    intersections = []
    # Using itertools.combinations to find pairs like in the teacher's code
    for (i, (k1, b1)), (j, (k2, b2)) in itertools.combinations(enumerate(lines), 2):
        if k1 is not None and k2 is not None:
            # Handle nearly parallel lines for numerical stability [cite: 35]
            if abs(k1 - k2) < 1e-8:
                continue

            # Analytical computation based on line equations [cite: 33]
            x = (b2 - b1) / (k1 - k2)
            y = k1 * x + b1
            intersections.append((x, y, i + 1, j + 1))

    return intersections


# 2. PROCESSING PIPELINE [cite: 22, 24]
def process_image(image_path):
    image = cv2.imread(image_path)
    # Convert to grayscale [cite: 23]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Gaussian Blur and Canny Edge Detection [cite: 24]
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 100, 200)

    # 3. LINE DETECTION (HoughLinesP) [cite: 26, 27]
    lines_p = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10
    )

    slopes_intercepts = []
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            # Convert segment to general form ax + by + c = 0
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2

            res = line_to_slope_intercept(a, b, c)
            if res:
                slopes_intercepts.append(res)

    # 4. INTERSECTION AND VISUALIZATION [cite: 37, 38]
    intersections = line_intersections(slopes_intercepts)

    for x, y, i, j in intersections:
        # Print intersections, even those outside image boundaries [cite: 34, 50]
        print(f"Lines {i} and {j} intersect at point ({x:.2f}, {y:.2f})")
        # Draw point if within visible range
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

    cv2.imshow("Detected Lines and Intersections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Execute with the image filename provided in the instructions [cite: 55]
process_image("20251206_115810.jpg")
