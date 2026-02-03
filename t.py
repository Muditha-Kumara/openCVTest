import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, HORIZONTAL
import itertools

# --- CONFIGURATION & INITIALIZATION ---
# Based on Screenshot 18-42-02
roi_pts = [(5, 349), (1014, 581)]
image_path = "20251206_115810.jpg"
x_multiplier = 0.25
y_multiplier = 0.25

# Load and resize the initial image [cite: 14]
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image at {image_path}")
    exit()

resized_image = cv2.resize(
    image, (int(image.shape[1] * x_multiplier), int(image.shape[0] * y_multiplier))
)

# Define the ROI (Region of Interest) based on teacher's snippet [cite: 18-42-02]
x1, y1 = roi_pts[0]
x2, y2 = roi_pts[1]
roi = resized_image[min(y1, y2) : max(y1, y2), min(x1, x2) : max(x1, x2)]

# --- MATHEMATICAL FUNCTIONS ---


def line_to_slope_intercept(a, b, c):
    """
    Converts general form ax + by + c = 0 to slope-intercept y = kx + b0.
    Follows teacher's logic in Screenshot 18-41-52.
    """
    if b == 0:
        return None  # Vertical line handling
    k = -a / b
    intercept = -c / b
    return k, intercept


def line_intersections(lines):
    """
    Computes analytical intersection points for all pairs of lines.
    Follows teacher's logic in Screenshot 18-41-52[cite: 32].
    """
    intersections = []
    # Use itertools to compare every line against every other line once [cite: 18-41-52]
    for (i, (k1, b1)), (j, (k2, b2)) in itertools.combinations(enumerate(lines), 2):
        if k1 is not None and k2 is not None:
            # Check for nearly parallel lines for numerical stability
            if abs(k1 - k2) < 1e-8:
                continue

            # Analytical intersection formula: x = (b2 - b1) / (k1 - k2) [cite: 33]
            x = (b2 - b1) / (k1 - k2)
            y = k1 * x + b1
            intersections.append((x, y, i + 1, j + 1))

    return intersections


# --- PROCESSING PIPELINE ---


def update_image(*args):
    """
    Main processing loop triggered by the Tkinter slider.
    Follows flow in Screenshot 18-42-11[cite: 42].
    """
    # 1. Preprocessing [cite: 23, 24]
    blur_val = blur_size_scale.get()
    # Kernel size must be odd
    if blur_val % 2 == 0:
        blur_val += 1

    gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (blur_val, blur_val), 0)
    edges = cv2.Canny(blurred_image, 100, 200)

    display_image = roi.copy()

    # 2. Line Detection [cite: 25, 27]
    # Using Probabilistic Hough Transform as seen in the teacher's GUI title
    lines_p = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10
    )

    slopes_intercepts = []
    if lines_p is not None:
        for line in lines_p:
            lx1, ly1, lx2, ly2 = line[0]
            # Draw detected lines on image [cite: 37]
            cv2.line(display_image, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)

            # Mathematical representation: a = y2-y1, b = x1-x2, c = x2y1 - x1y2 [cite: 18-42-02]
            a = ly2 - ly1
            b = lx1 - lx2
            c = lx2 * ly1 - lx1 * ly2

            res = line_to_slope_intercept(a, b, c)
            if res:
                slopes_intercepts.append(res)

    # 3. Intersection Computation [cite: 31]
    intersections = line_intersections(slopes_intercepts)

    print("\nIntersections:")
    for x, y, i, j in intersections:
        # F-string formatting from Screenshot 18-42-51 [cite: 18-42-51]
        print(f"Lines {i} ja {j} intersect at point ({x:.2f}, {y:.2f})")

        # Visualize intersection point if it is within the ROI [cite: 38]
        if 0 <= x < display_image.shape[1] and 0 <= y < display_image.shape[0]:
            cv2.circle(display_image, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Show results [cite: 18-42-51]
    cv2.imshow("edges", edges)
    cv2.imshow("display_image", display_image)
    cv2.waitKey(1)


# --- TKINTER GUI SETUP ---
# Based on Screenshot 18-42-15 [cite: 18-42-15]
root = tk.Tk()
root.title("HoughLinesP Controls (Tkinter sliders + cv2.imshow)")

blur_size_scale = Scale(
    root, from_=1, to=100, orient=HORIZONTAL, label="blur size", command=update_image
)
blur_size_scale.set(7)
blur_size_scale.pack()

# Initial call to populate windows
update_image()

# Start the loop
root.mainloop()
cv2.destroyAllWindows()
