import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, HORIZONTAL
import itertools

# Add scikit-image import
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from skimage.color import rgb2gray

# --- 1. SETUP & ROI ---
image_path = "20251206_115810.jpg"
original_img = cv2.imread(image_path)

if original_img is None:
    print(f"Error: Could not load image at {image_path}")
    exit()

# Scale image for GUI stability
scale = 0.25
resized_image = cv2.resize(original_img, None, fx=scale, fy=scale)

# ROI coordinates based on field markings
roi_pts = [(5, 349), (1014, 581)]
x1, y1 = roi_pts[0]
x2, y2 = roi_pts[1]
roi = resized_image[min(y1, y2) : max(y1, y2), min(x1, x2) : max(x1, x2)]


def postprocess_lines(lines, dist_threshold=100, angle_threshold=10):

    if lines is None:
        print(f"Initial detected lines: 0")
        return []

    print(f"Initial detected lines: {len(lines)}")

    # Convert lines to a more workable list format
    lines = [l[0] for l in lines]

    # Show input lines
    vis_height = roi.shape[0]
    vis_width = roi.shape[1]
    input_img = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
    i = 0
    for lx1, ly1, lx2, ly2 in lines:
        cv2.line(input_img, (lx1, ly1), (lx2, ly2), (i, 255 - i, 255), 2)
        i += 40
    cv2.imshow("Input Lines", input_img)
    cv2.waitKey(1)

    final_lines = []
    used = np.zeros(len(lines), dtype=bool)

    for i in range(len(lines)):
        if used[i]:
            continue

        group = [lines[i]]
        used[i] = True

        x1, y1, x2, y2 = lines[i]
        angle_i = np.rad2deg(np.arctan2(y2 - y1, x2 - x1)) % 180

        for j in range(i + 1, len(lines)):
            if used[j]:
                continue

            x3, y3, x4, y4 = lines[j]
            angle_j = np.rad2deg(np.arctan2(y4 - y3, x4 - x3)) % 180

            angle_diff = abs(angle_i - angle_j)
            if angle_diff < angle_threshold or angle_diff > (180 - angle_threshold):
                mid_i = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                mid_j = np.array([(x3 + x4) / 2, (y3 + y4) / 2])
                dist = np.linalg.norm(mid_i - mid_j)
                if dist < dist_threshold:
                    group.append(lines[j])
                    used[j] = True

        group = np.array(group)
        final_lines.append(np.mean(group, axis=0).astype(int))

    print(f"Final unique lines: {len(final_lines)}")

    # Show output lines
    output_img = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
    for lx1, ly1, lx2, ly2 in final_lines:
        cv2.line(output_img, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
    cv2.imshow("Output Lines", output_img)
    cv2.waitKey(1)

    return final_lines


def line_to_slope_intercept(a, b, c):
    """Mathematical representation y = kx + b0 [cite: 29]"""
    if b == 0:
        return None
    k = -a / b
    intercept = -c / b
    return k, intercept


def compute_least_squares_intersection(lines_slopes_intercepts):
    """
    BONUS FEATURE: Compute intersection of MULTIPLE lines using least squares
    Finds the point that best fits all lines simultaneously

    For each line y = kx + b, rewrite as: kx - y + b = 0
    Or in matrix form: A @ p = 0, where p = [x, y]^T

    Solves using least squares: minimize ||A @ p||^2

    Args:
        lines_slopes_intercepts: List of (k, b) tuples representing y = kx + b

    Returns:
        (x, y, uncertainty): Intersection point and uncertainty estimate
    """
    if len(lines_slopes_intercepts) < 2:
        return None

    # Build matrix A where each row is [k, -1] for line y = kx + b
    # We want to solve: k*x - y + b = 0, or [k, -1] @ [x, y]^T = -b
    A = []
    b_vec = []

    for k, b in lines_slopes_intercepts:
        A.append([k, -1])
        b_vec.append(-b)

    A = np.array(A, dtype=float)
    b_vec = np.array(b_vec, dtype=float)

    # Solve using least squares: A @ p = b_vec
    # Solution: p = (A^T @ A)^-1 @ A^T @ b_vec
    try:
        p, residuals, rank, s = np.linalg.lstsq(A, b_vec, rcond=None)
        x, y = p[0], p[1]

        # Estimate uncertainty from residuals
        if len(residuals) > 0:
            mse = residuals[0] / len(lines_slopes_intercepts)  # Mean squared error
            uncertainty = np.sqrt(mse)
        else:
            # If perfect fit, estimate from singular values
            uncertainty = 1.0 / (s[-1] + 1e-10) if len(s) > 0 else 0.0

        return x, y, uncertainty
    except np.linalg.LinAlgError:
        return None


def compute_and_log_results(main_lines, pairwise_count):
    """Print assignment results only once"""
    print("\n=== ASSIGNMENT REQUIREMENTS RESULTS ===")
    print(f"✓ Line Detection: {len(main_lines)} lines detected using HoughLinesP")
    print(f"✓ Mathematical Representation: y = kx + b (Slope-Intercept Form)")
    print(
        f"✓ Analytical Intersection: {pairwise_count} pairwise intersections computed"
    )

    # --- BONUS FEATURE ---
    print(f"\n=== BONUS FEATURE: LEAST SQUARES (+1 pt) ===")
    if len(main_lines) >= 2:
        result = compute_least_squares_intersection(main_lines)
        if result:
            x_ls, y_ls, uncertainty = result
            print(f"✓ Multi-line Intersection: ({x_ls:.2f}, {y_ls:.2f})")
            print(f"✓ Uncertainty Estimation: ±{uncertainty:.4f} pixels")
            print(f"✓ Lines Used: {len(main_lines)}")
    print("=" * 50 + "\n")


def update_image(*args):
    try:
        blur_val = blur_size_scale.get()
        if blur_val % 2 == 0:
            blur_val += 1

        # --- 2. PREPROCESSING [cite: 23-24] ---
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)
        edges = cv2.Canny(blurred, 50, 150)

        display_image = roi.copy()

        # --- 3. LINE DETECTION (Probabilistic Hough) [cite: 27] ---
        # Increased minLineLength and maxLineGap to connect fragmented markings
        # Get dynamic HoughLinesP parameters from sliders
        threshold_val = hough_threshold_scale.get()
        min_line_length_val = min_line_length_scale.get()
        max_line_gap_val = max_line_gap_scale.get()

        lines_p = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=threshold_val,
            minLineLength=min_line_length_val,
            maxLineGap=max_line_gap_val,
        )

        lines_p = postprocess_lines(lines_p)

        main_lines = []
        if lines_p is not None:
            for line in lines_p:
                # Ensure line is a 1D array of 4 elements
                if isinstance(line, np.ndarray) and line.shape == (1, 4):
                    lx1, ly1, lx2, ly2 = line[0]
                elif isinstance(line, (np.ndarray, list)) and len(line) == 4:
                    lx1, ly1, lx2, ly2 = line
                else:
                    continue
                # General form coefficients: ax + by + c = 0
                a = ly2 - ly1
                b = lx1 - lx2
                c = lx2 * ly1 - lx1 * ly2

                res = line_to_slope_intercept(a, b, c)

                if res:
                    k_curr, b_curr = res

                    # --- FINE-TUNING: REMOVE DOUBLE EDGES ---
                    # Check if this line is too close to a line we already found
                    is_duplicate = False
                    for k_old, b_old in main_lines:
                        # If slope is similar AND vertical intercept is close, it's the same marking strip
                        if abs(k_curr - k_old) < 0.15 and abs(b_curr - b_old) < 30:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        main_lines.append((k_curr, b_curr))
                        # Draw the clean, unique line [cite: 37]
                cv2.line(display_image, (lx1, ly1), (lx2, ly2), (0, 255, 0), 3)

        # --- 4. ANALYTICAL INTERSECTION [cite: 32-35] ---
        pairwise_count = 0
        for (i, (k1, b1)), (j, (k2, b2)) in itertools.combinations(
            enumerate(main_lines), 2
        ):
            # Numerical stability check for nearly parallel lines [cite: 35]
            if abs(k1 - k2) > 0.05:
                pairwise_count += 1
                # Solve: k1*x + b1 = k2*x + b2
                x = (b2 - b1) / (k1 - k2)
                y = k1 * x + b1

                # Visualize intersection point [cite: 38]
                if 0 <= x < display_image.shape[1] and 0 <= y < display_image.shape[0]:
                    cv2.circle(display_image, (int(x), int(y)), 10, (0, 0, 255), -1)

        # --- BONUS: LEAST SQUARES INTERSECTION (Multiple Lines) ---
        if len(main_lines) >= 2:
            result = compute_least_squares_intersection(main_lines)
            if result:
                x_ls, y_ls, uncertainty = result

                # Visualize least squares intersection point (yellow circle with larger radius)
                if (
                    0 <= x_ls < display_image.shape[1]
                    and 0 <= y_ls < display_image.shape[0]
                ):
                    cv2.circle(
                        display_image, (int(x_ls), int(y_ls)), 15, (0, 255, 255), -1
                    )
                    # Draw uncertainty circle (dashed representation)
                    cv2.circle(
                        display_image,
                        (int(x_ls), int(y_ls)),
                        int(uncertainty) + 5,
                        (0, 255, 255),
                        2,
                    )
                    # Add text label
                    cv2.putText(
                        display_image,
                        f"LS: ({x_ls:.1f},{y_ls:.1f}) +/-{uncertainty:.2f}",
                        (int(x_ls) + 20, int(y_ls) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )

        cv2.imshow("Edges View", edges)
        cv2.imshow("Filtered Result", display_image)
        cv2.waitKey(1)

        # Store results globally for one-time logging
        global last_logged_state
        last_logged_state = (main_lines, pairwise_count)

    except Exception as e:
        print(f"Error: {e}")


# --- 5. GUI CONTROLS ---
root = tk.Tk()
root.title("Line Intersection Fine-Tuning")

# Global variables to store results for logging once
last_logged_state = None


# Blur size slider
blur_size_scale = Scale(
    root, from_=1, to=100, orient=HORIZONTAL, label="Blur Size", command=update_image
)
blur_size_scale.set(7)
blur_size_scale.pack()

# HoughLinesP threshold slider
hough_threshold_scale = Scale(
    root,
    from_=1,
    to=200,
    orient=HORIZONTAL,
    label="Hough Threshold",
    command=update_image,
)
hough_threshold_scale.set(100)
hough_threshold_scale.pack()

# HoughLinesP minLineLength slider
min_line_length_scale = Scale(
    root,
    from_=10,
    to=500,
    orient=HORIZONTAL,
    label="Min Line Length",
    command=update_image,
)
min_line_length_scale.set(120)
min_line_length_scale.pack()

# HoughLinesP maxLineGap slider
max_line_gap_scale = Scale(
    root, from_=1, to=100, orient=HORIZONTAL, label="Max Line Gap", command=update_image
)
max_line_gap_scale.set(40)
max_line_gap_scale.pack()

# Call once at startup to populate last_logged_state
update_image()

# Log the final results once
if last_logged_state:
    compute_and_log_results(last_logged_state[0], last_logged_state[1])

root.mainloop()
cv2.destroyAllWindows()
