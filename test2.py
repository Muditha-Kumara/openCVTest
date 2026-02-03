# --- Refactored for stability ---
import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, HORIZONTAL
import itertools

image_path = "20251206_115810.jpg"
original_img = cv2.imread(image_path)
if original_img is None:
    print(f"Error: Could not load image at {image_path}")
    exit()

scale = 0.25
resized_image = cv2.resize(original_img, None, fx=scale, fy=scale)
roi_pts = [(5, 349), (1014, 581)]
x1, y1 = roi_pts[0]
x2, y2 = roi_pts[1]
roi = resized_image[min(y1, y2) : max(y1, y2), min(x1, x2) : max(x1, x2)]


def line_to_slope_intercept(a, b, c):
    if b == 0:
        return None
    k = -a / b
    intercept = -c / b
    return k, intercept


def update_image(val):
    blur_val = int(val)
    if blur_val % 2 == 0:
        blur_val += 1
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)
    edges = cv2.Canny(blurred, 50, 150)
    display_image = roi.copy()
    lines_p = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=120, maxLineGap=40
    )
    main_lines = []
    if lines_p is not None:
        for line in lines_p:
            lx1, ly1, lx2, ly2 = line[0]
            a = ly2 - ly1
            b = lx1 - lx2
            c = lx2 * ly1 - lx1 * ly2
            res = line_to_slope_intercept(a, b, c)
            if res:
                k_curr, b_curr = res
                is_duplicate = False
                for k_old, b_old in main_lines:
                    if abs(k_curr - k_old) < 0.15 and abs(b_curr - b_old) < 30:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    main_lines.append((k_curr, b_curr))
                    cv2.line(display_image, (lx1, ly1), (lx2, ly2), (0, 255, 0), 3)
    for (i, (k1, b1)), (j, (k2, b2)) in itertools.combinations(
        enumerate(main_lines), 2
    ):
        if abs(k1 - k2) > 0.05:
            x = (b2 - b1) / (k1 - k2)
            y = k1 * x + b1
            if 0 <= x < display_image.shape[1] and 0 <= y < display_image.shape[0]:
                cv2.circle(display_image, (int(x), int(y)), 10, (0, 0, 255), -1)
    cv2.imshow("Edges View", edges)
    cv2.imshow("Filtered Result", display_image)
    cv2.waitKey(1)


def on_closing():
    cv2.destroyAllWindows()
    root.destroy()


root = tk.Tk()
root.title("Line Intersection Fine-Tuning")
blur_size_scale = Scale(
    root, from_=1, to=100, orient=HORIZONTAL, label="Blur Size", command=update_image
)
blur_size_scale.set(7)
blur_size_scale.pack()
root.protocol("WM_DELETE_WINDOW", on_closing)
update_image(blur_size_scale.get())
root.mainloop()
