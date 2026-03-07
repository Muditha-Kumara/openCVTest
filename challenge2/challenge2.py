import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

import cv2
import numpy as np
import csv

# Global variables
img = None
gray = None
row_index = None


def update_image(x):
    """Callback function for trackbar changes"""
    global img, gray, row_index

    if gray is None:
        return

    # Get trackbar values
    kernel_size = cv2.getTrackbarPos('Kernel Size', 'Controls')
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 1:
        kernel_size = 1

    sigma = cv2.getTrackbarPos('Sigma x10', 'Controls') / 10.0
    clahe_limit = cv2.getTrackbarPos('CLAHE x10', 'Controls') / 10.0
    if clahe_limit < 0.1:
        clahe_limit = 0.1

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
    contrasted = clahe.apply(blurred)

    # Convert to binary image using Otsu's thresholding
    _, binary = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Show results
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Contrasted", contrasted)
    cv2.imshow("Binary", binary)

    # Show result with parameters overlay
    result_img = img.copy()
    cv2.line(result_img, (0, row_index), (img.shape[1], row_index), (0, 255, 0), 2)
    text = f"Kernel: {kernel_size}x{kernel_size}, Sigma: {sigma:.1f}, CLAHE: {clahe_limit:.1f}"
    cv2.putText(result_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(result_img, "Press 's' to save CSV, 'q' to quit", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Result", result_img)
    
    # Update histograms
    update_histograms(gray, blurred, contrasted, binary, kernel_size, sigma, clahe_limit)


def update_histograms(gray_img, blurred_img, contrasted_img, binary_img, kernel_size, sigma, clahe_limit):
    """Render histogram panels with OpenCV for a clean, professional layout"""
    global img

    if img is None:
        return

    # Calculate histograms for all stages
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist_original = cv2.calcHist([grayscale_img], [0], None, [256], [0, 256]).flatten()
    hist_gray = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()
    hist_blurred = cv2.calcHist([blurred_img], [0], None, [256], [0, 256]).flatten()
    hist_contrasted = cv2.calcHist([contrasted_img], [0], None, [256], [0, 256]).flatten()
    hist_binary = cv2.calcHist([binary_img], [0], None, [256], [0, 256]).flatten()

    hist_data = [
        ("Original", hist_original, (52, 86, 240)),
        ("Grayscale", hist_gray, (120, 120, 120)),
        ("Blurred", hist_blurred, (46, 160, 67)),
        ("Contrasted", hist_contrasted, (220, 68, 55)),
        ("Binary", hist_binary, (233, 163, 48)),
    ]

    # Layout
    panel_w = 320
    panel_h = 220
    pad = 12
    header_h = 36
    cols = 3
    rows = 2
    canvas_w = cols * panel_w + (cols + 1) * pad
    canvas_h = rows * panel_h + (rows + 1) * pad + header_h

    # Colors (BGR)
    bg = (238, 240, 242)
    panel_bg = (255, 255, 255)
    border = (210, 214, 219)
    grid = (228, 231, 235)
    axis = (90, 95, 102)
    label = (50, 55, 60)

    canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)

    # Header
    title = "Image Histograms"
    subtitle = f"Kernel {kernel_size}x{kernel_size} | Sigma {sigma:.1f} | CLAHE {clahe_limit:.1f}"
    cv2.putText(canvas, title, (pad, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label, 2)
    cv2.putText(canvas, subtitle, (pad, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis, 1)

    def draw_hist_panel(x0, y0, title_text, hist, color):
        # Panel background and border
        cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h), panel_bg, -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h), border, 1)

        # Inner plot area
        left = x0 + 34
        right = x0 + panel_w - 12
        top = y0 + 30
        bottom = y0 + panel_h - 28
        plot_w = right - left
        plot_h = bottom - top

        # Gridlines
        for i in range(5):
            y = top + int(i * plot_h / 4)
            cv2.line(canvas, (left, y), (right, y), grid, 1)
        for i in range(4):
            x = left + int(i * plot_w / 3)
            cv2.line(canvas, (x, top), (x, bottom), grid, 1)

        # Axes
        cv2.line(canvas, (left, bottom), (right, bottom), axis, 1)
        cv2.line(canvas, (left, top), (left, bottom), axis, 1)

        # Title
        cv2.putText(
            canvas,
            title_text,
            (x0 + 10, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            label,
            1,
        )

        # Histogram plot
        max_val = np.max(hist) if np.max(hist) > 0 else 1
        x_vals = np.linspace(0, 255, plot_w).astype(int)
        hist_scaled = (hist / max_val) * (plot_h - 2)

        points = []
        for i in range(plot_w):
            x_bin = x_vals[i]
            y = bottom - int(hist_scaled[x_bin])
            points.append((left + i, y))
        cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, color, 2)

        # Axis labels
        cv2.putText(
            canvas,
            "Intensity",
            (left + plot_w // 2 - 35, y0 + panel_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            axis,
            1,
        )
        cv2.putText(
            canvas,
            "Count",
            (x0 + 6, top + plot_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            axis,
            1,
        )

    for idx, (title_text, hist, color) in enumerate(hist_data):
        row = idx // cols
        col = idx % cols
        x0 = pad + col * (panel_w + pad)
        y0 = header_h + pad + row * (panel_h + pad)
        draw_hist_panel(x0, y0, title_text, hist, color)

    cv2.imshow("Histograms", canvas)


def process_grid(image_path):
    """Process grid image with interactive OpenCV trackbars"""
    global img, gray, row_index

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    print(f"Image shape: {img.shape}")
    cv2.imshow("Original", img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", gray)

    # Extract row from middle of image
    row_index = img.shape[0] // 2

    # Histogram window rendered with OpenCV
    cv2.namedWindow("Histograms")

    # Create control window with trackbars
    cv2.namedWindow('Controls')
    cv2.createTrackbar('Kernel Size', 'Controls', 5, 51, update_image)
    cv2.createTrackbar('Sigma x10', 'Controls', 0, 100, update_image)
    cv2.createTrackbar('CLAHE x10', 'Controls', 20, 100, update_image)

    print("\n=== Interactive Parameter Tuning ===")
    print("Adjust trackbars in 'Controls' window")
    print("Histograms update in real-time")
    print("Press 'q' to quit")

    # Initial update
    update_image(0)

    # Main loop
    while True:
        key = cv2.waitKey(100) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    # Histogram window is managed by OpenCV


if __name__ == "__main__":
    process_grid("image.png")
