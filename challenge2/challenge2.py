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
    """Render histogram images with OpenCV to avoid Qt backend issues"""
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
        ("Original", hist_original, (255, 0, 0)),
        ("Grayscale", hist_gray, (128, 128, 128)),
        ("Blurred", hist_blurred, (0, 255, 0)),
        ("Contrasted", hist_contrasted, (0, 0, 255)),
        ("Binary", hist_binary, (0, 165, 255)),
    ]

    canvas_height = 220
    canvas_width = 256
    gap = 10
    label_height = 24
    total_width = len(hist_data) * (canvas_width + gap) - gap
    total_height = canvas_height + label_height + 10
    strip = np.full((total_height, total_width, 3), 245, dtype=np.uint8)

    for idx, (title, hist, color) in enumerate(hist_data):
        x0 = idx * (canvas_width + gap)
        hist_img = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)

        max_val = np.max(hist) if np.max(hist) > 0 else 1
        hist_norm = (hist / max_val) * (canvas_height - 1)

        for x in range(1, 256):
            y1 = int(canvas_height - hist_norm[x - 1])
            y2 = int(canvas_height - hist_norm[x])
            cv2.line(hist_img, (x - 1, y1), (x, y2), color, 1)

        cv2.putText(hist_img, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
        strip[10:10 + canvas_height, x0:x0 + canvas_width] = hist_img

    header = f"Histograms | Kernel={kernel_size}x{kernel_size}, Sigma={sigma:.1f}, CLAHE={clahe_limit:.1f}"
    cv2.putText(strip, header, (10, total_height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)

    cv2.imshow("Histograms", strip)


def save_csv():
    """Save current pixel values to CSV"""
    global img, gray, row_index
    
    if gray is None:
        print("No image loaded!")
        return
    
    # Get current trackbar values
    kernel_size = cv2.getTrackbarPos('Kernel Size', 'Controls')
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 1:
        kernel_size = 1
    
    sigma = cv2.getTrackbarPos('Sigma x10', 'Controls') / 10.0
    clahe_limit = cv2.getTrackbarPos('CLAHE x10', 'Controls') / 10.0
    if clahe_limit < 0.1:
        clahe_limit = 0.1
    
    # Recompute with current parameters
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
    contrasted = clahe.apply(blurred)
    
    # Convert to binary image
    _, binary = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Extract pixel values
    rgb_values = img[row_index, :]
    gray_values = gray[row_index, :]
    blurred_values = blurred[row_index, :]
    contrasted_values = contrasted[row_index, :]
    binary_values = binary[row_index, :]
    rgb_sums = np.sum(rgb_values, axis=1)
    
    # Write to CSV
    output_csv = "pixel_values.csv"
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['x', 'RGB_sum', 'Gray', 'Blurred', 'Contrasted', 'Binary'])
        
        for x in range(len(gray_values)):
            writer.writerow([
                x,
                int(rgb_sums[x]),
                int(gray_values[x]),
                int(blurred_values[x]),
                int(contrasted_values[x]),
                int(binary_values[x])
            ])
    
    print(f"\n✓ Saved to {output_csv}")
    print(f"  Parameters: Kernel={kernel_size}x{kernel_size}, Sigma={sigma:.1f}, CLAHE={clahe_limit:.1f}")
    print(f"  Total pixels: {len(gray_values)}")


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
    print("Press 's' to save CSV")
    print("Press 'q' to quit")
    
    # Initial update
    update_image(0)
    
    # Main loop
    while True:
        key = cv2.waitKey(100) & 0xFF
        
        if key == ord('s'):
            save_csv()
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    # Histogram window is managed by OpenCV


if __name__ == "__main__":
    process_grid("image.png")
