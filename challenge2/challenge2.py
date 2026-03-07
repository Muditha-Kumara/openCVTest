import cv2
import numpy as np


def nothing(x):
    pass


# Load the attached image
img = cv2.imread("image.png")  # Update filename if needed
if img is None:
    raise ValueError("Image not found. Please check the filename.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("Edge")
cv2.createTrackbar("Min", "Edge", 68, 255, nothing)
cv2.createTrackbar("Max", "Edge", 200, 255, nothing)


while True:
    min_val = cv2.getTrackbarPos("Min", "Edge")
    max_val = cv2.getTrackbarPos("Max", "Edge")
    edges = cv2.Canny(gray, min_val, max_val)

    # Overlay values on the edge image
    display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        display,
        f"Min: {min_val}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        display,
        f"Max: {max_val}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    # Overlay edges on real image (green edges)
    overlay = img.copy()
    overlay[edges != 0] = (0, 255, 0)

    # Print values in terminal
    print(f"Min: {min_val}, Max: {max_val}", end="\r")

    cv2.imshow("Edge", display)
    cv2.imshow("Overlay", overlay)
    if cv2.waitKey(100) & 0xFF == 27:
        break

cv2.destroyAllWindows()
