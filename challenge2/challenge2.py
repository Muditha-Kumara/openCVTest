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

    h, w = display.shape[:2]
    # Draw permanent vertical and horizontal lines
    cv2.line(display, (185, 0), (185, h - 1), (255, 0, 255), 1)
    cv2.line(display, (480, 0), (480, h - 1), (255, 0, 255), 1)
    cv2.line(display, (0, 100), (w - 1, 100), (255, 0, 255), 1)
    cv2.line(display, (0, 360), (w - 1, 360), (255, 0, 255), 1)

    # Find intersection points of horizontal lines with rightmost edge
    p1, p2 = None, None
    # y=100
    if 0 <= 100 < h:
        row = edges[100]
        for x in range(w - 1, -1, -1):
            if row[x] > 0:
                p1 = (x, 100)
                break
    # y=360
    if 0 <= 360 < h:
        row = edges[360]
        for x in range(w - 1, -1, -1):
            if row[x] > 0:
                p2 = (x, 360)
                break

    # Draw points if found
    if p1:
        cv2.circle(display, p1, 8, (0, 0, 255), -1)
        cv2.putText(
            display,
            "p1",
            (p1[0] + 10, p1[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
    if p2:
        cv2.circle(display, p2, 8, (0, 0, 255), -1)
        cv2.putText(
            display,
            "p2",
            (p2[0] + 10, p2[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    # Find intersection points of horizontal lines with leftmost edge (search left to right)
    p3, p4 = None, None
    # y=100
    if 0 <= 100 < h:
        row = edges[100]
        for x in range(w):
            if row[x] > 0:
                p3 = (x, 100)
                break
    # y=360
    if 0 <= 360 < h:
        row = edges[360]
        for x in range(w):
            if row[x] > 0:
                p4 = (x, 360)
                break

    # Draw points if found
    if p3:
        cv2.circle(display, p3, 8, (255, 0, 0), -1)
        cv2.putText(
            display,
            "p3",
            (p3[0] + 10, p3[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )
    if p4:
        cv2.circle(display, p4, 8, (255, 0, 0), -1)
        cv2.putText(
            display,
            "p4",
            (p4[0] + 10, p4[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
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
