import cv2
import numpy as np

# Utility function for trackbar callback (required by OpenCV)
def nothing(x):
    pass


# Load and validate input image
def load_image(filename):
    img = cv2.imread(filename)
    if img is None:
        raise ValueError("Image not found. Please check the filename.")
    return img


# Find intersection points for vertical and horizontal lines on edges
def find_intersections(edges, w, h):
    # Vertical lines
    v1 = v2 = v3 = v4 = None
    if 0 <= 185 < w:
        col = edges[:, 185]
        for y in range(h):
            if col[y] > 0:
                v1 = (185, y)
                break
        for y in range(h - 1, -1, -1):
            if col[y] > 0:
                v3 = (185, y)
                break
    if 0 <= 480 < w:
        col = edges[:, 480]
        for y in range(h):
            if col[y] > 0:
                v2 = (480, y)
                break
        for y in range(h - 1, -1, -1):
            if col[y] > 0:
                v4 = (480, y)
                break
    # Horizontal lines
    x1 = x2 = x3 = x4 = None
    if 0 <= 100 < h:
        row = edges[100]
        for x in range(w - 1, -1, -1):
            if row[x] > 0:
                x1 = (x, 100)
                break
    if 0 <= 360 < h:
        row = edges[360]
        for x in range(w - 1, -1, -1):
            if row[x] > 0:
                x2 = (x, 360)
                break
    if 0 <= 100 < h:
        row = edges[100]
        crossings = 0
        for x in range(w):
            if row[x] > 0:
                crossings += 1
                if crossings == 2:
                    x3 = (x, 100)
                    break
    if 0 <= 360 < h:
        row = edges[360]
        crossings = 0
        for x in range(w):
            if row[x] > 0:
                crossings += 1
                if crossings == 2:
                    x4 = (x, 360)
                    break
    return v1, v2, v3, v4, x1, x2, x3, x4


# Draw infinite lines (grid) across the display
def draw_infinite_line(display, pt1, pt2, color, w, h):
    if pt1 and pt2:
        x1, y1 = pt1
        x2, y2 = pt2
        if x1 == x2:
            cv2.line(display, (x1, 0), (x1, h - 1), color, 2)
        elif y1 == y2:
            cv2.line(display, (0, y1), (w - 1, y1), color, 2)
        else:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            points = []
            y_left = int(m * 0 + b)
            if 0 <= y_left < h:
                points.append((0, y_left))
            y_right = int(m * (w - 1) + b)
            if 0 <= y_right < h:
                points.append((w - 1, y_right))
            if m != 0:
                x_top = int(-b / m)
                if 0 <= x_top < w:
                    points.append((x_top, 0))
                x_bottom = int(((h - 1) - b) / m)
                if 0 <= x_bottom < w:
                    points.append((x_bottom, h - 1))
            if len(points) >= 2:
                cv2.line(display, points[0], points[1], color, 2)


# Calculate line parameters for intersection
def line_params(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    A = y2 - y1
    B = x1 - x2
    C = A * x1 + B * y1
    return A, B, C


# Find intersection point of two lines
def intersection(l1pt1, l1pt2, l2pt1, l2pt2):
    A1, B1, C1 = line_params(l1pt1, l1pt2)
    A2, B2, C2 = line_params(l2pt1, l2pt2)
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None
    x = int((B2 * C1 - B1 * C2) / det)
    y = int((A1 * C2 - A2 * C1) / det)
    return (x, y)


# Main processing loop
def main():
    img = load_image("image.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("Edge")
    cv2.createTrackbar("Min", "Edge", 68, 255, nothing)
    cv2.createTrackbar("Max", "Edge", 200, 255, nothing)
    while True:
        min_val = cv2.getTrackbarPos("Min", "Edge")
        max_val = cv2.getTrackbarPos("Max", "Edge")
        edges = cv2.Canny(gray, min_val, max_val)
        display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # Overlay trackbar values
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
        # Draw static grid lines
        cv2.line(display, (185, 0), (185, h - 1), (255, 0, 255), 1)
        cv2.line(display, (480, 0), (480, h - 1), (255, 0, 255), 1)
        cv2.line(display, (0, 100), (w - 1, 100), (255, 0, 255), 1)
        cv2.line(display, (0, 360), (w - 1, 360), (255, 0, 255), 1)
        # Find intersection points
        v1, v2, v3, v4, x1, x2, x3, x4 = find_intersections(edges, w, h)
        # Draw infinite grid lines
        draw_infinite_line(display, v1, v2, (0, 255, 0), w, h)
        draw_infinite_line(display, v3, v4, (0, 255, 0), w, h)
        draw_infinite_line(display, x1, x2, (255, 0, 0), w, h)
        draw_infinite_line(display, x3, x4, (255, 0, 0), w, h)
        # Find intersection points of infinite lines
        p1 = intersection(v1, v2, x1, x2) if v1 and v2 and x1 and x2 else None
        p2 = intersection(v1, v2, x3, x4) if v1 and v2 and x3 and x4 else None
        p3 = intersection(v3, v4, x1, x2) if v3 and v4 and x1 and x2 else None
        p4 = intersection(v3, v4, x3, x4) if v3 and v4 and x3 and x4 else None
        # Draw intersection points
        for idx, pt in enumerate([p1, p2, p3, p4], 1):
            if pt and 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(display, pt, 10, (0, 255, 255), -1)
                cv2.putText(
                    display,
                    f"p{idx}",
                    (pt[0] + 10, pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
        # Perspective rectification overlay
        p_points = [p1, p2, p3, p4]
        valid_p = [pt for pt in p_points if pt is not None]
        if len(valid_p) == 4:
            pts = np.array(valid_p, dtype=np.float32)
            pts_sorted = sorted(pts, key=lambda p: (p[1], p[0]))
            top = sorted(pts_sorted[:2], key=lambda p: p[0])
            bottom = sorted(pts_sorted[2:], key=lambda p: p[0])
            rect = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
            widthA = np.linalg.norm(rect[2] - rect[3])
            widthB = np.linalg.norm(rect[1] - rect[0])
            heightA = np.linalg.norm(rect[1] - rect[2])
            heightB = np.linalg.norm(rect[0] - rect[3])
            maxSide = int(max(widthA, widthB, heightA, heightB))
            dst = np.array(
                [
                    [0, 0],
                    [maxSide - 1, 0],
                    [maxSide - 1, maxSide - 1],
                    [0, maxSide - 1],
                ],
                dtype=np.float32,
            )
            M = cv2.getPerspectiveTransform(rect, dst)
            overlay = cv2.warpPerspective(img, M, (maxSide, maxSide))
            cv2.imshow("Overlay", overlay)
        else:
            cv2.imshow("Overlay", np.zeros_like(img))
        # Print values in terminal
        print(f"Min: {min_val}, Max: {max_val}", end="\r")
        cv2.imshow("Edge", display)
        if cv2.waitKey(100) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
