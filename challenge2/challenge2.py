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
    # Initialize all intersection points to None
    x1 = x2 = x3 = x4 = None
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

    # Find intersection points for vertical lines
    v1 = v2 = v3 = v4 = None
    # x=185
    if 0 <= 185 < w:
        col = edges[:, 185]
        # From top (v1)
        for y in range(h):
            if col[y] > 0:
                v1 = (185, y)
                break
        # From bottom (v3)
        for y in range(h - 1, -1, -1):
            if col[y] > 0:
                v3 = (185, y)
                break
    # x=480
    if 0 <= 480 < w:
        col = edges[:, 480]
        # From top (v2)
        for y in range(h):
            if col[y] > 0:
                v2 = (480, y)
                break
        # From bottom (v4)
        for y in range(h - 1, -1, -1):
            if col[y] > 0:
                v4 = (480, y)
                break

    # Find intersection points of horizontal lines with rightmost edge
    # x1, x2 = None, None
    # y=100
    if 0 <= 100 < h:
        row = edges[100]
        for x in range(w - 1, -1, -1):
            if row[x] > 0:
                x1 = (x, 100)
                break
    # y=360
    if 0 <= 360 < h:
        row = edges[360]
        for x in range(w - 1, -1, -1):
            if row[x] > 0:
                x2 = (x, 360)
                break

    # Find intersection points of horizontal lines with leftmost edge (search left to right)
    # Find intersection points of horizontal lines with second vertical edge line (x=185)
    # x3, x4 = None, None
    # y=100
    if 0 <= 100 < h:
        row = edges[100]
        crossings = 0
        for x in range(w):
            if row[x] > 0:
                crossings += 1
                if crossings == 2:
                    x3 = (x, 100)
                    break
    # y=360
    if 0 <= 360 < h:
        row = edges[360]
        crossings = 0
        for x in range(w):
            if row[x] > 0:
                crossings += 1
                if crossings == 2:
                    x4 = (x, 360)
                    break

    # Draw points if found
    def draw_infinite_line(pt1, pt2, color):
        if pt1 and pt2:
            x1, y1 = pt1
            x2, y2 = pt2
            # Calculate line equation: y = m*x + b
            if x1 == x2:
                # vertical line
                cv2.line(display, (x1, 0), (x1, h - 1), color, 2)
            elif y1 == y2:
                # horizontal line
                cv2.line(display, (0, y1), (w - 1, y1), color, 2)
            else:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                # Find intersection with image borders
                points = []
                # Left border (x=0)
                y_left = int(m * 0 + b)
                if 0 <= y_left < h:
                    points.append((0, y_left))
                # Right border (x=w-1)
                y_right = int(m * (w - 1) + b)
                if 0 <= y_right < h:
                    points.append((w - 1, y_right))
                # Top border (y=0)
                if m != 0:
                    x_top = int(-b / m)
                    if 0 <= x_top < w:
                        points.append((x_top, 0))
                # Bottom border (y=h-1)
                if m != 0:
                    x_bottom = int(((h - 1) - b) / m)
                    if 0 <= x_bottom < w:
                        points.append((x_bottom, h - 1))
                # Draw line between two valid border points
                if len(points) >= 2:
                    cv2.line(display, points[0], points[1], color, 2)

    if v1:
        cv2.circle(display, v1, 8, (0, 255, 255), -1)
        cv2.putText(
            display,
            "v1",
            (v1[0] + 10, v1[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
    if v2:
        cv2.circle(display, v2, 8, (0, 255, 255), -1)
        cv2.putText(
            display,
            "v2",
            (v2[0] + 10, v2[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
    if v3:
        cv2.circle(display, v3, 8, (0, 255, 255), -1)
        cv2.putText(
            display,
            "v3",
            (v3[0] + 10, v3[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
    if v4:
        cv2.circle(display, v4, 8, (0, 255, 255), -1)
        cv2.putText(
            display,
            "v4",
            (v4[0] + 10, v4[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
    if x1:
        cv2.circle(display, x1, 8, (0, 0, 255), -1)
        cv2.putText(
            display,
            "x1",
            (x1[0] + 10, x1[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
    if x2:
        cv2.circle(display, x2, 8, (0, 0, 255), -1)
        cv2.putText(
            display,
            "x2",
            (x2[0] + 10, x2[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
    if x3:
        cv2.circle(display, x3, 8, (255, 0, 0), -1)
        cv2.putText(
            display,
            "x3",
            (x3[0] + 10, x3[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )
    if x4:
        cv2.circle(display, x4, 8, (255, 0, 0), -1)
        cv2.putText(
            display,
            "x4",
            (x4[0] + 10, x4[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )

    draw_infinite_line(v1, v2, (0, 255, 0))
    draw_infinite_line(v3, v4, (0, 255, 0))
    draw_infinite_line(x1, x2, (255, 0, 0))
    draw_infinite_line(x3, x4, (255, 0, 0))

    # Find intersection points of the infinite lines
    def line_params(pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        A = y2 - y1
        B = x1 - x2
        C = A * x1 + B * y1
        return A, B, C

    def intersection(l1pt1, l1pt2, l2pt1, l2pt2):
        A1, B1, C1 = line_params(l1pt1, l1pt2)
        A2, B2, C2 = line_params(l2pt1, l2pt2)
        det = A1 * B2 - A2 * B1
        if det == 0:
            return None
        x = int((B2 * C1 - B1 * C2) / det)
        y = int((A1 * C2 - A2 * C1) / det)
        return (x, y)

    # Only compute if all points exist
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

        # Move infinite line drawing after all points are defined
        # ...existing code...
        # Draw points if found
        if v1:
            cv2.circle(display, v1, 8, (0, 255, 255), -1)
            cv2.putText(
                display,
                "v1",
                (v1[0] + 10, v1[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        if v2:
            cv2.circle(display, v2, 8, (0, 255, 255), -1)
            cv2.putText(
                display,
                "v2",
                (v2[0] + 10, v2[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        if v3:
            cv2.circle(display, v3, 8, (0, 255, 255), -1)
            cv2.putText(
                display,
                "v3",
                (v3[0] + 10, v3[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        if v4:
            cv2.circle(display, v4, 8, (0, 255, 255), -1)
            cv2.putText(
                display,
                "v4",
                (v4[0] + 10, v4[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        if x1:
            cv2.circle(display, x1, 8, (0, 0, 255), -1)
            cv2.putText(
                display,
                "x1",
                (x1[0] + 10, x1[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        if x2:
            cv2.circle(display, x2, 8, (0, 0, 255), -1)
            cv2.putText(
                display,
                "x2",
                (x2[0] + 10, x2[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        if x3:
            cv2.circle(display, x3, 8, (255, 0, 0), -1)
            cv2.putText(
                display,
                "x3",
                (x3[0] + 10, x3[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )
        if x4:
            cv2.circle(display, x4, 8, (255, 0, 0), -1)
            cv2.putText(
                display,
                "x4",
                (x4[0] + 10, x4[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )

        draw_infinite_line(v1, v2, (0, 255, 0))
        draw_infinite_line(v3, v4, (0, 255, 0))
        draw_infinite_line(x1, x2, (255, 0, 0))
        draw_infinite_line(x3, x4, (255, 0, 0))

        # Find intersection points of the infinite lines
        def line_params(pt1, pt2):
            x1, y1 = pt1
            x2, y2 = pt2
            A = y2 - y1
            B = x1 - x2
            C = A * x1 + B * y1
            return A, B, C

        def intersection(l1pt1, l1pt2, l2pt1, l2pt2):
            A1, B1, C1 = line_params(l1pt1, l1pt2)
            A2, B2, C2 = line_params(l2pt1, l2pt2)
            det = A1 * B2 - A2 * B1
            if det == 0:
                return None
            x = int((B2 * C1 - B1 * C2) / det)
            y = int((A1 * C2 - A2 * C1) / det)
            return (x, y)

        # Only compute if all points exist
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
    if v1:
        cv2.circle(display, v1, 8, (0, 255, 255), -1)
        cv2.putText(display, "v1", (v1[0] + 10, v1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if v2:
        cv2.circle(display, v2, 8, (0, 255, 255), -1)
        cv2.putText(display, "v2", (v2[0] + 10, v2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if v3:
        cv2.circle(display, v3, 8, (0, 255, 255), -1)
        cv2.putText(display, "v3", (v3[0] + 10, v3[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if v4:
        cv2.circle(display, v4, 8, (0, 255, 255), -1)
        cv2.putText(display, "v4", (v4[0] + 10, v4[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Find intersection points of horizontal lines with rightmost edge
    x1, x2 = None, None
    # y=100
    if 0 <= 100 < h:
        row = edges[100]
        for x in range(w - 1, -1, -1):
            if row[x] > 0:
                x1 = (x, 100)
                break
    # y=360
    if 0 <= 360 < h:
        row = edges[360]
        for x in range(w - 1, -1, -1):
            if row[x] > 0:
                x2 = (x, 360)
                break

    # Draw points if found
    if x1:
        cv2.circle(display, x1, 8, (0, 0, 255), -1)
        cv2.putText(
            display,
            "x1",
            (x1[0] + 10, x1[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
    if x2:
        cv2.circle(display, x2, 8, (0, 0, 255), -1)
        cv2.putText(
            display,
            "x2",
            (x2[0] + 10, x2[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    # Find intersection points of horizontal lines with leftmost edge (search left to right)
    # Find intersection points of horizontal lines with second vertical edge line (x=185)
    x3, x4 = None, None
    # y=100
    if 0 <= 100 < h:
        row = edges[100]
        crossings = 0
        for x in range(w):
            if row[x] > 0:
                crossings += 1
                if crossings == 2:
                    x3 = (x, 100)
                    break
    # y=360
    if 0 <= 360 < h:
        row = edges[360]
        crossings = 0
        for x in range(w):
            if row[x] > 0:
                crossings += 1
                if crossings == 2:
                    x4 = (x, 360)
                    break

    # Draw points if found
    if x3:
        cv2.circle(display, x3, 8, (255, 0, 0), -1)
        cv2.putText(
            display,
            "x3",
            (x3[0] + 10, x3[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )
    if x4:
        cv2.circle(display, x4, 8, (255, 0, 0), -1)
        cv2.putText(
            display,
            "x4",
            (x4[0] + 10, x4[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )

    # Draw points if found

    # Overlay window: perspective rectification using p points
    p_points = [p1, p2, p3, p4]
    valid_p = [pt for pt in p_points if pt is not None]
    if len(valid_p) == 4:
        # Order points: top-left, top-right, bottom-right, bottom-left
        pts = np.array(valid_p, dtype=np.float32)
        # Sort by y, then x
        pts_sorted = sorted(pts, key=lambda p: (p[1], p[0]))
        top = sorted(pts_sorted[:2], key=lambda p: p[0])
        bottom = sorted(pts_sorted[2:], key=lambda p: p[0])
        rect = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
        # Compute square side length for rectified view
        widthA = np.linalg.norm(rect[2] - rect[3])
        widthB = np.linalg.norm(rect[1] - rect[0])
        heightA = np.linalg.norm(rect[1] - rect[2])
        heightB = np.linalg.norm(rect[0] - rect[3])
        maxSide = int(max(widthA, widthB, heightA, heightB))
        dst = np.array(
            [[0, 0], [maxSide - 1, 0], [maxSide - 1, maxSide - 1], [0, maxSide - 1]],
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
