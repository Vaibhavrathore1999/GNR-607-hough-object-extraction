"""Reusable OpenCV pipeline for approximate vehicle rectangle detection."""

from pathlib import Path

import cv2
import numpy as np


SOURCE_PATH = Path(__file__).resolve().parent / "data" / "parking_lot_original.jpg"
CROP_XYXY = (480, 690, 2640, 1700)
DISPLAY_SCALE = 0.5

DEFAULT_PARAMS = {
    "canny_low": 55,
    "canny_high": 145,
    "closing_kernel": 5,
    "hough_votes": 28,
    "min_line_length": 24,
    "max_line_gap": 12,
}


def load_crop(source_path=SOURCE_PATH):
    """Load the credited photograph and return the reproducible resized crop."""
    original = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(f"Could not read source image: {source_path}")
    x_min, y_min, x_max, y_max = CROP_XYXY
    crop = original[y_min:y_max, x_min:x_max]
    return cv2.resize(
        crop, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA
    )


def preprocess(crop_bgr, params):
    """Return grayscale, denoised, Canny, and morphologically closed images."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.1)
    edges = cv2.Canny(
        denoised, params["canny_low"], params["canny_high"], L2gradient=True
    )
    size = params["closing_kernel"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    return gray, denoised, edges, closed


def _angle_delta(a, b):
    return abs((a - b + 90.0) % 180.0 - 90.0)


def _rectangle_iou(rect_a, rect_b):
    status, points = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if status == cv2.INTERSECT_NONE or points is None:
        return 0.0
    intersection = abs(cv2.contourArea(points))
    area_a = rect_a[1][0] * rect_a[1][1]
    area_b = rect_b[1][0] * rect_b[1][1]
    return intersection / (area_a + area_b - intersection + 1e-9)


def _rectangle_appearance(rect, gray, edges):
    box = np.int32(np.round(cv2.boxPoints(rect)))
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, box, 255)
    pixel_count = cv2.countNonZero(mask)
    if pixel_count == 0:
        return 0.0, 0.0, 0.0

    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
    edge_density = cv2.countNonZero(masked_edges) / pixel_count
    inside_mean, inside_std = cv2.meanStdDev(gray, mask=mask)

    expanded = (rect[0], (rect[1][0] + 16, rect[1][1] + 16), rect[2])
    ring_mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(
        ring_mask, np.int32(np.round(cv2.boxPoints(expanded))), 255
    )
    ring_mask[mask > 0] = 0
    ring_mean = cv2.mean(gray, mask=ring_mask)[0]
    local_contrast = abs(float(inside_mean[0, 0]) - ring_mean)
    return float(edge_density), float(inside_std[0, 0]), local_contrast


def detect_rectangles(gray, edges, closed, params, max_rectangles=8):
    """Group compatible probabilistic-Hough segments into rotated rectangles."""
    raw = cv2.HoughLinesP(
        closed,
        rho=1,
        theta=np.pi / 180,
        threshold=params["hough_votes"],
        minLineLength=params["min_line_length"],
        maxLineGap=params["max_line_gap"],
    )
    raw_lines = np.empty((0, 4), dtype=np.int32) if raw is None else raw.reshape(-1, 4)

    segments = []
    for x1, y1, x2, y2 in raw_lines:
        vector = np.array([x2 - x1, y2 - y1], dtype=float)
        length = float(np.linalg.norm(vector))
        if not 24 <= length <= 150:
            continue
        unit = vector / length
        if unit[0] < 0:
            unit = -unit
        segments.append(
            {
                "p1": np.array([x1, y1], dtype=float),
                "p2": np.array([x2, y2], dtype=float),
                "unit": unit,
                "angle": np.degrees(np.arctan2(unit[1], unit[0])) % 180,
                "midpoint": np.array([(x1 + x2) / 2, (y1 + y2) / 2]),
                "length": length,
            }
        )

    candidates = []
    for index, first in enumerate(segments):
        for second in segments[index + 1:]:
            if _angle_delta(first["angle"], second["angle"]) > 9:
                continue

            direction = first["unit"]
            normal = np.array([-direction[1], direction[0]])
            midpoint_delta = second["midpoint"] - first["midpoint"]
            width = abs(float(midpoint_delta @ normal))
            along_offset = abs(float(midpoint_delta @ direction))
            if not 18 <= width <= 65:
                continue
            if along_offset > 0.75 * max(first["length"], second["length"]):
                continue

            endpoints = np.vstack(
                [first["p1"], first["p2"], second["p1"], second["p2"]]
            )
            projected = endpoints @ direction
            length = float(projected.max() - projected.min())
            aspect = length / width
            if not (1.55 <= aspect <= 4.2 and 45 <= length <= 145):
                continue

            center = tuple(np.mean(endpoints, axis=0))
            angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
            rectangle = (center, (length, width), angle)
            density, standard_deviation, contrast = _rectangle_appearance(
                rectangle, gray, edges
            )
            if density < 0.045 or standard_deviation < 16 or contrast < 7:
                continue

            area = length * width
            shape_score = max(0.0, 1 - abs(aspect - 2.35) / 2.0)
            area_score = max(0.0, 1 - abs(area - 2600) / 3000)
            score = (
                contrast / 35
                + 1.5 * density
                + min(standard_deviation / 120, 0.5)
                + 0.25 * shape_score
                + 0.15 * area_score
            )
            candidates.append(
                {
                    "score": score,
                    "rect": rectangle,
                    "edge_density": density,
                    "contrast": contrast,
                }
            )

    height, width = gray.shape
    kept = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        center_x, center_y = candidate["rect"][0]
        if not (40 <= center_x <= width - 90 and 24 <= center_y <= height - 18):
            continue
        duplicate = False
        for prior in kept:
            center_distance = np.linalg.norm(
                np.asarray(candidate["rect"][0]) - np.asarray(prior["rect"][0])
            )
            if center_distance < 34 or _rectangle_iou(candidate["rect"], prior["rect"]) >= 0.15:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(candidate)
        if len(kept) >= max_rectangles:
            break
    return kept, raw_lines, len(candidates)


def run_pipeline(crop_bgr, params):
    gray, denoised, edges, closed = preprocess(crop_bgr, params)
    detections, lines, candidate_count = detect_rectangles(
        gray, edges, closed, params
    )
    return {
        "gray": gray,
        "denoised": denoised,
        "edges": edges,
        "closed": closed,
        "detections": detections,
        "lines": lines,
        "candidate_count": candidate_count,
    }


def draw_overlay(crop_bgr, detections, show_labels=True):
    canvas = crop_bgr.copy()
    for label, detection in enumerate(detections, start=1):
        box = np.int32(np.round(cv2.boxPoints(detection["rect"])))
        cv2.polylines(canvas, [box], True, (0, 230, 255), 3, cv2.LINE_AA)
        for point in box:
            cv2.circle(canvas, tuple(point), 3, (0, 110, 255), -1, cv2.LINE_AA)
        if show_labels:
            anchor = tuple(box[np.argmin(box[:, 1])])
            cv2.circle(canvas, anchor, 11, (24, 35, 55), -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(label),
                (anchor[0] - 5, anchor[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return canvas


def draw_hough_segments(crop_bgr, lines):
    canvas = crop_bgr.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(canvas, (x1, y1), (x2, y2), (255, 145, 35), 1, cv2.LINE_AA)
    return canvas
