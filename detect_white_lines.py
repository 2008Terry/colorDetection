#!/usr/bin/env python3
"""Robust soccer-field line detector (v2).

This version is designed for difficult images (fisheye, uneven lighting, low contrast),
and can detect lines that appear bright, dark, or both.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect soccer-field boundary lines")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument(
        "--line-polarity",
        choices=["bright", "dark", "both"],
        default="both",
        help="Detect bright lines, dark lines, or both (default: both)",
    )
    parser.add_argument(
        "--min-line-length",
        type=int,
        default=45,
        help="Minimum line length for Hough extraction",
    )
    parser.add_argument(
        "--max-line-gap",
        type=int,
        default=14,
        help="Maximum gap allowed when connecting line segments",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write extra intermediate maps for tuning",
    )
    return parser.parse_args()


def build_field_mask(image: np.ndarray) -> np.ndarray:
    """Estimate valid field area and suppress circular border/background."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Suppress very dark border/background first.
    _, non_black = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    # Turf is usually medium-to-high saturation with medium brightness.
    turf_like = cv2.inRange(hsv, (25, 10, 40), (110, 255, 255))

    merged = cv2.bitwise_or(non_black, turf_like)
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel_big, iterations=2)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full(gray.shape, 255, dtype=np.uint8)

    largest = max(contours, key=cv2.contourArea)
    field_mask = np.zeros_like(gray)
    cv2.drawContours(field_mask, [largest], -1, 255, thickness=cv2.FILLED)
    return cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel_big, iterations=1)


def line_enhancement(gray: np.ndarray, polarity: str) -> np.ndarray:
    """Enhance thin line structures under non-uniform illumination."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    k_large = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    white_small = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k_small)
    white_large = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k_large)
    black_small = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k_small)
    black_large = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k_large)

    bright_map = cv2.addWeighted(white_small, 0.55, white_large, 0.45, 0)
    dark_map = cv2.addWeighted(black_small, 0.55, black_large, 0.45, 0)

    if polarity == "bright":
        enhanced = bright_map
    elif polarity == "dark":
        enhanced = dark_map
    else:
        enhanced = cv2.max(bright_map, dark_map)

    # Local contrast map helps recover low-contrast segments.
    smooth = cv2.GaussianBlur(g, (0, 0), sigmaX=4.0)
    local_contrast = cv2.absdiff(g, smooth)
    enhanced = cv2.addWeighted(enhanced, 0.8, local_contrast, 0.2, 0)
    return enhanced


def detect_line_candidates(image: np.ndarray, field_mask: np.ndarray, polarity: str) -> tuple[np.ndarray, np.ndarray]:
    """Return binary candidate mask and enhancement map."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    enhanced = line_enhancement(gray, polarity)

    _, enh_bin = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Color prior: white paint should be low saturation; in difficult frames it may look darker.
    white_like = cv2.inRange(hsv, (0, 0, 100), (180, 105, 255))
    dark_desat = cv2.inRange(hsv, (0, 0, 0), (180, 70, 120))

    if polarity == "bright":
        color_prior = white_like
    elif polarity == "dark":
        color_prior = dark_desat
    else:
        color_prior = cv2.bitwise_or(white_like, dark_desat)

    merged = cv2.bitwise_or(enh_bin, color_prior)
    merged = cv2.bitwise_and(merged, field_mask)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, k, iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, k, iterations=2)
    return merged, enhanced


def keep_line_like_components(mask: np.ndarray) -> np.ndarray:
    """Filter blobs and keep line-like connected components."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)

    for label in range(1, n_labels):
        x, y, w, h, area = stats[label]
        if area < 25:
            continue

        ratio = max(w, h) / max(1, min(w, h))
        fill = area / max(1, w * h)

        # Keep elongated components; also keep medium fill components to avoid over-pruning curves.
        if ratio >= 2.2 or (ratio >= 1.4 and fill < 0.42):
            out[labels == label] = 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)


def extract_lines(binary_mask: np.ndarray, min_line_length: int, max_line_gap: int) -> np.ndarray:
    edges = cv2.Canny(binary_mask, 40, 125)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    line_mask = np.zeros_like(binary_mask)
    if lines is None:
        return line_mask

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, line)
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3, lineType=cv2.LINE_AA)

    return line_mask


def make_overlay(image: np.ndarray, mask: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    out = image.copy()
    out[mask > 0] = color
    return cv2.addWeighted(image, 0.7, out, 0.3, 0)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    field_mask = build_field_mask(image)
    candidates, enhanced = detect_line_candidates(image, field_mask, args.line_polarity)
    pruned = keep_line_like_components(candidates)
    hough_lines = extract_lines(pruned, args.min_line_length, args.max_line_gap)

    final_mask = cv2.bitwise_or(pruned, hough_lines)
    final_mask = cv2.bitwise_and(final_mask, field_mask)

    cv2.imwrite(str(outdir / "01_field_mask.png"), field_mask)
    cv2.imwrite(str(outdir / "02_enhanced.png"), enhanced)
    cv2.imwrite(str(outdir / "03_candidates.png"), candidates)
    cv2.imwrite(str(outdir / "04_pruned.png"), pruned)
    cv2.imwrite(str(outdir / "05_hough_lines.png"), hough_lines)
    cv2.imwrite(str(outdir / "06_final_mask.png"), final_mask)
    cv2.imwrite(str(outdir / "07_overlay.png"), make_overlay(image, final_mask))

    if args.debug:
        edges = cv2.Canny(pruned, 40, 125)
        cv2.imwrite(str(outdir / "08_edges_debug.png"), edges)

    print("Done. Outputs:")
    for name in [
        "01_field_mask.png",
        "02_enhanced.png",
        "03_candidates.png",
        "04_pruned.png",
        "05_hough_lines.png",
        "06_final_mask.png",
        "07_overlay.png",
    ]:
        print(" -", outdir / name)


if __name__ == "__main__":
    main()
