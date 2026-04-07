#!/usr/bin/env python3
"""Detect white field boundary lines from a single image.

Usage:
    python detect_white_lines.py --input input.jpg --outdir output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def build_valid_region_mask(img: np.ndarray, black_thr: int = 20) -> np.ndarray:
    """Build a mask for non-black visible region (helpful for fisheye borders)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > black_thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    keep = (labels == largest).astype(np.uint8) * 255
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=1)
    return keep


def white_candidate_mask(img: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Generate robust white-candidate mask from HSV + Lab + local contrast."""
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # White lines are bright with low saturation; adapt thresholds to scene.
    v_thr = max(160, int(np.percentile(v[valid_mask > 0], 78)))
    s_thr = min(75, int(np.percentile(s[valid_mask > 0], 45)))

    mask_hsv = ((v >= v_thr) & (s <= s_thr)).astype(np.uint8) * 255

    # Top-hat improves thin bright structures under uneven lighting.
    tophat = cv2.morphologyEx(l_channel, cv2.MORPH_TOPHAT, np.ones((17, 17), np.uint8))
    _, mask_tophat = cv2.threshold(tophat, 20, 255, cv2.THRESH_BINARY)

    # Additional bright cutoff on L-channel.
    l_thr = max(165, int(np.percentile(l_channel[valid_mask > 0], 80)))
    _, mask_l = cv2.threshold(l_channel, l_thr, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(mask_hsv, mask_tophat)
    mask = cv2.bitwise_and(mask, mask_l)
    mask = cv2.bitwise_and(mask, valid_mask)

    # Clean up.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def detect_lines(mask: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return line visualization and line-only binary mask."""
    edges = cv2.Canny(mask, 50, 150)
    edges = cv2.bitwise_and(edges, valid_mask)

    h, w = mask.shape[:2]
    min_len = max(30, int(min(h, w) * 0.08))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=min_len,
        maxLineGap=18,
    )

    line_mask = np.zeros_like(mask)
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    if lines is not None:
        for ln in lines[:, 0]:
            x1, y1, x2, y2 = map(int, ln)
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Merge detected segments with white candidates for more complete coverage.
    merged = cv2.bitwise_or(mask, line_mask)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    # Keep elongated connected components only.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    filtered = np.zeros_like(merged)
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if area < 150:
            continue
        elongation = max(ww, hh) / max(1, min(ww, hh))
        if elongation > 1.6 or area > 1000:
            filtered[labels == i] = 255

    filtered = cv2.bitwise_and(filtered, valid_mask)
    return overlay, filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect white boundary lines from field image")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(input_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    valid_mask = build_valid_region_mask(img)
    white_mask = white_candidate_mask(img, valid_mask)
    line_overlay, line_mask = detect_lines(white_mask, valid_mask)

    result = img.copy()
    result[line_mask > 0] = (255, 255, 255)
    result = cv2.addWeighted(img, 0.72, result, 0.28, 0)
    result[line_mask > 0] = (255, 255, 255)

    overlay = img.copy()
    overlay[line_mask > 0] = (0, 255, 255)
    overlay = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    overlay = cv2.addWeighted(overlay, 1.0, line_overlay, 1.0, 0)

    cv2.imwrite(str(outdir / "01_valid_mask.png"), valid_mask)
    cv2.imwrite(str(outdir / "02_white_candidate_mask.png"), white_mask)
    cv2.imwrite(str(outdir / "03_line_mask.png"), line_mask)
    cv2.imwrite(str(outdir / "04_overlay.png"), overlay)
    cv2.imwrite(str(outdir / "05_result_white_lines.png"), result)

    print(f"Done. Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
