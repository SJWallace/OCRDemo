# app/line_segmenter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image
import cv2


@dataclass
class LineImage:
    """Represents a single line cropped from the page."""
    image: Image.Image
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def segment_lines(page: Image.Image) -> List[LineImage]:
    """
    Simple horizontal projection-based line segmentation.
    Not as fancy as Kraken, but works well on your samples.
    """
    gray = np.array(page.convert("L"))
    # Binarize: text black on white
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection: sum of ink per row
    projection = bw.sum(axis=1)

    # Threshold to find "ink-heavy" rows (potential text)
    thresh = projection.max() * 0.1
    is_text_row = projection > thresh

    lines: List[LineImage] = []
    in_line = False
    start_row = 0

    for i, flag in enumerate(is_text_row):
        if flag and not in_line:
            in_line = True
            start_row = i
        elif not flag and in_line:
            in_line = False
            end_row = i
            # Add a bit of padding
            pad = 3
            y1 = max(0, start_row - pad)
            y2 = min(gray.shape[0], end_row + pad)
            line_img = page.crop((0, y1, page.width, y2))
            lines.append(LineImage(image=line_img, bbox=(0, y1, page.width, y2)))

    # In case the last line goes to the bottom
    if in_line:
        pad = 3
        y1 = max(0, start_row - pad)
        y2 = gray.shape[0]
        line_img = page.crop((0, y1, page.width, y2))
        lines.append(LineImage(image=line_img, bbox=(0, y1, page.width, y2)))

    return lines
