"""Load an SVG logo as a mask, cropped to the artwork.

Drawings exported from Inkscape carry the page they were drawn on: the KVA
file is an A4 sheet whose graphic occupies about 41% of the width and 11%
of the height, with two thirds of the page empty below it. Honouring the
viewBox therefore centres a lot of nothing and shrinks the logo to a
stamp, so everything here works from the ink's own bounding box instead.

Rendering goes through QtSvg, which ships with PyQt6 -- no cairo or
Inkscape needed at runtime.
"""
from __future__ import annotations

import numpy as np

# the logo's letterforms are near-black; the ring around them is drawn in a
# very light grey, so a mid threshold separates the two cleanly
INK = 200


def render_svg(path: str, px: int = 1024) -> np.ndarray:
    """Rasterize an SVG to (h, w) luminance, 0 = ink, 255 = paper."""
    from PyQt6.QtCore import QSize
    from PyQt6.QtGui import QColor, QImage, QPainter
    from PyQt6.QtSvg import QSvgRenderer

    r = QSvgRenderer(path)
    if not r.isValid():
        raise ValueError(f"not a readable SVG: {path}")
    size = r.defaultSize()
    if size.width() <= 0 or size.height() <= 0:
        size = QSize(px, px)
    h = max(1, int(px * size.height() / size.width()))
    img = QImage(px, h, QImage.Format.Format_ARGB32)
    img.fill(QColor(255, 255, 255))
    p = QPainter(img)
    r.render(p)
    p.end()

    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    a = np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine() // 4, 4)
    a = a[:, :px, :3].astype(np.float32)          # BGRA -> drop alpha
    return a.mean(2)


def crop_to_ink(lum: np.ndarray, thresh: int = INK,
                pad: float = 0.02) -> np.ndarray:
    """Trim the page away, leaving the drawing plus a small margin."""
    ink = lum < thresh
    if not ink.any():
        return lum
    ys, xs = np.nonzero(ink)
    ph = int(pad * (ys.max() - ys.min() + 1))
    pw = int(pad * (xs.max() - xs.min() + 1))
    y0, y1 = max(0, ys.min() - ph), min(lum.shape[0], ys.max() + ph + 1)
    x0, x1 = max(0, xs.min() - pw), min(lum.shape[1], xs.max() + pw + 1)
    return lum[y0:y1, x0:x1]


def logo_mask(path: str, px: int = 1024, thresh: int = INK) -> np.ndarray:
    """A boolean mask of the logo's ink, cropped to the artwork."""
    return crop_to_ink(render_svg(path, px), thresh) < thresh
