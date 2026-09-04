"""Loader for the string-art rosette SVGs (assets/Lows|mids|higs.svg).

The three drawings are one shape family, not three unrelated pictures: each
is 262 chords running from a point on an outer circle to the point at 52x
that angle on an inner circle, and they differ only in the inner radius
(45 / 84.6 / 162 against an outer 180). Segment i means the same thing in
all three files, so corresponding vertices can be blended directly.

That makes the blend exact rather than approximate. Both endpoints of a
chord keep their angle across the family and only the inner radius moves, so
a linear blend of two files is precisely the rosette whose inner radius is
the same blend of theirs -- including outside [0, 1], which is why `morph()`
is allowed to extrapolate past the end shapes.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np

from ..config import asset_path

SVG_NS = "{http://www.w3.org/2000/svg}"
# the three files in inner-radius order: tight core -> tight rim
FAMILY = ("Lows.svg", "mids.svg", "higs.svg")

_TRANSLATE = re.compile(r"translate\(\s*([-\d.eE]+)[\s,]+([-\d.eE]+)\s*\)")


def _translation(elem) -> np.ndarray:
    m = _TRANSLATE.search(elem.get("transform", "") or "")
    return np.array([float(m.group(1)), float(m.group(2))]) if m else np.zeros(2)


def _viewbox_center(root) -> np.ndarray:
    vb = root.get("viewBox")
    if vb:
        p = [float(v) for v in vb.replace(",", " ").split()]
        if len(p) == 4:
            return np.array([p[0] + p[2] / 2.0, p[1] + p[3] / 2.0])
    w, h = root.get("width"), root.get("height")
    if w and h:
        return np.array([float(re.sub(r"[^\d.]", "", w)) / 2.0,
                         float(re.sub(r"[^\d.]", "", h)) / 2.0])
    return np.zeros(2)


def load_rosette(path: str) -> np.ndarray:
    """Read one SVG into (N, 2, 2) segments: [segment][end][x, y].

    Coordinates come back centred on the drawing's own centre, scaled so the
    outermost point sits at radius 1, and with y flipped -- SVG counts y
    downward, the scene counts it up.
    """
    root = ET.parse(path).getroot()
    origin = _viewbox_center(root)

    segs: list[list[list[float]]] = []

    def walk(elem, offset):
        offset = offset + _translation(elem)
        for child in elem:
            if child.tag == SVG_NS + "line":
                try:
                    a = np.array([float(child.get("x1")), float(child.get("y1"))])
                    b = np.array([float(child.get("x2")), float(child.get("y2"))])
                except (TypeError, ValueError):
                    continue
                segs.append([(a + offset).tolist(), (b + offset).tolist()])
            else:
                walk(child, offset)

    walk(root, np.zeros(2))
    if not segs:
        raise ValueError(f"no <line> elements in {path}")

    pts = np.asarray(segs, np.float64) - origin
    pts[..., 1] *= -1.0
    scale = np.linalg.norm(pts.reshape(-1, 2), axis=1).max()
    return (pts / max(scale, 1e-9)).astype(np.float32)


def load_family(names=FAMILY) -> np.ndarray:
    """Stack the family into (K, N, 2, 2), truncated to a common length.

    The files are expected to match segment-for-segment. If a redrawn file
    disagrees the extra chords are dropped rather than raising, so a bad
    export degrades the morph instead of killing the mode.
    """
    shapes = [load_rosette(asset_path(n)) for n in names]
    n = min(len(s) for s in shapes)
    return np.stack([s[:n] for s in shapes])


def morph(family: np.ndarray, m: float) -> np.ndarray:
    """Blend the family at position `m` (0 = first file, K-1 = last).

    Extrapolates outside that range along the end pair, which stays a valid
    rosette until the inner radius would pass through zero.
    """
    k = family.shape[0]
    if k == 1:
        return family[0]
    i = int(np.floor(np.clip(m, 0.0, k - 2)))
    t = float(m) - i
    return family[i] + (family[i + 1] - family[i]) * t


def outer_angles(shape: np.ndarray) -> np.ndarray:
    """Angle of each chord's outer endpoint, in [0, 1) turns.

    This is the handle the mode uses to bind a chord to a spectrum bin: the
    rosette becomes a circular analyzer with the drawing's own geometry.
    """
    outer = shape[:, 0, :]
    return (np.arctan2(outer[:, 1], outer[:, 0]) / (2 * np.pi)) % 1.0
