"""Shared pieces for the monochrome "data-viz" modes (blueprint / fiber).

These modes imitate generative scientific-diagram art: white hairline
geometry, additive particle glow, and tiny technical annotations with
leader lines, all grayscale on near-black.
"""
from __future__ import annotations

import numpy as np
from vispy import scene


def additive(visual) -> None:
    """Configure a visual for glowing additive blending, no depth writes."""
    visual.set_gl_state("additive", depth_test=False, cull_face=False)


_LABEL_STYLES = [
    lambda r: f"{r.integers(100000, 999999)}",
    lambda r: f"{r.integers(10, 99)}.{r.integers(100, 999)}",
    lambda r: f"SIG.{r.integers(1000, 9999)}_{chr(65 + r.integers(0, 26))}",
    lambda r: f"{r.integers(0, 9)}:{r.integers(10, 99)}",
]


class AnnotationLayer:
    """Tiny technical labels with thin leader lines, pinned to scene points.

    `retarget(points)` re-pins the labels to new anchor points (call it every
    couple of seconds / on strong beats, not every frame).
    """

    def __init__(self, parent, n: int = 10, horizontal: bool = False,
                 seed: int = 3):
        self.n = n
        self.horizontal = horizontal
        self.rng = np.random.default_rng(seed)
        self.text = scene.visuals.Text(
            text=[""] * n, pos=np.zeros((n, 3), np.float32),
            font_size=6, color=(1.0, 1.0, 1.0, 0.6),
            anchor_x="left", anchor_y="bottom", parent=parent)
        self.lines = scene.visuals.Line(
            pos=np.zeros((n * 2, 3), np.float32), connect="segments",
            color=(1.0, 1.0, 1.0, 0.25), width=1, parent=parent)
        additive(self.lines)
        self.visuals = [self.text, self.lines]

    def retarget(self, points: np.ndarray) -> None:
        """points: (k, 3) candidate anchors; n are sampled from them."""
        if len(points) == 0:
            return
        idx = self.rng.choice(len(points), size=min(self.n, len(points)),
                              replace=len(points) < self.n)
        anchors = np.asarray(points, np.float32)[idx]

        if self.horizontal:
            # long horizontal callouts (x-ray style)
            dirs = np.zeros_like(anchors)
            dirs[:, 0] = self.rng.choice([-1.0, 1.0], len(anchors)) * \
                self.rng.uniform(0.8, 2.2, len(anchors))
        else:
            dirs = self.rng.normal(0, 0.35, anchors.shape).astype(np.float32)
            dirs[:, 0] += 0.4
            dirs[:, 2] = np.abs(dirs[:, 2]) * 0.3

        ends = anchors + dirs
        segs = np.empty((len(anchors) * 2, 3), np.float32)
        segs[0::2] = anchors
        segs[1::2] = ends
        self.lines.set_data(pos=segs, connect="segments",
                            color=(1.0, 1.0, 1.0, 0.25))
        labels = [self.rng.choice(_LABEL_STYLES)(self.rng)
                  for _ in range(len(anchors))]
        self.text.text = labels
        self.text.pos = ends

    def set_visible(self, v: bool) -> None:
        for vis in self.visuals:
            vis.visible = v


def morph_sphere(verts: np.ndarray, t: float, amount: float,
                 twist: float) -> np.ndarray:
    """Audio-driven morph for a unit sphere mesh.

    `amount` (0..~0.6) blends in a low-order lobed radial displacement
    (breathing blob), `twist` (radians) shears the sphere around its z axis
    proportional to height. Both animate with time t.
    """
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    lobes = (0.5 * np.sin(2.6 * x + 1.9 * t)
             + 0.3 * np.sin(3.3 * y - 1.4 * t + 1.0)
             + 0.2 * np.sin(4.1 * z + 0.8 * t + 2.0))
    r = 1.0 + amount * lobes
    a = twist * z + 0.0
    ca, sa = np.cos(a), np.sin(a)
    out = np.empty_like(verts)
    out[:, 0] = (x * ca - y * sa) * r
    out[:, 1] = (x * sa + y * ca) * r
    out[:, 2] = z * r
    return out


def glow_marker_data(pos: np.ndarray, alpha: np.ndarray, base_size: float,
                     halo_scale: float = 4.0, halo_alpha: float = 0.10):
    """Duplicate each particle as sharp core + big faint halo (fake bloom).

    Returns (positions, colors, sizes) ready for one Markers.set_data call.
    """
    n = len(pos)
    out_pos = np.vstack([pos, pos]).astype(np.float32)
    sizes = np.empty(2 * n, np.float32)
    sizes[:n] = base_size
    sizes[n:] = base_size * halo_scale
    colors = np.ones((2 * n, 4), np.float32)
    colors[:n, 3] = alpha
    colors[n:, 3] = alpha * halo_alpha
    return out_pos, colors, sizes
