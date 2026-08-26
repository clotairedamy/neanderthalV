"""Mode 10 — Reactive 3D Text.

Your text is rasterized with PIL, and every lit pixel of the glyphs becomes a
point in a 3D cloud, extruded into several depth layers so the letters have
real thickness.

The word doubles as a spectrum analyzer: a point's horizontal position picks
its frequency band, so the left of the word answers the bass and the right
answers the highs — each zone lit in its own color (lows / mids / highs) and
pushed toward the viewer by that band's energy. Kicks blow the letters apart
into drifting dust that springs back into legible type.
"""
from __future__ import annotations

import os

import numpy as np
from vispy import scene

from ..physics.velocity import VelocityValue
from .base import BaseMode
from .mono import additive

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def load_font(size: int):
    from PIL import ImageFont
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def text_points(text: str, max_points: int = 12000,
                px: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize `text` and return (points Nx2 in [-1,1], u in [0,1]).

    `u` is the horizontal position across the whole word, used to map each
    point onto a frequency band.
    """
    from PIL import Image, ImageDraw

    text = (text or "GEOVIZ").strip() or "GEOVIZ"
    font = load_font(px)
    tmp = Image.new("L", (8, 8))
    d = ImageDraw.Draw(tmp)
    try:
        box = d.textbbox((0, 0), text, font=font)
    except Exception:
        box = (0, 0, px * len(text) // 2, px)
    w = max(box[2] - box[0], 1) + 20
    h = max(box[3] - box[1], 1) + 20

    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((10 - box[0], 10 - box[1]), text, fill=255,
                             font=font)
    mask = np.asarray(img) > 96
    if not mask.any():
        return np.zeros((0, 2), np.float32), np.zeros(0, np.float32)

    # Trace the letterforms rather than filling them: a solid slab of
    # additive points washes out under bloom, while contours stay crisp.
    from scipy.ndimage import binary_erosion
    inner = binary_erosion(mask, np.ones((3, 3), bool))
    edge = mask & ~inner
    ey, ex = np.nonzero(edge)
    iy, ix = np.nonzero(inner)

    # keep every contour point, scatter a light fill inside for body
    fill_budget = max(0, max_points - len(ex)) // 3
    if len(ix) > fill_budget and fill_budget > 0:
        sel = np.linspace(0, len(ix) - 1, fill_budget).astype(int)
        ix, iy = ix[sel], iy[sel]
    elif fill_budget <= 0:
        ix, iy = ix[:0], iy[:0]

    xs = np.concatenate([ex, ix])
    ys = np.concatenate([ey, iy])
    if len(xs) > max_points:
        idx = np.linspace(0, len(xs) - 1, max_points).astype(int)
        xs, ys = xs[idx], ys[idx]

    u = (xs - xs.min()) / max(xs.max() - xs.min(), 1)
    # normalize to [-1,1] on the long axis, preserving aspect
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    scale = 2.0 / max(w, h)
    pts = np.stack([(xs - cx) * scale, -(ys - cy) * scale], axis=1)
    return pts.astype(np.float32), u.astype(np.float32)


class TextMode(BaseMode):
    name = "Reactive 3D Text"
    camera_distance = 4.4
    camera_elevation = 4.0

    LAYERS = 3          # extrusion depth slices

    def build(self):
        self.rng = np.random.default_rng(5)
        self._text = None
        self._build_points(self.settings.text_content)

        self.markers = scene.visuals.Markers(parent=self.view.scene,
                                             antialias=1)
        additive(self.markers)
        self.visuals = [self.markers]

        d = self.settings.damping
        self.spin = VelocityValue(0.0, accel=3.0, damping=d)
        self.pulse = VelocityValue(1.0, accel=18.0, damping=d)
        self.explode = VelocityValue(0.0, accel=6.0, damping=0.72)
        self.spin_angle = 0.0
        self._t = 0.0
        self._last_ft = None

    def _build_points(self, text: str) -> None:
        pts, u = text_points(text)
        self._text = text
        n = len(pts)
        if n == 0:
            self.base = np.zeros((0, 3), np.float32)
            self.u = np.zeros(0, np.float32)
            self.band = np.zeros(0, int)
            self.jitter = np.zeros((0, 3), np.float32)
            self.n = 0
            return

        # extrude: repeat the glyph across LAYERS depth slices
        layers = self.LAYERS
        depth = np.linspace(-0.16, 0.16, layers).astype(np.float32)
        base = np.empty((n * layers, 3), np.float32)
        for i, z in enumerate(depth):
            base[i * n:(i + 1) * n, 0] = pts[:, 0]
            base[i * n:(i + 1) * n, 1] = z
            base[i * n:(i + 1) * n, 2] = pts[:, 1]
        self.base = base
        self.u = np.tile(u, layers)
        self.layer = np.repeat(np.arange(layers), n)
        self.band = np.clip((self.u * 7).astype(int), 0, 6)
        # per-point explosion direction
        # scatter distance is small relative to the ~2-unit-wide word: any
        # larger and a routine kick dissolves the type into an unreadable blob
        j = self.rng.normal(0, 1, base.shape).astype(np.float32)
        j /= np.linalg.norm(j, axis=1, keepdims=True) + 1e-9
        self.jitter = j * self.rng.uniform(0.12, 0.85,
                                           (len(base), 1)).astype(np.float32)
        self.n = len(base)

    def update(self, frame, dt):
        if self.settings.text_content != self._text:
            self._build_points(self.settings.text_content)
        if self.n == 0:
            return
        dt = min(dt, 0.05)
        self._t += dt
        d = self.settings.damping
        for v in (self.spin, self.pulse):
            v.damping = d

        new_frame = frame.time != self._last_ft
        self._last_ft = frame.time
        punch = frame.punch if new_frame else 0.0
        bass = float(frame.bands[:2].mean())

        self.spin.set_target(0.10 + frame.rms * 0.45)
        self.pulse.set_target(1.0 + bass * 0.22)
        self.explode.set_target(max(0.0, frame.rms - 0.7) * 0.7)
        if punch > 0.05:
            kb = punch * self.settings.beat_impulse
            self.pulse.impulse(kb * 1.2)
            self.explode.impulse(kb * self.settings.text_explode * 0.45)

        self.spin_angle += self.spin.update(dt) * dt
        scale = max(0.2, self.pulse.update(dt))
        boom = float(np.clip(self.explode.update(dt), 0.0, 3.0))

        # per-band forward push: the word reads as a spectrum analyzer
        band_e = frame.bands[self.band]
        pos = self.base * scale
        pos = pos.copy()
        pos[:, 1] += band_e * self.settings.text_depth
        pos += self.jitter * boom

        ca, sa = np.cos(self.spin_angle), np.sin(self.spin_angle)
        rot = np.empty_like(pos)
        rot[:, 0] = pos[:, 0] * ca - pos[:, 1] * sa
        rot[:, 1] = pos[:, 0] * sa + pos[:, 1] * ca
        rot[:, 2] = pos[:, 2]

        # colour by frequency zone, brightened by that band's energy
        lut = self.palette.lut(7)
        colors = np.ones((self.n, 4), np.float32)
        colors[:, :3] = np.clip(lut[self.band] * (0.55 + 0.5 * band_e)[:, None],
                                0, 1)
        # back extrusion layers sit dimmer, so the type reads as solid
        layer_fade = 1.0 - 0.45 * (self.layer / max(self.LAYERS - 1, 1))
        # additive blending over thousands of overlapping points saturates
        # fast — keep per-point alpha low and let the overlap build the glow
        colors[:, 3] = np.clip((0.16 + 0.30 * band_e) * layer_fade
                               * (1.0 - 0.4 * min(boom, 1.0)), 0.03, 0.7)

        self.markers.set_data(rot.astype(np.float32), face_color=colors,
                              size=1.5 + 1.3 * frame.rms + band_e * 1.1,
                              edge_width=0)

    def velocity_magnitude(self):
        return self.spin.speed + self.pulse.speed + self.explode.speed
