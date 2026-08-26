"""Mode 10 — Reactive 3D Text.

Your text is rasterized with PIL and every lit pixel of the glyphs becomes a
point in a 3D cloud, extruded into depth layers so the letters are solid
type rather than a flat sheet.

The word doubles as a spectrum analyzer: a point's horizontal position picks
its frequency band, so the left of the word answers the bass and the right
answers the highs — each zone lit in its own color and pushed toward the
viewer by that band's energy. Kicks blow the letters apart into drifting
dust that springs back into legible type.

The word *sways* within a bounded angle rather than spinning freely: a
continuously rotating word is edge-on (and unreadable) half the time, while
a bounded sway keeps it facing you and still shows the extruded sides.
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


def wrap_text(text: str, max_aspect: float = 3.2) -> str:
    """Break a long word/phrase over two lines.

    A single long line has to be scaled down so far to fit the frame that
    its letters become an unreadable stripe; two lines keep them large.
    """
    if len(text) < 7:
        return text
    est_aspect = len(text) * 0.62      # rough glyph aspect for a bold face
    if est_aspect <= max_aspect:
        return text
    mid = len(text) // 2
    spaces = [i for i, c in enumerate(text) if c == " "]
    if spaces:                          # prefer breaking at a real space
        cut = min(spaces, key=lambda i: abs(i - mid))
        return text[:cut] + "\n" + text[cut + 1:]
    return text[:mid] + "\n" + text[mid:]


def text_points(text: str, max_points: int = 9000, px: int = 150,
                target_w: float = 3.0,
                target_h: float = 1.7) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize `text` and return (points Nx2, u in [0,1]).

    `u` is the horizontal position across the whole word, used to map each
    point onto a frequency band.
    """
    from PIL import Image, ImageDraw

    text = (text or "NEANDERTHALV").strip() or "NEANDERTHALV"
    text = wrap_text(text)
    font = load_font(px)
    tmp = Image.new("L", (8, 8))
    d = ImageDraw.Draw(tmp)
    try:
        box = d.multiline_textbbox((0, 0), text, font=font, spacing=12)
    except Exception:
        box = (0, 0, px * len(text) // 2, px)
    w = max(box[2] - box[0], 1) + 20
    h = max(box[3] - box[1], 1) + 20

    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).multiline_text((10 - box[0], 10 - box[1]), text,
                                       fill=255, font=font, spacing=12,
                                       align="center")
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
    # Fit into a target box: normalizing the long axis alone leaves a long
    # word's letters a few pixels tall, while width-only overflows the frame.
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    scale = min(target_w / w, target_h / h)
    pts = np.stack([(xs - cx) * scale, -(ys - cy) * scale], axis=1)
    return pts.astype(np.float32), u.astype(np.float32)


class TextMode(BaseMode):
    name = "Reactive 3D Text"
    camera_distance = 3.2       # framed so the word fills the view
    camera_elevation = 8.0
    trail_scale = 0.4           # heavy trails smear type illegible

    LAYERS = 7              # extrusion slices, front-to-back
    EXTRUDE = 0.30          # half-thickness of the slab
    BACK_DENSITY = 0.45     # rear slices are sparser: they only add body

    def build(self):
        self.rng = np.random.default_rng(5)
        self._text = None
        self._build_points(self.settings.text_content)

        self.markers = scene.visuals.Markers(parent=self.view.scene,
                                             antialias=1)
        additive(self.markers)
        self.visuals = [self.markers]

        d = self.settings.damping
        self.sway_rate = VelocityValue(0.0, accel=3.0, damping=d)
        self.pulse = VelocityValue(1.0, accel=18.0, damping=d)
        self.explode = VelocityValue(0.0, accel=6.0, damping=0.72)
        self.sway_phase = 0.0
        self._t = 0.0
        self._last_ft = None

    def _build_points(self, text: str) -> None:
        pts, u = text_points(text)
        self._text = text
        n = len(pts)
        if n == 0:
            self.base = np.zeros((0, 3), np.float32)
            self.band = np.zeros(0, int)
            self.jitter = np.zeros((0, 3), np.float32)
            self.depth_t = np.zeros(0, np.float32)
            self.n = 0
            return

        # Extrude front-to-back. The front slice keeps every point so the
        # letterforms stay sharp; rear slices are thinned since they only
        # need to fill in the sides of the slab.
        layers = self.LAYERS
        zs = np.linspace(self.EXTRUDE, -self.EXTRUDE, layers).astype(np.float32)
        chunks_p, chunks_u, chunks_d = [], [], []
        for i, z in enumerate(zs):
            if i == 0:
                sel = np.arange(n)
            else:
                k = max(1, int(n * self.BACK_DENSITY))
                sel = self.rng.choice(n, k, replace=False)
            block = np.empty((len(sel), 3), np.float32)
            block[:, 0] = pts[sel, 0]
            block[:, 1] = z                     # depth axis (vispy is Z-up)
            block[:, 2] = pts[sel, 1]
            chunks_p.append(block)
            chunks_u.append(u[sel])
            chunks_d.append(np.full(len(sel), i / max(layers - 1, 1),
                                    np.float32))

        self.base = np.concatenate(chunks_p)
        uu = np.concatenate(chunks_u)
        self.depth_t = np.concatenate(chunks_d)    # 0 = front, 1 = back
        self.band = np.clip((uu * 7).astype(int), 0, 6)
        # scatter distance is small relative to the ~2-unit-wide word: any
        # larger and a routine kick dissolves the type into an unreadable blob
        j = self.rng.normal(0, 1, self.base.shape).astype(np.float32)
        j /= np.linalg.norm(j, axis=1, keepdims=True) + 1e-9
        self.jitter = j * self.rng.uniform(
            0.12, 0.85, (len(self.base), 1)).astype(np.float32)
        self.n = len(self.base)

    def update(self, frame, dt):
        if self.settings.text_content != self._text:
            self._build_points(self.settings.text_content)
        if self.n == 0:
            return
        dt = min(dt, 0.05)
        self._t += dt
        d = self.settings.damping
        for v in (self.sway_rate, self.pulse):
            v.damping = d

        new_frame = frame.time != self._last_ft
        self._last_ft = frame.time
        punch = frame.punch if new_frame else 0.0
        bass = float(frame.bands[:2].mean())

        self.sway_rate.set_target(0.35 + frame.rms * 1.1)
        self.pulse.set_target(1.0 + bass * 0.22)
        self.explode.set_target(max(0.0, frame.rms - 0.7) * 0.7)
        if punch > 0.05:
            kb = punch * self.settings.beat_impulse
            self.pulse.impulse(kb * 1.2)
            self.explode.impulse(kb * self.settings.text_explode * 0.45)

        # bounded sway: the word turns to show its extruded sides but always
        # comes back to face the viewer, so it stays readable throughout
        self.sway_phase += max(0.0, self.sway_rate.update(dt)) * dt
        amp = float(self.settings.text_sway)
        yaw = np.sin(self.sway_phase) * amp
        pitch = np.sin(self.sway_phase * 0.53 + 1.1) * amp * 0.16

        scale = max(0.2, self.pulse.update(dt))
        boom = float(np.clip(self.explode.update(dt), 0.0, 3.0))

        # per-band forward push: the word reads as a spectrum analyzer
        band_e = frame.bands[self.band]
        pos = self.base * scale
        pos[:, 1] += band_e * self.settings.text_depth
        pos += self.jitter * boom

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        x1 = x * cy - y * sy            # yaw about the vertical (Z) axis
        y1 = x * sy + y * cy
        rot = np.empty_like(pos)
        rot[:, 0] = x1
        rot[:, 1] = y1 * cp - z * sp    # slight pitch for extra parallax
        rot[:, 2] = y1 * sp + z * cp

        # colour by frequency zone, brightened by that band's energy
        lut = self.palette.lut(7)
        colors = np.ones((self.n, 4), np.float32)
        colors[:, :3] = np.clip(lut[self.band] * (0.55 + 0.5 * band_e)[:, None],
                                0, 1)
        # depth shading: the front face is bright, the slab recedes into dark
        depth_fade = 1.0 - 0.65 * self.depth_t
        # additive blending over thousands of overlapping points saturates
        # fast — keep per-point alpha low and let the overlap build the glow
        colors[:, 3] = np.clip((0.16 + 0.30 * band_e) * depth_fade
                               * (1.0 - 0.4 * min(boom, 1.0)), 0.03, 0.7)
        colors[:, :3] *= depth_fade[:, None]

        self.markers.set_data(rot.astype(np.float32), face_color=colors,
                              size=1.5 + 1.3 * frame.rms + band_e * 1.1,
                              edge_width=0)

    def velocity_magnitude(self):
        return self.sway_rate.speed + self.pulse.speed + self.explode.speed
