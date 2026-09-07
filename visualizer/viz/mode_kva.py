"""Mode 14 -- KVA Ring.

The logo artwork, made to move: a polar ring of dots around the KVA mark.

The source drawing is an Inkscape A4 page carrying a fine polar grid --
1606 radial spokes and 1656 arcs filling an annulus from 0.64 to 1.0 of
its outer radius -- with the logo in the middle. The ring is rebuilt here
rather than traced, because 3262 static line segments cannot react to
anything, while a polar grid regenerated each frame can carry the
spectrum around its circumference. The measured annulus ratio is kept, so
it still reads as the same drawing.

The logo itself *is* the artwork: it is rasterized from the SVG and cropped
to its ink, since the page it was drawn on is mostly empty and honouring
the viewBox would shrink the mark to a stamp in the middle of nothing.

The spectrum is laid around the ring mirrored about the vertical axis --
left and right halves reflect -- which keeps the figure symmetrical and
avoids a seam where the last bin meets the first. Beats fire a shockwave
that travels outward through the rings, and the logo pulses on the low
end.
"""
from __future__ import annotations

import os

import numpy as np
from vispy import scene

from ..config import asset_path
from ..physics.velocity import VelocityValue
from .base import BaseMode
from .svg_logo import logo_mask

# measured from the drawing: the ring occupies this fraction of its radius
R_IN, R_OUT = 0.64, 1.0

# which spectrum bins each band selection reads (64-bin spectrum, 7 bands)
BANDS = {
    "all": (0, 64),
    "lows": (0, 18),
    "mids": (18, 45),
    "highs": (45, 64),
}
BAND_ORDER = ["all", "lows", "mids", "highs"]


def ring_points(sectors: int, rings: int):
    """Polar grid over the annulus: angles, radii and the per-dot arrays."""
    th = np.linspace(0.0, 2.0 * np.pi, sectors, endpoint=False)
    rr = np.linspace(R_IN, R_OUT, rings)
    T, R = np.meshgrid(th, rr)                    # (rings, sectors)
    return T, R


def mirrored_bins(sectors: int, lo: int, hi: int) -> np.ndarray:
    """Map each sector to a spectrum bin, mirrored about the vertical.

    Running the bins straight round the circle puts bin 0 hard against bin
    63, which shows as a seam; reflecting them makes the figure symmetric
    and puts the loudest end of the range at the top.
    """
    half = sectors // 2
    idx = np.linspace(lo, hi - 1, half)
    full = np.concatenate([idx, idx[::-1]])
    if len(full) < sectors:                        # odd sector counts
        full = np.concatenate([full, full[-1:]])
    return np.clip(full[:sectors].astype(int), 0, 63)


class KvaRingMode(BaseMode):
    name = "KVA Ring"
    camera = "panzoom"          # a logo is front-facing; never orbit it
    trail_scale = 0.45
    bloom_scale = 1.3           # the ring should bloom

    def build(self):
        r = self.profile.fractal_resolution
        self.extent = r / 2 + 20
        # the view is only guaranteed to show +-extent on its narrow axis,
        # and the ring has to leave room for the spokes and for the beat
        # pushing it outward, or it clips against the sides
        self.rad = self.extent * 0.72
        d = self.settings.damping

        self.spin = 0.0
        self.pulse = VelocityValue(1.0, accel=16.0, damping=d)
        self.logo_pulse = VelocityValue(1.0, accel=14.0, damping=d)
        self._waves: list[float] = []       # birth times of active shockwaves
        self._t = 0.0
        self._last_ft = -1.0
        self._sectors = 0

        self.spokes = scene.visuals.Line(connect="segments", width=1,
                                         parent=self.view.scene)
        self.spokes.set_gl_state("additive", depth_test=False)
        self.spokes.order = 0
        self.dots = scene.visuals.Markers(parent=self.view.scene, antialias=1)
        self.dots.set_gl_state("additive", depth_test=False)
        self.dots.order = 1

        self.logo = scene.visuals.Image(np.zeros((2, 2, 4), np.float32),
                                        parent=self.view.scene)
        self.logo.set_gl_state("translucent", depth_test=False)
        self.logo.order = 2
        self._load_logo()
        self.visuals = [self.spokes, self.dots, self.logo]
        self._alloc(int(self.settings.kva_sectors))

    # ----------------------------------------------------------- the mark

    def _load_logo(self) -> None:
        path = asset_path("kva.svg")
        try:
            mask = logo_mask(path, px=900) if os.path.exists(path) else None
        except Exception:
            mask = None
        if mask is None or not mask.any():
            self._logo_rgba = None
            self.logo.visible = False
            return
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), np.float32)
        rgba[..., :3] = 1.0
        rgba[..., 3] = mask.astype(np.float32)
        # row 0 of an image is its top, and this view has y increasing up
        self._logo_rgba = rgba[::-1].copy()
        self._logo_wh = (w, h)
        self.logo.set_data(self._logo_rgba)

    def _place_logo(self, scale: float, alpha: float) -> None:
        from vispy.visuals.transforms import STTransform
        w, h = self._logo_wh
        # sit inside the ring's hole, with margin
        width = 2.0 * R_IN * self.rad * 0.92 * scale
        s = width / w
        self.logo.transform = STTransform(scale=(s, s),
                                          translate=(-w * s / 2, -h * s / 2))
        rgba = self._logo_rgba.copy()
        rgba[..., 3] *= alpha
        self.logo.set_data(rgba)

    # ------------------------------------------------------------ topology

    def _alloc(self, sectors: int) -> None:
        sectors = int(np.clip(sectors, 24, 360))
        rings = int(np.clip(self.settings.kva_rings, 2, 12))
        self._sectors, self._rings = sectors, rings
        self.T, self.R = ring_points(sectors, rings)
        self._bin_key = None
        # spokes: one short radial tick per sector, drawn under the dots
        self._spoke_idx = np.arange(sectors)

    def _bins(self, band: str) -> np.ndarray:
        key = (band, self._sectors)
        if key != self._bin_key:
            lo, hi = BANDS.get(band, BANDS["all"])
            self._bin = mirrored_bins(self._sectors, lo, hi)
            self._bin_key = key
        return self._bin

    # --------------------------------------------------------------- frame

    def update(self, frame, dt):
        s = self.settings
        if (int(s.kva_sectors) != self._sectors
                or int(s.kva_rings) != self._rings):
            self._alloc(int(s.kva_sectors))
        dt = min(dt, 0.05)
        self._t += dt
        fresh = frame.time != self._last_ft
        self._last_ft = frame.time

        bands = np.clip(frame.bands * s.sensitivity, 0, 1.5)
        lows = float(bands[:2].mean())
        spec = np.clip(frame.spectrum, 0.0, 1.0)

        if fresh and frame.beat:
            self.pulse.impulse(frame.beat_strength * s.beat_impulse * 0.9)
            self.logo_pulse.impulse(frame.beat_strength * s.beat_impulse * 0.5)
            self._waves.append(self._t)
        self.pulse.set_target(1.0)
        self.logo_pulse.set_target(1.0 + 0.10 * lows)
        beat = float(np.clip(self.pulse.update(dt), 0.4, 2.2))
        self._waves = [w for w in self._waves if self._t - w < 1.2][-4:]

        # the ring turns, faster when the track is busier
        self.spin += dt * (0.10 + 0.55 * frame.rms) * float(s.kva_spin)

        sect = self._bins(str(s.kva_band))
        energy = spec[sect]                                # (sectors,)
        # a shockwave is a bump in radius travelling outward
        boost = np.zeros_like(self.R)
        wave = np.zeros_like(self.R)          # brightness lift on the front
        for w in self._waves:
            age = self._t - w
            front = R_IN + (R_OUT - R_IN + 0.25) * (age / 0.55)
            ring_hit = np.exp(-((self.R - front) / 0.075) ** 2)
            boost += ring_hit * np.exp(-age * 2.0) * 0.16
            wave = np.maximum(wave, ring_hit * np.exp(-age * 2.0))
        # per-dot displacement: the band drives the ring outward
        push = energy[None, :] * float(s.kva_reach) * (0.35 + 0.65 * beat)
        rad = (self.R + push * 0.22 + boost) * self.rad

        ang = self.T + self.spin
        x = (rad * np.cos(ang)).ravel()
        y = (rad * np.sin(ang)).ravel()
        pos = np.stack([x, y, np.zeros_like(x)], 1).astype(np.float32)

        # some palettes carry values a hair outside [0, 1]
        lut = np.clip(self.palette.lut(256), 0.0, 1.0)
        # colour by the sector's own energy, so the ring reads as a dial
        # the shockwave lights the dots it passes, not just displaces them
        e = np.clip(np.repeat(energy[None, :], self._rings, 0)
                    + wave * 0.55, 0, 1).ravel()
        idx = np.clip((e * 255).astype(int), 0, 255)
        col = np.empty((len(x), 4), np.float32)
        col[:, :3] = lut[idx]
        # clip after the beat multiplier, not before: the pulse runs past 1
        col[:, 3] = np.clip((0.28 + 0.72 * e) * (0.6 + 0.5 * beat), 0, 1)
        size = np.clip(2.0 + 9.0 * e * float(s.kva_dot), 1.2, 20.0) \
            * (0.85 + 0.35 * beat)
        self.dots.set_data(pos, edge_width=0, face_color=col,
                           size=size.astype(np.float32))

        # radial ticks at the outer edge, brightest where the band is loud
        r0 = (R_OUT + 0.04) * self.rad
        r1 = r0 + energy * float(s.kva_reach) * 0.22 * self.rad
        a = self.T[0] + self.spin
        seg = np.empty((self._sectors * 2, 3), np.float32)
        seg[0::2, 0] = r0 * np.cos(a)
        seg[0::2, 1] = r0 * np.sin(a)
        seg[1::2, 0] = r1 * np.cos(a)
        seg[1::2, 1] = r1 * np.sin(a)
        seg[:, 2] = 0.0
        sc = np.empty((self._sectors * 2, 4), np.float32)
        sc[:, :3] = lut[np.clip((energy * 255).astype(int), 0, 255)].repeat(2, 0)
        sc[:, 3] = np.repeat(np.clip(0.10 + 0.75 * energy, 0, 1), 2)
        self.spokes.set_data(pos=seg, color=sc)

        if self._logo_rgba is not None:
            lp = float(np.clip(self.logo_pulse.update(dt), 0.7, 1.35))
            self._place_logo(lp, float(np.clip(0.88 + 0.12 * lows, 0, 1)))

    def velocity_magnitude(self):
        return self.pulse.speed + self.logo_pulse.speed
