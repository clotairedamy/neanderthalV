"""Drum-triggered grain overlay effects.

A fullscreen noise layer drawn on top of every visualization mode. Its
intensity is a velocity envelope: drum-stem transients and beats punch it up
instantly, damping lets it decay — so grain flashes exactly when a drum hits.

Styles:
- film:      fine monochrome film grain
- static:    coarse blocky TV static
- scanlines: horizontal interference lines with row jitter
- burst:     violent full-frame noise slams that die out fast
"""
from __future__ import annotations

import numpy as np
from vispy import scene
from vispy.visuals.transforms import STTransform

from ..physics.velocity import VelocityValue

GRAIN_STYLES = ["off", "film", "static", "scanlines", "burst"]

_RES = {
    "film": (420, 620),        # fine grain
    "static": (72, 108),       # chunky blocks
    "scanlines": (360, 480),
    "burst": (140, 210),
}
_INTERP = {"film": "linear", "static": "nearest",
           "scanlines": "linear", "burst": "nearest"}


class GrainOverlay:
    def __init__(self, view, settings):
        self.settings = settings
        self.view = view
        self.rng = np.random.default_rng(0)
        self.image = scene.visuals.Image(
            np.zeros((8, 8, 4), np.float32), parent=view.scene,
            interpolation="nearest")
        self.image.set_gl_state("additive", depth_test=False)
        self.image.visible = False

        # velocity envelope: impulses on drum hits, damped decay
        self.env = VelocityValue(0.0, accel=4.0, damping=0.82)
        self._prev_drums = 0.0
        self._hold = np.zeros((8, 8), np.float32)
        self._static_t = 0.0

    def _fit(self, h, w):
        self.image.transform = STTransform(scale=(1.0 / w, 1.0 / h))

    def update(self, frame, dt: float) -> None:
        style = self.settings.grain_mode
        if style == "off" or frame is None:
            self.image.visible = False
            return

        drums = frame.stem_energy.get("drums", 0.0)
        transient = max(0.0, drums - self._prev_drums)
        self._prev_drums = drums

        # drum hit -> instant velocity spike; sustain follows drum level
        self.env.set_target(drums * 0.5)
        if transient > 0.04:
            self.env.impulse(transient * (30.0 if style == "burst" else 18.0))
        if frame.beat:
            self.env.impulse(frame.beat_strength *
                             (6.0 if style == "burst" else 3.0))
        level = np.clip(self.env.update(dt), 0.0, 1.0) * \
            self.settings.grain_intensity

        if level < 0.015:
            self.image.visible = False
            return
        self.image.visible = True

        h, w = _RES[style]
        if self.image.interpolation != _INTERP[style]:
            self.image.interpolation = _INTERP[style]
        rng = self.rng
        if style == "film":
            noise = rng.normal(0.5, 0.32, (h, w)).clip(0, 1).astype(np.float32)
            alpha = (noise ** 2) * level * 0.34
            gray = np.full((h, w), 1.0, np.float32)
        elif style == "static":
            # chunky static that only re-rolls ~20x/sec for a stuttery feel
            self._static_t += dt
            if self._static_t > 0.05 or not self._hold.shape == (h, w):
                self._static_t = 0.0
                self._hold = rng.random((h, w)).astype(np.float32)
            noise = self._hold
            alpha = (noise > (1.0 - 0.45 * level)).astype(np.float32) * \
                (0.22 + 0.32 * level)
            gray = 0.6 + 0.4 * noise
        elif style == "scanlines":
            rows = rng.random((h, 1)).astype(np.float32)
            lines = (np.arange(h) % 3 == 0).astype(np.float32)[:, None]
            tear = (rows > 1.0 - 0.30 * level).astype(np.float32)
            noise = rng.random((h, w)).astype(np.float32)
            alpha = (lines * 0.14 + tear * 0.45) * level * (0.4 + 0.6 * noise)
            gray = np.full((h, w), 1.0, np.float32)
        else:  # burst
            noise = rng.random((h, w)).astype(np.float32)
            alpha = (noise ** 1.5) * (level ** 1.4) * 0.7
            gray = 0.75 + 0.25 * noise

        img = np.empty((h, w, 4), np.float32)
        img[..., 0] = img[..., 1] = img[..., 2] = gray
        img[..., 3] = np.clip(alpha, 0.0, 0.95)
        self.image.set_data(img)
        self._fit(h, w)

    @property
    def level(self) -> float:
        return float(np.clip(self.env.value, 0, 1))
