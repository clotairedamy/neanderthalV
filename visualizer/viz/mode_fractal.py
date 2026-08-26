"""Mode 4 — Reactive Fractal (Julia set).

The Julia parameter c orbits the Mandelbrot cardioid; its angle and radius are
velocity-integrated targets driven by bass and centroid, so parameter changes
glide instead of jumping. Colors cycle through the image palette LUT.
"""
from __future__ import annotations

import numpy as np
from vispy import scene

from ..physics.velocity import VelocityValue
from .base import BaseMode


class FractalMode(BaseMode):
    name = "Reactive Fractal"
    camera = "panzoom"

    ITERS = 48

    def build(self):
        res = self.profile.fractal_resolution
        self.res = res
        x = np.linspace(-1.6, 1.6, res)
        y = np.linspace(-1.6, 1.6, res)
        X, Y = np.meshgrid(x, y)
        self.grid = (X + 1j * Y).astype(np.complex64)

        self.angle = VelocityValue(0.0, accel=2.0, damping=self.settings.damping)
        self.radius = VelocityValue(0.75, accel=6.0, damping=self.settings.damping)
        self.color_shift = VelocityValue(0.0, accel=3.0, damping=self.settings.damping)
        self._angle_pos = 0.0

        self.image = scene.visuals.Image(
            np.zeros((res, res, 3), np.float32), parent=self.view.scene,
            interpolation="linear")
        # center the image in the panzoom view
        from vispy.visuals.transforms import STTransform
        self.image.transform = STTransform(translate=(-res / 2, -res / 2))
        self.visuals = [self.image]

    def _julia(self, c: complex) -> np.ndarray:
        Z = self.grid.copy()
        count = np.zeros(Z.shape, np.float32)
        alive = np.ones(Z.shape, bool)
        for i in range(self.ITERS):
            Z[alive] = Z[alive] * Z[alive] + c
            escaped = np.abs(Z) > 2.0
            newly = escaped & alive
            count[newly] = i + 1 - np.log2(np.maximum(
                np.log(np.abs(Z[newly]) + 1e-9), 1e-9))
            alive &= ~escaped
            if not alive.any():
                break
        count[alive] = self.ITERS
        return count / self.ITERS

    def update(self, frame, dt):
        for v in (self.angle, self.radius, self.color_shift):
            v.damping = self.settings.damping

        bass = float(frame.bands[:2].mean())
        self.angle.set_target(0.15 + bass * 1.6)          # angular speed target
        self.radius.set_target(0.62 + frame.centroid * 0.28)
        self.color_shift.set_target(frame.rms * 4.0)
        if frame.beat:
            kick = self.beat_kick(frame, 1.0) * self.settings.beat_impulse
            self.angle.impulse(kick * 2.5)
            self.color_shift.impulse(kick * 6.0)

        self._angle_pos += self.angle.update(dt) * dt
        r = np.clip(self.radius.update(dt), 0.3, 0.95)
        shift = self.color_shift.update(dt)

        c = complex(r * np.cos(self._angle_pos), r * np.sin(self._angle_pos))
        m = self._julia(c)

        lut = self.palette.lut(256)
        idx = ((m * 255) + shift * 40) % 255
        img = lut[idx.astype(int)]
        img *= (0.35 + 0.65 * m[..., None])               # darken exteriors
        self.image.set_data(img.astype(np.float32))

    def velocity_magnitude(self):
        return self.angle.speed + self.radius.speed + self.color_shift.speed
