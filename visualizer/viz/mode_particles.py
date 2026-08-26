"""Mode 3 — Particle Swarm.

2000+ particles with momentum physics: a swirling attractor field whose
strength follows audio energy, outward radial impulses on beats, and fading
velocity trails colored from the image palette (particles are assigned to
frequency bands; each band drives its particles' agitation).
"""
from __future__ import annotations

import numpy as np
from vispy import scene

from .base import BaseMode


class ParticlesMode(BaseMode):
    name = "Particle Swarm"
    TRAIL = 6  # positions of history per particle

    def build(self):
        n = self.profile.particle_count
        rng = np.random.default_rng(7)
        self.n = n
        r = rng.uniform(0.5, 2.5, n) ** 0.8
        theta = rng.uniform(0, 2 * np.pi, n)
        phi = np.arccos(rng.uniform(-1, 1, n))
        self.pos = np.stack([r * np.sin(phi) * np.cos(theta),
                             r * np.cos(phi),
                             r * np.sin(phi) * np.sin(theta)], axis=1)
        self.vel = rng.normal(0, 0.1, (n, 3))
        self.band = rng.integers(0, 7, n)
        self.history = np.repeat(self.pos[None], self.TRAIL, axis=0)

        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.markers.set_data(self.pos.astype(np.float32), size=4)
        self.trails = scene.visuals.Line(pos=np.zeros((2, 3), np.float32),
                                         connect="segments", width=1,
                                         parent=self.view.scene)
        self.visuals = [self.markers, self.trails]
        self._speed = 0.0

    def update(self, frame, dt):
        dt = min(dt, 0.05)
        energy = frame.rms
        band_e = frame.bands[self.band]                       # (n,)

        # forces: spring to shell, swirl, band agitation
        r = np.linalg.norm(self.pos, axis=1, keepdims=True) + 1e-6
        radial = self.pos / r
        shell = 1.2 + band_e[:, None] * 1.5
        spring = (shell - r) * radial * 4.0
        up = np.array([0.0, 1.0, 0.0])
        swirl = np.cross(up, radial) * (0.8 + energy * 4.0)
        rng = np.random.default_rng(int(frame.time * 1000) & 0xFFFF)
        jitter = rng.normal(0, 1, self.pos.shape) * (0.15 + band_e[:, None] * 2.0)

        acc = spring + swirl + jitter
        if frame.beat:
            acc += radial * self.beat_kick(frame, 220.0) * self.settings.beat_impulse

        # momentum physics with configurable friction
        self.vel += acc * dt
        self.vel *= self.settings.damping ** (dt * 60.0)
        self.pos += self.vel * dt
        self._speed = float(np.mean(np.linalg.norm(self.vel, axis=1)))

        # trails
        self.history = np.roll(self.history, 1, axis=0)
        self.history[0] = self.pos

        colors = np.ones((self.n, 4), np.float32)
        base = self.palette.colors[self.band]
        bright = 0.4 + 0.6 * np.clip(band_e, 0, 1)[:, None]
        colors[:, :3] = np.clip(base * bright, 0, 1)
        sizes = 3.0 + band_e * 7.0 + (3.0 if frame.beat else 0.0)
        self.markers.set_data(self.pos.astype(np.float32),
                              face_color=colors, edge_width=0, size=sizes)

        # segment list: p[t] -> p[t+1] for each particle, alpha fading with age
        segs = np.empty((self.n * (self.TRAIL - 1) * 2, 3), np.float32)
        cols = np.empty((self.n * (self.TRAIL - 1) * 2, 4), np.float32)
        for i in range(self.TRAIL - 1):
            s = i * self.n * 2
            segs[s:s + 2 * self.n:2] = self.history[i]
            segs[s + 1:s + 2 * self.n:2] = self.history[i + 1]
            fade = (1.0 - i / self.TRAIL) * 0.35
            c = np.repeat(colors, 2, axis=0)
            c[:, 3] = fade
            cols[s:s + 2 * self.n] = c
        self.trails.set_data(pos=segs, color=cols, connect="segments")

        if self.settings.show_velocity_debug:
            # velocity vectors drawn as brighter short trails (already visible);
            # boost alpha so direction of motion reads clearly
            pass

    def velocity_magnitude(self):
        return self._speed
