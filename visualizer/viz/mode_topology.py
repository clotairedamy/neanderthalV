"""Mode 5 — Audio Waveform 3D Topology.

A radial surface where waves propagate outward from the center at velocity-
controlled speed: energy is injected at the innermost ring each frame and
advected outward through a ring buffer. Height layers pick colors from the
image palette.
"""
from __future__ import annotations

import numpy as np
from vispy import scene

from ..physics.velocity import VelocityValue
from .base import BaseMode
from .geometry import lambert


class TopologyMode(BaseMode):
    name = "Waveform 3D Topology"
    camera_distance = 9.0

    def build(self):
        g = self.profile.topology_grid
        self.rings = g
        self.sectors = g
        r = np.linspace(0.15, 4.0, self.rings)
        a = np.linspace(0, 2 * np.pi, self.sectors, endpoint=False)
        R, A = np.meshgrid(r, a, indexing="ij")
        self.X = (R * np.cos(A)).astype(np.float32)
        self.Y = (R * np.sin(A)).astype(np.float32)

        self.heights = np.zeros((self.rings, self.sectors), np.float32)
        self.wave_speed = VelocityValue(1.0, accel=6.0, damping=self.settings.damping)
        self._advect_acc = 0.0

        # triangulate the radial grid
        faces = []
        for i in range(self.rings - 1):
            for j in range(self.sectors):
                j2 = (j + 1) % self.sectors
                v00 = i * self.sectors + j
                v01 = i * self.sectors + j2
                v10 = (i + 1) * self.sectors + j
                v11 = (i + 1) * self.sectors + j2
                faces += [[v00, v10, v11], [v00, v11, v01]]
        self.faces = np.asarray(faces, np.int64)

        verts = np.stack([self.X.ravel(), self.Y.ravel(),
                          self.heights.ravel()], axis=1)
        self.mesh = scene.visuals.Mesh(
            vertices=verts.astype(np.float32), faces=self.faces,
            vertex_colors=np.ones((verts.shape[0], 4), np.float32),
            shading=None, parent=self.view.scene)
        self.visuals = [self.mesh]

    def update(self, frame, dt):
        self.wave_speed.damping = self.settings.damping
        self.wave_speed.set_target(1.0 + frame.rms * 6.0)
        if frame.beat:
            self.wave_speed.impulse(self.beat_kick(frame, 8.0)
                                    * self.settings.beat_impulse)
        speed = max(0.2, self.wave_speed.update(dt))

        # advect rings outward at `speed` rings/sec (velocity-based propagation)
        self._advect_acc += speed * dt * 22.0
        steps = int(self._advect_acc)
        self._advect_acc -= steps
        for _ in range(min(steps, 6)):
            self.heights[1:] = self.heights[:-1] * 0.985
            # inject: waveform shaped around the circle + band accents
            w = frame.waveform
            idx = np.linspace(0, len(w) - 1, self.sectors).astype(int)
            ring = w[idx] * 2.5 * (0.4 + frame.rms)
            band_mod = np.repeat(frame.bands, int(np.ceil(self.sectors / 7)))[:self.sectors]
            self.heights[0] = ring + band_mod * 0.8

        verts = np.stack([self.X.ravel(), self.Y.ravel(),
                          self.heights.ravel()], axis=1)

        # color by height layer through the palette
        h = self.heights.ravel()
        norm = np.clip((h + 1.2) / 2.4, 0, 1)
        lut = self.palette.lut(64)
        colors = np.ones((len(h), 4), np.float32)
        # surface normal from the height gradient, then baked lighting
        gy, gx = np.gradient(self.heights)
        nrm = np.stack([-gx.ravel(), -gy.ravel(), np.full(h.size, 0.35)], 1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        colors[:, :3] = lut[(norm * 63).astype(int)] * lambert(nrm)
        self.mesh.set_data(vertices=verts.astype(np.float32),
                           faces=self.faces, vertex_colors=colors)

    def velocity_magnitude(self):
        return self.wave_speed.speed
