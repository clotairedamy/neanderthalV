"""Mode 1 — Frequency Mesh Icosphere.

Faces are binned by latitude into the 7 frequency bands. Each band's energy
drives a velocity-integrated radial displacement of its vertices; colors come
from the active (image) palette per band.
"""
from __future__ import annotations

import numpy as np
from vispy import scene

from ..physics.velocity import VelocityArray, VelocityValue
from .base import BaseMode
from .geometry import icosphere, rotation_matrix, vertex_normals_sphere


class IcosphereMode(BaseMode):
    name = "Frequency Mesh Icosphere"

    def build(self):
        self.verts0, self.faces = icosphere(self.profile.icosphere_subdivisions)
        self.normals = vertex_normals_sphere(self.verts0)

        # assign each vertex a band by latitude (z from -1..1 -> 7 bins)
        z = self.verts0[:, 2]
        self.vert_band = np.clip(((z + 1) / 2 * 7).astype(int), 0, 6)

        # velocity-driven displacement per band + slow whole-body spin
        self.displace = VelocityArray(7, accel=18.0, damping=self.settings.damping)
        self.spin = VelocityValue(0.0, accel=4.0, damping=self.settings.damping)
        self.spin_angle = 0.0

        self.mesh = scene.visuals.Mesh(
            vertices=self.verts0.astype(np.float32), faces=self.faces,
            vertex_colors=np.ones((len(self.verts0), 4), np.float32),
            shading="smooth", parent=self.view.scene)
        self.wire = scene.visuals.Mesh(
            vertices=self.verts0.astype(np.float32) * 1.001, faces=self.faces,
            color=(1, 1, 1, 0.08), mode="lines", parent=self.view.scene)
        self.visuals = [self.mesh, self.wire]

    def update(self, frame, dt):
        d = self.settings.damping
        self.displace.damping = d
        self.spin.damping = d

        self.displace.set_target(frame.bands * 0.9)
        if frame.beat:
            self.displace.impulse(np.full(7, self.beat_kick(frame, 3.0)
                                          * self.settings.beat_impulse))
            self.spin.impulse(self.beat_kick(frame, 1.2) * self.settings.beat_impulse)
        self.spin.set_target(0.3 + frame.rms * 1.5)

        disp = self.displace.update(dt)
        self.spin_angle += self.spin.update(dt) * dt

        amount = disp[self.vert_band][:, None]
        verts = self.verts0 * (1.0 + 0.55 * amount)
        R = rotation_matrix(np.array([0.3, 1.0, 0.2]), self.spin_angle)
        verts = verts @ R.T

        colors = np.ones((len(verts), 4), np.float32)
        base = self.palette.colors[self.vert_band]
        bright = 0.35 + 0.65 * np.clip(disp[self.vert_band], 0, 1)[:, None]
        colors[:, :3] = np.clip(base * bright, 0, 1)

        self.mesh.set_data(vertices=verts.astype(np.float32), faces=self.faces,
                           vertex_colors=colors)
        self.wire.set_data(vertices=(verts * 1.003).astype(np.float32),
                           faces=self.faces, color=(1, 1, 1, 0.08))

    def velocity_magnitude(self):
        return self.displace.speed + self.spin.speed
