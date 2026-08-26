"""Mode 6 — Geometric Kaleidoscope.

A central icosahedron surrounded by two counter-rotating rings of 6-fold
mirrored satellites. Rotation is velocity-driven; every beat injects a
rotational velocity spike that decays with damping. Colors walk the palette.

The whole kaleidoscope is one batched Mesh (single GPU upload per frame).
"""
from __future__ import annotations

import numpy as np
from vispy import scene

from ..physics.velocity import VelocityValue
from .base import BaseMode
from .geometry import (icosahedron, lambert, octahedron,
                       rotation_matrix, subdivide)


class KaleidoscopeMode(BaseMode):
    name = "Geometric Kaleidoscope"
    FOLD = 6

    def build(self):
        v, f = icosahedron()
        self.center_verts, cf = subdivide(v, f, 1)
        self.sat_verts, sf = octahedron()

        blocks = [(self.center_verts, cf)]
        self.sat_meta = []
        for ring in range(2):
            for k in range(self.FOLD):
                blocks.append((self.sat_verts, sf))
                self.sat_meta.append({"ring": ring, "k": k})

        faces, offset = [], 0
        self.slices = []
        for bv, bf in blocks:
            faces.append(bf + offset)
            self.slices.append(slice(offset, offset + len(bv)))
            offset += len(bv)
        self.faces = np.vstack(faces)
        self.n_verts = offset

        self.mesh = scene.visuals.Mesh(
            vertices=np.zeros((offset, 3), np.float32), faces=self.faces,
            vertex_colors=np.ones((offset, 4), np.float32),
            shading=None, parent=self.view.scene)
        self.visuals = [self.mesh]

        self.rot = VelocityValue(0.4, accel=4.0, damping=self.settings.damping)
        self.counter_rot = VelocityValue(-0.3, accel=4.0, damping=self.settings.damping)
        self.pulse = VelocityValue(1.0, accel=25.0, damping=self.settings.damping)
        self.angle = 0.0
        self.counter_angle = 0.0

    def update(self, frame, dt):
        for v in (self.rot, self.counter_rot, self.pulse):
            v.damping = self.settings.damping

        self.rot.set_target(0.4 + frame.rms * 2.5)
        self.counter_rot.set_target(-(0.3 + frame.centroid * 2.0))
        self.pulse.set_target(0.9 + frame.rms * 0.6)
        if frame.beat:
            kick = self.beat_kick(frame, 1.0) * self.settings.beat_impulse
            self.rot.impulse(kick * 5.0)                 # velocity spike
            self.counter_rot.impulse(-kick * 4.0)
            self.pulse.impulse(kick * 4.0)

        self.angle += self.rot.update(dt) * dt
        self.counter_angle += self.counter_rot.update(dt) * dt
        s = max(0.1, self.pulse.update(dt))

        verts = np.empty((self.n_verts, 3), np.float32)
        colors = np.ones((self.n_verts, 4), np.float32)

        Rc = rotation_matrix(np.array([0.2, 1.0, 0.3]), self.angle * 2.0)
        cv = self.center_verts @ Rc.T
        verts[self.slices[0]] = cv * s * 0.9
        c0 = self.palette.band_color(3)
        colors[self.slices[0], :3] = np.clip(
            c0 * (0.5 + frame.rms) * lambert(cv), 0, 1)

        for meta, sl in zip(self.sat_meta, self.slices[1:]):
            ring, k = meta["ring"], meta["k"]
            base_a = 2 * np.pi * k / self.FOLD
            a = base_a + (self.angle if ring == 0 else self.counter_angle)
            radius = 2.0 + ring * 1.3
            tilt = 0.5 * np.sin(self.angle + k)
            center = np.array([np.cos(a) * radius,
                               np.sin(tilt) * (0.8 + ring),
                               np.sin(a) * radius])
            Rs = rotation_matrix(np.array([1.0, 0.7, 0.3]),
                                 self.angle * 3.0 + k)
            band = (k + ring * 3) % 7
            e = frame.bands[band]
            sv = self.sat_verts @ Rs.T
            verts[sl] = sv * (0.35 + e * 0.5) * s + center
            c = self.palette.band_color(band)
            colors[sl, :3] = np.clip(c * (0.45 + 0.55 * e) * lambert(sv), 0, 1)

        self.mesh.set_data(vertices=verts, faces=self.faces, vertex_colors=colors)

    def velocity_magnitude(self):
        return self.rot.speed + self.counter_rot.speed + self.pulse.speed
