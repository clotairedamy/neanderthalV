"""Mode 2 — Polyhedra Harmonic Array.

Four rotating platonic solids, one per stem (vocals/drums/bass/other), on
velocity-controlled orbits. Orbit speed follows stem energy; beats add angular
impulses; scale pulses with stem level; colors map stems to the image palette.

All four solids live in one batched Mesh (single GPU upload per frame).
"""
from __future__ import annotations

import numpy as np
from vispy import scene

from ..audio.analyzer import STEMS
from ..physics.velocity import VelocityValue
from .base import BaseMode
from .geometry import (cube, icosahedron, lambert, octahedron,
                       rotation_matrix, tetrahedron)


class PolyhedraMode(BaseMode):
    name = "Polyhedra Harmonic Array"

    SHAPES = [tetrahedron, cube, octahedron, icosahedron]

    def build(self):
        self.items = []
        all_faces = []
        offset = 0
        for i, (stem, shape_fn) in enumerate(zip(STEMS, self.SHAPES)):
            v, f = shape_fn()
            all_faces.append(f + offset)
            self.items.append({
                "stem": stem, "verts": v, "slice": slice(offset, offset + len(v)),
                "orbit": VelocityValue(0.0, accel=3.0, damping=self.settings.damping),
                "spin": VelocityValue(0.0, accel=5.0, damping=self.settings.damping),
                "scale": VelocityValue(0.6, accel=20.0, damping=self.settings.damping),
                "orbit_angle": i * np.pi / 2,
                "spin_angle": 0.0,
                "axis": np.array([np.sin(i * 1.7) + 0.2, 1.0, np.cos(i * 2.3)]),
                "radius": 2.2,
            })
            offset += len(v)

        self.faces = np.vstack(all_faces)
        self.n_verts = offset
        self.mesh = scene.visuals.Mesh(
            vertices=np.zeros((offset, 3), np.float32), faces=self.faces,
            vertex_colors=np.ones((offset, 4), np.float32),
            shading=None, parent=self.view.scene)
        self.visuals = [self.mesh]

    def update(self, frame, dt):
        verts = np.empty((self.n_verts, 3), np.float32)
        colors = np.ones((self.n_verts, 4), np.float32)

        for it in self.items:
            e = frame.stem_energy.get(it["stem"], 0.0)
            for k in ("orbit", "spin", "scale"):
                it[k].damping = self.settings.damping

            it["orbit"].set_target(0.25 + e * 2.2)
            it["spin"].set_target(0.5 + e * 5.0)
            it["scale"].set_target(0.45 + e * 0.8)
            if frame.beat:
                kick = self.beat_kick(frame, 1.0) * self.settings.beat_impulse
                it["orbit"].impulse(kick * 2.0)
                it["spin"].impulse(kick * 6.0)
                it["scale"].impulse(kick * 1.5)

            it["orbit_angle"] += it["orbit"].update(dt) * dt
            it["spin_angle"] += it["spin"].update(dt) * dt
            s = max(0.05, it["scale"].update(dt))

            a = it["orbit_angle"]
            center = np.array([np.cos(a) * it["radius"],
                               np.sin(a * 0.7) * 0.8,
                               np.sin(a) * it["radius"]])
            R = rotation_matrix(it["axis"], it["spin_angle"])
            rv = it["verts"] @ R.T
            verts[it["slice"]] = rv * s + center

            c = self.palette.stem_color(it["stem"])
            # these solids' vertices sit on the unit sphere, so the rotated
            # vertex direction is a usable normal for baked lighting
            shade = lambert(rv / (np.linalg.norm(rv, axis=1, keepdims=True) + 1e-9))
            colors[it["slice"], :3] = np.clip(c * (0.4 + 0.6 * e) * shade, 0, 1)

        self.mesh.set_data(vertices=verts, faces=self.faces, vertex_colors=colors)

    def velocity_magnitude(self):
        return sum(it["orbit"].speed + it["spin"].speed for it in self.items)
