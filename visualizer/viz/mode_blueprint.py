"""Mode 7 — Blueprint Constellation (monochrome data-viz style).

Thin white sacred-geometry scaffolding (circles, hexagram, radial spokes)
slowly rotating under a dense spray of white particles that stream outward
from a dark vortex center, leaving ray-like streaks. Tiny technical labels
pin themselves to bright particles. Everything grayscale, additive glow.

Audio mapping: bass -> scaffold pulse & rotation velocity, energy -> particle
flow speed, beats -> radial burst impulses, treble -> particle sparkle.
"""
from __future__ import annotations

import numpy as np
from vispy import scene
from vispy.visuals.transforms import MatrixTransform

from ..physics.velocity import VelocityValue
from .base import BaseMode
from .geometry import icosphere
from .mono import AnnotationLayer, additive, glow_marker_data, morph_sphere


def _circle(cx, cy, r, n=72, z=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=True)
    pts = np.stack([cx + r * np.cos(a), cy + r * np.sin(a),
                    np.full(n, z)], axis=1)
    segs = np.empty(((n - 1) * 2, 3))
    segs[0::2] = pts[:-1]
    segs[1::2] = pts[1:]
    return segs


def _polygon(verts_2d, z=0.0):
    v = np.asarray(verts_2d, float)
    n = len(v)
    segs = np.empty((n * 2, 3))
    for i in range(n):
        segs[2 * i, :2] = v[i]
        segs[2 * i + 1, :2] = v[(i + 1) % n]
    segs[:, 2] = z
    return segs


def _build_scaffold(rng: np.random.Generator) -> np.ndarray:
    segs = []
    R = 3.0
    # concentric circles
    for r in (R, R * 0.62, R * 0.33):
        segs.append(_circle(0, 0, r))
    # metatron ring: six circles on hexagon vertices
    hexa = [(R * 0.62 * np.cos(a), R * 0.62 * np.sin(a))
            for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
    for hx, hy in hexa:
        segs.append(_circle(hx, hy, R * 0.30, n=48))
    # hexagram (two triangles) + hexagon
    tri1 = [(R * np.cos(a), R * np.sin(a))
            for a in np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 3, endpoint=False)]
    tri2 = [(R * np.cos(a), R * np.sin(a))
            for a in np.linspace(-np.pi / 2, -np.pi / 2 + 2 * np.pi, 3, endpoint=False)]
    segs.append(_polygon(tri1))
    segs.append(_polygon(tri2))
    segs.append(_polygon(hexa))
    # radial spokes
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        segs.append(np.array([[0.3 * np.cos(a), 0.3 * np.sin(a), 0],
                              [1.12 * R * np.cos(a), 1.12 * R * np.sin(a), 0]]))
    # scattered detail circles
    for _ in range(14):
        a = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.8, 1.15) * R
        segs.append(_circle(r * np.cos(a), r * np.sin(a),
                            rng.uniform(0.05, 0.16), n=24))
    return np.vstack(segs).astype(np.float32)


class BlueprintMode(BaseMode):
    name = "Blueprint Constellation"
    camera_distance = 8.5
    camera_elevation = 55.0    # look down onto the diagram like the reference

    def build(self):
        rng = np.random.default_rng(11)
        self.rng = rng
        n = int(self.profile.particle_count * 1.6)
        self.n = n

        # scaffold
        self.scaffold_segs = _build_scaffold(rng)
        self.scaffold = scene.visuals.Line(
            pos=self.scaffold_segs, connect="segments",
            color=(1.0, 1.0, 1.0, 0.20), width=1, parent=self.view.scene)
        additive(self.scaffold)
        self.scaffold_tr = MatrixTransform()
        self.scaffold.transform = self.scaffold_tr

        # particles
        self.pos = self._spawn(n)
        self.vel = self._initial_vel(self.pos)
        self.sparkle = rng.uniform(0, 2 * np.pi, n)
        # power-law brightness hierarchy: most particles dim dust, a few stars
        self.lum = (rng.uniform(0.08, 1.0, n) ** 3).astype(np.float32)

        self.markers = scene.visuals.Markers(parent=self.view.scene,
                                             antialias=1)
        additive(self.markers)
        self.streaks = scene.visuals.Line(
            pos=np.zeros((2 * n, 3), np.float32), connect="segments",
            width=1, parent=self.view.scene)
        additive(self.streaks)

        # geodesic wireframe sphere in the vortex center
        sv, sf = icosphere(2)
        self._sv, self._sf = sv.astype(np.float32), sf
        self.sphere_tr = MatrixTransform()
        self.sphere_body = scene.visuals.Mesh(
            vertices=(sv * 0.97).astype(np.float32), faces=sf,
            color=(0.05, 0.055, 0.07, 1.0), parent=self.view.scene)
        self.sphere_body.set_gl_state(depth_test=True,
                                      polygon_offset=(1.0, 1.0),
                                      polygon_offset_fill=True)
        self.sphere_body.transform = self.sphere_tr
        self.sphere_wire = scene.visuals.Mesh(
            vertices=sv.astype(np.float32), faces=sf, mode="lines",
            color=(1.0, 1.0, 1.0, 0.35), parent=self.view.scene)
        self.sphere_wire.set_gl_state(depth_test=True, blend=True,
                                      blend_func=("src_alpha", "one"))
        self.sphere_wire.transform = self.sphere_tr

        self.annotations = AnnotationLayer(self.view.scene, n=10, seed=5)
        self._anno_t = 0.0

        self.rot = VelocityValue(0.05, accel=2.0, damping=self.settings.damping)
        self.pulse = VelocityValue(1.0, accel=18.0, damping=self.settings.damping)
        self.flow = VelocityValue(0.4, accel=5.0, damping=self.settings.damping)
        self.sphere_spin = VelocityValue(0.3, accel=4.0,
                                         damping=self.settings.damping)
        self.sphere_breath = VelocityValue(1.0, accel=18.0,
                                           damping=self.settings.damping)
        self.flash = VelocityValue(0.0, accel=2.0, damping=0.78)
        self.morph = VelocityValue(0.0, accel=12.0, damping=self.settings.damping)
        self.twist = VelocityValue(0.0, accel=8.0, damping=self.settings.damping)
        self.tilt = VelocityValue(0.0, accel=5.0, damping=0.88)
        self.rot_angle = 0.0
        self.sphere_angle = 0.0
        self._mesh_tick = 0
        self._t = 0.0
        self._last_ft = None

        self.visuals = [self.scaffold, self.sphere_body, self.sphere_wire,
                        self.markers, self.streaks, *self.annotations.visuals]

    def _spawn(self, k):
        r = np.abs(self.rng.normal(0.15, 0.35, k)) + 0.1
        theta = self.rng.uniform(0, 2 * np.pi, k)
        z = self.rng.normal(0, 0.18, k)     # flattened: same plane as diagram
        return np.stack([r * np.cos(theta), r * np.sin(theta), z],
                        axis=1).astype(np.float64)

    def _initial_vel(self, pos):
        r = np.linalg.norm(pos, axis=1, keepdims=True) + 1e-6
        outward = pos / r
        return outward * self.rng.uniform(0.2, 1.2, (len(pos), 1))

    def update(self, frame, dt):
        dt = min(dt, 0.05)
        self._t += dt
        for v in (self.rot, self.pulse, self.flow, self.sphere_spin,
                  self.sphere_breath, self.morph, self.twist):
            v.damping = self.settings.damping

        # impulses fire once per analysis frame (render runs at 2x)
        new_frame = frame.time != self._last_ft
        self._last_ft = frame.time
        punch = frame.punch if new_frame else 0.0
        beat_now = frame.beat and new_frame

        bass = float(frame.bands[:2].mean())
        mids = float(frame.bands[2:5].mean())
        treble = float(frame.bands[5:].mean())

        self.rot.set_target(0.05 + bass * 1.0 + frame.rms * 0.4)
        self.pulse.set_target(1.0 + bass * 0.18)
        self.flow.set_target(0.35 + frame.rms * 3.2)
        self.sphere_spin.set_target(0.3 + mids * 3.0)
        self.sphere_breath.set_target(0.40 + bass * 0.28)
        self.morph.set_target(bass * 0.24)
        self.twist.set_target((mids - 0.25) * 1.8)
        self.tilt.set_target(np.sin(self._t * 0.35) * (4.0 + bass * 10.0))
        self.flash.set_target(0.0)
        if punch > 0.05:
            kb = punch * self.settings.beat_impulse
            self.flow.impulse(kb * 6.0)
            self.flash.impulse(kb * 4.5)
            self.sphere_breath.impulse(kb * 0.6)
            self.morph.impulse(kb * 1.8)
            self.tilt.impulse(kb * 25.0)
        if beat_now:
            kick = self.beat_kick(frame, 1.0) * self.settings.beat_impulse
            self.rot.impulse(kick * 0.9)
            self.pulse.impulse(kick * 0.35)
            self.flow.impulse(kick * 10.0)
            self.sphere_spin.impulse(kick * 4.0)
            self.sphere_breath.impulse(kick * 0.7)
            self.flash.impulse(kick * 7.0)

        self.rot_angle += self.rot.update(dt) * dt
        s = max(0.2, self.pulse.update(dt))
        flow = max(0.05, self.flow.update(dt))
        flash = float(np.clip(self.flash.update(dt), 0.0, 1.2))
        morph_amt = float(np.clip(self.morph.update(dt), 0.0, 0.32))
        twist_amt = float(np.clip(self.twist.update(dt), -1.0, 1.0))
        tilt_deg = float(np.clip(self.tilt.update(dt), -30.0, 30.0))

        self.scaffold_tr.reset()
        self.scaffold_tr.scale((s, s, 1.0))
        self.scaffold_tr.rotate(np.degrees(self.rot_angle), (0, 0, 1))
        self.scaffold_tr.rotate(tilt_deg, (1, 0, 0))
        self.scaffold.set_data(color=(1.0, 1.0, 1.0,
                                      np.clip(0.24 + 0.22 * bass
                                              + 0.25 * flash, 0, 0.8)))

        # central geodesic sphere: counter-rotates, breathes on bass
        self.sphere_angle += self.sphere_spin.update(dt) * dt
        sb = max(0.15, self.sphere_breath.update(dt))
        self.sphere_tr.reset()
        self.sphere_tr.scale((sb, sb, sb))
        self.sphere_tr.rotate(np.degrees(-self.sphere_angle), (0.3, 0.4, 1.0))
        # Mesh rebuilds cost ~3.5 ms; the morph is slow, so refresh the
        # geometry every 3rd frame and let the transform handle spin/scale.
        self._mesh_tick += 1
        if self._mesh_tick % 3 == 0:
            mv = morph_sphere(self._sv, self._t, morph_amt,
                              twist_amt).astype(np.float32)
            self.sphere_body.set_data(vertices=mv * 0.97, faces=self._sf,
                                      color=(0.05, 0.055, 0.07, 1.0))
            self.sphere_wire.set_data(vertices=mv, faces=self._sf,
                                      color=(1.0, 1.0, 1.0,
                                             np.clip(0.3 + 0.4 * mids
                                                     + 0.4 * flash, 0, 0.95)))

        # particle physics: outward stream + swirl, momentum + damping
        r = np.linalg.norm(self.pos, axis=1, keepdims=True) + 1e-6
        outward = self.pos / r
        swirl = np.cross(np.array([0.0, 0.0, 1.0]), outward)
        acc = outward * flow * 2.2 + swirl * (0.4 + bass * 1.6)
        acc += self.rng.normal(0, 0.35 + treble * 1.2, self.pos.shape)
        self.vel += acc * dt
        self.vel *= self.settings.damping ** (dt * 60.0)
        self.pos += self.vel * dt

        # respawn escaped particles near the center
        far = (r[:, 0] > 5.0)
        if far.any():
            k = int(far.sum())
            self.pos[far] = self._spawn(k)
            self.vel[far] = self._initial_vel(self.pos[far])

        # brightness: dark vortex center, power-law hierarchy, sparkle flicker
        self.sparkle += dt * (3.0 + treble * 25.0)
        rr = r[:, 0]
        center_fade = np.clip((rr - 0.85) / 1.1, 0.0, 1.0)
        edge_fade = np.clip((5.0 - rr) / 1.2, 0.0, 1.0)
        flicker = 0.75 + 0.25 * np.sin(self.sparkle)
        alpha = (0.15 + 0.9 * frame.rms + 0.4 * treble) * self.lum * \
            center_fade * edge_fade * flicker * (1.0 + flash)
        alpha = np.clip(alpha, 0.0, 0.9).astype(np.float32)

        p, c, sz = glow_marker_data(self.pos.astype(np.float32), alpha,
                                    base_size=2.2 + 1.5 * frame.rms,
                                    halo_scale=3.0, halo_alpha=0.05)
        self.markers.set_data(p, face_color=c, size=sz, edge_width=0)

        # long ray streaks along velocity (the reference's radial spray)
        streak = self.pos - self.vel * (0.22 + 0.30 * frame.rms)
        segs = np.empty((2 * self.n, 3), np.float32)
        segs[0::2] = self.pos
        segs[1::2] = streak
        scol = np.zeros((2 * self.n, 4), np.float32)
        scol[:, :3] = 1.0
        scol[0::2, 3] = alpha * 0.45
        scol[1::2, 3] = 0.0
        self.streaks.set_data(pos=segs, color=scol, connect="segments")

        # re-pin annotations occasionally / on strong beats
        self._anno_t += dt
        if self._anno_t > 2.5 or (frame.beat and frame.beat_strength > 0.7
                                  and self._anno_t > 0.8):
            self._anno_t = 0.0
            bright = self.pos[alpha > np.percentile(alpha, 80)]
            self.annotations.retarget(bright)

    def set_visible(self, visible):
        super().set_visible(visible)
        if self.built:
            self.annotations.set_visible(visible)

    def velocity_magnitude(self):
        return self.rot.speed + self.flow.speed + self.flash.speed + \
            float(np.mean(np.linalg.norm(self.vel, axis=1)))
