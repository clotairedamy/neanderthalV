"""Mode 8 — Fiber Nebula (monochrome smoke-and-sphere composition).

A wireframe geodesic sphere hangs above a spinning spiral vortex disc while
plumes of fine particle smoke churn and wrap around it — grayscale, additive,
like ink in water lit by an x-ray.

Every element is velocity-driven and strongly audio-coupled:
- bass      -> sphere breathing + vortex radius pulse
- mids      -> plume turbulence (flow-field churn)
- treble    -> particle sparkle / vortex shimmer
- energy    -> plume flow speed, sphere & vortex spin
- beats     -> emission bursts, velocity kicks, white flash envelope
"""
from __future__ import annotations

import numpy as np
from vispy import scene
from vispy.visuals.transforms import MatrixTransform

from ..physics.velocity import VelocityValue
from .base import BaseMode
from .geometry import icosphere
from .mono import AnnotationLayer, additive, morph_sphere

N_VORTEX = 1600
VORTEX_Z = -2.1
N_EMITTERS = 4


class FiberMode(BaseMode):
    name = "Fiber Nebula"
    camera_distance = 9.0
    camera_elevation = 16.0

    # ------------------------------------------------------------------ build

    def build(self):
        rng = np.random.default_rng(23)
        self.rng = rng

        # --- geodesic sphere: dark occluding body + hairline wireframe
        v, f = icosphere(3)
        self._sv, self._sf = v.astype(np.float32), f
        patches = 0.05 + 0.06 * rng.random(len(v)) ** 2
        body_colors = np.ones((len(v), 4), np.float32)
        body_colors[:, 0] = body_colors[:, 1] = patches
        body_colors[:, 2] = patches * 1.15
        self._body_colors = body_colors
        self.sphere_tr = MatrixTransform()
        self.sphere_body = scene.visuals.Mesh(
            vertices=(v * 0.97).astype(np.float32), faces=f,
            vertex_colors=body_colors, parent=self.view.scene)
        self.sphere_body.set_gl_state(depth_test=True,
                                      polygon_offset=(1.0, 1.0),
                                      polygon_offset_fill=True)
        self.sphere_body.transform = self.sphere_tr
        self.sphere_wire = scene.visuals.Mesh(
            vertices=v.astype(np.float32), faces=f, mode="lines",
            color=(1.0, 1.0, 1.0, 0.3), parent=self.view.scene)
        self.sphere_wire.set_gl_state(depth_test=True, blend=True,
                                      blend_func=("src_alpha", "one"))
        self.sphere_wire.transform = self.sphere_tr

        # --- smoke plumes
        n = int(self.profile.particle_count * 1.2)
        self.n = n
        self.emitter_phase = rng.uniform(0, 2 * np.pi, N_EMITTERS)
        self.emitter_incl = rng.uniform(-0.7, 0.7, N_EMITTERS)
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))
        self.age = rng.uniform(0, 1, n)          # 0..1 of lifetime
        self.life = rng.uniform(1.5, 3.5, n)     # short lives keep ribbons tight
        self.shell_r = rng.uniform(1.25, 2.3, n)  # preferred orbit radius
        self.p_phase = rng.uniform(0, 2 * np.pi, n)
        self._respawn(np.ones(n, bool), 0.0)
        self.prev = self.pos.copy()

        self.markers = scene.visuals.Markers(parent=self.view.scene, antialias=1)
        additive(self.markers)
        self.trails = scene.visuals.Line(pos=np.zeros((2 * n, 3), np.float32),
                                         connect="segments", width=1,
                                         parent=self.view.scene)
        additive(self.trails)

        # --- spiral vortex disc
        self.v_arm = rng.integers(0, 4, N_VORTEX)
        self.v_s = rng.uniform(0, 1, N_VORTEX) ** 0.8
        self.v_jit = rng.normal(0, 0.06, (N_VORTEX, 2))
        self.v_zjit = rng.normal(0, 0.05, N_VORTEX) * (1 + self.v_s)
        self.v_phase = rng.uniform(0, 2 * np.pi, N_VORTEX)
        self.v_angle = 0.0
        self.vortex = scene.visuals.Markers(parent=self.view.scene, antialias=1)
        additive(self.vortex)

        self.annotations = AnnotationLayer(self.view.scene, n=6,
                                           horizontal=True, seed=9)
        self._anno_t = 0.0

        # --- velocity state
        d = self.settings.damping
        self.spin = VelocityValue(0.15, accel=3.0, damping=d)
        self.breath = VelocityValue(1.0, accel=16.0, damping=d)
        self.flow_speed = VelocityValue(0.8, accel=5.0, damping=d)
        self.turb = VelocityValue(0.5, accel=6.0, damping=d)
        self.vortex_w = VelocityValue(0.5, accel=4.0, damping=d)
        self.vortex_pulse = VelocityValue(1.0, accel=18.0, damping=d)
        self.flash = VelocityValue(0.0, accel=2.0, damping=0.78)
        self.morph = VelocityValue(0.0, accel=12.0, damping=d)
        self.twist = VelocityValue(0.0, accel=8.0, damping=d)
        self.spin_angle = 0.0
        self._t = 0.0
        self._last_ft = None

        self.visuals = [self.sphere_body, self.sphere_wire, self.markers,
                        self.trails, self.vortex, *self.annotations.visuals]

    # ------------------------------------------------------------- particles

    def _emitters(self, t: float) -> np.ndarray:
        a = self.emitter_phase + t * 0.35
        r = 1.7
        return np.stack([r * np.cos(a),
                         r * np.sin(a) * np.cos(self.emitter_incl),
                         r * np.sin(a) * np.sin(self.emitter_incl) - 0.4],
                        axis=1)

    def _respawn(self, mask: np.ndarray, t: float) -> None:
        k = int(mask.sum())
        if k == 0:
            return
        em = self._emitters(t)
        which = self.rng.integers(0, N_EMITTERS, k)
        base = em[which] + self.rng.normal(0, 0.22, (k, 3))
        self.pos[mask] = base
        # initial velocity: tangential around the sphere
        r = np.linalg.norm(base, axis=1, keepdims=True) + 1e-9
        tang = np.cross(base / r, np.array([0.15, 0.2, 1.0]))
        self.vel[mask] = tang * self.rng.uniform(0.5, 1.4, (k, 1))
        self.age[mask] = 0.0
        self.life[mask] = self.rng.uniform(1.5, 3.5, k)

    def _flow(self, p: np.ndarray, t: float, turb: float) -> np.ndarray:
        """Circulating field: swirl round a precessing axis + arcs + churn."""
        r = np.linalg.norm(p, axis=1, keepdims=True) + 1e-9
        rhat = p / r
        axis = np.array([np.sin(t * 0.21) * 0.5, np.cos(t * 0.17) * 0.4, 1.0])
        axis /= np.linalg.norm(axis)
        swirl = np.cross(np.broadcast_to(axis, p.shape), rhat)
        arc = np.cross(swirl, rhat)          # poloidal: wraps over the poles
        arc_amt = np.sin(p[:, 2:3] * 1.3 + t * 0.6)
        churn = np.stack([
            np.sin(p[:, 1] * 2.1 + t * 1.7) * np.cos(p[:, 2] * 1.7 - t),
            np.sin(p[:, 2] * 2.4 - t * 1.3) * np.cos(p[:, 0] * 1.9 + t),
            np.sin(p[:, 0] * 2.2 + t * 1.1) * np.cos(p[:, 1] * 2.3 + t * 0.7),
        ], axis=1)
        spring = (self.shell_r[:, None] - r) * rhat * 2.5
        return swirl * 2.0 + arc * arc_amt * 0.9 + churn * turb * 1.1 + spring

    # ------------------------------------------------------------- update

    def update(self, frame, dt):
        dt = min(dt, 0.05)
        self._t += dt
        t = self._t
        d = self.settings.damping
        for vv in (self.spin, self.breath, self.flow_speed, self.turb,
                   self.vortex_w, self.vortex_pulse):
            vv.damping = d

        bass = float(frame.bands[:2].mean())
        mids = float(frame.bands[2:5].mean())
        treble = float(frame.bands[5:].mean())

        # impulses must fire once per analysis frame (render runs at 2x)
        new_frame = frame.time != self._last_ft
        self._last_ft = frame.time
        punch = frame.punch if new_frame else 0.0
        beat_now = frame.beat and new_frame

        self.spin.set_target(0.15 + frame.rms * 1.8)
        self.breath.set_target(0.78 + bass * 0.28)
        self.morph.set_target(bass * 0.22 + frame.rms * 0.06)
        self.twist.set_target((mids - 0.25) * 1.6)
        self.flow_speed.set_target(0.5 + frame.rms * 3.2)
        self.turb.set_target(0.25 + mids * 2.2)
        self.vortex_w.set_target(0.4 + frame.rms * 3.5)
        self.vortex_pulse.set_target(1.0 + bass * 0.15)
        self.flash.set_target(0.0)
        if punch > 0.05:
            kb = punch * self.settings.beat_impulse
            self.breath.impulse(kb * 0.8)
            self.flash.impulse(kb * 4.5)
            self.flow_speed.impulse(kb * 5.0)
            self.morph.impulse(kb * 1.8)
            self.twist.impulse(kb * 1.5)
        if beat_now:
            kick = self.beat_kick(frame, 1.0) * self.settings.beat_impulse
            self.spin.impulse(kick * 2.5)
            self.breath.impulse(kick * 1.6)
            self.flow_speed.impulse(kick * 8.0)
            self.vortex_w.impulse(kick * 5.0)
            self.vortex_pulse.impulse(kick * 1.2)
            self.flash.impulse(kick * 7.0)
            # burst of fresh smoke on the hit
            burst = self.rng.random(self.n) < 0.10 * kick
            self._respawn(burst, t)

        self.spin_angle += self.spin.update(dt) * dt
        breath = max(0.3, self.breath.update(dt))
        morph_amt = float(np.clip(self.morph.update(dt), 0.0, 0.32))
        twist_amt = float(np.clip(self.twist.update(dt), -1.0, 1.0))
        flow_speed = max(0.05, self.flow_speed.update(dt))
        turb = max(0.0, self.turb.update(dt))
        flash = float(np.clip(self.flash.update(dt), 0.0, 1.2))

        # --- sphere: audio-morphing blob with height twist
        self.sphere_tr.reset()
        self.sphere_tr.scale((breath, breath, breath))
        self.sphere_tr.rotate(np.degrees(self.spin_angle), (0.25, 0.9, 0.35))
        mv = morph_sphere(self._sv, t, morph_amt, twist_amt).astype(np.float32)
        wire_a = np.clip(0.28 + 0.4 * bass + 0.35 * flash, 0, 0.9)
        self.sphere_body.set_data(vertices=mv * 0.97, faces=self._sf,
                                  vertex_colors=self._body_colors)
        self.sphere_wire.set_data(vertices=mv, faces=self._sf,
                                  color=(1.0, 1.0, 1.0, wire_a))

        # --- plumes
        self.prev[:] = self.pos
        acc = self._flow(self.pos, t, turb)
        self.vel += acc * dt * flow_speed
        self.vel *= d ** (dt * 60.0)
        self.pos += self.vel * dt * (0.6 + 0.6 * flow_speed)

        self.age += dt / self.life
        dead = (self.age >= 1.0) | (np.linalg.norm(self.pos, axis=1) > 4.5)
        self._respawn(dead, t)
        self.prev[dead] = self.pos[dead]

        # smoke alpha: fade in/out over life, sparkle with treble, flash
        fade = np.clip(self.age * 6, 0, 1) * np.clip((1 - self.age) * 2.5, 0, 1)
        sparkle = 0.7 + 0.3 * np.sin(self.age * 40 + self.p_phase +
                                     t * (3 + treble * 20))
        a = (0.06 + 0.24 * frame.rms) * fade * sparkle * (1.0 + flash)
        a = np.clip(a, 0, 0.7).astype(np.float32)

        cols = np.ones((self.n, 4), np.float32)
        cols[:, :3] = 0.88
        cols[:, 3] = a
        self.markers.set_data(self.pos.astype(np.float32), face_color=cols,
                              size=1.8 + 2.0 * frame.rms + treble * 1.5,
                              edge_width=0)

        segs = np.empty((2 * self.n, 3), np.float32)
        segs[0::2] = self.pos
        segs[1::2] = self.prev - self.vel * dt * 3.0   # motion streaks
        tcol = np.zeros((2 * self.n, 4), np.float32)
        tcol[:, :3] = 0.85
        tcol[0::2, 3] = a * 0.5
        tcol[1::2, 3] = 0.0
        self.trails.set_data(pos=segs, color=tcol, connect="segments")

        # --- vortex disc
        self.v_angle += self.vortex_w.update(dt) * dt
        pulse = max(0.4, self.vortex_pulse.update(dt))
        ang = (self.v_arm * (np.pi / 2) + self.v_s * 6.5 * np.pi
               + self.v_angle * (1.6 - self.v_s))    # inner spins faster
        rad = (0.25 + 2.9 * self.v_s) * pulse
        vx = rad * np.cos(ang) + self.v_jit[:, 0]
        vy = rad * np.sin(ang) + self.v_jit[:, 1]
        vz = VORTEX_Z + self.v_zjit + 0.15 * np.sin(ang * 2 + t)
        vp = np.stack([vx, vy, vz], axis=1).astype(np.float32)
        shimmer = 0.65 + 0.35 * np.sin(self.v_phase + t * (2 + treble * 25))
        va = ((0.09 + 0.40 * frame.rms) * (0.35 + 0.65 * self.v_s)
              * shimmer * (1.0 + 0.8 * flash))
        vcols = np.ones((N_VORTEX, 4), np.float32)
        vcols[:, :3] = 0.8
        vcols[:, 3] = np.clip(va, 0, 0.7)
        self.vortex.set_data(vp, face_color=vcols,
                             size=1.6 + 1.6 * self.v_s + treble * 1.5,
                             edge_width=0)

        # --- annotations
        self._anno_t += dt
        if self._anno_t > 3.0:
            self._anno_t = 0.0
            bright = self.pos[a > np.percentile(a, 85)]
            self.annotations.retarget(bright)

    def set_visible(self, visible):
        super().set_visible(visible)
        if self.built:
            self.annotations.set_visible(visible)

    def velocity_magnitude(self):
        return (self.spin.speed + self.flow_speed.speed + self.morph.speed +
                self.vortex_w.speed + self.flash.speed)
