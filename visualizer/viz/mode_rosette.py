"""Mode 11 -- Harmonic Rosette.

Three string-art rosettes drawn from assets/Lows.svg, mids.svg and higs.svg,
one bound to each third of the spectrum. The files are one shape family that
differs only in inner radius, so instead of cross-fading between three
pictures each ring *morphs* continuously along that family: energy pulls a
ring's inner radius in and its chords swing open across the disc, silence
lets it collapse back to a tight band hugging the rim.

The rings nest, counter-rotate at band-driven rates and sit on separate
depth planes, so the chords cross into a moire that never repeats. Each
chord's outer endpoint keeps the angle it has in the drawing, which is used
to bind it to a spectrum bin -- the rosette doubles as a circular analyzer,
its rim rippling in the shape of the current spectrum.

Audio mapping: lows/mids/highs -> one ring each (morph, spin, brightness);
kick transients bloom the bass ring open; beats kick the spin; a build-up
pulls all three into superposition, and the drop blows them apart.
"""
from __future__ import annotations

import numpy as np
from vispy import scene

from ..physics.velocity import VelocityValue
from .base import BaseMode
from .svg_rosette import load_family, morph, outer_angles

# per ring: which spectrum third drives it, palette index, nesting scale,
# base spin rate (rad/s, alternating sign for the moire) and base morph
RINGS = (
    {"band": slice(0, 2), "color": 0, "scale": 1.00, "spin": 0.17, "base": 0.0},
    {"band": slice(2, 5), "color": 3, "scale": 0.86, "spin": -0.26, "base": 1.0},
    {"band": slice(5, 7), "color": 6, "scale": 0.72, "spin": 0.38, "base": 2.0},
)


class RosetteMode(BaseMode):
    name = "Harmonic Rosette"
    # The rings are unit-radius, so the camera sits close. The margin is not
    # spare room: kick punches and the build-up dolly the camera in by up to
    # ~27%, and at 3.3 that clips the rim.
    camera_distance = 3.7
    camera_elevation = 66.0
    # the moire between the rings is the whole point; long trails smear it
    trail_scale = 0.30

    def build(self):
        self.family = load_family()
        self.n_seg = self.family.shape[1]

        # bind every chord to a spectrum bin by the angle it already has in
        # the drawing, so the rim ripples in the shape of the spectrum
        turns = outer_angles(self.family[0])
        self.bin_idx = np.minimum((turns * 64).astype(np.int32), 63)

        d = self.settings.damping
        self.rings = []
        for spec in RINGS:
            line = scene.visuals.Line(
                pos=np.zeros((self.n_seg * 2, 3), np.float32),
                color=np.zeros((self.n_seg * 2, 4), np.float32),
                connect="segments", width=1, parent=self.view.scene)
            line.set_gl_state("additive", depth_test=False, cull_face=False)
            self.rings.append({
                **spec,
                "line": line,
                "morph": VelocityValue(spec["base"], accel=9.0, damping=d),
                "rate": VelocityValue(spec["spin"], accel=4.0, damping=d),
                "angle": 0.0,
            })

        # the shared outer rim: one marker per chord, a plain circular readout
        # of the spectrum that stays put while the rings turn under it
        self.rim = scene.visuals.Markers(parent=self.view.scene, antialias=1)
        self.rim.set_gl_state("additive", depth_test=False)
        rim_a = turns * 2 * np.pi
        self.rim_xy = np.stack([np.cos(rim_a), np.sin(rim_a)], axis=1).astype(np.float32)

        # depth separation of the three planes: collapses through a build-up,
        # detonates on the drop
        self.spread = VelocityValue(0.55, accel=6.0, damping=d)
        self.visuals = [r["line"] for r in self.rings] + [self.rim]
        self._last_ft = -1.0

    # ------------------------------------------------------------------ tick

    def update(self, frame, dt):
        s = self.settings
        bands = frame.bands
        spec = frame.spectrum
        cell = np.clip(spec[self.bin_idx], 0.0, 1.5)      # (n_seg,) per chord

        ant = float(getattr(frame, "anticipation", 0.0)) if s.anticipation else 0.0
        # analysis runs at 30 Hz under a render loop that may run at 120, so
        # transients are gated on the analysis timestamp or they fire 4x
        fresh = frame.time != self._last_ft
        self._last_ft = frame.time

        # a build-up drags the rings into one plane and one shape; the drop
        # throws them back apart
        self.spread.set_target(0.55 * (1.0 - 0.9 * ant))
        if fresh and getattr(frame, "section", False):
            self.spread.impulse(3.0)
        sep = float(np.clip(self.spread.update(dt), -0.4, 2.2))

        for k, r in enumerate(self.rings):
            e = float(np.clip(bands[r["band"]].mean() * s.sensitivity, 0.0, 1.5))

            # energy pulls the inner radius in, swinging the chords open
            target = r["base"] - s.rose_morph * e
            self.morph_to_common(r, target, ant)
            if fresh and k == 0:
                r["morph"].impulse(-frame.punch * s.rose_morph * 1.4)
            r["morph"].update(dt)
            m = float(np.clip(r["morph"].value, -1.0, 2.6))
            r["morph"].value = m

            r["rate"].set_target(r["spin"] * s.rose_spin * (0.5 + 1.8 * e))
            if fresh and frame.beat:
                r["rate"].impulse(np.sign(r["spin"]) * frame.beat_strength
                                  * s.beat_impulse * 0.9)
            r["angle"] += r["rate"].update(dt) * dt

            shape = morph(self.family, m) * r["scale"]
            ca, sa = np.cos(r["angle"]), np.sin(r["angle"])
            rot = np.array([[ca, -sa], [sa, ca]], np.float32)
            xy = shape.reshape(-1, 2) @ rot.T                # (n_seg*2, 2)

            pos = np.empty((self.n_seg * 2, 3), np.float32)
            pos[:, :2] = xy
            base_z = (k - 1) * sep
            pos[0::2, 2] = base_z + s.rose_relief * cell     # outer end lifts
            pos[1::2, 2] = base_z

            col = np.empty((self.n_seg * 2, 4), np.float32)
            tint = np.clip(self.palette.band_color(r["color"]), 0, 1)
            col[:, :3] = tint
            # bright at the rim where the spectrum reads, fading into the core
            col[0::2, 3] = np.clip((0.04 + 0.30 * cell) * (0.35 + 0.7 * e), 0, 1)
            col[1::2, 3] = col[0::2, 3] * 0.30
            r["line"].set_data(pos=pos, color=col)

        # rim readout, on the outermost ring's own circle
        rim = np.empty((self.n_seg, 3), np.float32)
        rim[:, :2] = self.rim_xy * RINGS[0]["scale"]
        rim[:, 2] = s.rose_relief * cell
        rim_c = np.ones((self.n_seg, 4), np.float32)
        rim_c[:, :3] = np.clip(self.palette.band_color(6) * 0.5 + 0.5, 0, 1)
        rim_c[:, 3] = np.clip(0.05 + 0.45 * cell, 0, 1)
        self.rim.set_data(rim, face_color=rim_c, edge_width=0,
                          size=float(np.clip(1.2 + 3.5 * frame.rms, 1.2, 5.0)))

    @staticmethod
    def morph_to_common(ring, target: float, ant: float) -> None:
        """Blend a ring's own target toward the shape the others are heading
        for, so a build-up ends with all three superimposed."""
        ring["morph"].set_target(target * (1.0 - ant) + 1.0 * ant)

    def velocity_magnitude(self):
        return sum(r["morph"].speed + r["rate"].speed for r in self.rings)
