"""Mode 4 — Reactive Fractal (Julia set), evaluated on the GPU.

The Julia parameter c orbits the Mandelbrot cardioid; its angle and radius
are velocity-integrated targets driven by bass and centroid, so parameter
changes glide instead of jumping. Colors cycle through the image palette.

The escape-time loop runs in a fragment shader: at 320x320x48 iterations it
cost ~9 ms/frame in numpy, which alone blew the frame budget.
"""
from __future__ import annotations

import numpy as np
from vispy import gloo, scene
from vispy.scene.visuals import create_visual_node
from vispy.visuals import Visual

from ..physics.velocity import VelocityValue
from .base import BaseMode

VERT = """
attribute vec2 a_pos;
varying vec2 v_z;
uniform float u_extent;
void main() {
    v_z = a_pos * u_extent;
    gl_Position = $transform(vec4(a_pos, 0.0, 1.0));
}
"""

FRAG = """
uniform vec2 u_c;
uniform float u_shift;
uniform int u_iters;
uniform sampler2D u_lut;
varying vec2 v_z;

void main() {
    vec2 z = v_z;
    float n = 0.0;
    for (int i = 0; i < 256; i++) {
        if (i >= u_iters) { break; }
        // z = z^2 + c
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + u_c;
        float m = dot(z, z);
        if (m > 4.0) {
            // smooth (fractional) escape count, avoids visible banding
            n = float(i) + 1.0 - log2(max(log(sqrt(m)), 1e-9));
            break;
        }
        n = float(i) + 1.0;
    }
    float t = clamp(n / float(u_iters), 0.0, 1.0);
    // most pixels escape in the first few iterations, so a linear ramp
    // puts nearly everything at the dark end; pull the low range up
    float tc = pow(t, 0.32);
    vec3 col = texture2D(u_lut, vec2(fract(tc * 2.0 + u_shift), 0.5)).rgb;
    gl_FragColor = vec4(col * (0.30 + 0.85 * tc), 1.0);
}
"""


class JuliaVisual(Visual):
    """Full-quad Julia set; the escape loop lives in the fragment shader."""

    def __init__(self, extent: float = 1.6, iters: int = 96):
        Visual.__init__(self, vcode=VERT, fcode=FRAG)
        quad = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], np.float32)
        self.shared_program["a_pos"] = gloo.VertexBuffer(quad)
        self.shared_program["u_extent"] = float(extent)
        self.shared_program["u_c"] = (0.0, 0.0)
        self.shared_program["u_shift"] = 0.0
        self.shared_program["u_iters"] = int(iters)
        self._lut = gloo.Texture2D(np.zeros((1, 256, 3), np.float32),
                                   interpolation="linear",
                                   wrapping="repeat")
        self.shared_program["u_lut"] = self._lut
        self._draw_mode = "triangle_strip"
        self.set_gl_state(depth_test=False, blend=False, cull_face=False)

    def set_lut(self, lut: np.ndarray) -> None:
        self._lut.set_data(np.ascontiguousarray(
            lut.reshape(1, -1, 3).astype(np.float32)))

    def set_uniform(self, name, value):
        self.shared_program[name] = value

    def _prepare_transforms(self, view):
        view.view_program.vert["transform"] = view.get_transform()

    def _prepare_draw(self, view):
        return True

    def _compute_bounds(self, axis, view):
        return (-1.0, 1.0)


Julia = create_visual_node(JuliaVisual)


class FractalMode(BaseMode):
    name = "Reactive Fractal"
    camera = "panzoom"

    def build(self):
        self.julia = Julia(extent=1.6,
                           iters=96 if self.profile.name == "macos" else 48,
                           parent=self.view.scene)
        # the quad is in [-1,1]; scale it to fill the 2D view
        from vispy.visuals.transforms import STTransform
        r = self.profile.fractal_resolution
        self.julia.transform = STTransform(scale=(r / 2 + 20, r / 2 + 20))
        self.visuals = [self.julia]

        self.angle = VelocityValue(0.0, accel=2.0, damping=self.settings.damping)
        self.radius = VelocityValue(0.75, accel=6.0, damping=self.settings.damping)
        self.color_shift = VelocityValue(0.0, accel=3.0, damping=self.settings.damping)
        self._angle_pos = 0.0
        self._last_ft = None

    def update(self, frame, dt):
        for v in (self.angle, self.radius, self.color_shift):
            v.damping = self.settings.damping

        new_frame = frame.time != self._last_ft
        self._last_ft = frame.time
        punch = frame.punch if new_frame else 0.0

        bass = float(frame.bands[:2].mean())
        self.angle.set_target(0.15 + bass * 1.6)
        self.radius.set_target(0.62 + frame.centroid * 0.28)
        self.color_shift.set_target(frame.rms * 4.0)
        if punch > 0.05:
            kick = punch * self.settings.beat_impulse
            self.angle.impulse(kick * 2.5)
            self.color_shift.impulse(kick * 6.0)

        self._angle_pos += self.angle.update(dt) * dt
        r = float(np.clip(self.radius.update(dt), 0.3, 0.95))
        shift = self.color_shift.update(dt)

        self.julia.set_uniform("u_c", (r * np.cos(self._angle_pos),
                                       r * np.sin(self._angle_pos)))
        self.julia.set_uniform("u_shift", float(shift * 0.16 % 1.0))
        self.julia.set_lut(self.palette.lut(256))

    def velocity_magnitude(self):
        return self.angle.speed + self.radius.speed + self.color_shift.speed
