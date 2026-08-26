"""Mode 9 — Depth Point Cloud (Kinect-style volumetric video).

Based on the three.js `webgl_video_kinect` example: every pixel of a video
becomes a 3D point, un-projected by its depth value in a vertex shader using
the Kinect's field-of-view constants. Faithful to the original (a grayscale
depth video renders as the familiar ghostly figure), then extended:

- **Any source**: a depth video, an ordinary video (luminance as relief),
  the live webcam, a still photo, or — with nothing loaded — a scrolling
  spectrogram terrain built from the music itself.
- **RGBD layouts**: side-by-side / top-bottom videos use one half as color
  and the other as true depth.
- **Audio-reactive depth**: bass exaggerates depth contrast; each kick
  spawns a ripple that travels outward through the surface.
- **Disintegration**: loud passages scatter points along their view rays, so
  the figure explodes into dust and reassembles as the energy drops.
- **Palette tinting**: video luminance is remapped onto the image palette.

All of it runs through the same velocity integrators as the other modes, so
nothing snaps — it accelerates, overshoots and glides back.
"""
from __future__ import annotations

import os

import numpy as np
from vispy import gloo, scene
from vispy.scene.visuals import create_visual_node
from vispy.visuals import Visual

from ..physics.velocity import VelocityValue
from .base import BaseMode

# Kinect field-of-view constants, straight from the three.js example.
VERT = """
uniform sampler2D u_tex;
uniform vec2 u_res;
uniform vec2 u_uv_scale;
uniform vec2 u_uv_color;
uniform vec2 u_uv_depth;
uniform float u_near;
uniform float u_far;
uniform float u_zoffset;
uniform float u_point;
uniform float u_depth_scale;
uniform float u_scatter;
uniform float u_sparkle;
uniform float u_cutoff;
uniform float u_flip;
uniform float u_persp;
uniform float u_aspect;
uniform float u_alpha;
uniform float u_tint_mix;
uniform vec3 u_tint;
uniform vec4 u_wave0;
uniform vec4 u_wave1;
uniform vec4 u_wave2;
// frequency-band colouring: lows / mids / highs painted across an axis
uniform float u_band_mode;   // 0 off, 1 vertical, 2 depth, 3 horizontal, 4 radial
uniform float u_band_push;   // how far each band's energy displaces its zone
uniform vec3 u_col_low;
uniform vec3 u_col_mid;
uniform vec3 u_col_high;
uniform float u_e_low;
uniform float u_e_mid;
uniform float u_e_high;

attribute vec2 a_pix;

varying vec4 v_color;
varying float v_drop;

const float XtoZ = 1.11146;
const float YtoZ = 0.83359;

float hash12(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 band_color(float f) {
    if (f < 0.5) { return mix(u_col_low, u_col_mid, smoothstep(0.0, 0.5, f)); }
    return mix(u_col_mid, u_col_high, smoothstep(0.5, 1.0, f));
}

float band_energy(float f) {
    if (f < 0.5) { return mix(u_e_low, u_e_mid, smoothstep(0.0, 0.5, f)); }
    return mix(u_e_mid, u_e_high, smoothstep(0.5, 1.0, f));
}

float wave_at(vec4 w, vec2 uv) {
    if (w.w <= 0.0) { return 0.0; }
    float d = distance(uv, w.xy);
    float x = (d - w.z) / 0.05;
    return w.w * exp(-x * x);
}

void main() {
    vec2 uv = (a_pix + 0.5) / u_res;
    vec3 rgb  = texture2DLod(u_tex, u_uv_color + uv * u_uv_scale, 0.0).rgb;
    vec3 drgb = texture2DLod(u_tex, u_uv_depth + uv * u_uv_scale, 0.0).rgb;

    float depth = (drgb.r + drgb.g + drgb.b) / 3.0;
    v_drop = (depth < u_cutoff) ? 1.0 : 0.0;

    float vy0 = mix(uv.y, 1.0 - uv.y, u_flip);
    // band coordinate: which frequency zone this point belongs to.
    // Taken from the raw depth so the axis=depth case isn't self-referential.
    float bf = 0.0;
    if (u_band_mode > 0.5) {
        if (u_band_mode < 1.5)      { bf = vy0; }
        else if (u_band_mode < 2.5) { bf = depth; }
        else if (u_band_mode < 3.5) { bf = uv.x; }
        else { bf = clamp(length(uv - vec2(0.5)) * 2.0, 0.0, 1.0); }
        depth += u_band_push * band_energy(bf);
    }

    // audio: exaggerate depth contrast, then add travelling ripples.
    // Stay strictly inside [0,1): z must never approach 0 or the
    // un-projection collapses every point onto the view axis.
    depth = 0.5 + (depth - 0.5) * u_depth_scale;
    depth += wave_at(u_wave0, uv) + wave_at(u_wave1, uv) + wave_at(u_wave2, uv);
    depth = max(depth, 0.0);
    // soft knee instead of a hard clamp: clipping the peaks flattens the
    // nearest surfaces into one plateau and they blow out as a solid slab
    if (depth > 0.75) {
        depth = 0.75 + 0.22 * (1.0 - exp(-(depth - 0.75) / 0.22));
    }

    // Kinect un-projection (three.js webgl_video_kinect). u_persp blends
    // between an orthographic relief (0) and the exact frustum (1): at full
    // perspective the reconstruction is a cone that only reads correctly
    // from its apex, which fights an orbiting camera.
    float z = (1.0 - depth) * (u_far - u_near) + u_near;
    float z_ref = (u_near + u_far) * 0.5;
    float zs = mix(z_ref, z, u_persp);
    float vy = vy0;
    // u_aspect corrects for footage that isn't the Kinect's native 4:3.
    // NOTE vispy's turntable camera is Z-up: the video's vertical axis must
    // become world Z, and depth becomes Y, or the subject lies on its back.
    vec3 pos = vec3((uv.x - 0.5) * zs * XtoZ * u_aspect,
                    -z + u_zoffset,
                    (vy - 0.5) * zs * YtoZ);

    float rnd = hash12(a_pix);
    if (u_scatter > 0.001) {
        vec3 dir = normalize(pos - vec3(0.0, u_zoffset - 1.0, 0.0));
        pos += dir * u_scatter * (0.35 + rnd);
    }

    gl_Position = $transform(vec4(pos, 1.0));
    gl_PointSize = u_point * (1.0 + u_sparkle * rnd) * clamp(3.0 / z, 0.4, 2.5);

    // Shade by reconstructed depth, not raw luminance: thousands of
    // overlapping near-white points otherwise stack into a solid blob.
    float lum = dot(rgb, vec3(0.299, 0.587, 0.114));
    float shade = clamp(depth, 0.0, 1.0);
    vec3 tinted = u_tint * (0.30 + 1.15 * lum);
    vec3 base = mix(rgb, tinted, u_tint_mix);
    float boost = 1.0;
    if (u_band_mode > 0.5) {
        float be = band_energy(bf);
        base = mix(base, band_color(bf) * (0.35 + 1.1 * lum), 0.88);
        boost = 0.55 + 1.15 * be;      // each zone flares with its own band
    }
    v_color = vec4(base * (0.42 + 0.58 * shade) * boost,
                   u_alpha * (0.32 + 0.68 * shade) * min(boost, 1.4));
}
"""

FRAG = """
varying vec4 v_color;
varying float v_drop;

void main() {
    if (v_drop > 0.5) { discard; }
    vec2 c = gl_PointCoord - vec2(0.5);
    float r2 = dot(c, c);
    if (r2 > 0.25) { discard; }
    float soft = 1.0 - smoothstep(0.06, 0.25, r2);
    gl_FragColor = vec4(v_color.rgb, v_color.a * soft);
}
"""


class DepthCloudVisual(Visual):
    """Point cloud whose per-point depth is fetched from a texture in the
    vertex shader (vertex texture fetch), as in the three.js original."""

    def __init__(self, grid_w: int, grid_h: int):
        Visual.__init__(self, vcode=VERT, fcode=FRAG)
        ys, xs = np.mgrid[0:grid_h, 0:grid_w]
        pix = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
        self._n_points = len(pix)
        self.shared_program["a_pix"] = gloo.VertexBuffer(pix)
        self.shared_program["u_res"] = (float(grid_w), float(grid_h))

        self._tex = gloo.Texture2D(np.zeros((4, 4, 3), np.float32),
                                   interpolation="linear",
                                   wrapping="clamp_to_edge")
        self._tex_shape = (4, 4)
        self.shared_program["u_tex"] = self._tex

        defaults = {
            "u_uv_scale": (1.0, 1.0), "u_uv_color": (0.0, 0.0),
            "u_uv_depth": (0.0, 0.0), "u_near": 3.0, "u_far": 6.0,
            "u_zoffset": 4.5, "u_point": 2.0, "u_depth_scale": 1.0,
            "u_scatter": 0.0, "u_sparkle": 0.0, "u_cutoff": 0.02,
            "u_flip": 1.0, "u_persp": 0.35, "u_aspect": 1.0,
            "u_alpha": 0.85, "u_tint_mix": 0.0,
            "u_tint": (1.0, 1.0, 1.0), "u_wave0": (0.0, 0.0, 0.0, 0.0),
            "u_wave1": (0.0, 0.0, 0.0, 0.0), "u_wave2": (0.0, 0.0, 0.0, 0.0),
            "u_band_mode": 0.0, "u_band_push": 0.10,
            "u_col_low": (1.0, 0.3, 0.3), "u_col_mid": (0.4, 1.0, 0.5),
            "u_col_high": (0.4, 0.6, 1.0),
            "u_e_low": 0.0, "u_e_mid": 0.0, "u_e_high": 0.0,
        }
        for k, v in defaults.items():
            self.shared_program[k] = v

        self._draw_mode = "points"
        self.set_gl_state("translucent", depth_test=True, blend=True,
                          blend_func=("src_alpha", "one_minus_src_alpha"))

    def set_frame(self, rgb: np.ndarray) -> None:
        """Upload a HxWx3 uint8/float frame as the color+depth source."""
        data = rgb if rgb.dtype == np.float32 else \
            (rgb.astype(np.float32) / 255.0)
        if data.shape[:2] != self._tex_shape:
            self._tex = gloo.Texture2D(data, interpolation="linear",
                                       wrapping="clamp_to_edge")
            self._tex_shape = data.shape[:2]
            self.shared_program["u_tex"] = self._tex
        else:
            self._tex.set_data(data)

    def set_uniform(self, name: str, value) -> None:
        self.shared_program[name] = value

    def _prepare_transforms(self, view):
        view.view_program.vert["transform"] = view.get_transform()

    def _prepare_draw(self, view):
        return True

    def _compute_bounds(self, axis, view):
        return (-4.0, 4.0)


DepthCloud = create_visual_node(DepthCloudVisual)


class PointCloudMode(BaseMode):
    name = "Depth Point Cloud"
    camera_distance = 7.5
    camera_elevation = 3.0      # near eye-level: the subject stands upright
    consumes_video = True          # the video feeds the cloud, not a backdrop

    SPEC_W, SPEC_H = 128, 96       # audio-terrain fallback texture

    def build(self):
        # grid resolution ~ point count; the texture can be any size
        if self.profile.name == "rpi5":
            gw, gh = 200, 150
        else:
            gw, gh = 440, 330
        self.grid = (gw, gh)
        self.cloud = DepthCloud(gw, gh, parent=self.view.scene)
        self.visuals = [self.cloud]
        self.n_points = gw * gh

        d = self.settings.damping
        self.depth_scale = VelocityValue(1.0, accel=12.0, damping=d)
        self.point_size = VelocityValue(2.0, accel=14.0, damping=d)
        self.scatter = VelocityValue(0.0, accel=8.0, damping=d)
        self.sparkle = VelocityValue(0.0, accel=10.0, damping=d)
        self.tilt = VelocityValue(0.0, accel=4.0, damping=0.88)

        self.waves: list[dict] = []
        self._demo = None              # None = unopened, False = unavailable
        self._last_uploaded = None
        self._last_ft = None
        self._t = 0.0
        self._spec = np.zeros((self.SPEC_H, self.SPEC_W, 3), np.float32)
        self._tr = None

    # ------------------------------------------------------------- sources

    def _demo_frame(self):
        """The three.js `webgl_video_kinect` clip, bundled and looping.

        Driven by the mode's own clock, not the audio playhead: it is a demo
        loop rather than a user-supplied video, so it plays with no song
        loaded and keeps going while playback is paused.
        """
        if self._demo is False:
            return None
        if self._demo is None:
            from ..config import asset_path
            from ..video.player import VideoSource
            path = asset_path("kinect.mp4")
            if not os.path.exists(path):
                self._demo = False
                return None
            try:
                self._demo = VideoSource(path, max_height=720)
            except Exception:
                self._demo = False
                return None
        dur = self._demo.duration or 1.0
        return self._demo.frame_at(self._t % dur)

    def _source_frame(self, frame):
        """Pick the active source and return (rgb, is_new, layout)."""
        mgr = getattr(self, "manager", None)
        want = self.settings.pc_source

        if want == "kinect":
            rgb = self._demo_frame()
            if rgb is not None:
                return rgb, rgb is not self._last_uploaded, "luminance"
            return self._audio_terrain(frame), True, "luminance"

        if want in ("auto", "camera") and mgr is not None and \
                getattr(mgr, "camera", None) is not None:
            rgb = mgr.camera.frame_at(0.0)
            if rgb is not None:
                return rgb, True, self.settings.pc_layout
        if want == "camera":
            return self._audio_terrain(frame), True, "luminance"

        if want in ("auto", "video") and mgr is not None and \
                getattr(mgr, "video", None) is not None:
            rgb = mgr.video.frame_at(mgr.last_audio_time)
            if rgb is not None:
                is_new = rgb is not self._last_uploaded
                return rgb, is_new, self.settings.pc_layout
        if want == "video":
            return self._audio_terrain(frame), True, "luminance"

        if want in ("auto", "image") and mgr is not None and \
                getattr(mgr, "still_image", None) is not None:
            rgb = mgr.still_image
            return rgb, rgb is not self._last_uploaded, self.settings.pc_layout
        if want == "image":
            return self._audio_terrain(frame), True, "luminance"

        # auto: nothing of the user's is loaded — show the Kinect demo person
        if want == "auto":
            rgb = self._demo_frame()
            if rgb is not None:
                return rgb, rgb is not self._last_uploaded, "luminance"

        return self._audio_terrain(frame), True, "luminance"

    def _audio_terrain(self, frame) -> np.ndarray:
        """Scrolling spectrogram relief — used when no visual source exists."""
        if frame is not None and frame.time != self._last_ft:
            spec = np.asarray(frame.spectrum, np.float32)
            row_v = np.interp(np.linspace(0, len(spec) - 1, self.SPEC_W),
                              np.arange(len(spec)), spec).astype(np.float32)
            lut = self.palette.lut(64)
            idx = np.clip((row_v * 63).astype(int), 0, 63)
            row = (lut[idx] * (0.25 + 0.75 * row_v)[:, None]).astype(np.float32)
            self._spec[1:] = self._spec[:-1] * 0.992
            self._spec[0] = row
        return self._spec

    # ------------------------------------------------------------- update

    def update(self, frame, dt):
        dt = min(dt, 0.05)
        self._t += dt
        d = self.settings.damping
        for v in (self.depth_scale, self.point_size, self.scatter,
                  self.sparkle):
            v.damping = d

        new_frame = frame.time != self._last_ft
        punch = frame.punch if new_frame else 0.0
        beat_now = frame.beat and new_frame

        bass = float(frame.bands[:2].mean())
        treble = float(frame.bands[5:].mean())

        rgb, is_new, layout = self._source_frame(frame)
        self._last_ft = frame.time
        if rgb is not None and is_new:
            self.cloud.set_frame(rgb)
            self._last_uploaded = rgb
        if rgb is not None:
            h, w = rgb.shape[:2]
            if layout == "side_by_side":
                w *= 0.5
            elif layout == "top_bottom":
                h *= 0.5
            self.cloud.set_uniform("u_aspect", float((w / h) / (4.0 / 3.0)))

        # --- audio-driven uniforms
        self.depth_scale.set_target(1.0 + bass * 0.55 + frame.rms * 0.2)
        self.point_size.set_target(self.settings.pc_point_size *
                                   (1.0 + frame.rms * 0.7))
        # disintegration kicks in only on the loud passages
        self.scatter.set_target(max(0.0, frame.rms - 0.6) * 1.6)
        self.sparkle.set_target(treble * 1.6)
        self.tilt.set_target(np.sin(self._t * 0.27) * (3.0 + bass * 9.0))
        if punch > 0.05:
            kb = punch * self.settings.beat_impulse
            self.depth_scale.impulse(kb * 1.6)
            self.point_size.impulse(kb * 4.0)
            self.scatter.impulse(kb * 0.9)
            self.tilt.impulse(kb * 14.0)
            if punch > 0.25:            # only real hits ripple the surface
                self._spawn_wave(kb)
        if beat_now:
            self.sparkle.impulse(frame.beat_strength * 2.0)

        ds = float(np.clip(self.depth_scale.update(dt), 0.35, 2.0))
        ps = float(np.clip(self.point_size.update(dt), 0.5, 14.0))
        sc = float(np.clip(self.scatter.update(dt), 0.0, 2.5))
        sp = float(np.clip(self.sparkle.update(dt), 0.0, 3.0))
        tilt = float(np.clip(self.tilt.update(dt), -25.0, 25.0))

        self._advance_waves(dt)

        # --- layout: which half of the frame is color, which is depth
        if layout == "side_by_side":
            uv_scale, uv_c, uv_d = (0.5, 1.0), (0.0, 0.0), (0.5, 0.0)
        elif layout == "top_bottom":
            uv_scale, uv_c, uv_d = (1.0, 0.5), (0.0, 0.0), (0.0, 0.5)
        else:
            uv_scale, uv_c, uv_d = (1.0, 1.0), (0.0, 0.0), (0.0, 0.0)

        tint = self.palette.band_color(3)
        if self.settings.use_image_colors:
            tint_mix = 0.9
        elif layout == "luminance":
            # a depth map carries no real color — tint it rather than
            # rendering a flat gray mass
            tint_mix = 0.55
        else:
            tint_mix = 0.0          # RGBD footage: keep the true video color

        u = self.cloud.set_uniform
        u("u_uv_scale", uv_scale)
        u("u_uv_color", uv_c)
        u("u_uv_depth", uv_d)
        near = float(self.settings.pc_near)
        far = float(max(self.settings.pc_far, near + 0.5))
        u("u_near", near)
        u("u_far", far)
        u("u_zoffset", (near + far) * 0.5)      # keeps the cloud centred
        u("u_persp", float(self.settings.pc_perspective))
        u("u_point", ps)
        u("u_depth_scale", ds)
        u("u_scatter", sc)
        u("u_sparkle", sp)
        u("u_cutoff", float(self.settings.pc_cutoff))
        u("u_tint", tuple(float(c) for c in np.clip(tint, 0, 1)))
        u("u_tint_mix", tint_mix)
        u("u_alpha", 0.85)
        for i in range(3):
            w = self.waves[i] if i < len(self.waves) else None
            u(f"u_wave{i}", (w["x"], w["y"], w["r"], w["a"]) if w
              else (0.0, 0.0, 0.0, 0.0))

        # --- lows / mids / highs painted across an axis of the cloud
        axis = self.settings.pc_band_axis
        mode_id = {"off": 0.0, "vertical": 1.0, "depth": 2.0,
                   "horizontal": 3.0, "radial": 4.0}.get(axis, 0.0)
        u("u_band_mode", mode_id)
        if mode_id > 0.0:
            b = frame.bands
            u("u_e_low", float(b[:2].mean()))
            u("u_e_mid", float(b[2:5].mean()))
            u("u_e_high", float(b[5:].mean()))
            for name, idx in (("u_col_low", 0), ("u_col_mid", 3),
                              ("u_col_high", 6)):
                c = np.clip(self.palette.band_color(idx), 0, 1)
                u(name, tuple(float(v) for v in c))
            u("u_band_push", float(self.settings.pc_band_push))

        # gentle bass-driven sway of the whole cloud
        from vispy.visuals.transforms import MatrixTransform
        if self._tr is None:
            self._tr = MatrixTransform()
            self.cloud.transform = self._tr
        self._tr.reset()
        self._tr.rotate(tilt, (0, 1, 0))

    # -------------------------------------------------------------- waves

    def _spawn_wave(self, strength: float) -> None:
        # Keep ripples subtle: a tall ring reads as a wall sweeping across the
        # surface and swamps the subject entirely.
        rng = np.random.default_rng(int(self._t * 1000) & 0xFFFF)
        self.waves.insert(0, {
            "x": 0.5 + float(rng.normal(0, 0.10)),
            "y": 0.5 + float(rng.normal(0, 0.10)),
            "r": 0.0,
            "a": float(np.clip(strength * 0.07, 0.012, 0.11)),
        })
        del self.waves[3:]

    def _advance_waves(self, dt: float) -> None:
        for w in self.waves:
            w["r"] += dt * 0.75
            w["a"] *= float(np.exp(-dt * 1.6))
        self.waves = [w for w in self.waves if w["r"] < 1.6 and w["a"] > 0.004]

    def velocity_magnitude(self):
        return (self.depth_scale.speed + self.scatter.speed +
                self.point_size.speed)
