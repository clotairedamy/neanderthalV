"""Mode 10 - Reactive 3D Text, tessellated.

After the three.js `webgl_modifier_tessellation` example: the text is built
as solid geometry broken into many small chunks, and each chunk is rotated
and pushed away from its own centre by an audio-driven amplitude. At rest
the chunks sit flush and the word is crisp, opaque, readable type; on a kick
they shatter outward and spring back.

A point-cloud style sits alongside it. The first version of this mode was a
point cloud and was abandoned because it could not be read while a track
played -- uniform area sampling turns a word into a featureless slab of
dust. This one samples by distance to the glyph boundary instead, so the
contours stay dense and bright and the letterforms survive; the tessellated
style remains the default for when legibility matters most.

Geometry is uploaded once and every frame's animation happens in the vertex
shader, so the per-frame CPU cost is a handful of uniforms.
"""
from __future__ import annotations

import os

import numpy as np
from vispy import gloo
from vispy.scene.visuals import create_visual_node
from vispy.visuals import Visual

from ..physics.velocity import VelocityValue
from .base import BaseMode

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

VERT = """
attribute vec3 a_pos;
attribute vec3 a_centroid;
attribute vec3 a_seed;
attribute vec3 a_norm;
attribute float a_u;

uniform float u_explode;
uniform float u_scale;
uniform float u_yaw;
uniform float u_pitch;
uniform float u_band_depth;
uniform float u_cell;
uniform vec3 u_light;
uniform float u_mono;
uniform sampler2D u_lut;
uniform sampler2D u_energy;

varying vec4 v_color;

vec3 rot_axis(vec3 v, vec3 axis, float a) {
    float c = cos(a);
    float s = sin(a);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

void main() {
    float e = texture2DLod(u_energy, vec2(a_u, 0.5), 0.0).r;

    vec3 p = a_pos;
    vec3 n = a_norm;
    if (u_explode > 0.001) {
        vec3 axis = normalize(a_seed + vec3(0.0001, 0.0001, 1.0));
        float amt = u_explode * (0.35 + fract(a_seed.x * 13.73));
        vec3 local = rot_axis(p - a_centroid, axis, amt * 1.5);
        n = rot_axis(n, axis, amt * 1.5);
        p = a_centroid + local + a_seed * amt * u_cell * 5.0;
    }

    // depth axis is Y: vispy's turntable camera is Z-up
    p.y += e * u_band_depth;
    p *= u_scale;

    float cy = cos(u_yaw), sy = sin(u_yaw);
    vec3 q  = vec3(p.x * cy - p.y * sy, p.x * sy + p.y * cy, p.z);
    vec3 qn = vec3(n.x * cy - n.y * sy, n.x * sy + n.y * cy, n.z);
    float cp = cos(u_pitch), sp = sin(u_pitch);
    q  = vec3(q.x,  q.y * cp - q.z * sp,  q.y * sp + q.z * cp);
    qn = vec3(qn.x, qn.y * cp - qn.z * sp, qn.y * sp + qn.z * cp);

    gl_Position = $transform(vec4(q, 1.0));

    vec3 base = texture2DLod(u_lut, vec2(a_u, 0.5), 0.0).rgb;
    // monochrome is a flat white substrate, not the palette drained of
    // saturation: a dark palette colour would desaturate to muddy grey
    base = mix(base, vec3(0.86), u_mono);
    float diff = max(dot(normalize(qn), normalize(u_light)), 0.0);
    v_color = vec4(base * (0.32 + 0.68 * diff) * (0.60 + 0.70 * e), 1.0);
}
"""

FRAG = """
varying vec4 v_color;
void main() { gl_FragColor = v_color; }
"""


POINT_VERT = """
attribute vec3 a_pos;
attribute vec3 a_seed;
attribute float a_u;
attribute float a_edge;

uniform float u_explode;
uniform float u_scale;
uniform float u_yaw;
uniform float u_pitch;
uniform float u_band_depth;
uniform float u_size;
uniform float u_mono;
uniform sampler2D u_lut;
uniform sampler2D u_energy;

varying vec4 v_color;

void main() {
    float e = texture2DLod(u_energy, vec2(a_u, 0.5), 0.0).r;

    // the chunked style rotates a chunk about its own centre and nudges it
    // by a fraction of a cell; a point has no body to rotate, so all it can
    // do is drift -- and it has to be a small drift, or a kick throws the
    // word into a ball instead of roughing up its edges
    vec3 p = a_pos + a_seed * u_explode * 0.10;
    p.y += e * u_band_depth;
    p *= u_scale;

    float cy = cos(u_yaw), sy = sin(u_yaw);
    vec3 q = vec3(p.x * cy - p.y * sy, p.x * sy + p.y * cy, p.z);
    float cp = cos(u_pitch), sp = sin(u_pitch);
    q = vec3(q.x, q.y * cp - q.z * sp, q.y * sp + q.z * cp);

    gl_Position = $transform(vec4(q, 1.0));
    // contour points are the big bright ones; that is what carries the
    // letterform once the stroke interior thins out
    gl_PointSize = u_size * (0.40 + 1.15 * a_edge) * (0.78 + 0.75 * e);

    vec3 base = texture2DLod(u_lut, vec2(a_u, 0.5), 0.0).rgb;
    base = mix(base, vec3(0.62 + 0.38 * a_edge), u_mono);
    float b = (0.22 + 0.95 * a_edge) * (0.55 + 0.80 * e);
    v_color = vec4(base * b, 1.0);
}
"""

POINT_FRAG = """
varying vec4 v_color;
void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    float r2 = dot(d, d);
    if (r2 > 0.25) discard;
    // a solid core with a tight edge. Falling off across the whole radius
    // makes every point a fuzzy blob, and thousands of overlapping blobs
    // are a haze rather than a letter.
    float a = 1.0 - smoothstep(0.15, 0.25, r2);
    gl_FragColor = vec4(v_color.rgb * a, a);
}
"""


def load_font(size: int):
    from PIL import ImageFont
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def wrap_text(text: str, max_aspect: float = 3.2) -> str:
    """Break a long word/phrase over two lines.

    A single long line has to be scaled down so far to fit the frame that
    its letters become an unreadable stripe; two lines keep them large.
    """
    if len(text) < 7:
        return text
    if len(text) * 0.62 <= max_aspect:
        return text
    mid = len(text) // 2
    spaces = [i for i, c in enumerate(text) if c == " "]
    if spaces:
        cut = min(spaces, key=lambda i: abs(i - mid))
        return text[:cut] + "\n" + text[cut + 1:]
    return text[:mid] + "\n" + text[mid:]


def text_mask(text: str, px: int = 150) -> np.ndarray:
    """Rasterize `text` (wrapped) to a boolean mask, row 0 = top."""
    from PIL import Image, ImageDraw
    text = (text or "NEANDERTHALV").strip() or "NEANDERTHALV"
    text = wrap_text(text)
    font = load_font(px)
    d = ImageDraw.Draw(Image.new("L", (8, 8)))
    try:
        box = d.multiline_textbbox((0, 0), text, font=font, spacing=12)
    except Exception:
        box = (0, 0, px * len(text) // 2, px)
    w = max(box[2] - box[0], 1) + 16
    h = max(box[3] - box[1], 1) + 16
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).multiline_text((8 - box[0], 8 - box[1]), text,
                                       fill=255, font=font, spacing=12,
                                       align="center")
    return np.asarray(img) > 100


def tessellate(mask: np.ndarray, max_cells: int = 1600,
               target_w: float = 3.0, target_h: float = 1.7,
               thickness: float = 0.16, seed: int = 5):
    """Turn a glyph mask into solid chunks: one small box per filled cell.

    Returns per-vertex arrays (pos, centroid, seed, normal, u). Geometry is
    non-indexed so every triangle of a chunk can share that chunk's centroid
    and random axis, which is what lets a chunk tumble as one piece.
    """
    h, w = mask.shape
    # pick a cell size that keeps the chunk count near the budget
    filled = int(mask.sum())
    cell = max(2, int(np.sqrt(max(filled, 1) / max_cells)))
    gh, gw = h // cell, w // cell
    if gh < 1 or gw < 1:
        cell, gh, gw = 1, h, w
    grid = mask[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell)
    occupied = grid.mean(axis=(1, 3)) > 0.35
    ys, xs = np.nonzero(occupied)
    if len(xs) == 0:
        ys, xs = np.array([0]), np.array([0])

    scale = min(target_w / max(gw, 1), target_h / max(gh, 1))
    cx, cy = (gw - 1) / 2.0, (gh - 1) / 2.0
    px = (xs - cx) * scale
    pz = -(ys - cy) * scale          # row 0 is the top of the image
    hw = scale * 0.5
    d = thickness * 0.5

    # unit cube: 6 faces x 2 triangles, with outward normals
    fa = [((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (0, 0, -1)),
          ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1), (0, 0, 1)),
          ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1), (-1, 0, 0)),
          ((1, -1, -1), (1, -1, 1), (1, 1, 1), (1, 1, -1), (1, 0, 0)),
          ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1), (0, -1, 0)),
          ((-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1), (0, 1, 0))]
    corners, normals = [], []
    for a, b, c, dd, n in fa:
        for tri in ((a, b, c), (a, c, dd)):
            corners.extend(tri)
            normals.extend([n, n, n])
    corners = np.asarray(corners, np.float32)      # (36, 3) in +-1
    normals = np.asarray(normals, np.float32)
    # scale the unit cube to cell size: x/z by hw, y (depth) by d
    corners = corners * np.array([hw, d, hw], np.float32)

    n_cells = len(xs)
    centres = np.stack([px, np.zeros(n_cells), pz], axis=1).astype(np.float32)
    pos = (centres[:, None, :] + corners[None, :, :]).reshape(-1, 3)
    cent = np.repeat(centres, 36, axis=0)
    norm = np.tile(normals, (n_cells, 1))
    rng = np.random.default_rng(seed)
    sd = rng.normal(0, 1, (n_cells, 3)).astype(np.float32)
    sd /= np.linalg.norm(sd, axis=1, keepdims=True) + 1e-9
    sd = np.repeat(sd, 36, axis=0)
    span = max(px.max() - px.min(), 1e-6)
    u = np.repeat(((px - px.min()) / span).astype(np.float32), 36)
    return (pos.astype(np.float32), cent, sd, norm, u, n_cells,
            float(scale))


def point_cloud(mask: np.ndarray, n_points: int = 90000,
                target_w: float = 3.0, target_h: float = 1.7,
                depth: float = 0.15, edge_tau: float = 2.2,
                seed: int = 5):
    """Scatter points through a glyph mask, weighted toward the boundary.

    Sampling the filled area uniformly -- what the first version of this
    mode did -- spends most of the points deep inside thick strokes, where
    they say nothing about the shape, and the word reads as a slab. Here the
    weight falls off with distance to the nearest background pixel, so the
    contours stay dense while the interior thins to a haze. The letterforms
    are then carried by their outlines and survive both the sway and the
    band displacement.

    Point count matters more than any of the weighting here: a twelve-letter
    word is about two thousand points per glyph at 26k, which is not enough
    to draw a contour, and the letters merge into dust. The default is set
    high because points are nearly free on the GPU and legibility is the
    whole problem this style has.

    Returns (pos, seed, u, edge) with `edge` at 1 on the outline and decaying
    inward; the shader uses it for both point size and brightness.
    """
    from scipy.ndimage import distance_transform_edt

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        ys, xs = np.array([0]), np.array([0])
        dist = np.ones(1, np.float32)
    else:
        dist = distance_transform_edt(mask)[ys, xs].astype(np.float32)

    # 1 on the contour, decaying into the stroke
    edge = np.exp(-dist / edge_tau).astype(np.float32)
    # a low floor: interior points say little about the shape, and every
    # one spent there is a point not spent on a contour
    w = edge + 0.06
    w /= w.sum()

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(xs), size=n_points, p=w)
    # sub-pixel jitter, or the cloud sits on a visible pixel lattice
    jx = xs[pick] + rng.uniform(-0.5, 0.5, n_points)
    jy = ys[pick] + rng.uniform(-0.5, 0.5, n_points)

    h, w_px = mask.shape
    scale = min(target_w / max(w_px, 1), target_h / max(h, 1))
    px = (jx - (w_px - 1) / 2.0) * scale
    pz = -(jy - (h - 1) / 2.0) * scale        # row 0 is the top of the image
    # a slab of depth rather than a flat sheet, thickest along the contour
    py = rng.normal(0.0, 1.0, n_points) * depth * (0.35 + 0.65 * edge[pick])

    pos = np.stack([px, py, pz], 1).astype(np.float32)
    sd = rng.normal(0, 1, (n_points, 3)).astype(np.float32)
    sd /= np.linalg.norm(sd, axis=1, keepdims=True) + 1e-9
    span = max(px.max() - px.min(), 1e-6)
    u = ((px - px.min()) / span).astype(np.float32)
    return pos, sd, u, edge[pick].astype(np.float32)


class PointTextVisual(Visual):
    """Additive point-cloud text; animation happens in the vertex shader."""

    def __init__(self):
        Visual.__init__(self, vcode=POINT_VERT, fcode=POINT_FRAG)
        self._n = 0
        self._lut = gloo.Texture2D(np.zeros((1, 256, 3), np.float32),
                                   interpolation="linear",
                                   wrapping="clamp_to_edge")
        self._energy = gloo.Texture2D(np.zeros((1, 64, 3), np.float32),
                                      interpolation="linear",
                                      wrapping="clamp_to_edge")
        self.shared_program["u_lut"] = self._lut
        self.shared_program["u_energy"] = self._energy
        for k, v in (("u_explode", 0.0), ("u_scale", 1.0), ("u_yaw", 0.0),
                     ("u_pitch", 0.0), ("u_band_depth", 0.0),
                     ("u_size", 3.4), ("u_mono", 0.0)):
            self.shared_program[k] = v
        self._draw_mode = "points"
        # additive, no depth write: the cloud should accumulate into glow
        self.set_gl_state("additive", depth_test=False, blend=True)

    def set_geometry(self, pos, seed, u, edge):
        self.shared_program["a_pos"] = gloo.VertexBuffer(pos)
        self.shared_program["a_seed"] = gloo.VertexBuffer(seed)
        self.shared_program["a_u"] = gloo.VertexBuffer(u)
        self.shared_program["a_edge"] = gloo.VertexBuffer(edge)
        self._n = len(pos)

    def set_lut(self, lut):
        self._lut.set_data(np.ascontiguousarray(
            lut.reshape(1, -1, 3).astype(np.float32)))

    def set_energy(self, e):
        self._energy.set_data(np.ascontiguousarray(
            np.repeat(e.reshape(1, -1, 1), 3, axis=2).astype(np.float32)))

    def set_uniform(self, name, value):
        self.shared_program[name] = value

    def _prepare_transforms(self, view):
        view.view_program.vert["transform"] = view.get_transform()

    def _prepare_draw(self, view):
        return self._n > 0

    def _compute_bounds(self, axis, view):
        return (-2.0, 2.0)


class TessTextVisual(Visual):
    """Solid chunked text; all animation happens in the vertex shader."""

    def __init__(self):
        Visual.__init__(self, vcode=VERT, fcode=FRAG)
        self._n = 0
        self._lut = gloo.Texture2D(np.zeros((1, 256, 3), np.float32),
                                   interpolation="linear",
                                   wrapping="clamp_to_edge")
        self._energy = gloo.Texture2D(np.zeros((1, 64, 3), np.float32),
                                      interpolation="linear",
                                      wrapping="clamp_to_edge")
        self.shared_program["u_lut"] = self._lut
        self.shared_program["u_energy"] = self._energy
        for k, v in (("u_explode", 0.0), ("u_scale", 1.0), ("u_yaw", 0.0),
                     ("u_pitch", 0.0), ("u_band_depth", 0.0),
                     ("u_cell", 0.03),
                     ("u_light", (0.35, -0.75, 0.55))):
            self.shared_program[k] = v
        self._draw_mode = "triangles"
        # opaque with depth: solid letters read far better than additive dust
        self.set_gl_state(depth_test=True, blend=False, cull_face=False)

    def set_geometry(self, pos, cent, seed, norm, u):
        self.shared_program["a_pos"] = gloo.VertexBuffer(pos)
        self.shared_program["a_centroid"] = gloo.VertexBuffer(cent)
        self.shared_program["a_seed"] = gloo.VertexBuffer(seed)
        self.shared_program["a_norm"] = gloo.VertexBuffer(norm)
        self.shared_program["a_u"] = gloo.VertexBuffer(u)
        self._n = len(pos)

    def set_lut(self, lut):
        self._lut.set_data(np.ascontiguousarray(
            lut.reshape(1, -1, 3).astype(np.float32)))

    def set_energy(self, e):
        self._energy.set_data(np.ascontiguousarray(
            np.repeat(e.reshape(1, -1, 1), 3, axis=2).astype(np.float32)))

    def set_uniform(self, name, value):
        self.shared_program[name] = value

    def _prepare_transforms(self, view):
        view.view_program.vert["transform"] = view.get_transform()

    def _prepare_draw(self, view):
        return self._n > 0

    def _compute_bounds(self, axis, view):
        return (-2.0, 2.0)


TessText = create_visual_node(TessTextVisual)
PointText = create_visual_node(PointTextVisual)


class TextMode(BaseMode):
    name = "Reactive 3D Text"
    camera_distance = 3.4
    camera_elevation = 8.0
    trail_scale = 0.35          # heavy trails smear type illegible

    def build(self):
        self.tess = TessText(parent=self.view.scene)
        self.points = PointText(parent=self.view.scene)
        self.visuals = [self.tess, self.points]
        self._text = None
        self._style = None
        self.n_cells = 0
        self._rebuild(self.settings.text_content,
                      bool(self.settings.text_points))

        d = self.settings.damping
        self.sway_rate = VelocityValue(0.0, accel=3.0, damping=d)
        self.pulse = VelocityValue(1.0, accel=18.0, damping=d)
        self.explode = VelocityValue(0.0, accel=6.0, damping=0.72)
        self.sway_phase = 0.0
        self._last_ft = None

    @property
    def visual(self):
        """Whichever style is currently showing."""
        return self.points if self._style else self.tess

    def _rebuild(self, text: str, as_points: bool) -> None:
        mask = text_mask(text)
        if as_points:
            pos, seed, u, edge = point_cloud(mask)
            self.points.set_geometry(pos, seed, u, edge)
            self.n_cells = len(pos)
        else:
            pos, cent, seed, norm, u, n_cells, cell = tessellate(mask)
            self.tess.set_geometry(pos, cent, seed, norm, u)
            self.tess.set_uniform("u_cell", cell)
            self.n_cells = n_cells
        # only build the style being shown, but keep the other parented so
        # switching back does not need a mode rebuild
        self.tess.visible = not as_points
        self.points.visible = as_points
        self._text = text
        self._style = as_points

    def update(self, frame, dt):
        want_points = bool(self.settings.text_points)
        if self.settings.text_content != self._text or want_points != self._style:
            self._rebuild(self.settings.text_content, want_points)
        dt = min(dt, 0.05)
        d = self.settings.damping
        for v in (self.sway_rate, self.pulse):
            v.damping = d

        new_frame = frame.time != self._last_ft
        self._last_ft = frame.time
        punch = frame.punch if new_frame else 0.0
        bass = float(frame.bands[:2].mean())

        self.sway_rate.set_target(0.35 + frame.rms * 1.1)
        self.pulse.set_target(1.0 + bass * 0.18)
        self.explode.set_target(max(0.0, frame.rms - 0.72) * 0.5)
        if punch > 0.05:
            kb = punch * self.settings.beat_impulse
            self.pulse.impulse(kb * 1.0)
            if punch > 0.35:
                self.explode.impulse(kb * self.settings.text_explode * 0.30)

        # bounded sway keeps the word facing the viewer and readable
        self.sway_phase += max(0.0, self.sway_rate.update(dt)) * dt
        amp = float(self.settings.text_sway)
        u = self.visual.set_uniform
        u("u_yaw", float(np.sin(self.sway_phase) * amp))
        u("u_pitch", float(np.sin(self.sway_phase * 0.53 + 1.1) * amp * 0.16))
        u("u_scale", float(max(0.2, self.pulse.update(dt))))
        u("u_explode", float(np.clip(self.explode.update(dt), 0.0, 2.5)))
        u("u_band_depth", float(self.settings.text_depth))
        u("u_mono", 1.0 if self.settings.text_mono else 0.0)
        if self._style:
            u("u_size", float(self.settings.text_point_size))

        # 64-wide ramps sampled by horizontal position -> spectrum analyzer
        bands = np.clip(frame.bands, 0, 1)
        ramp = np.interp(np.linspace(0, 6, 64), np.arange(7), bands)
        self.visual.set_energy(ramp.astype(np.float32))
        self.visual.set_lut(self.palette.lut(256))

    def velocity_magnitude(self):
        return self.sway_rate.speed + self.pulse.speed + self.explode.speed
