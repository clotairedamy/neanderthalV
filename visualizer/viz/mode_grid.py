"""Mode 11 -- Hiding Grid.

Combines the two reference drawings, which turn out to speak the same
language: a grid plus a coherent noise field, in black and white.

  hiding-squares      40x40 cells, each holding a square that noise scales
                      and slides diagonally; the cell clips it, so a square
                      leaving its box is progressively hidden and finally
                      gone. Measured from the file, in cell-normalized
                      coordinates: centre = (18/35)(s - s^2), half-size |s|,
                      with s over roughly [-1.75, 1].
  deformed-grid-mesh  48x48 lattice whose nodes a much smoother noise field
                      pushes off-grid by up to ~3 cells, drawn as edges and
                      node dots.

Here one field does both jobs: its fine detail sets each cell's square, and
a low-frequency band of it warps the lattice the squares live in, so the
squares deform with the mesh instead of floating over it.

Nothing is traced from the SVGs -- the field is evaluated live, which is
what makes column count and feature size controls rather than constants.

Audio mapping: beats and kicks swell every square (a square near the edge
of its cell can be pushed out of sight entirely, so the grid breathes
holes); bass drives the warp; the spectrum is laid across the columns, so
each column lights and grows with its own frequency; overall energy sets
how fast the field flows.
"""
from __future__ import annotations

import numpy as np
from vispy import scene

from ..physics.velocity import VelocityValue
from .base import BaseMode
from .noise import perlin3, permutation

# measured from hiding-squares.svg: the raw file uses t = (90/7)(1 - s) on a
# 50-unit cell, which in a cell normalized to [-1, 1] is exactly this
CENTRE_K = 18.0 / 35.0
S_HI, S_LO = 1.0, -1.75          # the file's observed range of s
# the warp field is the same noise an octave and a half down; measured at
# ~1.9 periods across the mesh against ~7 for the squares
WARP_SCALE = 0.27
WARP_CELLS = 3.5                 # peak node displacement, in cells


def hiding_square(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clip a cell's square to the cell, in normalized [-1, 1] coordinates.

    Returns (lo, hi) along one axis; the square is offset equally in x and y
    so both axes clip identically. lo >= hi means the square has slid fully
    out of its cell and is hidden.
    """
    centre = CENTRE_K * (s - s * s)
    half = np.abs(s)
    return np.maximum(-1.0, centre - half), np.minimum(1.0, centre + half)


class GridMode(BaseMode):
    name = "Hiding Grid"
    # the source drawings are flat graphics; a turntable tilt would shear the
    # lattice, so this renders face-on in the 2D view like the fractal does
    camera = "panzoom"
    trail_scale = 0.22
    # the source drawings are crisp vector work; glow just blurs the squares
    bloom_scale = 0.25

    MAX_COLS = 96
    cols_attr = "grid_cols"     # which setting drives the column count
    warp_cells = WARP_CELLS     # peak node displacement, in cells

    def node_mask(self, n: int):
        """Per-node multiplier on the lattice line and dot alpha, or None.

        A cell driven to nothing still sits inside a drawn lattice, which
        reads exactly like the holes the noise makes on its own. Clearing
        the lines too is what turns a carved region into an actual void.
        """
        return None

    def shape_scale(self, s, cols: int, swell: float):
        """Let a subclass reshape the per-cell square scale before clipping.

        `s` is the finished scale for every cell -- noise, beat swell and
        spectrum already folded in -- so an override has the last word.
        """
        return s

    def build(self):
        self.perm = permutation(11)
        self.extent = self.profile.fractal_resolution * 0.45
        self._cols = 0
        self._t = 0.0
        d = self.settings.damping
        # beats resize the squares; this is the multiplier on s
        self.swell = VelocityValue(1.0, accel=16.0, damping=d)
        self.warp_amt = VelocityValue(1.0, accel=6.0, damping=d)
        self._last_ft = -1.0

        self.mesh = scene.visuals.Mesh(parent=self.view.scene)
        # Every square is clipped to its own cell and the cells tile without
        # overlapping, so no square is ever drawn over another and blending
        # buys nothing -- while costing real fill rate when a beat swells
        # them all to full size. Hidden cells collapse to zero area, so they
        # need no alpha to disappear.
        self.mesh.set_gl_state(depth_test=False, cull_face=False, blend=False)
        self.mesh.order = 0
        self.lines = scene.visuals.Line(connect="segments", width=1,
                                        parent=self.view.scene)
        self.lines.set_gl_state("translucent", depth_test=False)
        self.lines.order = 1
        self.nodes = scene.visuals.Markers(parent=self.view.scene, antialias=1)
        self.nodes.set_gl_state("translucent", depth_test=False)
        self.nodes.order = 2
        self.visuals = [self.mesh, self.lines, self.nodes]
        self._alloc(int(getattr(self.settings, self.cols_attr)))

    # ------------------------------------------------------------ topology

    def _alloc(self, cols: int) -> None:
        """Rebuild the fixed index buffers for a new column count.

        Faces and edge indices only change when `cols` does, so they are
        built here rather than per frame; hidden cells are collapsed to
        zero-area quads instead of being removed, which keeps the face array
        constant while the field animates.
        """
        cols = int(np.clip(cols, 4, self.MAX_COLS))
        self._cols = cols
        n = cols + 1

        # lattice sample points in [-1, 1]
        g = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        self.gx, self.gy = np.meshgrid(g, g)
        self.cell = 2.0 / cols

        # two triangles per cell over four own vertices
        base = np.arange(cols * cols, dtype=np.uint32) * 4
        self.faces = np.empty((cols * cols * 2, 3), np.uint32)
        self.faces[0::2] = np.stack([base, base + 1, base + 2], 1)
        self.faces[1::2] = np.stack([base, base + 2, base + 3], 1)

        # lattice edges, as segment pairs into the node array
        idx = np.arange(n * n, dtype=np.uint32).reshape(n, n)
        h = np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1)
        v = np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], 1)
        self.edges = np.concatenate([h, v]).ravel()

        # each column reads its own slice of the spectrum
        self.col_bin = np.minimum((np.arange(cols) * 64 // cols), 63)

        # per-vertex colour buffers, filled in place each frame; at 96
        # columns these are ~37k vertices and reallocating them per frame
        # cost more than the noise did
        self._line_rgba = np.ones((len(self.edges), 4), np.float32)
        self._node_rgba = np.ones((n * n, 4), np.float32)
        self._nm_cache = None
        self._nm_edges = None

    # --------------------------------------------------------------- frame

    def update(self, frame, dt):
        s_set = self.settings
        want = int(getattr(s_set, self.cols_attr))
        if want != self._cols:
            self._alloc(want)
        cols, n = self._cols, self._cols + 1

        bands = frame.bands
        bass = float(np.clip(bands[:2].mean() * s_set.sensitivity, 0, 1.5))
        treble = float(np.clip(bands[5:].mean() * s_set.sensitivity, 0, 1.5))

        fresh = frame.time != self._last_ft
        self._last_ft = frame.time
        if fresh:
            if frame.beat:
                self.swell.impulse(frame.beat_strength * s_set.grid_beat
                                   * s_set.beat_impulse * 1.6)
            self.swell.impulse(-frame.punch * s_set.grid_beat * 0.8)
        self.swell.set_target(1.0)
        swell = float(np.clip(self.swell.update(dt), 0.15, 2.4))

        # the field flows faster when the track is busier
        self._t += dt * (s_set.grid_flow * (0.35 + 1.4 * frame.rms))

        # --- warp the lattice with the low-frequency band of the field
        self.warp_amt.set_target(0.55 + 1.1 * bass)
        wa = float(np.clip(self.warp_amt.update(dt), 0.0, 2.5)) * s_set.grid_warp
        ns = float(s_set.grid_noise)
        wx, wy = self.gx * ns * WARP_SCALE, self.gy * ns * WARP_SCALE
        dx = perlin3(wx, wy, self._t, self.perm)
        dy = perlin3(wx + 31.7, wy + 11.3, self._t, self.perm)
        amp = self.cell * self.warp_cells * wa
        nx = self.gx + dx * amp
        ny = self.gy + dy * amp

        # --- per-cell square scale from the field's fine detail
        cx = (self.gx[:-1, :-1] + self.cell * 0.5) * ns
        cy = (self.gy[:-1, :-1] + self.cell * 0.5) * ns
        fine = perlin3(cx, cy, self._t * 1.7 + 5.0, self.perm)
        s = S_HI - (fine * 0.5 + 0.5) * (S_HI - S_LO)

        # the spectrum lies across the columns: each grows and lights with
        # its own frequency, so the grid doubles as an analyzer
        col = np.clip(frame.spectrum[self.col_bin], 0.0, 1.0)[None, :]
        s = s * swell * (0.72 + 0.55 * col)

        # last, so an override is not undone by the swell or the spectrum
        s = self.shape_scale(s, cols, swell)

        lo, hi = hiding_square(s)
        hidden = lo >= hi
        lo = np.where(hidden, 0.0, lo)
        hi = np.where(hidden, 0.0, hi)

        # --- map each clipped square through its cell's warped quad
        a0, a1 = (lo + 1.0) * 0.5, (hi + 1.0) * 0.5      # cell-local [0,1]
        n00x, n00y = nx[:-1, :-1], ny[:-1, :-1]
        n10x, n10y = nx[:-1, 1:], ny[:-1, 1:]
        n01x, n01y = nx[1:, :-1], ny[1:, :-1]
        n11x, n11y = nx[1:, 1:], ny[1:, 1:]

        def bilinear(a, b):
            top_x = n00x + (n10x - n00x) * a
            top_y = n00y + (n10y - n00y) * a
            bot_x = n01x + (n11x - n01x) * a
            bot_y = n01y + (n11y - n01y) * a
            return top_x + (bot_x - top_x) * b, top_y + (bot_y - top_y) * b

        corners = [bilinear(a0, a0), bilinear(a1, a0),
                   bilinear(a1, a1), bilinear(a0, a1)]
        verts = np.empty((cols * cols, 4, 3), np.float32)
        verts[..., 2] = 0.0
        for i, (px, py) in enumerate(corners):
            verts[:, i, 0] = px.ravel() * self.extent
            verts[:, i, 1] = py.ravel() * self.extent

        # --- black and white throughout: tone, never hue
        fill = np.clip(0.42 + 0.30 * col + 0.22 * np.abs(s) / 1.75, 0, 1)
        vc = np.empty((cols * cols, 4, 4), np.float32)
        vc[..., 0] = vc[..., 1] = vc[..., 2] = fill.ravel()[:, None]
        vc[..., 3] = np.where(hidden.ravel(), 0.0, 0.95)[:, None]
        self.mesh.set_data(vertices=verts.reshape(-1, 3),
                           faces=self.faces,
                           vertex_colors=vc.reshape(-1, 4))

        node_xy = np.stack([nx.ravel() * self.extent,
                            ny.ravel() * self.extent,
                            np.zeros(n * n, np.float32)], 1).astype(np.float32)
        la = 0.10 + 0.22 * treble
        na = 0.18 + 0.35 * treble
        size = float(np.clip(1.0 + 2.4 * frame.rms, 1.0, 3.6))
        nm = self.node_mask(n)
        if nm is None:
            self.lines.set_data(pos=node_xy[self.edges],
                                color=(1.0, 1.0, 1.0, la))
            self.nodes.set_data(node_xy, edge_width=0,
                                face_color=(1.0, 1.0, 1.0, na), size=size)
        else:
            # the mask only changes when the word or the grid does, so the
            # edge gather is cached on its identity
            if nm is not self._nm_cache:
                f = nm.ravel().astype(np.float32)
                self._nm_cache = nm
                self._nm_flat = f
                self._nm_edges = f[self.edges]
            self._line_rgba[:, 3] = la * self._nm_edges
            self._node_rgba[:, 3] = na * self._nm_flat
            self.lines.set_data(pos=node_xy[self.edges],
                                color=self._line_rgba)
            self.nodes.set_data(node_xy, edge_width=0,
                                face_color=self._node_rgba, size=size)

    def velocity_magnitude(self):
        return self.swell.speed + self.warp_amt.speed
