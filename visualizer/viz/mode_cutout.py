"""Mode 12 -- Grid Cutout.

The Hiding Grid with a word carved through the middle of it.

Two things have to be true for the type to read, and both were wrong in the
obvious implementation of this:

Subtracting from the field cannot draw a shape. The grid is already about
half empty by design, so squares removed inside the letterforms are
indistinguishable from the holes the noise makes everywhere else. The word
only appears if its cells are driven to a *constant* -- solid where the
ground is varied, or empty against a ground given a density floor. Contrast
against the field's texture is what carries the letters, not absence.

And type cannot survive the field's warp. The plain grid displaces nodes by
up to 3.5 cells, which at this resolution is wider than a letter stroke, so
the word shreds. This mode warps a fraction of that: enough for the letters
to breathe with the mesh, not enough to break them.

The lattice has to go too. Cells driven to nothing still sit inside a drawn
mesh, and a dark patch of lattice is exactly what the noise produces all
over the field anyway -- so an emptied word reads as more of the same. With
the lines cleared as well the word becomes a true hole, which no amount of
noise ever makes, and it separates from the ground instantly. The letter
*boundary* keeps its lines, because a node is only cleared when all four of
its cells are inside the word; that leaves the type finely outlined.
"""
from __future__ import annotations

import numpy as np

from .mode_grid import GridMode
from .mode_text import text_mask

# s = 1 is exactly the cell-filling square (centre 0, half-size 1), so this
# is the value that makes a letter cell solid
SOLID = 1.0
# The ground never drops below this share of a cell, so the field stays
# dense enough to read as a ground. It is deliberately modest: the void is
# distinguished mainly by having no lattice in it, not by being darker, so
# this does not have to be pushed high enough to flatten the field into a
# sheet.
GROUND_FLOOR = 0.30


def cell_text_mask(text: str, cols: int, scale: float = 0.78) -> np.ndarray:
    """Rasterize `text` onto a cols x cols grid, 1 inside the letterforms."""
    from PIL import Image

    m = text_mask(text)
    h, w = m.shape
    # fit to `scale` of the width, then to the height if that binds first --
    # a wrapped two-line word is close to square
    tw = max(1, int(round(cols * scale)))
    th = max(1, int(round(tw * h / w)))
    if th > cols * scale:
        th = max(1, int(round(cols * scale)))
        tw = max(1, int(round(th * w / h)))

    img = Image.fromarray((m * 255).astype(np.uint8)).resize(
        (tw, th), Image.LANCZOS)
    glyph = np.asarray(img) > 110

    out = np.zeros((cols, cols), bool)
    y0, x0 = (cols - th) // 2, (cols - tw) // 2
    out[y0:y0 + th, x0:x0 + tw] = glyph
    # the mask's row 0 is the top of the image; the grid's row 0 sits at
    # y = -1, which the panzoom view puts at the bottom
    return out[::-1]


class GridCutoutMode(GridMode):
    name = "Grid Cutout"
    cols_attr = "cutout_cols"
    # a fraction of the plain grid's 3.5 cells: more than this and the
    # letter strokes come apart
    warp_cells = 1.15

    def build(self):
        super().build()
        self._mask_key = None
        self._word = None
        self._node_key = None
        self._nodes = None

    def _word_mask(self, cols: int) -> np.ndarray:
        s = self.settings
        key = (s.text_content, cols, float(s.cutout_scale))
        if key != self._mask_key:
            self._word = cell_text_mask(s.text_content, cols,
                                        float(s.cutout_scale))
            self._mask_key = key
        return self._word

    def node_mask(self, n: int):
        """Clear the lattice inside the letters, keeping their outline.

        A node is only cleared when all four cells around it are word cells,
        so the boundary nodes survive and draw a thin edge along the type.
        """
        if self.settings.cutout_invert:
            return None
        w = self._word_mask(n - 1)
        # returned by identity so the caller can cache its edge gather
        if self._node_key != (self._mask_key, n):
            keep = np.ones((n, n), np.float32)
            interior = w[:-1, :-1] & w[:-1, 1:] & w[1:, :-1] & w[1:, 1:]
            keep[1:-1, 1:-1] = ~interior
            self._nodes = keep
            self._node_key = (self._mask_key, n)
        return self._nodes

    def shape_scale(self, s, cols: int, swell: float):
        word = self._word_mask(cols)
        if self.settings.cutout_invert:
            # word solid against the untouched field; it keeps a little of
            # the beat so the type breathes without losing its edges
            lit = SOLID * float(np.clip(0.88 + 0.12 * swell, 0.5, 1.0))
            return np.where(word, lit, s)
        # word void. The ground is given a density floor -- still varied,
        # never small -- so the hole is unmistakably a hole
        # |s| past 1 already overflows the cell, so clipping there spreads
        # the ground's variation across the range that is actually visible
        ground = GROUND_FLOOR + (1.0 - GROUND_FLOOR) * np.clip(np.abs(s), 0, 1)
        return np.where(word, 0.0, ground)
