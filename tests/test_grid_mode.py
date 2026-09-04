import numpy as np
import pytest

from visualizer.viz.mode_grid import CENTRE_K, S_HI, S_LO, hiding_square
from visualizer.viz.noise import perlin3, permutation

# Real (scale, translate) pairs lifted from hiding-squares.svg, spanning the
# file's full range. The drawing nests scale(s) around translate(t, t) on a
# 50-unit cell, so the square's centre lands at s*t and its half-size at
# 25|s|; dividing by 25 puts both in a cell normalized to [-1, 1].
SOURCE_CELLS = [
    (-1.7552492145294951, 35.42463275823637),
    (0.9901905286408608, 0.12612177461750385),
    (-0.0009767525758643814, 12.869701104546829),
    (-0.08736772302407614, 13.980442153166695),
    (-0.9769323982956792, 25.41770226380159),
    (0.7617285794976614, 3.0634896921729258),
]


@pytest.mark.parametrize("s,t", SOURCE_CELLS)
def test_centre_formula_reproduces_the_source_file(s, t):
    """(18/35)(s - s^2) must equal the drawing's own s*t/25, or the mode is
    not reproducing the artwork it claims to."""
    assert CENTRE_K * (s - s * s) == pytest.approx(s * t / 25.0, abs=1e-9)


def test_square_is_clipped_to_its_cell():
    s = np.array([v for v, _ in SOURCE_CELLS])
    lo, hi = hiding_square(s)
    assert np.all(lo >= -1.0) and np.all(hi <= 1.0)


def test_a_square_pushed_far_enough_out_is_fully_hidden():
    """The hiding is the whole point: past some offset a cell goes empty."""
    s = np.linspace(S_LO * 1.6, S_HI, 400)
    lo, hi = hiding_square(s)
    assert np.any(lo >= hi), "no value of s ever hides a square"


def test_offset_is_a_fixed_fraction_of_the_square_size():
    """In the drawing a square is always shifted diagonally by ~51% of its
    own half-size, so small squares stay near their cell centre and only
    large ones can reach the edge and start hiding."""
    s = np.array([1e-3, 0.05, 0.4])
    lo, hi = hiding_square(s)
    centre, half = (lo + hi) / 2, (hi - lo) / 2
    assert np.allclose(half, s)
    assert np.allclose(centre / half, CENTRE_K * (1 - s), rtol=1e-9)


def test_extreme_scale_leaves_only_a_sliver():
    """s at the file's minimum leaves a thin band against one cell edge."""
    lo, hi = hiding_square(np.array([S_LO]))
    assert lo[0] == pytest.approx(-1.0)
    assert -1.0 < hi[0] < -0.5


def test_full_scale_fills_the_cell():
    lo, hi = hiding_square(np.array([1.0]))
    assert (lo[0], hi[0]) == pytest.approx((-1.0, 1.0))


# ------------------------------------------------------------------ noise

def test_noise_is_in_range_and_deterministic():
    p = permutation(3)
    g = np.linspace(0, 12, 64)
    x, y = np.meshgrid(g, g)
    a = perlin3(x, y, 0.5, p)
    assert a.min() >= -1.0 and a.max() <= 1.0
    assert np.array_equal(a, perlin3(x, y, 0.5, permutation(3)))


def test_noise_is_spatially_coherent_not_white():
    """The drawings' fields are smooth; white noise would not reproduce them."""
    p = permutation(3)
    g = np.linspace(0, 6, 80)
    x, y = np.meshgrid(g, g)
    a = perlin3(x, y, 0.0, p)
    corr = np.corrcoef(a[:, :-1].ravel(), a[:, 1:].ravel())[0, 1]
    assert corr > 0.9


def test_noise_scale_changes_feature_size():
    """`Noise scale` must actually change the field, not just reseed it."""
    p = permutation(3)
    g = np.linspace(0, 1, 96)
    x, y = np.meshgrid(g, g)
    def corr(k):
        a = perlin3(x * k, y * k, 0.0, p)
        return np.corrcoef(a[:, :-1].ravel(), a[:, 1:].ravel())[0, 1]
    assert corr(2.0) > corr(20.0)


def test_noise_moves_continuously_through_time():
    """The field flows; a discontinuity would read as a frame glitch."""
    p = permutation(3)
    g = np.linspace(0, 5, 40)
    x, y = np.meshgrid(g, g)
    a = perlin3(x, y, 1.0, p)
    b = perlin3(x, y, 1.0 + 1e-4, p)
    assert np.abs(a - b).max() < 1e-2


def test_scale_maps_over_the_files_observed_range():
    """s is mapped from noise onto the range measured in the SVG."""
    n = np.array([-1.0, 0.0, 1.0])
    s = S_HI - (n * 0.5 + 0.5) * (S_HI - S_LO)
    assert s[0] == pytest.approx(S_HI)
    assert s[-1] == pytest.approx(S_LO)
