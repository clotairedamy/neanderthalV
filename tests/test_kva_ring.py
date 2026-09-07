import numpy as np
import pytest

from visualizer.config import asset_path
from visualizer.viz.mode_kva import (BANDS, R_IN, R_OUT, mirrored_bins,
                                     ring_points)
from visualizer.viz.svg_logo import crop_to_ink, logo_mask


# ------------------------------------------------------------ the artwork

def _page(h=200, w=200):
    """A mostly-empty page with a small mark on it, like the export."""
    page = np.full((h, w), 255.0, np.float32)
    page[40:60, 30:120] = 10.0
    return page


def test_the_empty_page_is_cropped_away():
    """Inkscape exports carry the sheet they were drawn on: the KVA file is
    A4 with the graphic in 41% of the width and 11% of the height."""
    out = crop_to_ink(_page(), pad=0.0)
    assert out.shape == (20, 90)


def test_cropping_keeps_all_of_the_ink():
    before = (_page() < 200).sum()
    assert (crop_to_ink(_page(), pad=0.0) < 200).sum() == before


def test_a_margin_can_be_kept():
    tight = crop_to_ink(_page(), pad=0.0).shape
    padded = crop_to_ink(_page(), pad=0.1).shape
    assert padded[0] > tight[0] and padded[1] > tight[1]


def test_a_blank_page_is_left_alone():
    blank = np.full((50, 50), 255.0, np.float32)
    assert crop_to_ink(blank).shape == (50, 50)


def test_the_bundled_logo_loads_and_is_wordshaped():
    m = logo_mask(asset_path("kva.svg"), px=600)
    assert m.any(), "no ink found in the logo"
    h, w = m.shape
    assert 1.8 < w / h < 3.5, f"unexpected logo aspect {w / h:.2f}"
    # the ring around it is drawn far lighter, so it must not be picked up
    assert 0.2 < m.mean() < 0.7, m.mean()


def test_the_logo_crop_drops_most_of_the_page():
    from visualizer.viz.svg_logo import render_svg
    full = render_svg(asset_path("kva.svg"), px=600)
    m = logo_mask(asset_path("kva.svg"), px=600)
    assert m.size < 0.25 * full.size


# --------------------------------------------------------------- the ring

def test_the_ring_sits_in_the_measured_annulus():
    T, R = ring_points(64, 5)
    assert R.shape == (5, 64) == T.shape
    assert R.min() == pytest.approx(R_IN)
    assert R.max() == pytest.approx(R_OUT)


def test_the_ring_closes_without_a_duplicate_sector():
    T, _ = ring_points(64, 3)
    assert T[0, 0] == 0.0
    assert T[0, -1] < 2 * np.pi          # endpoint excluded, or it doubles up


@pytest.mark.parametrize("band", list(BANDS))
def test_every_band_maps_inside_the_spectrum(band):
    lo, hi = BANDS[band]
    b = mirrored_bins(128, lo, hi)
    assert b.min() >= lo and b.max() <= max(hi - 1, lo)
    assert b.min() >= 0 and b.max() <= 63


def test_the_spectrum_is_mirrored_so_the_ring_has_no_seam():
    """Running the bins straight round the circle butts bin 63 against
    bin 0, which shows as a hard edge."""
    b = mirrored_bins(128, 0, 64)
    assert np.array_equal(b[:64], b[64:][::-1])


def test_the_mapping_fills_the_requested_sector_count():
    for n in (24, 63, 128, 360):
        assert len(mirrored_bins(n, 0, 64)) == n


def test_the_bands_partition_the_spectrum_in_order():
    assert BANDS["lows"][0] == 0
    assert BANDS["highs"][1] == 64
    assert BANDS["lows"][1] <= BANDS["mids"][0]
    assert BANDS["mids"][1] <= BANDS["highs"][0]


def test_narrow_bands_still_span_the_ring():
    """Selecting 'highs' must spread those bins around the whole circle,
    not leave most of it dead."""
    b = mirrored_bins(180, *BANDS["highs"])
    assert len(np.unique(b)) > 5
