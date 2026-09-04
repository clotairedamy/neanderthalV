import numpy as np
import pytest

from visualizer.viz.svg_rosette import (FAMILY, load_family, load_rosette,
                                        morph, outer_angles)
from visualizer.config import asset_path


@pytest.fixture(scope="module")
def family():
    return load_family()


def test_family_loads_all_three_files(family):
    assert family.shape == (len(FAMILY), 262, 2, 2)
    assert np.all(np.isfinite(family))


def test_shapes_are_centred_and_normalized(family):
    """The mode nests the rings by scale, so each must arrive at radius 1."""
    for shape in family:
        outer = np.linalg.norm(shape[:, 0, :], axis=1)
        assert np.allclose(outer, 1.0, atol=1e-4)


def test_inner_radius_is_what_distinguishes_the_files(family):
    """The morph is only exact because the files differ in one scalar."""
    radii = [np.linalg.norm(s[:, 1, :], axis=1) for s in family]
    for r in radii:
        assert np.allclose(r, r[0], atol=1e-4)      # constant within a file
    means = [float(r.mean()) for r in radii]
    assert means[0] < means[1] < means[2]           # Lows -> mids -> higs


def test_chords_correspond_across_the_family(family):
    """Blending is vertex-for-vertex, so outer endpoints must line up."""
    for shape in family[1:]:
        assert np.allclose(shape[:, 0, :], family[0][:, 0, :], atol=1e-5)


def test_morph_hits_the_source_shapes(family):
    for i in range(len(family)):
        assert np.allclose(morph(family, i), family[i], atol=1e-6)


def test_morph_interpolates_the_inner_radius_linearly(family):
    """A blend of two rosettes is the rosette with the blended inner radius --
    the property the whole mode is built on."""
    r0 = np.linalg.norm(family[0][:, 1, :], axis=1).mean()
    r1 = np.linalg.norm(family[1][:, 1, :], axis=1).mean()
    for t in (0.25, 0.5, 0.75):
        got = np.linalg.norm(morph(family, t)[:, 1, :], axis=1)
        assert np.allclose(got, r0 + t * (r1 - r0), atol=1e-4)


def test_morph_extrapolates_below_the_first_shape(family):
    """Energy pushes the bass ring past Lows.svg; it must stay a rosette."""
    s = morph(family, -0.8)
    inner = np.linalg.norm(s[:, 1, :], axis=1)
    assert inner.min() > 0.0
    assert np.allclose(inner, inner[0], atol=1e-4)
    assert np.allclose(s[:, 0, :], family[0][:, 0, :], atol=1e-5)


def test_outer_angles_are_evenly_spaced(family):
    """Chord angle picks the spectrum bin, so the bins must be evenly covered."""
    a = np.sort(outer_angles(family[0]))
    assert 0.0 <= a[0] and a[-1] < 1.0
    assert np.allclose(np.diff(a), 1.0 / len(a), atol=1e-5)


def test_loader_recentres_and_flips_y(tmp_path):
    """A redrawn export may be offset; the loader must normalize it."""
    svg = tmp_path / "t.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<g transform="translate(50, 50)">'
        '<line x1="10" y1="0" x2="0" y2="0"/>'
        '<line x1="-10" y1="0" x2="0" y2="0"/>'
        '<line x1="0" y1="10" x2="0" y2="0"/></g></svg>')
    s = load_rosette(str(svg))
    assert s.shape == (3, 2, 2)
    assert np.isclose(np.linalg.norm(s[:, 0, :], axis=1).max(), 1.0)
    # SVG y points down, the scene's up: the third chord must end up at -1
    assert np.isclose(s[2, 0, 1], -1.0)


def test_loader_rejects_a_file_with_no_lines(tmp_path):
    svg = tmp_path / "empty.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"/>')
    with pytest.raises(ValueError):
        load_rosette(str(svg))


def test_assets_are_present():
    """The mode has no fallback shape, so a missing file is a hard failure."""
    import os
    for n in FAMILY:
        assert os.path.exists(asset_path(n)), n
