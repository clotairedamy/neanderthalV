import numpy as np

from visualizer.viz.mode_text import tessellate, text_mask, wrap_text


def test_long_text_wraps_to_stay_legible():
    """A long word must wrap, or its letters shrink to an unreadable stripe."""
    assert "\n" not in wrap_text("DROP")
    assert "\n" in wrap_text("NEANDERTHALV")
    # breaking at a real space is preferred over mid-word
    assert wrap_text("HELLO WORLD") == "HELLO\nWORLD"


def test_mask_is_rendered_and_wrapped():
    m = text_mask("DROP")
    assert m.dtype == bool and m.any()
    wide = text_mask("DROP")
    tall = text_mask("NEANDERTHALV")
    # the wrapped word is proportionally taller than the single-line one
    assert (tall.shape[0] / tall.shape[1]) > (wide.shape[0] / wide.shape[1])


def test_tessellate_geometry_is_well_formed():
    pos, cent, seed, norm, u, n_cells, cell = tessellate(text_mask("DROP"))
    assert n_cells > 50
    assert len(pos) == n_cells * 36          # 6 faces x 2 tris x 3 verts
    for arr in (cent, seed, norm):
        assert len(arr) == len(pos)
    assert len(u) == len(pos)
    assert cell > 0
    assert np.all(np.isfinite(pos))


def test_chunks_fit_the_target_box():
    pos, *_ = tessellate(text_mask("DROP"), target_w=3.0, target_h=1.7)
    assert pos[:, 0].max() - pos[:, 0].min() <= 3.2      # + chunk half-width
    assert pos[:, 2].max() - pos[:, 2].min() <= 1.9


def test_chunk_count_respects_budget():
    _, _, _, _, _, few, _ = tessellate(text_mask("DROP"), max_cells=200)
    _, _, _, _, _, many, _ = tessellate(text_mask("DROP"), max_cells=3000)
    assert few < many


def test_seeds_are_unit_length_per_chunk():
    """Explosion axes must be normalized or chunks fly at random speeds."""
    _, _, seed, _, _, _, _ = tessellate(text_mask("AB"))
    assert np.allclose(np.linalg.norm(seed, axis=1), 1.0, atol=1e-4)


def test_u_maps_left_to_right():
    """Horizontal position drives the frequency band, so u must track x."""
    pos, _, _, _, u, _, _ = tessellate(text_mask("ABCDEF"))
    assert np.corrcoef(pos[:, 0], u)[0, 1] > 0.99
    assert 0.0 <= u.min() and u.max() <= 1.0


def test_normals_are_unit_length():
    _, _, _, norm, _, _, _ = tessellate(text_mask("O"))
    assert np.allclose(np.linalg.norm(norm, axis=1), 1.0, atol=1e-5)


def test_empty_text_falls_back():
    pos, *_ = tessellate(text_mask(""))
    assert len(pos) > 0
