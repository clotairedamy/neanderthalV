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


# ------------------------------------------------ point-cloud style

def test_points_land_inside_the_letterforms():
    from visualizer.viz.mode_text import point_cloud, text_mask
    mask = text_mask("BASS")
    pos, sd, u, edge = point_cloud(mask, n_points=4000)
    h, w = mask.shape
    scale = min(3.0 / w, 1.7 / h)
    # invert the mapping back to pixel coordinates
    px = np.clip(np.rint(pos[:, 0] / scale + (w - 1) / 2).astype(int), 0, w - 1)
    py = np.clip(np.rint(-pos[:, 2] / scale + (h - 1) / 2).astype(int), 0, h - 1)
    inside = mask[py, px].mean()
    assert inside > 0.95, f"only {inside:.0%} of points are on the glyphs"


def test_sampling_is_weighted_toward_the_contours():
    """Uniform area sampling is what made the first point-cloud text
    unreadable; the contour bias is the whole fix."""
    from visualizer.viz.mode_text import point_cloud, text_mask
    pos, sd, u, edge = point_cloud(text_mask("BASS"), n_points=20000)
    # more points near the boundary than deep inside the strokes
    assert (edge > 0.5).sum() > (edge < 0.15).sum()


def test_edge_drives_size_and_brightness_over_a_usable_range():
    from visualizer.viz.mode_text import point_cloud, text_mask
    _, _, _, edge = point_cloud(text_mask("BASS"), n_points=8000)
    assert 0.0 <= edge.min() and edge.max() <= 1.0
    assert edge.max() - edge.min() > 0.4


def test_point_cloud_is_deterministic():
    from visualizer.viz.mode_text import point_cloud, text_mask
    m = text_mask("BASS")
    a = point_cloud(m, n_points=2000)[0]
    b = point_cloud(m, n_points=2000)[0]
    assert np.array_equal(a, b)


def test_point_cloud_matches_the_tessellated_extent():
    """Both styles must occupy the same box, or switching between them
    would jump the framing."""
    from visualizer.viz.mode_text import point_cloud, tessellate, text_mask
    m = text_mask("BASS")
    tp = tessellate(m)[0]
    pp = point_cloud(m, n_points=20000)[0]
    for ax in (0, 2):
        assert abs(pp[:, ax].max() - tp[:, ax].max()) < 0.12
        assert abs(pp[:, ax].min() - tp[:, ax].min()) < 0.12


def test_empty_text_still_produces_a_cloud():
    from visualizer.viz.mode_text import point_cloud, text_mask
    pos, sd, u, edge = point_cloud(text_mask(""), n_points=500)
    assert len(pos) == 500 and np.isfinite(pos).all()
