import numpy as np

from visualizer.viz.mode_text import text_points


def test_text_points_shape_and_range():
    pts, u = text_points("NEANDERTHALV")
    assert len(pts) > 500
    assert pts.shape[1] == 2
    assert len(u) == len(pts)
    # fitted into the target box (3.0 x 1.7 world units), centred
    assert pts[:, 0].max() - pts[:, 0].min() <= 3.05
    assert pts[:, 1].max() - pts[:, 1].min() <= 1.75
    assert abs(pts[:, 0].mean()) < 0.6 and abs(pts[:, 1].mean()) < 0.6
    assert 0.0 <= u.min() and u.max() <= 1.0


def test_wide_text_is_wider_than_tall():
    pts, _ = text_points("WIDEWORD")
    span_x = pts[:, 0].max() - pts[:, 0].min()
    span_y = pts[:, 1].max() - pts[:, 1].min()
    assert span_x > span_y


def test_u_tracks_horizontal_position():
    pts, u = text_points("ABCDEF")
    # u must increase with x so the word maps left-to-right onto bands
    order = np.argsort(pts[:, 0])
    assert np.corrcoef(pts[order, 0], u[order])[0, 1] > 0.99


def test_point_budget_respected():
    pts, _ = text_points("LONGER TEXT HERE", max_points=2000)
    assert len(pts) <= 2000


def test_empty_text_falls_back():
    pts, _ = text_points("")
    assert len(pts) > 0          # falls back to a default word, never blank


def test_contours_not_solid_fill():
    """Points should trace letterforms, not fill them solid."""
    pts, _ = text_points("O", max_points=12000)
    # an 'O' traced as contours leaves its centre empty
    cx = (pts[:, 0].max() + pts[:, 0].min()) / 2
    cy = (pts[:, 1].max() + pts[:, 1].min()) / 2
    r = 0.12 * (pts[:, 0].max() - pts[:, 0].min())
    near_centre = np.sum((np.abs(pts[:, 0] - cx) < r)
                         & (np.abs(pts[:, 1] - cy) < r))
    assert near_centre / len(pts) < 0.05


def test_long_text_wraps_to_stay_legible():
    """A long word must wrap, or its letters shrink to an unreadable stripe."""
    from visualizer.viz.mode_text import wrap_text
    assert "\n" not in wrap_text("DROP")
    assert "\n" in wrap_text("NEANDERTHALV")
    # breaking at a real space is preferred over mid-word
    assert wrap_text("HELLO WORLD") == "HELLO\nWORLD"
    short = text_points("DROP")[0]
    long_ = text_points("NEANDERTHALV")[0]
    h_short = short[:, 1].max() - short[:, 1].min()
    h_long = long_[:, 1].max() - long_[:, 1].min()
    # wrapping keeps the long word's glyphs comparable in height
    assert h_long > h_short * 0.9
