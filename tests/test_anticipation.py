import numpy as np

from visualizer.audio.beatgrid import BeatGrid


def _grid(sections):
    return BeatGrid(sections=np.asarray(sections, float))


def test_time_to_next_section():
    g = _grid([10.0, 25.0])
    assert g.time_to_next_section(0.0) == 10.0
    assert g.time_to_next_section(9.5) == 0.5
    assert g.time_to_next_section(10.0) == 15.0     # boundary already passed
    assert g.time_to_next_section(30.0) == float("inf")
    assert _grid([]).time_to_next_section(5.0) == float("inf")


def test_anticipation_ramps_then_releases():
    g = _grid([10.0])
    assert g.anticipation(0.0) == 0.0        # far away: no tension
    assert g.anticipation(6.9) == 0.0        # outside the 3s lead
    mid = g.anticipation(8.5)
    late = g.anticipation(9.7)
    assert 0.0 < mid < late < 1.0            # monotonic build
    assert g.anticipation(10.0) == 0.0       # released after the boundary


def test_anticipation_is_eased_not_linear():
    g = _grid([10.0])
    # a 1.6 exponent means the ramp starts gently and accelerates
    half = g.anticipation(8.5)               # exactly half the 3s lead
    assert half < 0.5


def test_anticipation_lead_is_configurable():
    g = _grid([10.0])
    assert g.anticipation(6.0, lead=5.0) > 0.0
    assert g.anticipation(6.0, lead=2.0) == 0.0


def test_no_sections_never_anticipates():
    g = _grid([])
    assert g.anticipation(5.0) == 0.0
