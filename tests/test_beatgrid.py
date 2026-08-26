import numpy as np

from visualizer.audio.beatgrid import compute_beat_grid

SR = 44100


def test_grid_bpm_and_beats(test_song):
    g = compute_beat_grid(test_song.mean(axis=1), SR)
    assert 115 <= g.bpm <= 125
    assert 12 <= len(g.beats) <= 18
    # kicks are on an exact 0.5 s grid in the fixture
    off = np.abs((g.beats % 0.5) - np.where(g.beats % 0.5 > 0.25, 0.5, 0))
    assert off.max() < 0.05


def test_grid_events_and_scaling(test_song):
    g = compute_beat_grid(test_song.mean(axis=1), SR)
    b, p = g.events_between(g.beats[2] - 0.02, g.beats[2] + 0.02)
    assert b > 0
    b2, _ = g.events_between(g.beats[2] + 0.1, g.beats[2] + 0.2)
    assert b2 == 0
    gs = g.scaled(0.5)
    assert np.allclose(gs.beats, g.beats * 0.5)
    assert abs(gs.bpm - g.bpm * 2) < 1e-6


def test_onsets_present(test_song):
    g = compute_beat_grid(test_song.mean(axis=1), SR)
    assert len(g.onsets) >= 10
    assert g.onset_strengths.max() <= 1.0
