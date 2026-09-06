"""Regression tests for what the beat grid tracks, and how.

Both were found on a bass-driven storm-sound track that the grid called
120 BPM when it is 69: the cuts derived from it could not line up with
anything audible.
"""
import numpy as np
import pytest

from visualizer.audio.beatgrid import compute_beat_grid

SR = 22050


def _click_track(bpm, seconds=16.0, sr=SR, seed=0):
    """A dry pulse train: unambiguous ground truth for tempo."""
    n = int(seconds * sr)
    y = np.zeros(n, np.float32)
    period = 60.0 / bpm
    env = np.exp(-np.linspace(0, 8, int(0.05 * sr))).astype(np.float32)
    rng = np.random.default_rng(seed)
    click = env * rng.normal(0, 1, len(env)).astype(np.float32)
    for i in range(int(seconds / period)):
        s = int(i * period * sr)
        y[s:s + len(click)] += click[:max(0, n - s)]
    return y


def _low_pulse(bpm, seconds=16.0, sr=SR):
    """A sub-bass pulse: the groove where a track has no real drum hits."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    period = 60.0 / bpm
    env = np.exp(-((t % period) / 0.18)).astype(np.float32)
    return (np.sin(2 * np.pi * 55 * t).astype(np.float32) * env)


def _bpm_close(got, want, tol=0.06):
    """True if `got` matches `want`, allowing the usual octave ambiguity."""
    return any(abs(got - want * f) <= want * f * tol
               for f in (0.5, 1.0, 2.0))


def test_a_plain_click_track_gets_its_tempo():
    g = compute_beat_grid(_click_track(100.0), SR)
    assert _bpm_close(g.bpm, 100.0), g.bpm


@pytest.mark.parametrize("bpm", [69.0, 128.0])
def test_tempo_survives_broadband_noise(bpm):
    """Storm rumble, room tone and tape hiss are broadband. Averaging the
    onset envelope across frequency lets that noise swamp the onsets;
    taking the median across bands does not, which is why librosa's own
    beat_track builds its envelope that way.
    """
    rng = np.random.default_rng(1)
    clicks = _click_track(bpm)
    noise = rng.normal(0, 0.35, len(clicks)).astype(np.float32)
    g = compute_beat_grid(clicks + noise, SR)
    assert _bpm_close(g.bpm, bpm), g.bpm


def test_the_beat_comes_from_the_bass_when_the_drum_stem_is_junk():
    """Separation on non-drum material leaves a drums stem of artifacts.
    It is not silent, so a 'drums exist' test passes it, and tracking it
    returns a tempo unrelated to the track."""
    rng = np.random.default_rng(2)
    bass = _low_pulse(69.0)
    drums = rng.normal(0, 0.05, len(bass)).astype(np.float32)   # artifacts
    g = compute_beat_grid(bass + drums, SR, drums=drums, bass=bass)
    assert _bpm_close(g.bpm, 69.0), g.bpm


def test_a_junk_drum_stem_alone_would_have_been_trusted_before():
    """Guard the old condition explicitly: the artifacts are far above
    digital silence, so amplitude alone cannot tell them from a real kit."""
    rng = np.random.default_rng(2)
    drums = rng.normal(0, 0.05, int(4 * SR)).astype(np.float32)
    assert float(np.abs(drums).max()) > 1e-4


def test_a_real_drum_stem_still_drives_the_beat():
    """The fix must not stop a drum-led track being tracked on its drums."""
    drums = _click_track(128.0)
    bass = _low_pulse(128.0) * 0.3
    g = compute_beat_grid(drums + bass, SR, drums=drums, bass=bass)
    assert _bpm_close(g.bpm, 128.0), g.bpm


def test_beats_land_on_the_pulses():
    bpm = 100.0
    g = compute_beat_grid(_click_track(bpm), SR)
    period = 60.0 / bpm
    truth = np.arange(0, 16.0, period)
    off = np.abs(np.asarray(g.beats)[:, None] - truth[None, :]).min(1)
    assert np.median(off) < 0.08, np.median(off)


def test_the_grid_says_where_it_got_the_beat():
    g = compute_beat_grid(_click_track(100.0), SR)
    assert g.source == "mix"
    d = _click_track(100.0)
    g2 = compute_beat_grid(d, SR, drums=d)
    assert g2.source == "rhythm"


def test_beat_strengths_stay_in_range():
    g = compute_beat_grid(_click_track(110.0), SR)
    assert len(g.beat_strengths) == len(g.beats)
    assert g.beat_strengths.min() >= 0.25 - 1e-6
    assert g.beat_strengths.max() <= 1.0 + 1e-6
