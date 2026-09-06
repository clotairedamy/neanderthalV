import numpy as np
import pytest

from visualizer.video.flashes import JOIN, find_flashes


def _track(n=600, base=0.10, spikes=(), width=6, amp=0.9):
    """A steady brightness track with flashes injected at given frames."""
    t = np.full(n, base, np.float32)
    for s in spikes:
        t[s:s + width] = base * (1.0 + amp)
    return t


def test_finds_an_injected_flash_at_its_onset():
    f = find_flashes(_track(spikes=(120,)), fps=30.0)
    assert len(f) == 1
    assert f[0]["t"] == pytest.approx(120 / 30.0, abs=1e-6)


def test_reports_duration_and_peak():
    f = find_flashes(_track(spikes=(200,), width=7, amp=1.3), fps=30.0)[0]
    assert f["dur"] == pytest.approx(7 / 30.0, abs=1e-6)
    assert f["peak"] > 1.0


def test_adjacent_frames_are_one_flash_not_many():
    """A flash lasts several frames; each must not be reported separately."""
    assert len(find_flashes(_track(spikes=(150,), width=8), 30.0)) == 1


def test_separate_flashes_stay_separate():
    f = find_flashes(_track(n=900, spikes=(120, 400, 700)), 30.0)
    assert len(f) == 3


def test_flashes_closer_than_the_join_window_merge():
    n = 600
    t = np.full(n, 0.1, np.float32)
    t[200:204] = 0.2
    t[204 + JOIN - 1:208 + JOIN] = 0.2       # within the join gap
    assert len(find_flashes(t, 30.0)) == 1


def test_a_quiet_clip_reports_no_flashes():
    assert find_flashes(np.full(600, 0.12, np.float32), 30.0) == []


def test_drifting_footage_does_not_read_as_flashing():
    """A sunset dimming, or a camera re-exposing, moves the whole track.
    A global threshold would fire on that; a local baseline must not."""
    n = 900
    t = np.linspace(0.05, 0.45, n).astype(np.float32)   # slow 9x brighten
    assert find_flashes(t, 30.0) == []


def test_a_flash_is_still_found_on_top_of_a_drift():
    n = 900
    t = np.linspace(0.05, 0.45, n).astype(np.float32)
    t[500:506] *= 2.0
    f = find_flashes(t, 30.0)
    assert len(f) == 1 and f[0]["t"] == pytest.approx(500 / 30.0, abs=1e-6)


def test_a_track_shorter_than_the_baseline_window_is_handled():
    assert find_flashes(np.full(5, 0.2, np.float32), 30.0) == []


def test_onsets_come_back_in_order():
    f = find_flashes(_track(n=900, spikes=(700, 120, 400)), 30.0)
    ts = [x["t"] for x in f]
    assert ts == sorted(ts)


# ------------------------------------------------------- cue scheduling

class _Stub:
    """Just the state VizManager._video_time touches."""
    from visualizer.viz.manager import VizManager
    _video_time = VizManager._video_time
    _cue_hit = VizManager._cue_hit

    def __init__(self, cues, gap=0.22, on=True):
        class S:
            video_beat_cue = on
            video_cue_gap = gap
        self.settings = S()
        self.flash_cues = cues
        self._cue_time = None
        self._cue_i = -1
        self._last_cue = -99.0
        self._cue_drum_ema = 0.0


class _F:
    def __init__(self, beat=False, punch=0.0, strength=0.0):
        self.beat = beat
        self.punch = punch
        self.beat_strength = strength
        self.stem_energy = {}


CUES = [{"t": 2.0, "dur": 0.2, "peak": 1.0},
        {"t": 5.0, "dur": 0.2, "peak": 1.0},
        {"t": 9.0, "dur": 0.2, "peak": 1.0}]


def test_without_cueing_the_video_follows_the_audio_playhead():
    m = _Stub(CUES, on=False)
    assert m._video_time(12.34, _F(beat=True, punch=1.0), 1 / 60) == 12.34


def test_a_clip_with_no_flashes_falls_back_to_the_playhead():
    m = _Stub([])
    assert m._video_time(7.5, _F(beat=True, punch=1.0), 1 / 60) == 7.5


def test_a_drum_hit_jumps_the_video_to_a_flash():
    m = _Stub(CUES)
    m._video_time(0.0, _F(), 1 / 60)              # settle
    t = m._video_time(1.0, _F(beat=True, punch=1.0, strength=0.9), 1 / 60)
    assert t in [c["t"] for c in CUES]


def test_between_hits_the_video_plays_forward_from_the_cue():
    """Cutting to a flash and freezing there would lose the flash decay."""
    m = _Stub(CUES)
    t0 = m._video_time(1.0, _F(beat=True, punch=1.0, strength=0.9), 1 / 60)
    t1 = m._video_time(1.02, _F(), 1 / 60)
    assert t1 > t0
    assert t1 == pytest.approx(t0 + 1 / 60, abs=1e-9)


def test_cuts_are_rate_limited():
    m = _Stub(CUES, gap=0.5)
    hit = _F(beat=True, punch=1.0, strength=0.9)
    a = m._video_time(1.0, hit, 1 / 60)
    b = m._video_time(1.1, hit, 1 / 60)        # too soon: must not re-cut
    assert b == pytest.approx(a + 1 / 60, abs=1e-9)
    c = m._video_time(1.7, hit, 1 / 60)        # past the gap: cuts again
    assert c in [x["t"] for x in CUES]


def test_cues_advance_in_order_rather_than_repeating():
    m = _Stub(CUES, gap=0.1)
    hit = _F(beat=True, punch=1.0, strength=0.9)
    seen = [m._video_time(1.0 + i * 0.5, hit, 1 / 60) for i in range(3)]
    assert seen == [c["t"] for c in CUES]


def test_cue_list_wraps_around():
    m = _Stub(CUES, gap=0.1)
    hit = _F(beat=True, punch=1.0, strength=0.9)
    seen = [m._video_time(1.0 + i * 0.5, hit, 1 / 60) for i in range(4)]
    assert seen[3] == seen[0]


def test_the_drums_stem_is_preferred_over_the_kick_transient():
    """'On the drums' means the drums stem when it exists."""
    m = _Stub(CUES)
    f = _F(punch=0.0)
    f.stem_energy = {"drums": 0.9}
    m._cue_drum_ema = 0.0
    assert m._cue_hit(f) > 0.35          # fires on drums despite punch 0


def test_it_falls_back_to_the_kick_when_stems_are_not_ready():
    m = _Stub(CUES)
    assert m._cue_hit(_F(punch=0.8)) == pytest.approx(0.8)
