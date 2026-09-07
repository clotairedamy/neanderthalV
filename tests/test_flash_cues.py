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

    def __init__(self, cues, gap=0.22, on=True, strength=0.45):
        class S:
            video_beat_cue = on
            video_cue_gap = gap
            video_cue_strength = strength
        self.settings = S()
        self.video = None          # no clip: nothing to wrap the cue time to
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


def test_a_beat_jumps_the_video_to_a_flash():
    m = _Stub(CUES)
    m._video_time(0.0, _F(), 1 / 60)              # settle
    t = m._video_time(1.0, _F(beat=True, punch=1.0, strength=0.9), 1 / 60)
    assert t in [c["t"] for c in CUES]


def test_a_transient_off_the_beat_does_not_cut():
    """The low-band 'punch' averages ~0.67 on real material and crosses any
    useful threshold about twenty times a second. Triggering on it made the
    cuts free-run at the rate limit instead of locking to the music."""
    m = _Stub(CUES)
    m._video_time(0.0, _F(), 1 / 60)
    before = m._cue_time
    after = m._video_time(1.0, _F(beat=False, punch=1.0), 1 / 60)
    assert after == pytest.approx(before + 1 / 60, abs=1e-9)


def test_a_weak_beat_does_not_cut():
    m = _Stub(CUES, strength=0.6)
    m._video_time(0.0, _F(), 1 / 60)
    before = m._cue_time
    after = m._video_time(1.0, _F(beat=True, strength=0.3), 1 / 60)
    assert after == pytest.approx(before + 1 / 60, abs=1e-9)


def test_the_strength_threshold_selects_how_many_beats_cut():
    def cuts(threshold):
        m = _Stub(CUES, gap=0.0, strength=threshold)
        m._video_time(0.0, _F(), 1 / 60)
        n = 0
        for i, st in enumerate([0.2, 0.5, 0.9, 0.4, 0.7]):
            before = m._cue_time
            got = m._video_time(1.0 + i, _F(beat=True, strength=st), 1 / 60)
            n += abs(got - (before + 1 / 60)) > 1e-9
            m._cue_time = got
        return n
    assert cuts(0.0) == 5
    assert cuts(0.45) == 3
    assert cuts(0.8) == 1


def test_a_loud_drum_can_carry_a_beat_the_detector_rated_weakly():
    m = _Stub(CUES, strength=0.5)
    m._video_time(0.0, _F(), 1 / 60)
    f = _F(beat=True, strength=0.2)
    f.stem_energy = {"drums": 0.95}
    m._cue_drum_ema = 0.0
    t = m._video_time(1.0, f, 1 / 60)
    assert t in [c["t"] for c in CUES]


def test_a_loud_drum_between_beats_still_does_not_cut():
    """Drums select among beats; they never invent one."""
    m = _Stub(CUES, strength=0.5)
    m._video_time(0.0, _F(), 1 / 60)
    f = _F(beat=False)
    f.stem_energy = {"drums": 0.99}
    m._cue_drum_ema = 0.0
    before = m._cue_time
    assert m._video_time(1.0, f, 1 / 60) == pytest.approx(before + 1 / 60, abs=1e-9)


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


# --------------------------------------------------- video orientation

from visualizer.viz.manager import background_transform


def _box(w, h, mode="background", aspect=1.0):
    """Unit-space box the frame occupies: (width, height, top_y, bottom_y)."""
    (sx, sy), (tx, ty) = background_transform(w, h, mode, aspect)
    return w * sx, -h * sy, ty, ty + h * sy


@pytest.mark.parametrize("mode", ["background", "pip"])
def test_the_top_of_the_frame_lands_above_the_bottom(mode):
    """An Image lays row 0 at y = 0 and the background camera has y going
    up, so an unflipped transform plays every video upside down."""
    _, _, top, bottom = _box(874, 874, mode, 1.5)
    assert top > bottom


@pytest.mark.parametrize("mode", ["background", "pip"])
def test_the_frame_is_flipped_not_merely_offset(mode):
    (sx, sy), _ = background_transform(640, 480, mode, 1.5)
    assert sy < 0 < sx


@pytest.mark.parametrize("w,h", [(874, 874), (1920, 1080), (1080, 1920),
                                 (640, 480), (2560, 1080)])
@pytest.mark.parametrize("aspect", [0.6, 1.0, 1.5, 2.4])
def test_the_clip_keeps_its_own_proportions(w, h, aspect):
    """The camera maps the unit square to the whole canvas, so filling that
    square stretches the clip to the window's shape."""
    u, v, _, _ = _box(w, h, "background", aspect)
    assert (u * aspect) / v == pytest.approx(w / h, rel=1e-9)


@pytest.mark.parametrize("aspect", [0.6, 1.0, 1.5, 2.4])
def test_the_letterboxed_clip_fits_and_is_centred(aspect):
    u, v, top, bottom = _box(874, 874, "background", aspect)
    assert u <= 1.0 + 1e-9 and v <= 1.0 + 1e-9
    assert max(u, v) == pytest.approx(1.0)      # touches the frame
    (_, _), (tx, _) = background_transform(874, 874, "background", aspect)
    assert tx == pytest.approx((1.0 - u) / 2)   # centred horizontally
    assert bottom == pytest.approx(1.0 - top)   # and vertically


def test_a_matching_aspect_fills_the_canvas_exactly():
    u, v, top, bottom = _box(1500, 1000, "background", 1.5)
    assert (u, v) == pytest.approx((1.0, 1.0))
    assert (bottom, top) == pytest.approx((0.0, 1.0))


def test_pip_keeps_its_proportions_and_stays_on_canvas():
    for aspect in (0.7, 1.5, 2.4):
        u, v, top, bottom = _box(874, 874, "pip", aspect)
        assert (u * aspect) / v == pytest.approx(1.0, rel=1e-9)
        (_, _), (tx, _) = background_transform(874, 874, "pip", aspect)
        assert 0.0 <= tx and tx + u <= 1.0
        assert 0.0 <= bottom < top <= 1.0


def test_orientation_holds_for_any_frame_size():
    for w, h in ((16, 9), (874, 874), (1080, 1920), (3840, 2160)):
        _, _, top, bottom = _box(w, h, "background", 1.5)
        assert top > bottom


def test_the_cue_time_wraps_inside_the_clip():
    """A clip used as a visual layer outlives itself whenever the track is
    longer; letting the cue time run past the end pinned it to the last
    frame, which reads as the video freezing."""
    class _Clip:
        duration = 10.0
    m = _Stub(CUES)
    m.video = _Clip()
    m._video_time(0.0, _F(), 1 / 60)
    m._cue_time = 9.999
    t = m._video_time(1.0, _F(), 1 / 60)
    assert 0.0 <= t < 10.0
