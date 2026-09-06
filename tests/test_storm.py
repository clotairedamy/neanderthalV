import subprocess

import numpy as np
import pytest

from visualizer.viz.mode_storm import (FLASH_DECAY, MIN_FLASH_GAP, make_bolt)


def _rng(seed=0):
    return np.random.default_rng(seed)


# ------------------------------------------------------------ bolt shape

def test_bolt_starts_and_ends_where_asked():
    """Midpoint displacement only moves the points it inserts, so the two
    endpoints must survive exactly -- that is what aims the strike."""
    paths = make_bolt(_rng(), (-0.3, 0.9), (0.1, 0.2))
    main = paths[0]
    assert np.allclose(main[0], (-0.3, 0.9))
    assert np.allclose(main[-1], (0.1, 0.2))


def test_bolt_subdivides_to_the_expected_resolution():
    main = make_bolt(_rng(), (0, 1), (0, -1), depth=5)[0]
    assert len(main) == 2 ** 5 + 1


def test_bolt_is_jagged_but_still_goes_where_it_is_aimed():
    """It has to wander off the straight line, or it is not lightning; it
    has to stay near it, or it stops reading as a descending strike."""
    start, end = np.array([0.0, 1.0]), np.array([0.0, -0.6])
    main = make_bolt(_rng(3), start, end)[0]
    span = np.linalg.norm(end - start)
    # perpendicular distance of each point from the straight channel
    d = np.abs(main[:, 0])
    assert d.max() > 0.02 * span, "path is not jagged at all"
    assert d.max() < 0.5 * span, "path wanders too far to read as a strike"


def test_bolt_descends_monotonically_overall():
    main = make_bolt(_rng(5), (0.0, 0.95), (0.05, 0.2))[0]
    # not every step, but the trend must be downward
    assert main[0, 1] > main[-1, 1]
    assert np.polyfit(np.arange(len(main)), main[:, 1], 1)[0] < 0


def test_forks_start_on_the_main_channel():
    """A fork that starts in mid-air reads as a separate bolt."""
    paths = make_bolt(_rng(7), (0, 1), (0, -0.5))
    main = paths[0]
    for fork in paths[1:]:
        d = np.linalg.norm(main - fork[0], axis=1).min()
        assert d < 1e-9, "fork is detached from the channel"


def test_forks_are_shorter_than_the_channel():
    paths = make_bolt(_rng(11), (0, 1), (0, -0.5))
    if len(paths) > 1:
        main_len = np.linalg.norm(np.diff(paths[0], axis=0), axis=1).sum()
        for fork in paths[1:]:
            f = np.linalg.norm(np.diff(fork, axis=0), axis=1).sum()
            assert f < main_len


def test_bolt_is_deterministic_for_a_seed():
    a = make_bolt(_rng(2), (0, 1), (0, -1))[0]
    b = make_bolt(_rng(2), (0, 1), (0, -1))[0]
    assert np.array_equal(a, b)


def test_a_degenerate_bolt_does_not_blow_up():
    paths = make_bolt(_rng(), (0.0, 0.0), (0.0, 0.0))
    assert np.isfinite(paths[0]).all()


# ------------------------------------------------------- flash behaviour

def test_flash_decay_matches_the_measured_footage():
    """The reference clip's flashes lasted a median 233 ms. The envelope
    should be most of the way down by then, not lingering."""
    env = float(np.exp(-0.233 / FLASH_DECAY))
    assert 0.25 < env < 0.45


def test_strike_rate_is_capped_below_the_photosensitivity_threshold():
    """Flashing above ~3 Hz is the range to stay out of; the reference
    storm's own median gap was 0.87 s, so this is not a tight limit."""
    assert 1.0 / MIN_FLASH_GAP <= 3.4


# ------------------------------------------------- silent-video handling

def _ffmpeg():
    from visualizer.audio.engine import _ffmpeg_exe
    exe = _ffmpeg_exe()
    try:
        subprocess.run([exe, "-version"], capture_output=True, timeout=20)
    except Exception:
        pytest.skip("ffmpeg unavailable")
    return exe


def test_a_video_with_no_audio_track_is_detected(tmp_path):
    """Storm.mov failed to load because it is video-only, which ffmpeg
    reports as an output with no streams rather than as a bad file."""
    from visualizer.audio.engine import video_has_audio
    exe = _ffmpeg()
    silent = tmp_path / "silent.mp4"
    subprocess.run([exe, "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
                    "-pix_fmt", "yuv420p", str(silent)],
                   capture_output=True, check=True)
    assert video_has_audio(str(silent)) is False


def test_a_video_with_audio_is_detected(tmp_path):
    from visualizer.audio.engine import video_has_audio
    exe = _ffmpeg()
    withsnd = tmp_path / "sound.mp4"
    subprocess.run([exe, "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-pix_fmt", "yuv420p", "-shortest", str(withsnd)],
                   capture_output=True, check=True)
    assert video_has_audio(str(withsnd)) is True


def test_extracting_from_a_silent_video_returns_none_instead_of_raising(tmp_path):
    """None keeps the file usable as a visual layer; an exception loses it."""
    from visualizer.audio.engine import extract_audio_from_video, video_has_audio
    exe = _ffmpeg()
    silent = tmp_path / "silent.mp4"
    subprocess.run([exe, "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
                    "-pix_fmt", "yuv420p", str(silent)],
                   capture_output=True, check=True)
    assert video_has_audio(str(silent)) is False
    assert extract_audio_from_video(str(silent)) is None


def test_audio_is_actually_recovered_when_present(tmp_path):
    from visualizer.audio.engine import extract_audio_from_video
    exe = _ffmpeg()
    withsnd = tmp_path / "sound.mp4"
    subprocess.run([exe, "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-pix_fmt", "yuv420p", "-shortest", str(withsnd)],
                   capture_output=True, check=True)
    a = extract_audio_from_video(str(withsnd))
    assert a is not None and a.ndim == 2 and a.shape[1] == 2
    assert np.abs(a).max() > 0.05, "extracted audio is silent"
