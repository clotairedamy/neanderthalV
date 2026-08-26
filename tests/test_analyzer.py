import numpy as np

from visualizer.audio.analyzer import FFT_SIZE, AudioAnalyzer

SR = 44100


def _frames(analyzer, mono, seconds, hz=30):
    out = []
    for i in range(int(seconds * hz)):
        t = i / hz
        s = int(t * SR)
        w = mono[s:s + FFT_SIZE]
        if len(w) < FFT_SIZE:
            break
        out.append(analyzer.analyze(t, w))
    return out


def test_beat_detection_and_bpm(test_song, settings):
    an = AudioAnalyzer(SR, settings)
    frames = _frames(an, test_song.mean(axis=1), 8)
    beats = sum(f.beat for f in frames)
    assert beats >= 5
    assert 100 <= frames[-1].bpm <= 140


def test_dynamic_range_and_punch(test_song, settings):
    an = AudioAnalyzer(SR, settings)
    frames = _frames(an, test_song.mean(axis=1), 8)[30:]   # skip warmup
    bands = np.array([f.bands for f in frames])
    rms = np.array([f.rms for f in frames])
    # adaptive floor/ceiling: bass band must span a wide range
    assert bands[:, 1].max() - bands[:, 1].min() > 0.35
    assert rms.max() - rms.min() > 0.3
    punches = sum(f.punch > 0.3 for f in frames)
    assert punches >= 10           # kicks at 2/s plus hats over 7 s


def test_frame_shapes(test_song, settings):
    an = AudioAnalyzer(SR, settings)
    f = an.analyze(0.0, test_song.mean(axis=1)[:FFT_SIZE])
    assert f.bands.shape == (7,)
    assert f.attack.shape == (7,)
    assert f.spectrum.shape == (64,)
    assert 0 <= f.centroid <= 1
    assert 0 <= f.punch <= 1
