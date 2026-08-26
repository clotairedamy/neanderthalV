import numpy as np
import pytest
import soundfile as sf

SR = 44100


def test_pseudo_stems(test_song):
    from visualizer.audio.stems import _pseudo_stems
    stems = _pseudo_stems(test_song, SR)
    assert set(stems) == {"vocals", "drums", "bass", "other"}
    for d in stems.values():
        assert d.shape == test_song.shape
    e = {k: float(np.mean(v ** 2)) for k, v in stems.items()}
    # the fixture's percussion must land mostly in drums, low end in bass
    assert e["drums"] > e["vocals"]
    assert e["bass"] > e["vocals"]


def test_hpss_pure_tone():
    from visualizer.audio.stems import _hpss
    tone = np.sin(2 * np.pi * 80 * np.arange(SR) / SR).astype(np.float32)
    h, p = _hpss(tone, SR)
    assert np.mean(h ** 2) > 10 * np.mean(p ** 2)   # steady tone is harmonic


def test_sp1_export(tmp_path, test_song):
    from visualizer.audio.sp1_export import (SP1_STEM_ORDER, export_sp1_wav,
                                             sp1_filename)
    stems = {s: test_song * g for s, g in
             zip(SP1_STEM_ORDER, (0.1, 0.5, 0.8, 0.3))}
    out = str(tmp_path / sp1_filename("Test Song.wav", 124.2))
    assert out.endswith("_124BPM.wav")
    export_sp1_wav(stems, SR, out)
    info = sf.info(out)
    assert (info.channels, info.samplerate, info.subtype) == (8, 48000, "PCM_24")
    data, _ = sf.read(out)
    # channel order must follow SP1_STEM_ORDER, relative balance preserved
    energies = [float(np.mean(data[:, 2 * i:2 * i + 2] ** 2))
                for i in range(4)]
    assert energies[2] == max(energies)          # bass loudest (0.8 gain)
    assert np.abs(data).max() <= 1.0


def test_stretch_stereo(test_song):
    from visualizer.audio.stretch import _stretch_stereo
    out = _stretch_stereo(test_song[:SR * 2], 2.0)
    assert abs(len(out) - SR) < SR * 0.05        # half the length at 2x
    assert out.shape[1] == 2


def test_load_audio_roundtrip(tmp_path, test_song):
    from visualizer.audio.engine import load_audio_file
    p = str(tmp_path / "song.wav")
    sf.write(p, test_song, SR)
    loaded = load_audio_file(p)
    assert loaded.shape == test_song.shape
    assert np.allclose(loaded, test_song, atol=1e-4)
