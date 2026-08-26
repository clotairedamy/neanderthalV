import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SR = 44100


@pytest.fixture(scope="session")
def test_song():
    """8 s synthetic 120 BPM track: kick + bass + melody + hats. (n,2) f32."""
    t = np.arange(SR * 8) / SR
    kick = np.sin(2 * np.pi * 55 * t) * np.exp(-(t % 0.5) * 22) * 1.4
    bass = 0.4 * np.sin(2 * np.pi * 82.4 * t) * \
        (0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t))
    melody = 0.25 * np.sin(2 * np.pi * 440 * t)
    rng = np.random.default_rng(0)
    hats = rng.normal(0, 1, len(t)) * np.exp(-((t + 0.25) % 0.25) * 40) * 0.15
    mix = np.clip(kick + bass + melody + hats, -1, 1).astype(np.float32)
    return np.stack([mix, mix], axis=1)


@pytest.fixture()
def settings():
    from visualizer.config import Settings
    s = Settings()          # defaults, no disk I/O
    s._path = "/tmp/neanderthalv_test_config.ini"
    return s
