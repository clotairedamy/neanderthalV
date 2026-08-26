import numpy as np


def _bare_mode(settings):
    """PointCloudMode instance without GL resources (build() needs a canvas)."""
    from visualizer.config import active_profile
    from visualizer.viz.mode_pointcloud import PointCloudMode
    m = PointCloudMode.__new__(PointCloudMode)
    m.settings = settings
    m.profile = active_profile()
    m.waves = []
    m._t = 0.0
    m._last_ft = None
    m.SPEC_H, m.SPEC_W = PointCloudMode.SPEC_H, PointCloudMode.SPEC_W
    m._spec = np.zeros((m.SPEC_H, m.SPEC_W, 3), np.float32)
    return m


def test_wave_spawn_and_decay(settings):
    m = _bare_mode(settings)
    m._spawn_wave(1.0)
    assert len(m.waves) == 1
    w = m.waves[0]
    assert 0.0 <= w["x"] <= 1.0 and 0.0 <= w["y"] <= 1.0
    assert 0.03 <= w["a"] <= 0.45
    for _ in range(10):
        m._t += 0.1
        m._spawn_wave(1.0)
    assert len(m.waves) <= 3          # only three ride the shader at once
    for _ in range(200):
        m._advance_waves(1 / 60)
    assert m.waves == []              # they expire


def test_wave_travels_outward(settings):
    m = _bare_mode(settings)
    m._spawn_wave(1.0)
    r0, a0 = m.waves[0]["r"], m.waves[0]["a"]
    for _ in range(30):
        m._advance_waves(1 / 60)
    assert m.waves[0]["r"] > r0       # radius expands
    assert m.waves[0]["a"] < a0       # amplitude decays


def test_audio_terrain_scrolls(settings):
    from visualizer.color.palette import PaletteManager

    class F:
        def __init__(self, t, v):
            self.time = t
            self.spectrum = np.full(64, v, np.float32)

    m = _bare_mode(settings)
    m.palette = PaletteManager(settings)
    out = m._audio_terrain(F(0.0, 0.9))
    assert out.shape == (m.SPEC_H, m.SPEC_W, 3)
    top_bright = out[0].mean()
    assert top_bright > 0
    m._last_ft = 0.0
    m._audio_terrain(F(1.0, 0.0))     # a silent frame scrolls the loud row down
    assert out[1].mean() > out[0].mean()
    assert np.all(np.isfinite(out))


def test_shader_uniforms_declared():
    """Every uniform the mode sets must exist in the shader source."""
    from visualizer.viz import mode_pointcloud as M
    for name in ("u_near", "u_far", "u_zoffset", "u_point", "u_depth_scale",
                 "u_scatter", "u_sparkle", "u_cutoff", "u_tint", "u_tint_mix",
                 "u_alpha", "u_persp", "u_uv_scale", "u_uv_color",
                 "u_uv_depth", "u_wave0", "u_wave1", "u_wave2"):
        assert f"uniform" in M.VERT and name in M.VERT, name
