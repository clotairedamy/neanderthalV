import numpy as np
from PIL import Image


def test_extract_image_palette(tmp_path):
    from visualizer.color.palette import extract_image_palette
    img = Image.new("RGB", (60, 60))
    px = img.load()
    for x in range(60):
        for y in range(60):
            px[x, y] = (255, 40, 40) if x < 20 else \
                (40, 255, 60) if x < 40 else (50, 80, 255)
    p = str(tmp_path / "pal.png")
    img.save(p)
    pal = extract_image_palette(p)
    assert pal.shape[1] == 3 and 2 <= len(pal) <= 8
    assert np.all((pal >= 0) & (pal <= 1))


def test_blend_modes():
    from visualizer.color.palette import blend
    a, b = np.random.rand(7, 3), np.random.rand(7, 3)
    for mode in ("overlay", "multiply", "screen"):
        out = blend(a, b, mode)
        assert np.all((out >= 0) & (out <= 1))


def test_palette_manager_updates(settings):
    from visualizer.color.palette import PaletteManager

    class F:
        centroid = 0.5
        rms = 0.6

    pm = PaletteManager(settings)
    for _ in range(60):
        pm.update(F(), 1 / 30)
    assert pm.colors.shape == (7, 3)
    assert pm.lut(64).shape == (64, 3)
    assert np.all(np.isfinite(pm.colors))


def test_presets_roundtrip(settings, monkeypatch, tmp_path):
    import visualizer.presets as P
    monkeypatch.setattr(P, "_path", lambda: str(tmp_path / "presets.ini"))
    settings.damping = 0.91
    settings.grain_mode = "burst"
    settings.viz_mode = 7
    P.save_preset("my look", settings)
    settings.damping = 0.7
    settings.grain_mode = "off"
    settings.viz_mode = 0
    assert P.load_preset("my look", settings)
    assert settings.damping == 0.91
    assert settings.grain_mode == "burst"
    assert settings.viz_mode == 7
    assert "my look" in P.list_presets()
    P.delete_preset("my look")
    assert "my look" not in P.list_presets()
