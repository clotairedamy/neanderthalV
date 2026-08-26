"""Named visual presets: snapshot/restore the look-related settings."""
from __future__ import annotations

import configparser
import os

from .config import config_dir

PRESET_FIELDS = [
    "viz_mode", "palette", "color_blend_mode", "use_image_colors",
    "grain_mode", "grain_intensity", "damping", "sensitivity",
    "beat_impulse", "smoothing", "auto_camera", "bloom", "video_display",
]

_BOOLS = {"use_image_colors", "auto_camera", "bloom"}
_INTS = {"viz_mode"}
_FLOATS = {"grain_intensity", "damping", "sensitivity", "beat_impulse",
           "smoothing"}


def _path() -> str:
    return os.path.join(config_dir(), "presets.ini")


def _read() -> configparser.ConfigParser:
    cp = configparser.ConfigParser()
    cp.read(_path())
    return cp


def list_presets() -> list[str]:
    return sorted(_read().sections())


def save_preset(name: str, settings) -> None:
    cp = _read()
    cp[name] = {f: str(getattr(settings, f)) for f in PRESET_FIELDS}
    with open(_path(), "w") as fh:
        cp.write(fh)


def load_preset(name: str, settings) -> bool:
    cp = _read()
    if name not in cp:
        return False
    sec = cp[name]
    for f in PRESET_FIELDS:
        if f not in sec:
            continue
        raw = sec[f]
        if f in _BOOLS:
            setattr(settings, f, raw.lower() in ("1", "true", "yes"))
        elif f in _INTS:
            setattr(settings, f, int(float(raw)))
        elif f in _FLOATS:
            setattr(settings, f, float(raw))
        else:
            setattr(settings, f, raw)
    return True


def delete_preset(name: str) -> None:
    cp = _read()
    if cp.remove_section(name):
        with open(_path(), "w") as fh:
            cp.write(fh)
