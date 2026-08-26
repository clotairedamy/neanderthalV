"""Settings persistence (config.ini) and platform detection / performance profiles."""
from __future__ import annotations

import configparser
import os
import platform
import sys
from dataclasses import dataclass, field, fields


def is_raspberry_pi() -> bool:
    if sys.platform != "linux":
        return False
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "raspberry pi" in f.read().lower()
    except OSError:
        return platform.machine() in ("aarch64", "armv7l")


def is_macos() -> bool:
    return sys.platform == "darwin"


def config_dir() -> str:
    if is_macos():
        d = os.path.expanduser("~/Library/Application Support/GeoViz")
    else:
        d = os.path.join(os.environ.get("XDG_CONFIG_HOME",
                                        os.path.expanduser("~/.config")), "geoviz")
    os.makedirs(d, exist_ok=True)
    return d


def asset_path(name: str) -> str:
    """Path to a bundled asset, working both from source and inside a
    PyInstaller bundle."""
    base = getattr(sys, "_MEIPASS", None)
    if base is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "assets", name)


def cache_dir() -> str:
    if is_macos():
        d = os.path.expanduser("~/Library/Caches/GeoViz")
    else:
        d = os.path.join(os.environ.get("XDG_CACHE_HOME",
                                        os.path.expanduser("~/.cache")), "geoviz")
    os.makedirs(d, exist_ok=True)
    return d


@dataclass
class PerformanceProfile:
    name: str
    target_fps: int
    particle_count: int
    fractal_resolution: int
    icosphere_subdivisions: int
    topology_grid: int
    analysis_hz: int
    video_max_height: int


MACOS_PROFILE = PerformanceProfile(
    name="macos", target_fps=60, particle_count=3200, fractal_resolution=320,
    icosphere_subdivisions=3, topology_grid=90, analysis_hz=30, video_max_height=1080,
)
RPI_PROFILE = PerformanceProfile(
    name="rpi5", target_fps=30, particle_count=800, fractal_resolution=160,
    icosphere_subdivisions=2, topology_grid=50, analysis_hz=30, video_max_height=720,
)


def active_profile() -> PerformanceProfile:
    return RPI_PROFILE if is_raspberry_pi() else MACOS_PROFILE


@dataclass
class Settings:
    """User-tunable settings, persisted to config.ini."""
    smoothing: float = 0.8          # EMA factor for analysis
    sensitivity: float = 1.0        # scales band energies
    damping: float = 0.85           # velocity friction (0.7-0.95)
    beat_impulse: float = 1.0       # scale of velocity spikes on beats
    color_blend_mode: str = "overlay"   # overlay | multiply | screen
    palette: str = "viridis"        # viridis|plasma|turbo|neon|image|custom
    use_image_colors: bool = True
    fft_min_hz: float = 20.0
    fft_max_hz: float = 16000.0
    viz_mode: int = 0
    video_display: str = "background"  # background|pip|texture|off
    chromakey: bool = False
    show_velocity_debug: bool = False
    auto_camera: bool = True
    last_dir: str = ""
    # depth point cloud (mode 9)
    pc_source: str = "auto"         # auto | video | camera | image | audio
    pc_layout: str = "luminance"    # luminance | side_by_side | top_bottom
    pc_near: float = 3.0
    pc_far: float = 6.0
    pc_point_size: float = 2.2
    pc_perspective: float = 0.35  # 0 = flat relief, 1 = exact Kinect frustum
    pc_band_axis: str = "off"     # off|vertical|depth|horizontal|radial
    pc_band_push: float = 0.10    # how far each band's energy displaces its zone
    # reactive 3D text (mode 10)
    text_content: str = "GEOVIZ"
    text_depth: float = 0.55      # per-band forward push
    text_explode: float = 1.0     # beat scatter strength
    pc_cutoff: float = 0.08
    bloom: bool = True              # GPU glow post-processing
    trails: bool = True             # feedback echo trails
    trail_amount: float = 1.0       # 0 = off-ish, 2 = heavy tunnel feedback
    anticipation: bool = True       # build tension before detected drops
    auto_choreograph: bool = False  # switch modes at song sections
    preserve_pitch: bool = False
    mode_prefs: str = "{}"          # json: per-mode palette/grain memory
    grain_mode: str = "off"         # off | film | static | scanlines | burst
    grain_intensity: float = 1.0
    export_dir: str = ""            # empty -> ~/Desktop
    export_format: str = "screen"   # screen | reel | square

    _path: str = field(default="", repr=False, compare=False)

    @classmethod
    def load(cls) -> "Settings":
        s = cls()
        s._path = os.path.join(config_dir(), "config.ini")
        cp = configparser.ConfigParser()
        if cp.read(s._path) and cp.has_section("geoviz"):
            sec = cp["geoviz"]
            for f in fields(cls):
                if f.name.startswith("_") or f.name not in sec:
                    continue
                raw = sec[f.name]
                try:
                    if f.type == "bool" or isinstance(getattr(s, f.name), bool):
                        setattr(s, f.name, raw.lower() in ("1", "true", "yes"))
                    elif isinstance(getattr(s, f.name), int):
                        setattr(s, f.name, int(float(raw)))
                    elif isinstance(getattr(s, f.name), float):
                        setattr(s, f.name, float(raw))
                    else:
                        setattr(s, f.name, raw)
                except ValueError:
                    pass
        return s

    def save(self) -> None:
        cp = configparser.ConfigParser()
        cp["geoviz"] = {
            f.name: str(getattr(self, f.name))
            for f in fields(self) if not f.name.startswith("_")
        }
        try:
            with open(self._path or os.path.join(config_dir(), "config.ini"), "w") as fh:
                cp.write(fh)
        except OSError:
            pass
