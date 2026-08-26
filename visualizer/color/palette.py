"""Color extraction from images (k-means) and audio-driven palette animation.

- Extract 5-8 dominant colors from a photo (k-means on downsampled pixels).
- Map colors to frequency bands / stems.
- Smooth transitions driven by audio energy; palette rotation driven by
  spectral centroid; overlay/multiply/screen blend modes.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2

STEMS = ["vocals", "drums", "bass", "other"]
N_BANDS = 7

BUILTIN = {
    # monochrome data-viz look (references: Ikeda-style generative art)
    "mono": np.array([
        [0.35, 0.36, 0.38], [0.48, 0.49, 0.51], [0.60, 0.61, 0.63],
        [0.72, 0.73, 0.75], [0.83, 0.84, 0.86], [0.92, 0.93, 0.94],
        [1.00, 1.00, 1.00],
    ]),
    "neon": np.array([
        [0.00, 1.00, 0.75], [1.00, 0.15, 0.80], [0.20, 0.55, 1.00],
        [1.00, 0.85, 0.10], [0.55, 0.10, 1.00], [0.10, 1.00, 0.30],
        [1.00, 0.35, 0.15],
    ]),
    "custom": np.array([
        [0.90, 0.20, 0.30], [0.95, 0.60, 0.15], [0.95, 0.90, 0.30],
        [0.30, 0.80, 0.45], [0.20, 0.60, 0.90], [0.45, 0.35, 0.90],
        [0.85, 0.40, 0.85],
    ]),
}


def _mpl_colormap(name: str, n: int = 7) -> np.ndarray:
    """Sample a matplotlib-style colormap via vispy (no matplotlib dependency)."""
    try:
        from vispy.color import get_colormap
        cm = get_colormap(name)
        return np.asarray(cm.map(np.linspace(0.05, 0.95, n)))[:, :3]
    except Exception:
        return BUILTIN["custom"][:n]


def extract_image_palette(path: str, k: int = 6) -> np.ndarray:
    """K-means dominant colors from an image, sorted dark -> bright. (k,3) in 0..1."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((120, 120))
    px = np.asarray(img, dtype=np.float64).reshape(-1, 3) / 255.0
    # drop near-duplicate work: subsample
    if len(px) > 6000:
        px = px[np.random.default_rng(0).choice(len(px), 6000, replace=False)]
    k = int(np.clip(k, 5, 8))
    # an image with few distinct colors can't fill k clusters
    n_unique = len(np.unique((px * 32).astype(np.uint8), axis=0))
    k = max(2, min(k, n_unique))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        centroids, labels = kmeans2(px, k, minit="++", seed=1, iter=20)
    # order by cluster weight then luminance for stable, pleasing mapping
    counts = np.bincount(labels, minlength=k)
    keep = centroids[counts > 0]
    lum = keep @ np.array([0.2126, 0.7152, 0.0722])
    return keep[np.argsort(lum)]


def extract_frame_palette(frame_rgb: np.ndarray, k: int = 6) -> np.ndarray:
    """Dominant colors from a video frame (H,W,3 uint8)."""
    small = frame_rgb[::8, ::8].reshape(-1, 3).astype(np.float64) / 255.0
    if len(small) < k * 4:
        return BUILTIN["custom"][:k]
    centroids, _ = kmeans2(small, k, minit="++", seed=1, iter=8)
    lum = centroids @ np.array([0.2126, 0.7152, 0.0722])
    return centroids[np.argsort(lum)]


# ---- blend modes -----------------------------------------------------------

def blend(a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    a = np.clip(a, 0, 1)
    b = np.clip(b, 0, 1)
    if mode == "multiply":
        return a * b
    if mode == "screen":
        return 1.0 - (1.0 - a) * (1.0 - b)
    # overlay
    return np.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))


class PaletteManager:
    """Holds the active palette and animates it from analysis frames.

    Colors returned are already smoothed (EMA) and rotated by spectral
    centroid, so visualizations just ask for band/stem colors each frame.
    """

    def __init__(self, settings):
        self.settings = settings
        self.image_palette: np.ndarray | None = None
        self.base = _mpl_colormap("viridis", N_BANDS)
        self._current = self.base.copy()
        self._rotation = 0.0
        self._energy_mix = 0.0
        self.set_builtin(settings.palette)

    # -- palette sources --

    def set_builtin(self, name: str) -> None:
        self.settings.palette = name
        if name == "image" and self.image_palette is not None:
            self.base = self._fit(self.image_palette)
        elif name in BUILTIN:
            self.base = BUILTIN[name][:N_BANDS].copy()
        else:
            self.base = _mpl_colormap(name, N_BANDS)

    def set_image(self, path: str) -> np.ndarray:
        self.image_palette = extract_image_palette(path)
        if self.settings.use_image_colors:
            self.settings.palette = "image"
            self.base = self._fit(self.image_palette)
        return self.image_palette

    def set_image_palette_colors(self, colors: np.ndarray) -> None:
        """Manual adjustment from the UI color picker."""
        self.image_palette = np.asarray(colors, float)
        if self.settings.palette == "image":
            self.base = self._fit(self.image_palette)

    @staticmethod
    def _fit(pal: np.ndarray, n: int = N_BANDS) -> np.ndarray:
        """Resample an arbitrary-length palette to n entries."""
        idx = np.linspace(0, len(pal) - 1, n)
        lo = np.floor(idx).astype(int)
        hi = np.ceil(idx).astype(int)
        frac = (idx - lo)[:, None]
        return pal[lo] * (1 - frac) + pal[hi] * frac

    # -- animation --

    def update(self, frame, dt: float) -> None:
        """Advance rotation (spectral centroid) and energy-driven transitions."""
        if frame is not None:
            # adaptive palette: rotate through colors as brightness shifts
            self._rotation += (0.2 + 2.5 * frame.centroid) * dt
            target_mix = np.clip(frame.rms * 2.0, 0, 1)
            self._energy_mix += (target_mix - self._energy_mix) * min(1.0, 3.0 * dt)

        rot = self._rotation % N_BANDS
        i0 = int(rot)
        f = rot - i0
        rolled = np.roll(self.base, -i0, axis=0)
        rolled2 = np.roll(self.base, -(i0 + 1), axis=0)
        animated = rolled * (1 - f) + rolled2 * f

        # blend animated palette against base by audio energy
        mode = self.settings.color_blend_mode
        blended = blend(self.base, animated, mode)
        target = self.base * (1 - self._energy_mix) + blended * self._energy_mix

        # smooth the actual output colors (no color popping)
        a = 1.0 - min(1.0, 4.0 * dt)
        self._current = self._current * a + target * (1 - a)

    # -- queries --

    @property
    def colors(self) -> np.ndarray:
        return self._current

    def band_color(self, i: int) -> np.ndarray:
        return self._current[i % N_BANDS]

    def stem_color(self, stem: str) -> np.ndarray:
        try:
            i = STEMS.index(stem)
        except ValueError:
            i = 0
        idx = int(round(i * (N_BANDS - 1) / max(1, len(STEMS) - 1)))
        return self._current[idx]

    def lut(self, n: int) -> np.ndarray:
        """Interpolated (n,3) lookup table across the current palette."""
        return self._fit(self._current, n)
