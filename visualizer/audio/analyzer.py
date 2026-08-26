"""Real-time audio analysis: FFT, 7 frequency bands, beat/BPM, energy, EMA."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FFT_SIZE = 2048
BAND_NAMES = ["sub_bass", "bass", "low_mid", "mid", "high_mid", "treble", "presence"]
BAND_EDGES = [(20, 60), (60, 250), (250, 500), (500, 2000),
              (2000, 4000), (4000, 8000), (8000, 16000)]
STEMS = ["vocals", "drums", "bass", "other"]


@dataclass
class AnalysisFrame:
    """One 30 Hz analysis snapshot consumed by the visualizations."""
    time: float = 0.0
    bands: np.ndarray = field(default_factory=lambda: np.zeros(7))
    stem_bands: dict = field(default_factory=dict)      # stem -> (7,) array
    stem_energy: dict = field(default_factory=dict)     # stem -> float
    rms: float = 0.0
    attack: np.ndarray = field(default_factory=lambda: np.zeros(7))
    punch: float = 0.0          # low-band transient (kick/bass hit), 0..1
    beat: bool = False
    beat_strength: float = 0.0
    anticipation: float = 0.0   # 0..1 build-up before a section boundary
    section: bool = False       # a section boundary lands in this tick
    bpm: float = 0.0
    centroid: float = 0.0                               # normalized 0..1
    spectrum: np.ndarray = field(default_factory=lambda: np.zeros(64))
    waveform: np.ndarray = field(default_factory=lambda: np.zeros(FFT_SIZE))


class EMA:
    def __init__(self, factor: float = 0.8, shape=None):
        self.factor = factor
        self.state = None if shape is None else np.zeros(shape)

    def __call__(self, x):
        if self.state is None:
            self.state = np.asarray(x, float).copy() if np.ndim(x) else float(x)
        else:
            self.state = self.factor * self.state + (1.0 - self.factor) * x
        return self.state


class BeatDetector:
    """Spectral-flux onset detection with adaptive threshold + BPM estimation."""

    REFRACTORY = 60.0 / 240.0  # never faster than 240 BPM

    def __init__(self, history: int = 43):
        self.history = history
        self.flux_hist: list[float] = []
        self.prev_mag: np.ndarray | None = None
        self.last_beat_t = -10.0
        self.beat_intervals: list[float] = []

    def process(self, mag: np.ndarray, t: float) -> tuple[bool, float, float]:
        """Returns (is_beat, beat_strength, bpm)."""
        if self.prev_mag is None or self.prev_mag.shape != mag.shape:
            self.prev_mag = mag.copy()
            return False, 0.0, 0.0

        # positive spectral flux, weighted toward low/mid where beats live
        diff = mag - self.prev_mag
        self.prev_mag = mag.copy()
        n = len(mag)
        weights = np.ones(n)
        weights[: n // 4] = 2.0
        flux = float(np.sum(np.maximum(diff, 0.0) * weights))

        self.flux_hist.append(flux)
        if len(self.flux_hist) > self.history:
            self.flux_hist.pop(0)

        is_beat, strength = False, 0.0
        if len(self.flux_hist) >= 12:
            arr = np.asarray(self.flux_hist)
            mean, std = arr.mean(), arr.std()
            thresh = mean + 1.4 * std + 1e-6
            if flux > thresh and (t - self.last_beat_t) > self.REFRACTORY:
                is_beat = True
                strength = float(np.clip((flux - mean) / (std + 1e-9) / 4.0, 0.2, 1.0))
                interval = t - self.last_beat_t
                if 60.0 / 220.0 <= interval <= 60.0 / 50.0:
                    self.beat_intervals.append(interval)
                    if len(self.beat_intervals) > 16:
                        self.beat_intervals.pop(0)
                self.last_beat_t = t

        bpm = 0.0
        if len(self.beat_intervals) >= 4:
            bpm = 60.0 / float(np.median(self.beat_intervals))
            while bpm > 180.0:
                bpm /= 2.0
            while bpm < 60.0:
                bpm *= 2.0
        return is_beat, strength, bpm


class AudioAnalyzer:
    """Windowed FFT analysis of the mix and of each stem."""

    def __init__(self, sr: int, settings):
        self.sr = sr
        self.settings = settings
        self.window = np.hanning(FFT_SIZE)
        self.freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sr)
        self.beat = BeatDetector()
        self._band_ema = EMA(settings.smoothing, shape=7)
        self._stem_emas = {s: EMA(settings.smoothing, shape=7) for s in STEMS}
        self._rms_ema = EMA(settings.smoothing)
        self._centroid_ema = EMA(settings.smoothing)
        # adaptive per-band normalization: track ceiling AND floor so real
        # music (which hovers in a narrow loudness window) still spans 0..1
        self._band_max = np.full(7, 1e-3)
        self._band_min = np.zeros(7)
        self._prev_norm = np.zeros(7)
        self._rms_max = 1e-3
        self._rms_min = 0.0
        self._precompute_band_bins()

    def _precompute_band_bins(self):
        lo = self.settings.fft_min_hz
        hi = self.settings.fft_max_hz
        self._band_masks = []
        for f0, f1 in BAND_EDGES:
            f0 = max(f0, lo)
            f1 = min(f1, hi)
            mask = (self.freqs >= f0) & (self.freqs < f1)
            if not mask.any():
                mask = np.zeros_like(self.freqs, dtype=bool)
                mask[1] = True
            self._band_masks.append(mask)

    def set_smoothing(self, factor: float):
        factor = float(np.clip(factor, 0.0, 0.98))
        self._band_ema.factor = factor
        self._rms_ema.factor = factor
        self._centroid_ema.factor = factor
        for e in self._stem_emas.values():
            e.factor = factor

    def refresh_fft_range(self):
        self._precompute_band_bins()

    def _magnitude(self, mono: np.ndarray) -> np.ndarray:
        if len(mono) < FFT_SIZE:
            mono = np.pad(mono, (0, FFT_SIZE - len(mono)))
        return np.abs(np.fft.rfft(mono[:FFT_SIZE] * self.window))

    def _band_energies(self, mag: np.ndarray, adapt: bool = False) -> np.ndarray:
        vals = np.array([mag[m].mean() if m.any() else 0.0 for m in self._band_masks])
        vals = np.log10(1.0 + 8.0 * vals)
        if adapt:
            # ceiling decays fairly fast, floor drifts up toward it: the
            # usable window keeps hugging the current music's dynamic range
            self._band_max = np.maximum(self._band_max * 0.993, vals)
            self._band_min = np.minimum(
                self._band_min + (self._band_max - self._band_min) * 0.008, vals)
            # denom guard: a near-constant band must not amplify noise to 1.0
            denom = np.maximum(self._band_max - self._band_min,
                               0.25 * self._band_max) + 1e-9
            out = (vals - self._band_min) / denom
        else:
            out = vals / (self._band_max + 1e-9)
        out = np.clip(out, 0.0, 1.0) ** 1.25       # gamma: punchier contrast
        return np.clip(out * self.settings.sensitivity, 0.0, 1.0)

    def analyze(self, t: float, mono: np.ndarray,
                stem_windows: dict[str, np.ndarray] | None = None) -> AnalysisFrame:
        mag = self._magnitude(mono)

        raw_bands = self._band_energies(mag, adapt=True)
        # per-band transient (attack) + low-band punch: fires on every kick /
        # bass hit even when the beat detector is unsure — drives impulses
        attack = np.clip((raw_bands - self._prev_norm) * 5.0, 0.0, 1.0)
        self._prev_norm = raw_bands
        punch = float(np.clip(attack[:3].max() * 1.3, 0.0, 1.0))
        bands = self._band_ema(raw_bands)

        raw_rms = float(np.sqrt(np.mean(mono ** 2)))
        self._rms_max = max(self._rms_max * 0.993, raw_rms)
        self._rms_min = min(self._rms_min + (self._rms_max - self._rms_min)
                            * 0.008, raw_rms)
        norm_rms = (raw_rms - self._rms_min) / \
            max(self._rms_max - self._rms_min, 0.25 * self._rms_max, 1e-9)
        rms = self._rms_ema(np.clip(norm_rms * self.settings.sensitivity, 0, 1))

        total = float(mag.sum()) + 1e-9
        centroid_hz = float((self.freqs * mag).sum() / total)
        centroid = self._centroid_ema(np.clip(centroid_hz / (self.sr / 4), 0, 1) ** 0.5)

        is_beat, strength, bpm = self.beat.process(mag, t)

        stem_bands, stem_energy = {}, {}
        if stem_windows:
            for name, w in stem_windows.items():
                smag = self._magnitude(w)
                stem_bands[name] = self._stem_emas[name](self._band_energies(smag))
                stem_energy[name] = float(np.clip(
                    np.sqrt(np.mean(w[:FFT_SIZE] ** 2)) * 3.0 * self.settings.sensitivity, 0, 1))
        else:
            for name in STEMS:
                stem_bands[name] = bands.copy()
                stem_energy[name] = float(np.clip(rms * 3.0, 0, 1))

        # compact display spectrum (64 log-spaced bins)
        edges = np.unique(np.geomspace(1, len(mag) - 1, 65).astype(int))
        spec = np.array([mag[a:b].mean() if b > a else mag[a]
                         for a, b in zip(edges[:-1], edges[1:])])
        spec = np.log10(1 + 8 * spec)
        spec = spec / (spec.max() + 1e-9)
        if len(spec) < 64:
            spec = np.pad(spec, (0, 64 - len(spec)))

        return AnalysisFrame(
            time=t, bands=np.asarray(bands), stem_bands=stem_bands,
            stem_energy=stem_energy, rms=float(np.clip(rms, 0, 1)),
            attack=attack, punch=punch,
            beat=is_beat, beat_strength=strength, bpm=bpm,
            centroid=float(centroid), spectrum=spec[:64],
            waveform=mono[:FFT_SIZE].copy() if len(mono) >= FFT_SIZE
            else np.pad(mono, (0, FFT_SIZE - len(mono))),
        )
