"""Stem separation: Demucs v4 with on-disk caching, plus a fast DSP fallback.

Cache layout: <cache>/stems/<sha1-of-file>/{vocals,drums,bass,other}.npy
(float32 stereo at the engine sample rate).

If Demucs (torch) isn't installed — typical on Raspberry Pi — we fall back to
pseudo-stems built from harmonic/percussive separation + band splitting so the
stem mixer and per-stem visuals still work.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from ..config import cache_dir

STEMS = ["vocals", "drums", "bass", "other"]


def _file_hash(path: str) -> str:
    h = hashlib.sha1()
    h.update(os.path.basename(path).encode())
    h.update(str(os.path.getsize(path)).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))  # first 1MB is plenty for identity
    return h.hexdigest()


def stem_cache_path(audio_path: str) -> str:
    d = os.path.join(cache_dir(), "stems", _file_hash(audio_path))
    os.makedirs(d, exist_ok=True)
    return d


def load_cached_stems(audio_path: str) -> dict[str, np.ndarray] | None:
    d = stem_cache_path(audio_path)
    out = {}
    for s in STEMS:
        p = os.path.join(d, f"{s}.npy")
        if not os.path.exists(p):
            return None
        out[s] = np.load(p, mmap_mode="r")
    return out


def demucs_available() -> bool:
    try:
        import demucs  # noqa: F401
        return True
    except ImportError:
        return False


def _hpss(mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Median-filtering harmonic/percussive separation (explicit implementation
    — librosa 1.0's effects.hpss mis-scales output on some scipy/numpy combos)."""
    import librosa
    from scipy.ndimage import median_filter

    S = librosa.stft(mono, n_fft=2048, hop_length=512)
    mag = np.abs(S)
    harm_env = median_filter(mag, size=(1, 17), mode="reflect")
    perc_env = median_filter(mag, size=(17, 1), mode="reflect")
    eps = 1e-10
    mask_h = (harm_env ** 2 + eps) / (harm_env ** 2 + perc_env ** 2 + 2 * eps)
    h = librosa.istft(S * mask_h, hop_length=512, length=len(mono))
    p = librosa.istft(S * (1.0 - mask_h), hop_length=512, length=len(mono))
    return h, p


def _pseudo_stems(audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Fast approximation: HPSS + frequency splits. audio is (n, 2) float32."""
    from scipy.signal import butter, sosfiltfilt

    mono = audio.mean(axis=1)
    harmonic, percussive = _hpss(mono, sr)

    # zero-phase filters so the residual "other" subtraction is meaningful
    sos_lp = butter(4, 180, btype="low", fs=sr, output="sos")
    sos_hp = butter(4, 500, btype="high", fs=sr, output="sos")
    bass = sosfiltfilt(sos_lp, harmonic)
    vocals = sosfiltfilt(sos_hp, harmonic) * 0.8
    other = harmonic - bass - vocals

    def stereo(x):
        return np.repeat(x[:, None], 2, axis=1).astype(np.float32)

    return {"vocals": stereo(vocals), "drums": stereo(percussive),
            "bass": stereo(bass), "other": stereo(other)}


class StemSeparationWorker(QThread):
    """Runs Demucs (or the fallback) off the GUI thread and caches results."""

    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)     # stem name -> (n,2) float32
    failed = pyqtSignal(str)

    def __init__(self, audio_path: str, audio: np.ndarray, sr: int, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.audio = audio  # (n, 2) float32 at engine sr
        self.sr = sr

    def run(self):
        try:
            cached = load_cached_stems(self.audio_path)
            if cached is not None:
                self.progress.emit("Loaded stems from cache")
                self.finished_ok.emit({k: np.asarray(v) for k, v in cached.items()})
                return

            if demucs_available():
                self.progress.emit("Separating stems with Demucs v4 (htdemucs)…")
                stems = self._run_demucs()
            else:
                self.progress.emit("Demucs not installed — using fast DSP pseudo-stems")
                stems = _pseudo_stems(self.audio, self.sr)

            d = stem_cache_path(self.audio_path)
            for name, data in stems.items():
                np.save(os.path.join(d, f"{name}.npy"), data.astype(np.float32))
            self.progress.emit("Stems ready")
            self.finished_ok.emit(stems)
        except Exception as e:  # separation must never crash the app
            self.failed.emit(f"Stem separation failed: {e}")

    def _run_demucs(self) -> dict[str, np.ndarray]:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        model = get_model("htdemucs")
        model.eval()
        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"

        wav = torch.from_numpy(self.audio.T.astype(np.float32))  # (2, n)
        if self.sr != model.samplerate:
            import torchaudio
            wav = torchaudio.functional.resample(wav, self.sr, model.samplerate)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)

        with torch.no_grad():
            sources = apply_model(model, wav[None], device=device,
                                  split=True, overlap=0.15, progress=False)[0]
        sources = sources * (ref.std() + 1e-8) + ref.mean()

        out = {}
        for i, name in enumerate(model.sources):  # drums, bass, other, vocals
            s = sources[i]
            if model.samplerate != self.sr:
                import torchaudio
                s = torchaudio.functional.resample(s, model.samplerate, self.sr)
            arr = s.cpu().numpy().T.astype(np.float32)  # (n, 2)
            # match engine length exactly
            n = len(self.audio)
            if len(arr) >= n:
                arr = arr[:n]
            else:
                arr = np.pad(arr, ((0, n - len(arr)), (0, 0)))
            out[name] = arr
        return {s: out[s] for s in STEMS if s in out}
