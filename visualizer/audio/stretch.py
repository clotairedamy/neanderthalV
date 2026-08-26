"""Pitch-preserving time stretch (phase vocoder), rendered off-thread.

Varispeed (the default) changes pitch like a turntable. When "preserve
pitch" is on, the engine keeps varispeed going while this worker renders a
time-stretched copy of every stem in the background, then swaps buffers.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


def _stretch_stereo(data: np.ndarray, rate: float) -> np.ndarray:
    import librosa
    chans = [librosa.effects.time_stretch(
        np.ascontiguousarray(data[:, c], dtype=np.float32), rate=rate)
        for c in range(data.shape[1])]
    n = min(len(c) for c in chans)
    return np.stack([c[:n] for c in chans], axis=1).astype(np.float32)


class StretchWorker(QThread):
    done = pyqtSignal(object, object, float)   # audio (n,2), stems dict|None, rate
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, audio: np.ndarray, stems: dict | None, sr: int,
                 rate: float, parent=None):
        super().__init__(parent)
        self.audio = audio
        self.stems = stems
        self.sr = sr
        self.rate = rate

    def run(self):
        try:
            if self.stems:
                out_stems = {}
                for name, data in self.stems.items():
                    self.progress.emit(f"Stretching {name} ({self.rate:.2f}x)…")
                    out_stems[name] = _stretch_stereo(np.asarray(data), self.rate)
                n = min(len(v) for v in out_stems.values())
                out_stems = {k: v[:n] for k, v in out_stems.items()}
                audio = np.clip(sum(out_stems.values()), -1.0, 1.0)
            else:
                self.progress.emit(f"Stretching audio ({self.rate:.2f}x)…")
                audio = _stretch_stereo(self.audio, self.rate)
                out_stems = None
            self.done.emit(audio, out_stems, self.rate)
        except Exception as e:
            self.failed.emit(f"Time-stretch failed: {e}")
