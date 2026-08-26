"""Offline beat grid: precomputed beats, onsets and song sections.

For file playback we know the whole song, so instead of reacting to beats a
frame late, we schedule them: the grid is computed once on load (and refined
from the drum stem when separation finishes), and the engine fires
beat/punch impulses exactly when the playhead crosses a grid entry.
Sections (energy-change boundaries) drive auto-choreography.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

HOP = 512


@dataclass
class BeatGrid:
    beats: np.ndarray = field(default_factory=lambda: np.zeros(0))      # times s
    beat_strengths: np.ndarray = field(default_factory=lambda: np.zeros(0))
    onsets: np.ndarray = field(default_factory=lambda: np.zeros(0))     # times s
    onset_strengths: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sections: np.ndarray = field(default_factory=lambda: np.zeros(0))   # times s
    bpm: float = 0.0
    source: str = "mix"          # "mix" or "drums"

    def scaled(self, factor: float) -> "BeatGrid":
        """Time-scale the grid (pitch-preserving stretch by rate r -> 1/r)."""
        return BeatGrid(self.beats * factor, self.beat_strengths.copy(),
                        self.onsets * factor, self.onset_strengths.copy(),
                        self.sections * factor,
                        self.bpm / factor if factor > 0 else self.bpm,
                        self.source)

    def events_between(self, t0: float, t1: float):
        """(beat_strength, punch_strength) for events in (t0, t1]; 0 if none."""
        beat = 0.0
        if len(self.beats):
            m = (self.beats > t0) & (self.beats <= t1)
            if m.any():
                beat = float(self.beat_strengths[m].max())
        punch = 0.0
        if len(self.onsets):
            m = (self.onsets > t0) & (self.onsets <= t1)
            if m.any():
                punch = float(self.onset_strengths[m].max())
        return beat, punch

    def section_crossed(self, t0: float, t1: float) -> bool:
        if not len(self.sections):
            return False
        return bool(((self.sections > t0) & (self.sections <= t1)).any())

    def time_to_next_section(self, t: float) -> float:
        """Seconds until the next section boundary (inf if none ahead).

        This is what makes choreography possible: because the whole track was
        analyzed up front, the visuals can start building *before* the drop
        instead of only reacting once it has already happened.
        """
        if not len(self.sections):
            return float("inf")
        ahead = self.sections[self.sections > t]
        return float(ahead[0] - t) if len(ahead) else float("inf")

    def anticipation(self, t: float, lead: float = 3.0) -> float:
        """0..1 tension ramp over the `lead` seconds before a boundary."""
        gap = self.time_to_next_section(t)
        if gap > lead:
            return 0.0
        return float(np.clip(1.0 - gap / lead, 0.0, 1.0)) ** 1.6


def compute_beat_grid(mono: np.ndarray, sr: int,
                      drums: np.ndarray | None = None,
                      bass: np.ndarray | None = None) -> BeatGrid:
    """Analyze a full track. `drums`/`bass` are optional mono stems that make
    beat tracking and punch scheduling far more precise."""
    import librosa

    beat_src = drums if drums is not None and float(np.abs(drums).max()) > 1e-4 \
        else mono
    env = librosa.onset.onset_strength(y=beat_src.astype(np.float32), sr=sr,
                                       hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=env, sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    bs = env[np.clip(beat_frames, 0, len(env) - 1)] if len(beat_frames) else \
        np.zeros(0)
    if len(bs):
        bs = np.clip(bs / (np.percentile(bs, 90) + 1e-9), 0.25, 1.0)

    # punch onsets from the low end (kick + bass hits)
    if bass is not None or drums is not None:
        low_src = (0.0 if bass is None else bass) + \
                  (0.0 if drums is None else drums)
    else:
        low_src = mono
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, 220, btype="low", fs=sr, output="sos")
    low = sosfiltfilt(sos, np.asarray(low_src, dtype=np.float64))
    lenv = librosa.onset.onset_strength(y=low.astype(np.float32), sr=sr,
                                        hop_length=HOP)
    on_frames = librosa.onset.onset_detect(
        onset_envelope=lenv, sr=sr, hop_length=HOP, backtrack=False)
    onsets = librosa.frames_to_time(on_frames, sr=sr, hop_length=HOP)
    os_ = lenv[np.clip(on_frames, 0, len(lenv) - 1)] if len(on_frames) else \
        np.zeros(0)
    if len(os_):
        os_ = np.clip(os_ / (np.percentile(os_, 90) + 1e-9), 0.3, 1.0)

    sections = _detect_sections(mono, sr, beats)

    return BeatGrid(beats=beats, beat_strengths=np.asarray(bs),
                    onsets=onsets, onset_strengths=np.asarray(os_),
                    sections=sections, bpm=tempo,
                    source="drums" if drums is not None else "mix")


def _detect_sections(mono: np.ndarray, sr: int,
                     beats: np.ndarray) -> np.ndarray:
    """Boundaries where the 4-beat average energy jumps — verse/drop edges."""
    if len(beats) < 12:
        return np.zeros(0)
    rms = []
    for b0, b1 in zip(beats[:-1], beats[1:]):
        seg = mono[int(b0 * sr):int(b1 * sr)]
        rms.append(float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0)
    rms = np.asarray(rms)
    k = 4
    smooth = np.convolve(rms, np.ones(k) / k, mode="same") + 1e-9

    bounds = []
    last = -1e9
    for i in range(k, len(smooth) - k):
        before = smooth[i - k:i].mean() + 1e-9
        after = smooth[i:i + k].mean()
        ratio = after / before
        if (ratio > 1.45 or ratio < 0.65) and beats[i] - last > 4.0:
            bounds.append(beats[i])
            last = beats[i]
    return np.asarray(bounds)


class BeatGridWorker(QThread):
    """Computes a BeatGrid off the GUI thread."""
    done = pyqtSignal(object)          # BeatGrid
    failed = pyqtSignal(str)

    def __init__(self, mono: np.ndarray, sr: int,
                 drums: np.ndarray | None = None,
                 bass: np.ndarray | None = None, parent=None):
        super().__init__(parent)
        self._args = (mono, sr, drums, bass)

    def run(self):
        try:
            self.done.emit(compute_beat_grid(*self._args))
        except Exception as e:
            self.failed.emit(f"Beat analysis failed: {e}")
