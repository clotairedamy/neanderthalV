"""Audio engine: file/mic input, stem-mixed playback, seek/speed, analysis clock.

Playback runs in a sounddevice callback thread mixing the four stems by their
gain/solo/mute state (variable speed via linear-interpolation varispeed).
The GUI polls `analysis_tick()` at ~30 Hz; it windows the audio at the current
playhead and returns an AnalysisFrame. The playhead sample counter is the
master clock — video sync reads `position` from here, so audio analysis is
frame-accurate against whatever the user hears.
"""
from __future__ import annotations

import os

import numpy as np
import sounddevice as sd
import soundfile as sf
from PyQt6.QtCore import QObject, pyqtSignal

from .analyzer import FFT_SIZE, AnalysisFrame, AudioAnalyzer, STEMS
from .beatgrid import BeatGrid, BeatGridWorker
from .stems import StemSeparationWorker

SR = 44100
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aiff", ".aif")


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def load_audio_file(path: str, sr: int = SR) -> np.ndarray:
    """Load any supported audio file as (n, 2) float32 at `sr`."""
    try:
        data, in_sr = sf.read(path, always_2d=True, dtype="float32")
    except Exception:
        # mp3/m4a on some platforms: go through librosa/audioread
        import librosa
        y, in_sr = librosa.load(path, sr=None, mono=False)
        data = y.T if y.ndim == 2 else y[:, None]
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    if in_sr != sr:
        import librosa
        data = np.stack([
            librosa.resample(data[:, ch].astype(np.float32), orig_sr=in_sr, target_sr=sr)
            for ch in range(2)], axis=1)
    return np.ascontiguousarray(data, dtype=np.float32)


def extract_audio_from_video(video_path: str, sr: int = SR) -> np.ndarray:
    """Extract the audio track of a video via ffmpeg -> (n, 2) float32."""
    import subprocess
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            [_ffmpeg_exe(), "-y", "-i", video_path, "-vn",
             "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "2", tmp.name],
            check=True, capture_output=True)
        return load_audio_file(tmp.name, sr)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


class AudioEngine(QObject):
    position_changed = pyqtSignal(float)
    duration_changed = pyqtSignal(float)
    playback_finished = pyqtSignal()
    stems_progress = pyqtSignal(str)
    stems_ready = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.sr = SR
        self.analyzer = AudioAnalyzer(self.sr, settings)

        self.audio: np.ndarray | None = None          # (n,2) full mix
        self.stems: dict[str, np.ndarray] | None = None
        self.gains = {s: 1.0 for s in STEMS}
        self.mutes = {s: False for s in STEMS}
        self.solo: str | None = None

        self._playhead = 0.0        # fractional sample position (source domain)
        self.speed = 1.0
        self.playing = False
        self.loop = False
        self._stream: sd.OutputStream | None = None
        self._worker: StemSeparationWorker | None = None

        # offline beat grid: scheduled beats/punches/sections for file playback
        self.beat_grid: BeatGrid | None = None
        self._grid_worker: BeatGridWorker | None = None
        self._last_tick_time = 0.0
        self.section_crossed = False    # set per analysis tick, UI consumes it

        # pitch-preserving speed (phase-vocoder stretch, swapped in when ready)
        self.preserve_pitch = False
        self._pitch_rate = 1.0          # rate of the currently installed buffers
        self._requested_speed = 1.0
        self._audio_orig: np.ndarray | None = None
        self._stems_orig: dict | None = None
        self._grid_orig: BeatGrid | None = None
        self._stretch_worker = None

        # microphone
        self.mic_enabled = False
        self._mic_stream: sd.InputStream | None = None
        self._mic_ring = np.zeros(self.sr * 2, dtype=np.float32)
        self._mic_write = 0
        self._mic_time = 0.0

        self._last_frame = AnalysisFrame()

    # ------------------------------------------------------------- loading

    def load_file(self, path: str, audio_override: np.ndarray | None = None) -> None:
        """Load an audio file (or pre-extracted video audio) and start stems."""
        self.stop_stream()
        try:
            audio = audio_override if audio_override is not None else load_audio_file(path)
        except Exception as e:
            self.error.emit(f"Could not load audio: {e}")
            return
        self.audio = audio
        self.stems = None
        self.beat_grid = None
        self._audio_orig = audio
        self._stems_orig = None
        self._grid_orig = None
        self._pitch_rate = 1.0
        self._playhead = 0.0
        self._last_tick_time = 0.0
        self.playing = False
        self.duration_changed.emit(self.duration)

        self._worker = StemSeparationWorker(path, audio, self.sr)
        self._worker.progress.connect(self.stems_progress)
        self._worker.failed.connect(self.stems_progress)
        self._worker.finished_ok.connect(self._on_stems)
        self._worker.start()

        # quick beat grid from the mix now; refined from drums when stems land
        self._start_grid_worker(audio.mean(axis=1))

    def _start_grid_worker(self, mono, drums=None, bass=None) -> None:
        self._grid_worker = BeatGridWorker(mono, self.sr, drums, bass)
        self._grid_worker.done.connect(self._on_grid)
        self._grid_worker.failed.connect(self.stems_progress)
        self._grid_worker.start()

    def _on_grid(self, grid) -> None:
        # never downgrade a drums-based grid to a mix-based one
        if self._grid_orig is None or grid.source == "drums":
            self._grid_orig = grid
            self.beat_grid = grid if self._pitch_rate == 1.0 else \
                grid.scaled(1.0 / self._pitch_rate)
            self.stems_progress.emit(
                f"Beat grid ready ({grid.source}): {grid.bpm:.0f} BPM, "
                f"{len(grid.beats)} beats, {len(grid.sections)} sections")

    def _on_stems(self, stems: dict) -> None:
        if self._audio_orig is None:
            return
        n = len(self._audio_orig)
        self._stems_orig = {k: np.asarray(v)[:n] for k, v in stems.items()}
        if self._pitch_rate == 1.0:
            self.stems = self._stems_orig
        else:
            self.apply_pitch_speed()     # re-stretch with real stems
        self.stems_ready.emit()
        self._start_grid_worker(self._audio_orig.mean(axis=1),
                                drums=self._stems_orig["drums"].mean(axis=1),
                                bass=self._stems_orig["bass"].mean(axis=1))

    # ------------------------------------------------------------- playback

    @property
    def duration(self) -> float:
        return 0.0 if self.audio is None else len(self.audio) / self.sr

    @property
    def position(self) -> float:
        return self._playhead / self.sr

    def play(self) -> None:
        if self.audio is None or self.playing:
            return
        if self._stream is None:
            self._stream = sd.OutputStream(
                samplerate=self.sr, channels=2, dtype="float32",
                blocksize=1024, latency="low", callback=self._callback)
            self._stream.start()
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def toggle(self) -> None:
        (self.pause if self.playing else self.play)()

    def stop_stream(self) -> None:
        self.playing = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def seek(self, seconds: float) -> None:
        if self.audio is not None:
            self._playhead = float(np.clip(seconds, 0, self.duration)) * self.sr
            self._last_tick_time = self.position

    def set_speed(self, x: float) -> None:
        self._requested_speed = float(np.clip(x, 0.5, 2.0))
        if not self.preserve_pitch:
            self.speed = self._requested_speed

    # -- pitch-preserving speed ------------------------------------------

    def set_preserve_pitch(self, on: bool) -> None:
        self.preserve_pitch = on
        if not on:
            self._restore_original()
            self.speed = self._requested_speed
        elif abs(self._requested_speed - 1.0) > 1e-3:
            self.apply_pitch_speed()

    def apply_pitch_speed(self) -> None:
        """Kick off a background stretch to the requested speed (call on
        slider release). Varispeed keeps playing until the swap."""
        if self.audio is None or not self.preserve_pitch:
            return
        rate = self._requested_speed
        if abs(rate - 1.0) < 1e-3:
            self._restore_original()
            self.speed = 1.0
            return
        self.speed = rate if self._pitch_rate == 1.0 else 1.0  # interim
        from .stretch import StretchWorker
        self._stretch_worker = StretchWorker(
            self._audio_orig, self._stems_orig, self.sr, rate)
        self._stretch_worker.progress.connect(self.stems_progress)
        self._stretch_worker.failed.connect(self.stems_progress)
        self._stretch_worker.done.connect(self._install_stretch)
        self._stretch_worker.start()

    def _install_stretch(self, audio, stems, rate: float) -> None:
        if self._audio_orig is None or not self.preserve_pitch:
            return
        if abs(rate - self._requested_speed) > 1e-3:
            return      # user moved the slider again; a newer stretch is coming
        rel = self.position / self.duration if self.duration else 0.0
        self.audio = audio
        self.stems = stems
        self.speed = 1.0
        self._pitch_rate = rate
        if self._grid_orig is not None:
            self.beat_grid = self._grid_orig.scaled(1.0 / rate)
        self.duration_changed.emit(self.duration)
        self.seek(rel * self.duration)
        self.stems_progress.emit(f"Pitch-preserved {rate:.2f}x ready")

    def _restore_original(self) -> None:
        if self._audio_orig is None or self._pitch_rate == 1.0:
            return
        rel = self.position / self.duration if self.duration else 0.0
        self.audio = self._audio_orig
        self.stems = self._stems_orig
        self._pitch_rate = 1.0
        self.beat_grid = self._grid_orig
        self.duration_changed.emit(self.duration)
        self.seek(rel * self.duration)

    # stem mixer
    def set_stem_gain(self, stem: str, g: float) -> None:
        self.gains[stem] = float(np.clip(g, 0, 1.5))

    def set_stem_mute(self, stem: str, mute: bool) -> None:
        self.mutes[stem] = mute

    def set_solo(self, stem: str | None) -> None:
        self.solo = stem

    def _effective_gain(self, stem: str) -> float:
        if self.solo is not None:
            return self.gains[stem] if stem == self.solo else 0.0
        return 0.0 if self.mutes[stem] else self.gains[stem]

    def _mix_at(self, idx: np.ndarray) -> np.ndarray:
        """Linear-interp read of the stem mix at fractional sample indices."""
        i0 = idx.astype(np.int64)
        frac = (idx - i0)[:, None].astype(np.float32)
        i1 = np.minimum(i0 + 1, len(self.audio) - 1)
        if self.stems:
            out = np.zeros((len(idx), 2), dtype=np.float32)
            for s, data in self.stems.items():
                g = self._effective_gain(s)
                if g > 1e-4:
                    out += g * (data[i0] * (1 - frac) + data[i1] * frac)
            return out
        return self.audio[i0] * (1 - frac) + self.audio[i1] * frac

    def _callback(self, out: np.ndarray, frames: int, time_info, status) -> None:
        if not self.playing or self.audio is None:
            out[:] = 0.0
            return
        idx = self._playhead + np.arange(frames) * self.speed
        valid = idx < len(self.audio) - 1
        if not valid.all():
            if self.loop:
                idx = idx % (len(self.audio) - 1)
                out[:] = np.clip(self._mix_at(idx), -1.0, 1.0)
                self._playhead = float(idx[-1] + self.speed)
                self._last_tick_time = 0.0
                return
            out[:] = 0.0
            if valid.any():
                out[valid] = self._mix_at(idx[valid])
            self._playhead = float(len(self.audio) - 1)
            self.playing = False
            self.playback_finished.emit()
            return
        out[:] = np.clip(self._mix_at(idx), -1.0, 1.0)
        self._playhead += frames * self.speed

    # ------------------------------------------------------------- microphone

    def set_mic_enabled(self, enabled: bool) -> None:
        self.mic_enabled = enabled
        if enabled:
            self.pause()
            try:
                self._mic_stream = sd.InputStream(
                    samplerate=self.sr, channels=1, dtype="float32",
                    blocksize=1024, callback=self._mic_callback)
                self._mic_stream.start()
            except Exception as e:
                self.mic_enabled = False
                self.error.emit(f"Microphone unavailable: {e}")
        elif self._mic_stream is not None:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None

    def _mic_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        mono = indata[:, 0]
        n = len(self._mic_ring)
        w = self._mic_write
        end = w + frames
        if end <= n:
            self._mic_ring[w:end] = mono
        else:
            k = n - w
            self._mic_ring[w:] = mono[:k]
            self._mic_ring[: end - n] = mono[k:]
        self._mic_write = end % n
        self._mic_time += frames / self.sr

    def _mic_window(self) -> np.ndarray:
        w = self._mic_write
        return np.concatenate([self._mic_ring[w:], self._mic_ring[:w]])[-FFT_SIZE:]

    _MIC_MASKS = {
        "bass":   np.array([1.0, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0]),
        "vocals": np.array([0.0, 0.0, 0.3, 1.0, 1.0, 0.4, 0.1]),
        "other":  np.array([0.1, 0.2, 1.0, 0.3, 0.3, 0.8, 1.0]),
    }

    def _apply_mic_proxy_stems(self, frame) -> None:
        """Live input has no stem separation; approximate stems from bands +
        transients so stem-driven effects (grain, polyhedra) still react."""
        b = frame.bands
        drums = float(np.clip(frame.attack.max() * 1.1 + frame.punch * 0.4,
                              0, 1))
        frame.stem_energy = {
            "drums": drums,
            "bass": float(b[:2].mean()),
            "vocals": float(b[3:5].mean()),
            "other": float((b[2] + b[5:].mean()) / 2),
        }
        frame.stem_bands = {
            "drums": np.clip(frame.attack * 1.2, 0, 1),
            **{name: b * mask for name, mask in self._MIC_MASKS.items()},
        }

    # ------------------------------------------------------------- analysis

    def analysis_tick(self) -> AnalysisFrame:
        """Called by the GUI timer (~30 Hz). Returns the current AnalysisFrame."""
        if self.mic_enabled:
            frame = self.analyzer.analyze(self._mic_time, self._mic_window())
            self._apply_mic_proxy_stems(frame)
            self._last_frame = frame
            return self._last_frame

        if self.audio is None:
            return self._last_frame

        pos = int(self._playhead)
        self.position_changed.emit(self.position)
        if not self.playing:
            # decay the last frame so visuals settle instead of freezing
            f = self._last_frame
            f.bands = f.bands * 0.92
            f.rms *= 0.92
            f.beat = False
            f.punch = 0.0
            f.attack = f.attack * 0.8
            for k in f.stem_energy:
                f.stem_energy[k] *= 0.92
            return f

        start = max(0, min(pos, len(self.audio) - FFT_SIZE))
        stem_windows = None
        if self.stems:
            stem_windows = {}
            for s, data in self.stems.items():
                g = self._effective_gain(s)
                stem_windows[s] = np.asarray(
                    data[start:start + FFT_SIZE].mean(axis=1)) * max(g, 1e-6)
            mono = sum(stem_windows.values())
        else:
            mono = self.audio[start:start + FFT_SIZE].mean(axis=1)

        frame = self.analyzer.analyze(self.position, mono, stem_windows)

        # scheduled sync: the precomputed grid overrides live beat detection,
        # firing exactly when the playhead crosses a beat/onset
        t_prev, t_now = self._last_tick_time, self.position
        self.section_crossed = False
        if self.beat_grid is not None and t_now > t_prev:
            beat_s, punch_s = self.beat_grid.events_between(t_prev, t_now)
            frame.beat = beat_s > 0.0
            frame.beat_strength = beat_s
            frame.punch = max(frame.punch, punch_s)
            if self.beat_grid.bpm > 0:
                frame.bpm = self.beat_grid.bpm * self.speed
            self.section_crossed = self.beat_grid.section_crossed(t_prev, t_now)
            frame.section = self.section_crossed
            frame.anticipation = self.beat_grid.anticipation(t_now)
        self._last_tick_time = t_now

        self._last_frame = frame
        return self._last_frame

    def shutdown(self) -> None:
        self.stop_stream()
        self.set_mic_enabled(False)
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
