"""Offline video export: renders every frame against the audio clock.

Unlike live recording (bounded by GPU readback speed), this steps time in
exact 1/fps increments, so the output is a true constant-60fps file with
every frame rendered — at whatever resolution the canvas has. Audio is the
current stem mix; the beat grid drives scheduled impulses identically to
live playback.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from .audio.analyzer import FFT_SIZE, AudioAnalyzer
from .audio.engine import _ffmpeg_exe


class OfflineFrameSource:
    """Reproduces engine.analysis_tick offline: fresh analyzer + beat grid."""

    def __init__(self, engine, settings, analysis_hz: int = 30):
        self.engine = engine
        self.analyzer = AudioAnalyzer(engine.sr, settings)
        self.analysis_dt = 1.0 / analysis_hz
        self._next_analysis = 0.0
        self._last_t = 0.0
        self._frame = None

    def frame_at(self, t: float):
        if self._frame is not None and t < self._next_analysis:
            return self._frame
        self._next_analysis = t + self.analysis_dt
        eng = self.engine
        sr = eng.sr
        pos = int(t * sr)
        start = max(0, min(pos, len(eng.audio) - FFT_SIZE))
        stem_windows = None
        if eng.stems:
            stem_windows = {}
            for s, data in eng.stems.items():
                g = eng._effective_gain(s)
                stem_windows[s] = np.asarray(
                    data[start:start + FFT_SIZE].mean(axis=1)) * max(g, 1e-6)
            mono = sum(stem_windows.values())
        else:
            mono = eng.audio[start:start + FFT_SIZE].mean(axis=1)
        frame = self.analyzer.analyze(t, mono, stem_windows)
        if eng.beat_grid is not None:
            beat_s, punch_s = eng.beat_grid.events_between(self._last_t, t)
            frame.beat = beat_s > 0.0
            frame.beat_strength = beat_s
            frame.punch = max(frame.punch, punch_s)
            if eng.beat_grid.bpm > 0:
                frame.bpm = eng.beat_grid.bpm
        self._last_t = t
        self._frame = frame
        return frame


def mixed_audio(engine) -> np.ndarray:
    """Full-length stereo mix honoring the current stem gains/solo/mutes."""
    if not engine.stems:
        return engine.audio
    out = np.zeros_like(engine.audio)
    for s, data in engine.stems.items():
        g = engine._effective_gain(s)
        if g > 1e-4:
            out += np.asarray(data) * g
    return np.clip(out, -1.0, 1.0)


def offline_export(viz, engine, settings, path: str, fmt: str = "screen",
                   fps: int = 60, progress=None, should_cancel=None,
                   process_events=None) -> str:
    """Render engine's loaded song frame-by-frame and encode with audio.

    Must run on the GUI thread (GL context). `progress(done, total)` is
    called per frame; `should_cancel()` aborts; `process_events()` keeps the
    UI alive.
    """
    if engine.audio is None:
        raise ValueError("No audio loaded")
    import cv2

    duration = engine.duration
    total = int(duration * fps)
    src = OfflineFrameSource(engine, settings)

    tmp_vid = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = None
    size = None
    cancelled = False
    try:
        for i in range(total):
            t = i / fps
            frame = src.frame_at(t)
            viz.render_tick(frame, 1.0 / fps, t)
            img = viz.grab_frame()
            h, w = img.shape[0] - img.shape[0] % 2, img.shape[1] - img.shape[1] % 2
            if writer is None:
                size = (w, h)
                writer = cv2.VideoWriter(
                    tmp_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(cv2.cvtColor(img[:h, :w, :3], cv2.COLOR_RGB2BGR))
            if progress:
                progress(i + 1, total)
            if process_events:
                process_events()
            if should_cancel and should_cancel():
                cancelled = True
                break
    finally:
        if writer is not None:
            writer.release()
    if cancelled or writer is None:
        try:
            os.unlink(tmp_vid)
        except OSError:
            pass
        return ""

    # encode: exact-fps input, target format scaling, mux current stem mix
    import soundfile as sf
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(tmp_wav, mixed_audio(engine), engine.sr)

    vf = []
    fmt_size = viz.FORMATS.get(fmt)
    if fmt_size:
        w, h = fmt_size
        vf = [f"scale={w}:{h}:force_original_aspect_ratio=increase",
              f"crop={w}:{h}", "setsar=1"]
    cmd = [_ffmpeg_exe(), "-y", "-i", tmp_vid, "-i", tmp_wav]
    if vf:
        cmd += ["-filter:v", ",".join(vf)]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", str(fps),
            "-c:a", "aac", "-b:a", "192k", "-shortest", path]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.unlink(tmp_vid)
    except Exception:
        os.replace(tmp_vid, path)
    finally:
        try:
            os.unlink(tmp_wav)
        except OSError:
            pass
    return path
