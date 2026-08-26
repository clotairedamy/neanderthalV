"""Export stems for the Teenage Engineering SP-1 stem player.

Format per https://solderless.engineering/stemloader/help/ :
- one WAV per song: 24-bit, 48 kHz, 8-channel PCM ("WAV (Microsoft)")
- 4 stereo stems interleaved as channels:
    ch 1/2 = stem 1 L/R,  ch 3/4 = stem 2 L/R,
    ch 5/6 = stem 3 L/R,  ch 7/8 = stem 4 L/R
- separate stem files must NOT be loaded onto the SP-1
- a "NNBPM" token in the filename makes the stem loader auto-set the song
  tempo (valid range 30-300, loader default is 80)
"""
from __future__ import annotations

import os
import re

import numpy as np
import soundfile as sf

SP1_SR = 48000
SP1_SUBTYPE = "PCM_24"

# SP-1 track buttons 1-4, in this stem order
SP1_STEM_ORDER = ["vocals", "drums", "bass", "other"]


def _resample_stereo(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.asarray(data, np.float32)
    import librosa
    return np.stack([
        librosa.resample(np.ascontiguousarray(data[:, ch], dtype=np.float32),
                         orig_sr=sr_in, target_sr=sr_out)
        for ch in range(2)], axis=1)


def sp1_filename(song_name: str, bpm: float | None) -> str:
    """Sanitized name with the loader's BPM auto-detect token when known."""
    base = os.path.splitext(os.path.basename(song_name))[0]
    base = re.sub(r"[^\w\- ]+", "", base).strip() or "song"
    base = re.sub(r"\s*\d+\s*BPM", "", base, flags=re.IGNORECASE).strip("_ ")
    if bpm and 30 <= round(bpm) <= 300:
        return f"{base}_{round(bpm)}BPM.wav"
    return f"{base}.wav"


def export_sp1_wav(stems: dict[str, np.ndarray], sr: int, out_path: str,
                   progress=None) -> str:
    """Write the 8-channel 24-bit 48 kHz WAV the SP-1 stem loader expects.

    `stems`: stem name -> (n, 2) float array at sample rate `sr`.
    Missing stems become silent channels (the loader zero-fills anyway).
    """
    def report(msg):
        if progress:
            progress(msg)

    resampled = []
    n_out = 0
    for name in SP1_STEM_ORDER:
        data = stems.get(name)
        if data is None:
            resampled.append(None)
            continue
        report(f"Resampling {name} to 48 kHz…")
        r = _resample_stereo(np.asarray(data), sr, SP1_SR)
        resampled.append(r)
        n_out = max(n_out, len(r))

    if n_out == 0:
        raise ValueError("No stems to export")

    out = np.zeros((n_out, 8), dtype=np.float32)
    for i, r in enumerate(resampled):
        if r is not None:
            out[: len(r), 2 * i: 2 * i + 2] = r

    # keep headroom identical across stems: one common normalization only
    peak = float(np.abs(out).max())
    if peak > 1.0:
        out /= peak * 1.005

    report("Writing 8-channel 24-bit WAV…")
    sf.write(out_path, out, SP1_SR, subtype=SP1_SUBTYPE)
    report(f"SP-1 file ready: {os.path.basename(out_path)}")
    return out_path
