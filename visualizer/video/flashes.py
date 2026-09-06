"""Find the flashes in a video, so playback can be cut to them on the beat.

Written for storm footage -- lightning is the obvious case -- but the test
is generic: a flash is a short, sharp rise in frame brightness above the
*local* baseline, not above a global one. That distinction matters, because
footage drifts (a sunset dims, a camera re-exposes) and a global threshold
either misses flashes in the bright stretches or fires constantly in the
dark ones.

Measured on the reference clip (86s of a thunderhead at dusk): 59 flashes,
median duration 233 ms, median 0.87 s apart, peaks around +230% over the
local baseline.

Results are cached next to the app's other caches, keyed by path, size and
mtime, because scanning is a few seconds and reloading the same clip
should not pay it twice.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

# a flash has to rise this far over the local baseline to count
RISE = 0.18
# frames closer together than this belong to the same flash
JOIN = 4
# the window the baseline is measured over: long enough to ignore a flash,
# short enough to track the footage drifting
BASELINE_FRAMES = 31


def _cache_path(video_path: str) -> str:
    from ..config import cache_dir
    try:
        st = os.stat(video_path)
        key = f"{os.path.abspath(video_path)}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        key = os.path.abspath(video_path)
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir(), f"flashes-{h}.json")


def brightness_track(video_path: str, size: int = 64) -> tuple[np.ndarray, float]:
    """Mean luminance per frame, decoded small. Returns (track, fps)."""
    import subprocess

    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    fps = 30.0
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    except Exception:
        pass
    # one greyscale decode at thumbnail size: far cheaper than stepping the
    # real decoder, and brightness is all this needs
    proc = subprocess.run(
        [exe, "-v", "error", "-i", video_path, "-vf", f"scale={size}:{size}",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True)
    buf = proc.stdout
    n = len(buf) // (size * size)
    if n == 0:
        return np.zeros(0, np.float32), fps
    v = np.frombuffer(buf, np.uint8)[:n * size * size]
    return v.reshape(n, -1).mean(1).astype(np.float32) / 255.0, fps


def find_flashes(track: np.ndarray, fps: float,
                 rise: float = RISE) -> list[dict]:
    """Flash onsets in a brightness track -> [{t, dur, peak}, ...]."""
    if len(track) < BASELINE_FRAMES:
        return []
    from scipy.ndimage import median_filter
    # the median ignores the flashes themselves, so the baseline is the
    # footage's ambient level rather than something the flashes drag up
    base = median_filter(track, size=BASELINE_FRAMES)
    excess = (track - base) / np.maximum(base, 1e-3)

    hits = np.flatnonzero(excess > rise)
    groups: list[list[int]] = []
    for i in hits:
        if groups and i - groups[-1][-1] <= JOIN:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return [{"t": g[0] / fps,
             "dur": (g[-1] - g[0] + 1) / fps,
             "peak": float(excess[g[0]:g[-1] + 1].max())}
            for g in groups]


def detect_flashes(video_path: str, use_cache: bool = True) -> list[dict]:
    """Flashes in a video, cached. Empty list if the clip has none."""
    cp = _cache_path(video_path)
    if use_cache and os.path.exists(cp):
        try:
            with open(cp) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    track, fps = brightness_track(video_path)
    out = find_flashes(track, fps)
    try:
        with open(cp, "w") as fh:
            json.dump(out, fh)
    except OSError:
        pass
    return out
