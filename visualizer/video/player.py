"""Video file support: frame-accurate playback synced to the audio clock.

The audio engine's playhead is the master clock. Each render tick we compute
the target frame index from the audio position and either step forward through
frames (cheap) or seek (when the gap is large). Sync accuracy is within one
frame (±1/fps).
"""
from __future__ import annotations

import numpy as np
import cv2

VIDEO_EXTS = (".mp4", ".mov", ".webm", ".avi", ".mkv")


def chromakey_mask(frame_rgb: np.ndarray) -> np.ndarray:
    """Alpha mask keying out green-screen pixels. Returns (h, w) float 0..1."""
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv, (40, 70, 70), (85, 255, 255))
    return 1.0 - (green.astype(np.float32) / 255.0)


class CameraSource:
    """Live webcam frames, same interface as VideoSource (time is ignored)."""

    def __init__(self, index: int = 0, max_height: int = 480):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise IOError("No webcam available (or camera access denied)")
        self.max_height = max_height
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame: np.ndarray | None = None

    def frame_at(self, t: float) -> np.ndarray | None:
        ok, bgr = self.cap.read()
        if ok and bgr is not None:
            h = bgr.shape[0]
            if h > self.max_height:
                s = self.max_height / h
                bgr = cv2.resize(bgr, None, fx=s, fy=s,
                                 interpolation=cv2.INTER_AREA)
            # mirror so it reads like a mirror, not a security camera
            self._frame = cv2.cvtColor(cv2.flip(bgr, 1), cv2.COLOR_BGR2RGB)
        return self._frame

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


class VideoSource:
    def __init__(self, path: str, max_height: int = 1080):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.max_height = max_height
        self._frame_idx = -1
        self._frame: np.ndarray | None = None

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    def thumbnail(self, at_fraction: float = 0.25) -> np.ndarray | None:
        """Preview frame for the UI before playback starts."""
        idx = int(self.n_frames * at_fraction)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._frame_idx = -1
        if not ok:
            return None
        return self._to_rgb(bgr)

    def _to_rgb(self, bgr: np.ndarray) -> np.ndarray:
        h = bgr.shape[0]
        if h > self.max_height:
            scale = self.max_height / h
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def frame_at(self, t: float) -> np.ndarray | None:
        """RGB frame for audio time t. Steps or seeks as needed; caches."""
        target = int(np.clip(t * self.fps, 0, max(0, self.n_frames - 1)))
        if target == self._frame_idx and self._frame is not None:
            return self._frame

        gap = target - self._frame_idx
        if gap < 0 or gap > int(self.fps):  # behind, or ahead by > 1s: seek
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self._frame_idx = target - 1
            gap = 1
        ok, bgr = False, None
        for _ in range(min(gap, int(self.fps) + 1)):  # bounded catch-up
            ok, bgr = self.cap.read()
            self._frame_idx += 1
        if ok and bgr is not None:
            self._frame = self._to_rgb(bgr)
        return self._frame

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass
