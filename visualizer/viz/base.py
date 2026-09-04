"""Base class shared by all six visualization modes."""
from __future__ import annotations

import numpy as np


class BaseMode:
    name = "base"
    camera = "turntable"      # "turntable" (3D) or "panzoom" (2D, e.g. fractal)
    camera_distance = 6.0
    camera_elevation: float | None = None   # set to force a viewing angle
    consumes_video = False    # True: mode renders the video itself (no backdrop)
    trail_scale = 1.0         # per-mode feedback-trail multiplier
    bloom_scale = 1.0         # per-mode glow multiplier (0 = crisp)

    def __init__(self, view, palette, settings, profile):
        self.view = view
        self.palette = palette
        self.settings = settings
        self.profile = profile
        self.visuals: list = []
        self.built = False

    def build(self) -> None:
        """Create vispy visuals (called once, lazily, on first activation)."""

    def update(self, frame, dt: float) -> None:
        """Advance animation from an AnalysisFrame."""

    def set_visible(self, visible: bool) -> None:
        if visible and not self.built:
            self.build()
            self.built = True
        for v in self.visuals:
            v.visible = visible

    def velocity_magnitude(self) -> float:
        """Aggregate speed of this mode's integrators (info display / debug)."""
        return 0.0

    @staticmethod
    def beat_kick(frame, scale: float = 1.0) -> float:
        return frame.beat_strength * scale if frame.beat else 0.0
