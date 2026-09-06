"""Mode 13 -- Footage.

Draws nothing at all, so the loaded video is the entire picture.

Every other mode paints something over the background layer, and the
opaque ones (the fractal quad, for instance) hide it completely. When the
clip *is* the piece -- storm footage cut to its own lightning on the beat
-- there needs to be a way to see it with nothing on top, and that is a
visualization that renders no geometry rather than a new video path.

The clip is still driven by the audio, and by the flash cueing in
VizManager, so this shows the real frames landing on the real beats.
"""
from __future__ import annotations

from .base import BaseMode


class FootageMode(BaseMode):
    name = "Footage (video only)"
    # 2D so the auto-camera has nothing to spin; there is no geometry for a
    # turntable to orbit anyway
    camera = "panzoom"
    trail_scale = 0.0           # trails would smear the footage
    bloom_scale = 0.0           # and glow would wash it out

    def build(self):
        self.visuals = []

    def update(self, frame, dt):
        # nothing to animate: the background video layer is the whole image
        pass

    def velocity_magnitude(self):
        return 0.0
