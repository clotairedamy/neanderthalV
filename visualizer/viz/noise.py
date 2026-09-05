"""Vectorized 3D gradient (Perlin) noise.

The grid modes are generative rather than traced from a file: the source
drawings are reproduced by evaluating a noise field, which is what makes
column count and feature size live controls instead of baked-in constants.
The third dimension is time, so the field flows rather than merely jitters.
"""
from __future__ import annotations

import numpy as np

# Perlin's improved fade, 6t^5-15t^4+10t^3: zero first and second derivative
# at the cell corners, so no creases show along the lattice
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def permutation(seed: int = 0) -> np.ndarray:
    p = np.random.default_rng(seed).permutation(256).astype(np.int32)
    return np.concatenate([p, p])


# Measured peak of this gradient set in 3D over 2.4M samples across six
# seeds. The often-quoted 1/sqrt(2) is the 2D figure and over-amplifies here
# by ~1.4x, which pushed the field outside the range callers map from.
_PEAK = 0.99107


def perlin3(x, y, z, perm: np.ndarray) -> np.ndarray:
    """Noise at the given coordinates. x/y/z broadcast together."""
    x, y, z = np.broadcast_arrays(np.asarray(x, np.float32),
                                  np.asarray(y, np.float32),
                                  np.asarray(z, np.float32))
    xi = np.floor(x).astype(np.int32)
    yi = np.floor(y).astype(np.int32)
    zi = np.floor(z).astype(np.int32)
    xf, yf, zf = x - xi, y - yi, z - zi
    u, v, w = _fade(xf), _fade(yf), _fade(zf)
    xi &= 255
    yi &= 255
    zi &= 255

    def corner(dx, dy, dz):
        # Perlin's own gradient selection: the twelve cube-edge gradients
        # have components in {-1, 0, 1}, so the dot product is a pair of
        # signed picks rather than a multiply. Indexing a gradient table
        # instead materializes an (N, 3) float array per corner, and with
        # eight corners per sample that dominated the frame -- this is the
        # same field, several times cheaper.
        h = perm[perm[perm[(xi + dx) & 255] + ((yi + dy) & 255)]
                 + ((zi + dz) & 255)] & 15
        x1, y1, z1 = xf - dx, yf - dy, zf - dz
        u = np.where(h < 8, x1, y1)
        v = np.where(h < 4, y1, np.where((h == 12) | (h == 14), x1, z1))
        return (np.where(h & 1, -u, u) + np.where(h & 2, -v, v))

    def lerp(a, b, t):
        return a + t * (b - a)

    x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), u)
    x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), u)
    x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), u)
    x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), u)
    out = lerp(lerp(x00, x10, v), lerp(x01, x11, v), w) / _PEAK
    # the peak is empirical, so clamp to make [-1, 1] a guarantee callers
    # can map from rather than an observation that usually holds
    return np.clip(out, -1.0, 1.0)
