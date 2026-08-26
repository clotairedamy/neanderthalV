"""Velocity/inertia system used by every visualization.

All animated quantities (position, rotation, scale, deformation) move through
one of these integrators instead of jumping to target values:

    value  <- value + velocity * dt
    velocity <- (velocity + (target - value) * accel * dt) * damping^(dt*60)

Beat impulses add instantaneous velocity ("acceleration bursts"); damping
gives momentum decay so geometry glides when the audio drops.
"""
from __future__ import annotations

import numpy as np


class VelocityValue:
    """Scalar with target-seeking acceleration, damping and impulses."""

    def __init__(self, value: float = 0.0, accel: float = 10.0,
                 damping: float = 0.85):
        self.value = float(value)
        self.target = float(value)
        self.velocity = 0.0
        self.accel = accel
        self.damping = damping

    def set_target(self, t: float) -> None:
        self.target = float(t)

    def impulse(self, amount: float) -> None:
        self.velocity += amount

    def update(self, dt: float) -> float:
        dt = min(dt, 0.1)
        self.velocity += (self.target - self.value) * self.accel * dt
        # frame-rate independent exponential damping
        self.velocity *= self.damping ** (dt * 60.0)
        self.value += self.velocity * dt
        return self.value

    @property
    def speed(self) -> float:
        return abs(self.velocity)


class VelocityVector:
    """N-dimensional (default 3D) vector version, vectorized with numpy."""

    def __init__(self, value=None, dims: int = 3, accel: float = 10.0,
                 damping: float = 0.85):
        self.value = np.zeros(dims) if value is None else np.asarray(value, float).copy()
        self.target = self.value.copy()
        self.velocity = np.zeros_like(self.value)
        self.accel = accel
        self.damping = damping

    def set_target(self, t) -> None:
        self.target = np.asarray(t, float)

    def impulse(self, vec) -> None:
        self.velocity += np.asarray(vec, float)

    def update(self, dt: float) -> np.ndarray:
        dt = min(dt, 0.1)
        self.velocity += (self.target - self.value) * self.accel * dt
        self.velocity *= self.damping ** (dt * 60.0)
        self.value += self.velocity * dt
        return self.value

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))


class VelocityArray:
    """Array-of-scalars integrator (e.g. one value per band, per face, per ring)."""

    def __init__(self, n: int, accel: float = 12.0, damping: float = 0.85):
        self.value = np.zeros(n)
        self.target = np.zeros(n)
        self.velocity = np.zeros(n)
        self.accel = accel
        self.damping = damping

    def set_target(self, t) -> None:
        self.target[:] = t

    def impulse(self, amounts) -> None:
        self.velocity += amounts

    def update(self, dt: float) -> np.ndarray:
        dt = min(dt, 0.1)
        self.velocity += (self.target - self.value) * self.accel * dt
        self.velocity *= self.damping ** (dt * 60.0)
        self.value += self.velocity * dt
        return self.value

    @property
    def speed(self) -> float:
        return float(np.mean(np.abs(self.velocity)))
