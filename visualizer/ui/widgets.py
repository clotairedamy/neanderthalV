"""Custom UI widgets: spectrum/info display, palette swatches, stem mixer."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QSize
from PyQt6.QtGui import (QBrush, QColor, QLinearGradient, QPainter,
                         QPolygonF)
from PyQt6.QtWidgets import (QColorDialog, QGridLayout, QGroupBox, QHBoxLayout,
                             QLabel, QPushButton, QSlider, QVBoxLayout, QWidget)

from ..audio.analyzer import STEMS
from .style import MONO_FONT


class SpectrumWidget(QWidget):
    """Frequency spectrum + energy meter + stem energy bars."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.spectrum = np.zeros(64)
        self.rms = 0.0
        self.stem_energy = {s: 0.0 for s in STEMS}
        self.stem_history = {s: np.zeros(120) for s in STEMS}

    def push(self, frame) -> None:
        self.spectrum = frame.spectrum
        self.rms = frame.rms
        self.stem_energy = dict(frame.stem_energy)
        for s in STEMS:
            h = self.stem_history[s]
            h[:-1] = h[1:]
            h[-1] = frame.stem_energy.get(s, 0.0)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(12, 12, 20))

        # spectrum (top 60%)
        sh = int(h * 0.58)
        n = len(self.spectrum)
        bw = w / n
        grad = QLinearGradient(0, sh, 0, 0)
        grad.setColorAt(0.0, QColor(60, 120, 255))
        grad.setColorAt(1.0, QColor(255, 80, 180))
        p.setBrush(QBrush(grad))
        p.setPen(Qt.PenStyle.NoPen)
        for i, v in enumerate(self.spectrum):
            bh = int(v * (sh - 4))
            p.drawRect(int(i * bw), sh - bh, max(1, int(bw) - 1), bh)

        # energy meter (thin strip)
        p.setBrush(QColor(90, 230, 140))
        p.drawRect(0, sh + 2, int(w * np.clip(self.rms, 0, 1)), 5)

        # stem energy waveforms (bottom)
        colors = {"vocals": QColor(255, 120, 200), "drums": QColor(255, 200, 80),
                  "bass": QColor(100, 160, 255), "other": QColor(140, 255, 160)}
        y0 = sh + 12
        lane = (h - y0) / 4
        pts_x = np.linspace(0, w, len(next(iter(self.stem_history.values()))))
        for k, s in enumerate(STEMS):
            hist = self.stem_history[s]
            p.setPen(colors[s])
            base = y0 + lane * (k + 0.9)
            ys = base - hist * (lane - 2)
            # one polyline beats ~120 drawLine calls per stem per tick
            p.drawPolyline(QPolygonF([QPointF(float(x), float(y))
                                      for x, y in zip(pts_x, ys)]))
        p.end()


class PaletteSwatches(QWidget):
    """Preview of extracted image colors; click a swatch to adjust it manually."""

    colors_changed = pyqtSignal(object)   # np.ndarray (k,3)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.colors: np.ndarray | None = None
        self.setMinimumHeight(28)
        self.setToolTip("Extracted image palette — click a swatch to edit")

    def set_colors(self, colors) -> None:
        self.colors = np.asarray(colors) if colors is not None else None
        self.update()

    def sizeHint(self):
        return QSize(200, 28)

    def paintEvent(self, ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        if self.colors is None or len(self.colors) == 0:
            p.fillRect(0, 0, w, h, QColor(30, 30, 38))
            p.setPen(QColor(120, 120, 130))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no image loaded")
        else:
            sw = w / len(self.colors)
            for i, c in enumerate(self.colors):
                p.fillRect(int(i * sw), 0, int(sw) + 1, h,
                           QColor(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)))
        p.end()

    def mousePressEvent(self, ev):
        if self.colors is None or len(self.colors) == 0:
            return
        i = int(ev.position().x() / self.width() * len(self.colors))
        i = min(i, len(self.colors) - 1)
        c = self.colors[i]
        initial = QColor(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
        picked = QColorDialog.getColor(initial, self, "Adjust palette color")
        if picked.isValid():
            self.colors[i] = [picked.redF(), picked.greenF(), picked.blueF()]
            self.update()
            self.colors_changed.emit(self.colors)


class StemMixer(QGroupBox):
    """Per-stem volume sliders with solo/mute."""

    gain_changed = pyqtSignal(str, float)
    mute_changed = pyqtSignal(str, bool)
    solo_changed = pyqtSignal(object)     # stem name or None

    def __init__(self, parent=None):
        super().__init__("Stem Mixer", parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(8, 8, 8, 8)
        self.solo_buttons: dict[str, QPushButton] = {}
        self.mute_buttons: dict[str, QPushButton] = {}
        self.sliders: dict[str, QSlider] = {}

        for col, stem in enumerate(STEMS):
            lbl = QLabel(stem.capitalize())
            lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            sl = QSlider(Qt.Orientation.Vertical)
            sl.setRange(0, 150)
            sl.setValue(100)
            sl.setMinimumHeight(80)
            sl.valueChanged.connect(
                lambda v, s=stem: self.gain_changed.emit(s, v / 100.0))
            solo = QPushButton("S")
            solo.setCheckable(True)
            solo.setFixedWidth(28)
            solo.setToolTip(f"Solo {stem}")
            solo.toggled.connect(lambda on, s=stem: self._on_solo(s, on))
            mute = QPushButton("M")
            mute.setCheckable(True)
            mute.setFixedWidth(28)
            mute.setToolTip(f"Mute {stem}")
            mute.toggled.connect(lambda on, s=stem: self.mute_changed.emit(s, on))

            grid.addWidget(lbl, 0, col)
            grid.addWidget(sl, 1, col, alignment=Qt.AlignmentFlag.AlignHCenter)
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.addWidget(solo)
            rl.addWidget(mute)
            grid.addWidget(row, 2, col, alignment=Qt.AlignmentFlag.AlignHCenter)

            self.sliders[stem] = sl
            self.solo_buttons[stem] = solo
            self.mute_buttons[stem] = mute

    def _on_solo(self, stem: str, on: bool) -> None:
        if on:
            for s, b in self.solo_buttons.items():
                if s != stem and b.isChecked():
                    b.blockSignals(True)
                    b.setChecked(False)
                    b.blockSignals(False)
            self.solo_changed.emit(stem)
        elif not any(b.isChecked() for b in self.solo_buttons.values()):
            self.solo_changed.emit(None)

    def toggle_mute(self, stem: str) -> None:
        b = self.mute_buttons[stem]
        b.setChecked(not b.isChecked())


class InfoBar(QWidget):
    """BPM / FPS / velocity / mode readout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        self.bpm = QLabel("BPM —")
        self.fps = QLabel("FPS —")
        self.vel = QLabel("vel —")
        self.mode = QLabel("")
        self.status = QLabel("")
        self.status.setStyleSheet("color: #8a8;")
        for w in (self.bpm, self.fps, self.vel, self.mode):
            w.setStyleSheet(f"font-family: {MONO_FONT}; color: #ccd;")
            lay.addWidget(w)
        lay.addStretch(1)
        lay.addWidget(self.status)

    def push(self, frame, fps: float, vel: float, mode_name: str) -> None:
        self.bpm.setText(f"BPM {frame.bpm:5.1f}" if frame.bpm else "BPM —")
        self.fps.setText(f"FPS {fps:3.0f}")
        self.vel.setText(f"vel {vel:5.2f}")
        self.mode.setText(mode_name)
