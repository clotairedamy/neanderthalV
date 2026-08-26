"""Optional MIDI control (VJ use): notes switch modes, CCs tweak parameters.

Requires `mido` + `python-rtmidi` (in requirements.txt). If they're missing
or no MIDI device is connected, this stays silently inactive.

Mapping:
  notes 36-43 (C1..G1)  -> visualization modes 1-8
  CC 1  (mod wheel)     -> sensitivity   0.1 .. 3.0
  CC 2                  -> velocity damping 0.70 .. 0.95
  CC 3                  -> beat impulse  0.0 .. 3.0
  CC 4                  -> grain intensity 0.1 .. 3.0
"""
from __future__ import annotations

import time

from PyQt6.QtCore import QThread, pyqtSignal


def midi_available() -> bool:
    try:
        import mido            # noqa: F401
        import rtmidi          # noqa: F401
        return True
    except Exception:
        return False


class MidiInput(QThread):
    note_on = pyqtSignal(int, int)     # note, velocity
    cc = pyqtSignal(int, int)          # controller, value
    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            import mido
            names = mido.get_input_names()
        except Exception as e:
            self.status.emit(f"MIDI unavailable: {e}")
            return
        if not names:
            self.status.emit("No MIDI devices found")
            return
        try:
            ports = [mido.open_input(n) for n in names]
        except Exception as e:
            self.status.emit(f"MIDI open failed: {e}")
            return
        self.status.emit("MIDI: " + ", ".join(names))
        try:
            while not self._stop:
                for p in ports:
                    for msg in p.iter_pending():
                        if msg.type == "note_on" and msg.velocity > 0:
                            self.note_on.emit(msg.note, msg.velocity)
                        elif msg.type == "control_change":
                            self.cc.emit(msg.control, msg.value)
                time.sleep(0.005)
        finally:
            for p in ports:
                try:
                    p.close()
                except Exception:
                    pass
