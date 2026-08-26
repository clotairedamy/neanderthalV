#!/usr/bin/env python3
"""Audio-Reactive 3D Geometric Visualizer — entry point.

Runs on macOS and Raspberry Pi 5. Auto-detects platform and applies a
performance profile (particle counts, fractal resolution, target FPS).
"""
import sys
import os


def main() -> int:
    # Vispy must know the backend before any canvas is created.
    os.environ.setdefault("VISPY_APP_BACKEND", "pyqt6")

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt

    import vispy
    vispy.use(app="pyqt6")

    app = QApplication(sys.argv)
    app.setApplicationName("NeanderthalV")
    app.setOrganizationName("NeanderthalV")

    from visualizer.config import Settings
    from visualizer.ui.main_window import MainWindow

    settings = Settings.load()
    win = MainWindow(settings)
    win.show()

    rc = app.exec()
    settings.save()
    return rc


if __name__ == "__main__":
    sys.exit(main())
