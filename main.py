#!/usr/bin/env python3
"""Audio-Reactive 3D Geometric Visualizer — entry point.

Runs on macOS and Raspberry Pi 5. Auto-detects platform and applies a
performance profile (particle counts, fractal resolution, target FPS).
"""
import sys
import os


def _quiet_third_party_warnings() -> None:
    """Silence two harmless-but-noisy warnings from dependencies.

    librosa's numba ufuncs get compiled once per worker thread, and numba
    warns that the signature was already compiled. It is not actionable from
    here and fires on every track load.
    """
    import warnings
    warnings.filterwarnings(
        "ignore", message=r".*previously compiled argument types.*")
    try:
        from numba.core.errors import NumbaWarning
        warnings.filterwarnings("ignore", category=NumbaWarning,
                                message=r".*Compilation requested.*")
    except Exception:
        pass


def main() -> int:
    # Vispy must know the backend before any canvas is created.
    os.environ.setdefault("VISPY_APP_BACKEND", "pyqt6")
    _quiet_third_party_warnings()

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt

    import vispy
    vispy.use(app="pyqt6")

    app = QApplication(sys.argv)
    from visualizer.ui.style import STYLESHEET
    app.setStyleSheet(STYLESHEET)
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
