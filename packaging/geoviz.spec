# PyInstaller spec — builds GeoViz.app for macOS.
# Usage: pyinstaller packaging/geoviz.spec  (run from the project root)
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = (
    collect_submodules("vispy")
    + collect_submodules("vispy.app.backends")
    + ["vispy.app.backends._pyqt6", "sounddevice", "soundfile",
       "scipy._cyutility", "PIL.Image"]
)
datas = (collect_data_files("vispy") + collect_data_files("librosa")
         + [("../assets/kinect.mp4", "assets")])

try:
    import demucs  # noqa: F401
    hiddenimports += collect_submodules("demucs")
except ImportError:
    pass

a = Analysis(
    ["../main.py"],
    pathex=[".."],
    hiddenimports=hiddenimports,
    datas=datas,
    excludes=["tkinter", "matplotlib.tests", "PyQt5"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts,
    exclude_binaries=True,
    name="GeoViz",
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="GeoViz")
app = BUNDLE(
    coll,
    name="GeoViz.app",
    icon="GeoViz.icns",
    bundle_identifier="com.geoviz.app",
    info_plist={
        "NSMicrophoneUsageDescription":
            "GeoViz analyzes live microphone audio to drive visualizations.",
        "NSHighResolutionCapable": True,
    },
)
