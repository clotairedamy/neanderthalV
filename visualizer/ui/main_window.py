"""Main application window: canvas + control panel, drag-drop, shortcuts."""
from __future__ import annotations

import json
import os
import time

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
    QMainWindow,
    QMessageBox, QProgressDialog, QPushButton, QRadioButton, QScrollArea,
    QSlider, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from ..audio.analyzer import STEMS, AnalysisFrame
from ..audio.engine import AUDIO_EXTS, AudioEngine, extract_audio_from_video
from ..color.palette import PaletteManager
from ..config import active_profile
from ..video.player import VIDEO_EXTS, VideoSource
from ..viz.manager import MODE_CLASSES, VizManager
from .style import MONO_FONT
from .widgets import InfoBar, PaletteSwatches, SpectrumWidget, StemMixer

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff")


class MainWindow(QMainWindow):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.profile = active_profile()
        self.setWindowTitle("NeanderthalV — Audio-Reactive 3D Visualizer")
        self.resize(1280, 800)
        self.setAcceptDrops(True)

        self.palette_mgr = PaletteManager(settings)
        self.engine = AudioEngine(settings, self)
        self.viz = VizManager(settings, self.palette_mgr, self.profile)

        self._frame = AnalysisFrame()
        self._last_render = time.time()
        self._seeking = False
        self._recording = False

        try:
            self._mode_prefs = json.loads(settings.mode_prefs or "{}")
        except ValueError:
            self._mode_prefs = {}
        self._queue: list[str] = []

        self._build_ui()
        self._connect_engine()
        self._install_shortcuts()
        self._start_timers()
        self._start_midi()

    # ------------------------------------------------------------------ UI

    @staticmethod
    def _scroll_tab():
        """A tab page that scrolls, returning (page widget, its layout)."""
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setSpacing(10)
        lay.setContentsMargins(10, 12, 10, 10)
        area = QScrollArea()
        area.setWidget(page)
        area.setWidgetResizable(True)
        return area, lay

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(splitter)

        # left: a header (presets) above tabbed control pages
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(10, 10, 10, 0)
        left_l.setSpacing(8)
        header_l = QVBoxLayout()
        left_l.addLayout(header_l)

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(False)
        self.tabs.tabBar().setElideMode(Qt.TextElideMode.ElideNone)
        self.tabs.tabBar().setExpanding(False)
        left_l.addWidget(self.tabs, stretch=1)
        src_tab, src_l = self._scroll_tab()
        mix_tab, mix_l = self._scroll_tab()
        vis_tab, vis_l = self._scroll_tab()
        mot_tab, mot_l = self._scroll_tab()
        fx_tab, fx_l = self._scroll_tab()
        exp_tab, exp_l = self._scroll_tab()

        # -- files
        files_box = QGroupBox("Files  (or drag && drop anywhere)")
        fl = QVBoxLayout(files_box)
        b_audio = QPushButton("Open Audio…  (MP3/WAV/FLAC/OGG)")
        b_audio.clicked.connect(self._browse_audio)
        b_image = QPushButton("Open Image…  (palette source)")
        b_image.clicked.connect(self._browse_image)
        b_video = QPushButton("Open Video…  (MP4/MOV/WebM)")
        b_video.clicked.connect(self._browse_video)
        self.mic_btn = QPushButton("🎤 Microphone: off")
        self.mic_btn.setCheckable(True)
        self.mic_btn.toggled.connect(self._toggle_mic)
        for b in (b_audio, b_image, b_video, self.mic_btn):
            fl.addWidget(b)
        self.file_label = QLabel("no file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: #99a;")
        fl.addWidget(self.file_label)
        self.queue_list = QListWidget()
        self.queue_list.setMaximumHeight(70)
        self.queue_list.setToolTip("Playlist — drop multiple audio files; "
                                   "double-click to play now")
        self.queue_list.itemDoubleClicked.connect(self._play_queued)
        self.queue_list.hide()
        fl.addWidget(self.queue_list)
        src_l.addWidget(files_box)

        # -- transport bar (built here, mounted under the canvas)
        self.transport = QWidget()
        self.transport.setObjectName("transport")
        tr = QHBoxLayout(self.transport)
        tr.setContentsMargins(12, 8, 12, 8)
        tr.setSpacing(10)
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(44)
        self.play_btn.setToolTip("Play / pause  (Space)")
        self.play_btn.clicked.connect(self._toggle_play)
        tr.addWidget(self.play_btn)
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setStyleSheet(f"font-family: {MONO_FONT};")
        tr.addWidget(self.time_label)
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.sliderPressed.connect(lambda: setattr(self, "_seeking", True))
        self.seek_slider.sliderReleased.connect(self._seek_released)
        tr.addWidget(self.seek_slider, stretch=1)
        tr.addWidget(QLabel("Speed"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(50, 200)
        self.speed_slider.setValue(100)
        self.speed_slider.setFixedWidth(110)
        self.speed_slider.valueChanged.connect(
            lambda v: (self.engine.set_speed(v / 100),
                       self.speed_label.setText(f"{v / 100:.2f}x")))
        self.speed_label = QLabel("1.00x")
        self.speed_label.setStyleSheet(f"font-family: {MONO_FONT};")
        self.speed_slider.sliderReleased.connect(self.engine.apply_pitch_speed)
        tr.addWidget(self.speed_slider)
        tr.addWidget(self.speed_label)
        self.pitch_check = QCheckBox("Pitch")
        self.pitch_check.setChecked(self.settings.preserve_pitch)
        self.pitch_check.setToolTip("Preserve pitch: phase-vocoder stretch "
                                    "instead of turntable-style varispeed")
        self.pitch_check.toggled.connect(self._toggle_pitch)
        self.loop_check = QCheckBox("Loop")
        self.loop_check.toggled.connect(
            lambda on: setattr(self.engine, "loop", on))
        tr.addWidget(self.pitch_check)
        tr.addWidget(self.loop_check)

        # -- presets
        preset_box = QGroupBox("Presets")
        prl = QHBoxLayout(preset_box)
        self.preset_combo = QComboBox()
        self._refresh_presets()
        self.preset_combo.activated.connect(self._apply_preset)
        b_psave = QPushButton("Save…")
        b_psave.clicked.connect(self._save_preset)
        b_pdel = QPushButton("✕")
        b_pdel.setFixedWidth(34)
        b_pdel.setStyleSheet("padding: 6px 0;")
        b_pdel.setToolTip("Delete selected preset")
        b_pdel.clicked.connect(self._delete_preset)
        prl.addWidget(self.preset_combo, stretch=1)
        prl.addWidget(b_psave)
        prl.addWidget(b_pdel)
        header_l.addWidget(preset_box)

        # -- stem mixer
        self.mixer = StemMixer()
        self.mixer.gain_changed.connect(self.engine.set_stem_gain)
        self.mixer.mute_changed.connect(self.engine.set_stem_mute)
        self.mixer.solo_changed.connect(self.engine.set_solo)
        mix_l.addWidget(self.mixer)

        # -- visualization modes
        mode_box = QGroupBox("Visualization")
        ml = QVBoxLayout(mode_box)
        self.mode_radios = []
        for i, cls in enumerate(MODE_CLASSES):
            rb = QRadioButton(f"{i + 1}. {cls.name}")
            rb.setChecked(i == self.settings.viz_mode)
            rb.toggled.connect(lambda on, k=i: on and self._mode_changed(k))
            ml.addWidget(rb)
            self.mode_radios.append(rb)
        self.choreo_check = QCheckBox("Auto-switch at song sections")
        self.choreo_check.setChecked(self.settings.auto_choreograph)
        self.choreo_check.setToolTip("Advance to the next mode when the beat "
                                     "grid detects a verse/drop boundary")
        self.choreo_check.toggled.connect(
            lambda on: setattr(self.settings, "auto_choreograph", on))
        ml.addWidget(self.choreo_check)
        vis_l.addWidget(mode_box)

        # -- image colors
        color_box = QGroupBox("Image Colors")
        cl = QVBoxLayout(color_box)
        self.swatches = PaletteSwatches()
        self.swatches.colors_changed.connect(
            self.palette_mgr.set_image_palette_colors)
        cl.addWidget(self.swatches)
        self.use_img_check = QCheckBox("Use image colors (I)")
        self.use_img_check.setChecked(self.settings.use_image_colors)
        self.use_img_check.toggled.connect(self._toggle_image_colors)
        cl.addWidget(self.use_img_check)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Palette"))
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(["viridis", "plasma", "turbo", "neon",
                                     "mono", "custom", "image"])
        self.palette_combo.setCurrentText(self.settings.palette)
        self.palette_combo.currentTextChanged.connect(self.palette_mgr.set_builtin)
        prow.addWidget(self.palette_combo)
        cl.addLayout(prow)
        brow = QHBoxLayout()
        brow.addWidget(QLabel("Blend"))
        self.blend_combo = QComboBox()
        self.blend_combo.addItems(["overlay", "multiply", "screen"])
        self.blend_combo.setCurrentText(self.settings.color_blend_mode)
        self.blend_combo.currentTextChanged.connect(
            lambda t: setattr(self.settings, "color_blend_mode", t))
        brow.addWidget(self.blend_combo)
        cl.addLayout(brow)
        vis_l.addWidget(color_box)

        # -- video display
        vid_box = QGroupBox("Video Display")
        vl = QVBoxLayout(vid_box)
        self.video_combo = QComboBox()
        self.video_combo.addItems(["background", "pip", "texture", "off"])
        self.video_combo.setCurrentText(self.settings.video_display)
        self.video_combo.currentTextChanged.connect(
            lambda t: setattr(self.settings, "video_display", t))
        vl.addWidget(self.video_combo)
        self.chroma_check = QCheckBox("Chromakey (green screen → palette)")
        self.chroma_check.setChecked(self.settings.chromakey)
        self.chroma_check.toggled.connect(
            lambda on: setattr(self.settings, "chromakey", on))
        vl.addWidget(self.chroma_check)
        self.video_thumb = QLabel()
        self.video_thumb.setFixedHeight(70)
        self.video_thumb.setStyleSheet("background:#111; border:1px solid #333;")
        self.video_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_thumb.setText("no video")
        vl.addWidget(self.video_thumb)
        src_l.addWidget(vid_box)

        # -- depth point cloud (mode 9)
        pc_box = QGroupBox("Depth Point Cloud  (mode 9)")
        pcl = QVBoxLayout(pc_box)
        srow2 = QHBoxLayout()
        srow2.addWidget(QLabel("Source"))
        self.pc_source_combo = QComboBox()
        self.pc_source_combo.addItem("Auto (your video → photo → Kinect demo)",
                                     "auto")
        self.pc_source_combo.addItem("Kinect demo — three.js clip", "kinect")
        self.pc_source_combo.addItem("Video", "video")
        self.pc_source_combo.addItem("Photo relief", "image")
        self.pc_source_combo.addItem("Audio terrain", "audio")
        self.pc_source_combo.addItem("Webcam (live)", "camera")
        i = self.pc_source_combo.findData(self.settings.pc_source)
        self.pc_source_combo.setCurrentIndex(max(0, i))
        self.pc_source_combo.currentIndexChanged.connect(self._pc_source_changed)
        srow2.addWidget(self.pc_source_combo)
        pcl.addLayout(srow2)
        lrow = QHBoxLayout()
        lrow.addWidget(QLabel("Depth from"))
        self.pc_layout_combo = QComboBox()
        self.pc_layout_combo.addItem("Luminance (any video)", "luminance")
        self.pc_layout_combo.addItem("RGBD side-by-side", "side_by_side")
        self.pc_layout_combo.addItem("RGBD top/bottom", "top_bottom")
        i = self.pc_layout_combo.findData(self.settings.pc_layout)
        self.pc_layout_combo.setCurrentIndex(max(0, i))
        self.pc_layout_combo.currentIndexChanged.connect(
            lambda k: setattr(self.settings, "pc_layout",
                              self.pc_layout_combo.itemData(k)))
        lrow.addWidget(self.pc_layout_combo)
        pcl.addLayout(lrow)

        def pc_spin(label, lo, hi, step, attr, tip=""):
            r = QHBoxLayout()
            lb = QLabel(label)
            if tip:
                lb.setToolTip(tip)
            r.addWidget(lb)
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setValue(getattr(self.settings, attr))
            sp.valueChanged.connect(lambda v, a=attr: setattr(self.settings, a, v))
            r.addWidget(sp)
            pcl.addLayout(r)

        pc_spin("Near clip", 0.5, 20.0, 0.25, "pc_near")
        pc_spin("Far clip", 1.0, 30.0, 0.25, "pc_far")
        pc_spin("Point size", 0.5, 10.0, 0.2, "pc_point_size")
        pc_spin("Depth cutoff", 0.0, 0.9, 0.02, "pc_cutoff",
                "Discard points darker than this (drops the background)")
        pc_spin("Perspective", 0.0, 1.0, 0.05, "pc_perspective",
                "0 = flat relief · 1 = exact Kinect frustum (three.js original)")
        brow2 = QHBoxLayout()
        lb = QLabel("Band colors")
        lb.setToolTip("Paint lows / mids / highs across an axis of the cloud; "
                      "each zone lights up and pushes out with its own band")
        brow2.addWidget(lb)
        self.pc_band_combo = QComboBox()
        for label, data in (("Off", "off"), ("Vertical (low→high)", "vertical"),
                            ("By depth", "depth"), ("Horizontal", "horizontal"),
                            ("Radial", "radial")):
            self.pc_band_combo.addItem(label, data)
        i = self.pc_band_combo.findData(self.settings.pc_band_axis)
        self.pc_band_combo.setCurrentIndex(max(0, i))
        self.pc_band_combo.currentIndexChanged.connect(
            lambda k: setattr(self.settings, "pc_band_axis",
                              self.pc_band_combo.itemData(k)))
        brow2.addWidget(self.pc_band_combo)
        pcl.addLayout(brow2)
        pc_spin("Band push", 0.0, 0.4, 0.02, "pc_band_push",
                "How far each band's energy displaces its own zone")
        self._pc_box = pc_box
        vis_l.addWidget(pc_box)

        # -- reactive 3D text (mode 10)
        txt_box = QGroupBox("Reactive 3D Text  (mode 10)")
        txl = QVBoxLayout(txt_box)
        self.text_edit = QLineEdit(self.settings.text_content)
        self.text_edit.setPlaceholderText("Type your text…")
        self.text_edit.setToolTip("Rasterized into a 3D point cloud; the word "
                                  "doubles as a spectrum analyzer")
        self.text_edit.textChanged.connect(
            lambda s: setattr(self.settings, "text_content", s))
        txl.addWidget(self.text_edit)

        def txt_spin(label, lo, hi, step, attr, tip=""):
            r = QHBoxLayout()
            lb2 = QLabel(label)
            if tip:
                lb2.setToolTip(tip)
            r.addWidget(lb2)
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setValue(getattr(self.settings, attr))
            sp.valueChanged.connect(lambda v, a=attr: setattr(self.settings, a, v))
            r.addWidget(sp)
            txl.addLayout(r)

        txt_spin("Band depth", 0.0, 2.0, 0.05, "text_depth",
                 "How far each frequency band pushes its part of the word")
        txt_spin("Beat explode", 0.0, 4.0, 0.2, "text_explode",
                 "How violently kicks scatter the letters into dust")
        self._txt_box = txt_box
        vis_l.addWidget(txt_box)

        # -- hiding grid (mode 11)
        grid_box = QGroupBox("Hiding Grid  (mode 11)")
        gl2 = QVBoxLayout(grid_box)
        cap = QLabel("Black and white. A noise field warps the lattice and "
                     "hides each cell\u2019s square; beats resize them.")
        cap.setWordWrap(True)
        cap.setStyleSheet("color:#889;")
        gl2.addWidget(cap)

        crow = QHBoxLayout()
        lbc = QLabel("Columns")
        lbc.setToolTip("Grid resolution. The source drawing used 40.")
        crow.addWidget(lbc)
        self.grid_cols_spin = QSpinBox()
        self.grid_cols_spin.setRange(4, 80)
        self.grid_cols_spin.setValue(int(self.settings.grid_cols))
        self.grid_cols_spin.valueChanged.connect(
            lambda v: setattr(self.settings, "grid_cols", int(v)))
        crow.addWidget(self.grid_cols_spin)
        gl2.addLayout(crow)

        def grid_spin(label, lo, hi, step, attr, tip=""):
            r = QHBoxLayout()
            lb3 = QLabel(label)
            if tip:
                lb3.setToolTip(tip)
            r.addWidget(lb3)
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setValue(getattr(self.settings, attr))
            sp.valueChanged.connect(lambda v, a=attr: setattr(self.settings, a, v))
            r.addWidget(sp)
            gl2.addLayout(r)

        grid_spin("Noise scale", 0.5, 24.0, 0.5, "grid_noise",
                  "Feature size of the field \u2014 low is broad and blocky, "
                  "high is fine and busy")
        grid_spin("Warp", 0.0, 2.5, 0.1, "grid_warp",
                  "How far the field pushes the lattice nodes off-grid")
        grid_spin("Beat resize", 0.0, 2.0, 0.1, "grid_beat",
                  "How hard each beat swells the squares")
        grid_spin("Flow", 0.0, 2.0, 0.05, "grid_flow",
                  "How fast the field drifts when the track is quiet")
        self._grid_box = grid_box
        vis_l.addWidget(grid_box)

        # -- settings
        set_box = QGroupBox("Analysis && Motion")
        sl = QVBoxLayout(set_box)

        def spin_row(label, lo, hi, step, value, cb):
            r = QHBoxLayout()
            r.addWidget(QLabel(label))
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setValue(value)
            sp.valueChanged.connect(cb)
            r.addWidget(sp)
            sl.addLayout(r)
            return sp

        spin_row("Smoothing", 0.0, 0.98, 0.05, self.settings.smoothing,
                 self._set_smoothing)
        spin_row("Sensitivity", 0.1, 4.0, 0.1, self.settings.sensitivity,
                 lambda v: setattr(self.settings, "sensitivity", v))
        spin_row("Velocity damping", 0.70, 0.95, 0.01, self.settings.damping,
                 lambda v: setattr(self.settings, "damping", v))
        spin_row("Beat impulse", 0.0, 3.0, 0.1, self.settings.beat_impulse,
                 lambda v: setattr(self.settings, "beat_impulse", v))
        spin_row("FFT min Hz", 20, 2000, 10, self.settings.fft_min_hz,
                 self._set_fft_min)
        spin_row("FFT max Hz", 2000, 20000, 250, self.settings.fft_max_hz,
                 self._set_fft_max)
        self.autocam_check = QCheckBox("Auto camera rotation")
        self.autocam_check.setChecked(self.settings.auto_camera)
        self.autocam_check.toggled.connect(
            lambda on: setattr(self.settings, "auto_camera", on))
        sl.addWidget(self.autocam_check)
        self.veldbg_check = QCheckBox("Velocity debug overlay")
        self.veldbg_check.setChecked(self.settings.show_velocity_debug)
        self.veldbg_check.toggled.connect(
            lambda on: setattr(self.settings, "show_velocity_debug", on))
        sl.addWidget(self.veldbg_check)
        mot_l.addWidget(set_box)

        # -- post-processing FX
        post_box = QGroupBox("Post-processing")
        sl = QVBoxLayout(post_box)
        self.bloom_check = QCheckBox("Bloom glow (GPU)")
        self.bloom_check.setChecked(self.settings.bloom)
        self.bloom_check.toggled.connect(self._toggle_bloom)
        sl.addWidget(self.bloom_check)
        self.trails_check = QCheckBox("Feedback trails")
        self.trails_check.setChecked(self.settings.trails)
        self.trails_check.setToolTip(
            "Each frame echoes the last, zoomed and rotated — motion smears "
            "into decaying trails. Length follows the music.")
        self.trails_check.toggled.connect(
            lambda on: setattr(self.settings, "trails", on))
        sl.addWidget(self.trails_check)
        spin_row("Trail strength", 0.0, 2.0, 0.1, self.settings.trail_amount,
                 lambda v: setattr(self.settings, "trail_amount", v))
        self.antic_check = QCheckBox("Anticipation (build before drops)")
        self.antic_check.setChecked(self.settings.anticipation)
        self.antic_check.setToolTip(
            "Uses the offline beat grid to see drops coming: color drains and "
            "the camera pushes in over the 3s before a section boundary, then "
            "everything detonates on the downbeat.")
        self.antic_check.toggled.connect(
            lambda on: setattr(self.settings, "anticipation", on))
        sl.addWidget(self.antic_check)
        fx_l.addWidget(post_box)

        # -- drum grain FX
        fx_box = QGroupBox("Drum Grain FX")
        fxl = QVBoxLayout(fx_box)
        grow = QHBoxLayout()
        grow.addWidget(QLabel("Style"))
        self.grain_combo = QComboBox()
        from ..viz.grain import GRAIN_STYLES
        self.grain_combo.addItems(GRAIN_STYLES)
        self.grain_combo.setCurrentText(self.settings.grain_mode)
        self.grain_combo.currentTextChanged.connect(
            lambda t: setattr(self.settings, "grain_mode", t))
        grow.addWidget(self.grain_combo)
        fxl.addLayout(grow)
        irow = QHBoxLayout()
        irow.addWidget(QLabel("Intensity"))
        gspin = QDoubleSpinBox()
        gspin.setRange(0.1, 3.0)
        gspin.setSingleStep(0.1)
        gspin.setValue(self.settings.grain_intensity)
        gspin.valueChanged.connect(
            lambda v: setattr(self.settings, "grain_intensity", v))
        irow.addWidget(gspin)
        fxl.addLayout(irow)
        fx_l.addWidget(fx_box)

        # -- export
        exp_box = QGroupBox("Export")
        el = QVBoxLayout(exp_box)
        frow = QHBoxLayout()
        frow.addWidget(QLabel("Format"))
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItem("Screen (native, 60fps)", "screen")
        self.fmt_combo.addItem("Instagram Reel 9:16 1080×1920 60fps", "reel")
        self.fmt_combo.addItem("Instagram Square 1:1 1080×1080 60fps", "square")
        idx = self.fmt_combo.findData(self.settings.export_format)
        self.fmt_combo.setCurrentIndex(max(0, idx))
        self.fmt_combo.currentIndexChanged.connect(
            lambda i: setattr(self.settings, "export_format",
                              self.fmt_combo.itemData(i)))
        frow.addWidget(self.fmt_combo)
        el.addLayout(frow)
        drow = QHBoxLayout()
        self.dir_label = QLabel()
        self.dir_label.setStyleSheet("color: #99a;")
        self._refresh_dir_label()
        b_dir = QPushButton("Folder…")
        b_dir.clicked.connect(self._choose_export_dir)
        drow.addWidget(self.dir_label, stretch=1)
        drow.addWidget(b_dir)
        el.addLayout(drow)
        brow2 = QHBoxLayout()
        b_shot = QPushButton("📷 Screenshot")
        b_shot.clicked.connect(self._screenshot)
        self.rec_btn = QPushButton("⏺ Record")
        self.rec_btn.setCheckable(True)
        self.rec_btn.toggled.connect(self._toggle_record)
        brow2.addWidget(b_shot)
        brow2.addWidget(self.rec_btn)
        el.addLayout(brow2)
        b_offline = QPushButton("🎬 Export Full Video (offline 60fps)…")
        b_offline.setToolTip("Renders every frame against the audio clock — "
                             "true constant 60fps at full quality, whole song")
        b_offline.clicked.connect(self._export_offline)
        el.addWidget(b_offline)
        self.sp1_btn = QPushButton("Export for TE SP-1 Stem Player…")
        self.sp1_btn.setToolTip(
            "Writes one 24-bit 48 kHz 8-channel WAV (4 stereo stems) for the\n"
            "solderless.engineering SP-1 stem loader. BPM goes in the filename.")
        self.sp1_btn.clicked.connect(self._export_sp1)
        el.addWidget(self.sp1_btn)
        exp_l.addWidget(exp_box)

        for lay in (src_l, mix_l, vis_l, mot_l, fx_l, exp_l):
            lay.addStretch(1)
        for tab, label in ((src_tab, "Source"), (mix_tab, "Mix"),
                           (vis_tab, "Visuals"), (mot_tab, "Motion"),
                           (fx_tab, "FX"), (exp_tab, "Export")):
            self.tabs.addTab(tab, label)
        left.setMinimumWidth(360)
        left.setMaximumWidth(430)
        splitter.addWidget(left)

        # right: canvas + info + spectrum + transport
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        self.info_bar = InfoBar()
        rl.addWidget(self.info_bar)
        rl.addWidget(self.viz.canvas.native, stretch=1)
        self.spectrum = SpectrumWidget()
        rl.addWidget(self.spectrum)
        rl.addWidget(self.transport)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 950])   # else the tab labels get elided
        splitter.setChildrenCollapsible(False)
        self._update_mode_panels()

    def _connect_engine(self):
        self.engine.duration_changed.connect(self._on_duration)
        self.engine.position_changed.connect(self._on_position)
        self.engine.stems_progress.connect(self.info_bar.status.setText)
        self.engine.stems_ready.connect(
            lambda: self.info_bar.status.setText("Stems ready ✓"))
        self.engine.playback_finished.connect(self._on_playback_finished)
        self.engine.error.connect(
            lambda msg: QMessageBox.warning(self, "NeanderthalV", msg))

    def _install_shortcuts(self):
        def sc(key, fn):
            s = QShortcut(QKeySequence(key), self)
            s.activated.connect(fn)

        sc(Qt.Key.Key_Space, self._toggle_play)
        for i in range(min(9, len(MODE_CLASSES))):
            sc(str(i + 1), lambda k=i: self._select_mode(k))
        if len(MODE_CLASSES) >= 10:
            sc("0", lambda: self._select_mode(9))     # mode 10
        # the digits run out at ten, so mode 11 onward take the keys beside them
        for j, key in enumerate(("-", "=")):
            if len(MODE_CLASSES) > 10 + j:
                sc(key, lambda k=10 + j: self._select_mode(k))
        # stem toggles: V/D/B/O
        for key, stem in (("V", "vocals"), ("D", "drums"),
                          ("B", "bass"), ("O", "other")):
            sc(key, lambda s=stem: self.mixer.toggle_mute(s))
        sc("F", self._toggle_fullscreen)
        sc("I", lambda: self.use_img_check.toggle())
        sc("Shift+V", self._cycle_video_display)
        sc("S", self._screenshot)

    def _start_timers(self):
        self.analysis_timer = QTimer(self)
        self.analysis_timer.setInterval(1000 // self.profile.analysis_hz)
        self.analysis_timer.timeout.connect(self._analysis_tick)
        self.analysis_timer.start()

        # Pace rendering to the display, not a hard-coded 60. A 16 ms timer on
        # a 120 Hz screen caps at 62 fps AND beats against the 8.3 ms vsync
        # interval, which reads as judder even when the average looks fine.
        screen = self.screen() or QApplication.primaryScreen()
        hz = screen.refreshRate() if screen else 60.0
        if not hz or hz < 24:
            hz = 60.0
        hz = min(hz, float(self.profile.max_fps))
        self.render_timer = QTimer(self)
        self.render_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.render_timer.setInterval(max(1, round(1000.0 / hz)))
        self.render_timer.timeout.connect(self._render_tick)
        self.render_timer.start()

    # ------------------------------------------------------------- timers

    def _analysis_tick(self):
        self._frame = self.engine.analysis_tick()
        self.spectrum.push(self._frame)
        self.info_bar.push(self._frame, self.viz.fps,
                           self.viz.velocity_magnitude, self.viz.mode_name)
        if (self.settings.auto_choreograph and self.engine.section_crossed
                and self.engine.playing):
            self._select_mode((self.viz.current + 1) % len(MODE_CLASSES))

    def _render_tick(self):
        now = time.time()
        dt = now - self._last_render
        self._last_render = now
        self.viz.render_tick(self._frame, dt, self.engine.position)

    # ------------------------------------------------------------- files

    def _browse_audio(self):
        exts = " ".join(f"*{e}" for e in AUDIO_EXTS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open audio", self.settings.last_dir, f"Audio ({exts})")
        if path:
            self._load_path(path)

    def _browse_image(self):
        exts = " ".join(f"*{e}" for e in IMAGE_EXTS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", self.settings.last_dir, f"Images ({exts})")
        if path:
            self._load_path(path)

    def _browse_video(self):
        exts = " ".join(f"*{e}" for e in VIDEO_EXTS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open video", self.settings.last_dir, f"Video ({exts})")
        if path:
            self._load_path(path)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        paths = [u.toLocalFile() for u in ev.mimeData().urls()]
        audio = [p for p in paths
                 if os.path.splitext(p)[1].lower() in AUDIO_EXTS]
        other = [p for p in paths if p not in audio]
        for p in other:
            self._load_path(p)
        if audio:
            self._load_path(audio[0])
            for p in audio[1:]:
                self._queue.append(p)
                self.queue_list.addItem(os.path.basename(p))
            self.queue_list.setVisible(bool(self._queue))

    def _load_path(self, path: str):
        if not path:
            return
        self.settings.last_dir = os.path.dirname(path)
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in AUDIO_EXTS:
                self._load_audio(path)
            elif ext in IMAGE_EXTS:
                self._load_image(path)
            elif ext in VIDEO_EXTS:
                self._load_video(path)
            else:
                self.info_bar.status.setText(f"Unsupported file type: {ext}")
        except Exception as e:
            QMessageBox.warning(self, "NeanderthalV", f"Failed to load {path}:\n{e}")

    def _load_audio(self, path: str):
        self.mic_btn.setChecked(False)
        self.viz.set_video(None)
        self.video_thumb.setText("no video")
        self.video_thumb.setPixmap(self._empty_pixmap())
        self.engine.load_file(path)
        self._current_path = path
        self.file_label.setText(os.path.basename(path))
        self.play_btn.setText("▶ Play")

    def _load_image(self, path: str):
        colors = self.palette_mgr.set_image(path)
        self.swatches.set_colors(colors)
        # the photo doubles as a depth-relief source for the point cloud
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            img.thumbnail((640, 640))
            self.viz.still_image = np.ascontiguousarray(np.asarray(img))
        except Exception:
            pass
        self.palette_combo.setCurrentText("image")
        self.info_bar.status.setText(
            f"Extracted {len(colors)} colors from {os.path.basename(path)}")

    def _load_video(self, path: str):
        self.mic_btn.setChecked(False)
        self.info_bar.status.setText("Extracting audio from video…")
        source = VideoSource(path, max_height=self.profile.video_max_height)
        thumb = source.thumbnail()
        if thumb is not None:
            self._show_thumb(thumb)
        audio = extract_audio_from_video(path)
        self.engine.load_file(path, audio_override=audio)
        self._current_path = path
        self.viz.set_video(source)
        if self.settings.video_display == "off":
            self.video_combo.setCurrentText("background")
        self.file_label.setText(os.path.basename(path))
        self.play_btn.setText("▶ Play")
        self.info_bar.status.setText("Video loaded — beat-matched to its audio")

    def _empty_pixmap(self):
        from PyQt6.QtGui import QPixmap
        return QPixmap()

    def _show_thumb(self, rgb: np.ndarray):
        from PyQt6.QtGui import QImage, QPixmap
        h, w = rgb.shape[:2]
        img = QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w,
                     QImage.Format.Format_RGB888).copy()
        self.video_thumb.setPixmap(QPixmap.fromImage(img).scaledToHeight(
            68, Qt.TransformationMode.SmoothTransformation))

    # ------------------------------------------------------------- actions

    def _toggle_play(self):
        self.engine.toggle()
        self.play_btn.setText("⏸ Pause" if self.engine.playing else "▶ Play")

    def _toggle_mic(self, on: bool):
        self.engine.set_mic_enabled(on)
        self.mic_btn.setText(f"🎤 Microphone: {'on' if self.engine.mic_enabled else 'off'}")
        if on and not self.engine.mic_enabled:  # failed to open
            self.mic_btn.setChecked(False)

    def _select_mode(self, i: int):
        self.mode_radios[i].setChecked(True)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _toggle_image_colors(self, on: bool):
        self.settings.use_image_colors = on
        if on and self.palette_mgr.image_palette is not None:
            self.palette_combo.setCurrentText("image")
        elif not on and self.settings.palette == "image":
            self.palette_combo.setCurrentText("viridis")

    def _cycle_video_display(self):
        order = ["background", "pip", "texture", "off"]
        cur = order.index(self.settings.video_display)
        self.video_combo.setCurrentText(order[(cur + 1) % len(order)])

    def _set_smoothing(self, v: float):
        self.settings.smoothing = v
        self.engine.analyzer.set_smoothing(v)

    def _set_fft_min(self, v: float):
        self.settings.fft_min_hz = v
        self.engine.analyzer.refresh_fft_range()

    def _set_fft_max(self, v: float):
        self.settings.fft_max_hz = v
        self.engine.analyzer.refresh_fft_range()

    # ---------------------------------------------------------- new features

    def _update_mode_panels(self):
        """Only the active mode's own settings card is shown, so the Visuals
        tab stays short instead of listing every mode's options at once."""
        cur = self.viz.current
        self._pc_box.setVisible(cur == 8)
        self._txt_box.setVisible(cur == 9)
        self._grid_box.setVisible(cur == 10)

    def _mode_changed(self, k: int):
        old = self.viz.current
        self._mode_prefs[str(old)] = {"palette": self.settings.palette,
                                      "grain": self.settings.grain_mode}
        self.viz.set_mode(k)
        self._update_mode_panels()
        prefs = self._mode_prefs.get(str(k))
        if prefs:
            self.palette_combo.setCurrentText(prefs["palette"])
            self.grain_combo.setCurrentText(prefs["grain"])
        self.settings.mode_prefs = json.dumps(self._mode_prefs)

    def _pc_source_changed(self, idx: int):
        src = self.pc_source_combo.itemData(idx)
        self.settings.pc_source = src
        if src == "camera":
            from ..video.player import CameraSource
            try:
                self.viz.set_camera(CameraSource())
                self.info_bar.status.setText("Webcam live — point cloud source")
            except Exception as e:
                self.viz.set_camera(None)
                self.pc_source_combo.setCurrentIndex(0)
                QMessageBox.warning(
                    self, "Webcam",
                    f"Could not open the webcam:\n{e}\n\nOn macOS, camera "
                    "access must be granted to the app running NeanderthalV "
                    "(System Settings ▸ Privacy & Security ▸ Camera).")
        else:
            self.viz.set_camera(None)

    def _toggle_pitch(self, on: bool):
        self.settings.preserve_pitch = on
        self.engine.set_preserve_pitch(on)

    def _toggle_bloom(self, on: bool):
        self.settings.bloom = on
        self.viz.canvas.bloom_enabled = on
        if on:
            self.viz.canvas._fx_failed = False

    # -- presets

    def _refresh_presets(self):
        from ..presets import list_presets
        self.preset_combo.clear()
        self.preset_combo.addItems(list_presets() or ["(no presets)"])

    def _apply_preset(self):
        from ..presets import load_preset
        name = self.preset_combo.currentText()
        if not load_preset(name, self.settings):
            return
        # push loaded values into the widgets (their handlers do the rest)
        self._select_mode(self.settings.viz_mode)
        self.palette_combo.setCurrentText(self.settings.palette)
        self.blend_combo.setCurrentText(self.settings.color_blend_mode)
        self.grain_combo.setCurrentText(self.settings.grain_mode)
        self.video_combo.setCurrentText(self.settings.video_display)
        self.autocam_check.setChecked(self.settings.auto_camera)
        self.bloom_check.setChecked(self.settings.bloom)
        self.use_img_check.setChecked(self.settings.use_image_colors)
        self.info_bar.status.setText(f"Preset '{name}' applied")

    def _save_preset(self):
        from ..presets import save_preset
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if ok and name.strip():
            save_preset(name.strip(), self.settings)
            self._refresh_presets()
            self.preset_combo.setCurrentText(name.strip())

    def _delete_preset(self):
        from ..presets import delete_preset
        delete_preset(self.preset_combo.currentText())
        self._refresh_presets()

    # -- playlist

    def _on_playback_finished(self):
        self.play_btn.setText("▶ Play")
        if self._queue and not self.engine.loop:
            QTimer.singleShot(100, self._advance_queue)

    def _advance_queue(self):
        if not self._queue:
            return
        path = self._queue.pop(0)
        self.queue_list.takeItem(0)
        self.queue_list.setVisible(bool(self._queue))
        self._load_path(path)
        QTimer.singleShot(400, self.engine.play)
        self.play_btn.setText("⏸ Pause")

    def _play_queued(self, item):
        row = self.queue_list.row(item)
        path = self._queue.pop(row)
        self.queue_list.takeItem(row)
        self.queue_list.setVisible(bool(self._queue))
        self._load_path(path)

    # -- MIDI

    def _start_midi(self):
        from ..midi import MidiInput, midi_available
        self._midi = None
        if not midi_available():
            return
        self._midi = MidiInput(self)
        self._midi.note_on.connect(self._on_midi_note)
        self._midi.cc.connect(self._on_midi_cc)
        self._midi.status.connect(self._midi_status)
        self._midi.start()

    def _midi_status(self, msg: str):
        """A machine with many MIDI ports produces a status line longer than
        the window; name the first and count the rest."""
        if msg.startswith("MIDI: "):
            names = [n.strip() for n in msg[6:].split(",") if n.strip()]
            if len(names) > 1:
                msg = f"MIDI: {names[0]} (+{len(names) - 1} more)"
        self.info_bar.status.setText(msg)

    def _on_midi_note(self, note: int, velocity: int):
        if 36 <= note < 36 + len(MODE_CLASSES):
            self._select_mode(note - 36)

    def _on_midi_cc(self, cc: int, value: int):
        f = value / 127.0
        if cc == 1:
            self.settings.sensitivity = 0.1 + f * 2.9
        elif cc == 2:
            self.settings.damping = 0.70 + f * 0.25
        elif cc == 3:
            self.settings.beat_impulse = f * 3.0
        elif cc == 4:
            self.settings.grain_intensity = 0.1 + f * 2.9

    # -- offline export

    def _export_offline(self):
        if self.engine.audio is None:
            QMessageBox.information(self, "Export", "Load a song first.")
            return
        from ..export import offline_export
        fmt = self.settings.export_format
        tag = {"screen": "", "reel": "_reel", "square": "_square"}.get(fmt, "")
        path = os.path.join(
            self._export_dir(),
            f"neanderthalv_{time.strftime('%Y%m%d_%H%M%S')}{tag}_60fps.mp4")
        was_playing = self.engine.playing
        self.engine.pause()
        self.analysis_timer.stop()
        self.render_timer.stop()
        total = int(self.engine.duration * 60)
        dlg = QProgressDialog("Rendering video offline…", "Cancel", 0, total,
                              self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        try:
            out = offline_export(
                self.viz, self.engine, self.settings, path, fmt=fmt,
                progress=lambda i, n: dlg.setValue(i),
                should_cancel=dlg.wasCanceled,
                process_events=QApplication.processEvents)
            self.info_bar.status.setText(
                f"Exported {out}" if out else "Export cancelled")
        except Exception as e:
            QMessageBox.warning(self, "Export", f"Export failed:\n{e}")
        finally:
            dlg.close()
            self.analysis_timer.start()
            self.render_timer.start()
            if was_playing:
                self.engine.play()

    def _export_dir(self) -> str:
        d = self.settings.export_dir or os.path.expanduser("~/Desktop")
        os.makedirs(d, exist_ok=True)
        return d

    def _refresh_dir_label(self):
        d = self.settings.export_dir or os.path.expanduser("~/Desktop")
        home = os.path.expanduser("~")
        self.dir_label.setText("Save to: " + d.replace(home, "~"))

    def _choose_export_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Choose export folder", self._export_dir())
        if d:
            self.settings.export_dir = d
            self._refresh_dir_label()

    def _screenshot(self):
        path = os.path.join(self._export_dir(),
                            f"neanderthalv_{time.strftime('%Y%m%d_%H%M%S')}.png")
        try:
            self.viz.screenshot(path)
            self.info_bar.status.setText(f"Saved {path}")
        except Exception as e:
            self.info_bar.status.setText(f"Screenshot failed: {e}")

    def _toggle_record(self, on: bool):
        if on:
            self._rec_audio_t0 = self.engine.position
            self.viz.start_recording()
            self.rec_btn.setText("⏹ Stop")
            self.info_bar.status.setText(
                f"Recording ({self.fmt_combo.currentText()})…")
        else:
            self.rec_btn.setText("⏺ Record")
            fmt = self.settings.export_format
            tag = {"screen": "", "reel": "_reel", "square": "_square"}.get(fmt, "")
            path = os.path.join(
                self._export_dir(),
                f"neanderthalv_{time.strftime('%Y%m%d_%H%M%S')}{tag}.mp4")
            self.info_bar.status.setText("Encoding 60fps video…")
            out = self.viz.stop_recording(
                path, audio=self.engine.audio, sr=self.engine.sr,
                audio_t0=getattr(self, "_rec_audio_t0", 0.0), fmt=fmt)
            self.info_bar.status.setText(f"Saved {out}" if out else "Nothing recorded")

    def _export_sp1(self):
        if not self.engine.stems:
            QMessageBox.information(
                self, "SP-1 export",
                "No stems available yet.\nLoad a song and wait for stem "
                "separation to finish first.")
            return
        from ..audio.sp1_export import export_sp1_wav, sp1_filename
        bpm = self._frame.bpm or 0.0
        default = os.path.join(
            self._export_dir(),
            sp1_filename(getattr(self, "_current_path", "song"), bpm))
        path, _ = QFileDialog.getSaveFileName(
            self, "Export SP-1 stem WAV", default, "WAV (*.wav)")
        if not path:
            return
        stems = {k: np.asarray(v) for k, v in self.engine.stems.items()}
        sr = self.engine.sr
        self.sp1_btn.setEnabled(False)

        from PyQt6.QtCore import QThread, pyqtSignal

        class _Worker(QThread):
            progress = pyqtSignal(str)
            done = pyqtSignal(str)
            fail = pyqtSignal(str)

            def run(w):
                try:
                    export_sp1_wav(stems, sr, path,
                                   progress=w.progress.emit)
                    w.done.emit(path)
                except Exception as e:
                    w.fail.emit(str(e))

        self._sp1_worker = _Worker(self)
        self._sp1_worker.progress.connect(self.info_bar.status.setText)
        self._sp1_worker.done.connect(lambda p: (
            self.sp1_btn.setEnabled(True),
            self.info_bar.status.setText(
                f"SP-1 WAV saved: {os.path.basename(p)} — load it with the "
                "solderless stem loader")))
        self._sp1_worker.fail.connect(lambda e: (
            self.sp1_btn.setEnabled(True),
            QMessageBox.warning(self, "SP-1 export", f"Export failed:\n{e}")))
        self._sp1_worker.start()

    # ------------------------------------------------------------- playback UI

    def _on_duration(self, dur: float):
        self.time_label.setText(f"0:00 / {int(dur // 60)}:{int(dur % 60):02d}")

    def _on_position(self, pos: float):
        dur = self.engine.duration
        if not self._seeking and dur > 0:
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(int(pos / dur * 1000))
            self.seek_slider.blockSignals(False)
        self.time_label.setText(
            f"{int(pos // 60)}:{int(pos % 60):02d} / "
            f"{int(dur // 60)}:{int(dur % 60):02d}")

    def _seek_released(self):
        self._seeking = False
        dur = self.engine.duration
        if dur > 0:
            self.engine.seek(self.seek_slider.value() / 1000 * dur)

    # ------------------------------------------------------------- lifecycle

    def closeEvent(self, ev):
        if getattr(self, "_midi", None) is not None:
            self._midi.stop()
            self._midi.wait(500)
        self.engine.shutdown()
        self.viz.set_video(None)
        self.viz.set_camera(None)
        self.settings.save()
        super().closeEvent(ev)
