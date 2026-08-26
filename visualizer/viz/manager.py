"""VizManager: owns the vispy canvas, background video layer, the six modes,
auto-camera, FPS tracking, screenshots and video export.
"""
from __future__ import annotations

import os
import subprocess
import time

import numpy as np
from vispy import scene
from vispy.visuals.transforms import STTransform, MatrixTransform

from ..color.palette import extract_frame_palette
from ..physics.velocity import VelocityValue
from ..video.player import VideoSource, chromakey_mask
from .grain import GrainOverlay
from .mode_icosphere import IcosphereMode
from .mode_polyhedra import PolyhedraMode
from .mode_particles import ParticlesMode
from .mode_fractal import FractalMode
from .mode_topology import TopologyMode
from .mode_kaleidoscope import KaleidoscopeMode
from .mode_blueprint import BlueprintMode
from .mode_fiber import FiberMode
from .mode_pointcloud import PointCloudMode
from .mode_text import TextMode

MODE_CLASSES = [IcosphereMode, PolyhedraMode, ParticlesMode,
                FractalMode, TopologyMode, KaleidoscopeMode,
                BlueprintMode, FiberMode, PointCloudMode, TextMode]


class VizManager:
    def __init__(self, settings, palette, profile):
        self.settings = settings
        self.palette = palette
        self.profile = profile

        from .bloom import BloomCanvas
        self.canvas = BloomCanvas(keys=None, bgcolor=(0.02, 0.02, 0.04, 1.0),
                                  show=False, vsync=True,
                                  bloom=settings.bloom, trails=settings.trails)
        # background view (video) — drawn first, main 3D view drawn on top
        self.bg_view = self.canvas.central_widget.add_view()
        self.bg_view.interactive = False
        self.bg_view.camera = scene.PanZoomCamera(rect=(0, 0, 1, 1))
        self.bg_view.camera.interactive = False
        self.bg_image = scene.visuals.Image(np.zeros((2, 2, 3), np.float32),
                                            parent=self.bg_view.scene)
        self.bg_image.visible = False

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(fov=45, distance=6.0,
                                                         elevation=18, azimuth=30)

        # dedicated 2D overlay view for image-based modes (fractal): vispy
        # breaks when a ViewBox's camera is swapped panzoom<->turntable, so
        # each camera type gets its own permanent view instead.
        self.view_2d = self.canvas.central_widget.add_view()
        self.view_2d.interactive = False
        r = profile.fractal_resolution
        self.view_2d.camera = scene.PanZoomCamera(
            rect=(-r / 2 - 20, -r / 2 - 20, r + 40, r + 40), aspect=1)
        self.view_2d.camera.interactive = False

        # topmost overlay view: drum-triggered grain effects
        self.fx_view = self.canvas.central_widget.add_view()
        self.fx_view.interactive = False
        self.fx_view.camera = scene.PanZoomCamera(rect=(0, 0, 1, 1))
        self.fx_view.camera.interactive = False
        self.grain = GrainOverlay(self.fx_view, settings)

        # fade-through-black masking mode transitions
        self._fade_img = scene.visuals.Image(np.zeros((2, 2, 4), np.float32),
                                             parent=self.fx_view.scene)
        self._fade_img.transform = STTransform(scale=(0.5, 0.5))
        self._fade_img.set_gl_state("translucent", depth_test=False)
        self._fade_img.visible = False
        self._fade = 0.0

        # texture-mode video quad lives in the 3D scene
        self.tex_image = scene.visuals.Image(np.zeros((2, 2, 3), np.float32),
                                             parent=self.view.scene)
        self.tex_image.visible = False
        self._tex_angle = 0.0

        self.modes = [cls(self.view_2d if cls.camera == "panzoom" else self.view,
                          palette, settings, profile)
                      for cls in MODE_CLASSES]
        for m in self.modes:
            m.manager = self        # modes may read video/camera/still image
        self.current = 0
        self._cam_spin = VelocityValue(0.15, accel=3.0, damping=0.9)
        # camera punch: bass hits push the camera in, damping glides it back
        self._zoom = VelocityValue(0.0, accel=3.0, damping=0.80)
        self._base_dist: float | None = None
        self._applied_dist: float | None = None
        self._last_punch_ft = -1.0
        # anticipation: tension builds before a section boundary, releases on it
        self._flash_env = 0.0
        self._trail_len = VelocityValue(0.0, accel=4.0, damping=0.85)
        self._spin_fx = VelocityValue(0.0, accel=3.0, damping=0.86)
        self._anticipation = 0.0
        self._tick_id = 0

        self.video: VideoSource | None = None
        self.camera = None                      # CameraSource when enabled
        self.still_image: np.ndarray | None = None   # last loaded photo (RGB)
        self.last_audio_time = 0.0
        self._video_pal_t = 0.0

        # perf tracking
        self.fps = 0.0
        self._fps_times: list[float] = []
        self.velocity_magnitude = 0.0

        # recording (frames stream to disk via cv2.VideoWriter)
        self._recording = False
        self._rec_writer = None
        self._rec_tmp = ""
        self._rec_frames = 0
        self._rec_t0 = 0.0
        self.NOMINAL_FPS = 60.0

        self.set_mode(int(np.clip(settings.viz_mode, 0, len(self.modes) - 1)))

    # ---------------------------------------------------------------- modes

    def set_mode(self, i: int) -> None:
        i = int(np.clip(i, 0, len(self.modes) - 1))
        if i != self.current and self.modes[self.current].built:
            self._fade = 1.0            # brief fade masks the cut
            self.canvas.reset_feedback()   # don't smear the old mode into the new
        for k, m in enumerate(self.modes):
            m.set_visible(k == i)
        self.current = i
        self.settings.viz_mode = i
        mode = self.modes[i]
        if mode.camera != "panzoom":
            self.view.camera.distance = mode.camera_distance
            if mode.camera_elevation is not None:
                self.view.camera.elevation = mode.camera_elevation

    @property
    def mode_name(self) -> str:
        return self.modes[self.current].name

    # ---------------------------------------------------------------- video

    def set_video(self, source: VideoSource | None) -> None:
        if self.video is not None:
            self.video.close()
        self.video = source
        if source is None:
            self.bg_image.visible = False
            self.tex_image.visible = False

    def set_camera(self, source) -> None:
        if self.camera is not None:
            self.camera.close()
        self.camera = source

    def _update_video(self, t: float, frame_analysis) -> None:
        mode = self.settings.video_display
        # the point cloud consumes the video itself — no flat backdrop too
        if self.modes[self.current].consumes_video:
            self.bg_image.visible = False
            self.tex_image.visible = False
            return
        if self.video is None or mode == "off":
            self.bg_image.visible = False
            self.tex_image.visible = False
            return
        frame = self.video.frame_at(t)
        if frame is None:
            return
        img = frame.astype(np.float32) / 255.0

        if self.settings.chromakey:
            alpha = chromakey_mask(frame)
            rgba = np.dstack([img, alpha.astype(np.float32)])
            img = rgba
            # adaptive palette from video colors, refreshed every 2s
            if t - self._video_pal_t > 2.0:
                self._video_pal_t = t
                self.palette.set_image_palette_colors(extract_frame_palette(frame))

        if mode == "texture":
            self.bg_image.visible = False
            self.tex_image.visible = True
            self.tex_image.set_data(img)
            h, w = frame.shape[:2]
            s = 3.0 / max(h, w)
            self._tex_angle += 0.01 + frame_analysis.rms * 0.05
            tr = MatrixTransform()
            tr.scale((s, s, 1))
            tr.translate((-w * s / 2, -h * s / 2, 0))
            tr.rotate(np.degrees(self._tex_angle), (0, 1, 0))
            self.tex_image.transform = tr
        else:
            self.tex_image.visible = False
            self.bg_image.visible = True
            self.bg_image.set_data(img)
            h, w = frame.shape[:2]
            if mode == "pip":
                # bottom-right quarter of the canvas
                self.bg_image.transform = STTransform(
                    scale=(0.3 / w, 0.3 / h), translate=(0.68, 0.02))
            else:  # full background
                self.bg_image.transform = STTransform(
                    scale=(1.0 / w, 1.0 / h), translate=(0, 0))

    # ---------------------------------------------------------------- frame

    def _update_fx(self, frame, dt: float) -> None:
        """Feedback trails + the anticipation build/release choreography."""
        c = self.canvas
        on = self.settings.anticipation
        ant = float(getattr(frame, "anticipation", 0.0)) if on else 0.0
        punch = frame.punch

        # --- release: a section boundary detonates everything at once.
        # The flash is set outright, not integrated: a flash that ramps in
        # over ten frames is not a flash.
        if on and getattr(frame, "section", False):
            self._flash_env = 1.0
            self._trail_len.impulse(2.6)
            self._spin_fx.impulse(2.2)
            self._zoom.impulse(-1.6)        # camera snaps back out
        self._flash_env *= float(np.exp(-dt * 6.0))
        self._spin_fx.set_target(0.0)

        # --- trails: length follows energy, stretches through the build-up
        strength = self.settings.trail_amount
        self._trail_len.set_target(frame.rms * 0.6 + ant * 1.6)
        tl = float(np.clip(self._trail_len.update(dt), 0.0, 3.0))
        c.trails_enabled = self.settings.trails
        c.trail_decay = float(np.clip(
            0.80 + 0.15 * np.tanh(tl) * strength, 0.0, 0.955)) \
            if strength > 0.01 else 0.0
        c.trail_zoom = 1.0 + (0.004 + 0.010 * ant) * strength
        c.trail_rot = (0.0015 + 0.004 * ant) * strength * \
            (1.0 + float(np.clip(self._spin_fx.update(dt), 0, 3)))

        # --- build-up: color drains and the frame dims, then it all returns
        c.desat = float(np.clip(ant * 0.75, 0.0, 0.85))
        c.dim = float(np.clip(1.0 - 0.28 * ant, 0.4, 1.0))
        # NB section boundaries are spaced >=4s apart by the grid, so this
        # never approaches the ~3Hz photosensitivity threshold
        c.flash = float(np.clip(self._flash_env * 0.42 + punch * 0.03, 0.0, 0.5))
        self._anticipation = ant

    def render_tick(self, frame, dt: float, audio_time: float) -> None:
        self.last_audio_time = audio_time
        self._tick_id += 1
        self.canvas.tick_id = self._tick_id
        self.palette.update(frame, dt)
        mode = self.modes[self.current]
        mode.update(frame, dt)
        self.velocity_magnitude = mode.velocity_magnitude()

        # auto camera rotation with velocity smoothing
        if (self.settings.auto_camera
                and isinstance(self.view.camera, scene.cameras.TurntableCamera)):
            self._cam_spin.set_target(0.1 + frame.rms * 0.6)
            if frame.beat:
                self._cam_spin.impulse(frame.beat_strength * 0.5)
            self.view.camera.azimuth += np.degrees(self._cam_spin.update(dt) * dt)

        # camera punch on bass transients (all 3D modes)
        cam = self.view.camera
        if (mode.camera != "panzoom"
                and isinstance(cam, scene.cameras.TurntableCamera)):
            if frame.time != self._last_punch_ft:
                self._last_punch_ft = frame.time
                self._zoom.impulse(frame.punch * 2.2)
            self._zoom.set_target(0.0)
            env = float(np.clip(self._zoom.update(dt), -1.5, 1.5))
            if (self._applied_dist is None
                    or abs(cam.distance - self._applied_dist) > 1e-7):
                self._base_dist = cam.distance   # user or mode moved the camera
            # anticipation dollies the camera in through the build-up
            ant = getattr(self, "_anticipation", 0.0)
            cam.distance = self._base_dist * (1.0 - 0.055 * env - 0.22 * ant)
            self._applied_dist = cam.distance
        else:
            self._applied_dist = None

        self._update_video(audio_time, frame)
        self.grain.update(frame, dt)
        self._update_fx(frame, dt)

        if self._fade > 0.004:
            self._fade *= float(np.exp(-dt * 5.0))
            img = np.zeros((2, 2, 4), np.float32)
            img[..., 3] = self._fade
            self._fade_img.set_data(img)
            self._fade_img.visible = True
        elif self._fade_img.visible:
            self._fade_img.visible = False

        now = time.time()
        self._fps_times.append(now)
        while self._fps_times and now - self._fps_times[0] > 1.0:
            self._fps_times.pop(0)
        self.fps = len(self._fps_times)

        if self._recording:
            self._capture_frame()

        self.canvas.update()

    # ---------------------------------------------------------------- export

    def screenshot(self, path: str) -> str:
        img = self.grab_frame()
        from PIL import Image
        Image.fromarray(img).save(path)
        return path

    def start_recording(self) -> None:
        import tempfile
        self._rec_tmp = tempfile.NamedTemporaryFile(suffix=".mp4",
                                                    delete=False).name
        self._rec_writer = None    # created lazily once the frame size is known
        self._rec_frames = 0
        self._rec_t0 = time.time()
        self._recording = True

    def grab_frame(self) -> np.ndarray:
        """Current composited frame (bloom included when enabled)."""
        render = getattr(self.canvas, "render_composited", None)
        if render is not None:
            return render()
        return self.canvas.render(alpha=False)

    def _capture_frame(self) -> None:
        import cv2
        img = self.grab_frame()
        h, w = img.shape[:2]
        h -= h % 2
        w -= w % 2
        if self._rec_writer is None:
            self._rec_writer = cv2.VideoWriter(
                self._rec_tmp, cv2.VideoWriter_fourcc(*"mp4v"),
                self.NOMINAL_FPS, (w, h))
            self._rec_size = (w, h)
        if (w, h) == self._rec_size:
            self._rec_writer.write(cv2.cvtColor(img[:h, :w, :3],
                                                cv2.COLOR_RGB2BGR))
            self._rec_frames += 1

    # Instagram-ready output specs (H.264 + AAC, 60 fps)
    FORMATS = {
        "screen": None,                      # native resolution
        "reel": (1080, 1920),                # 9:16 Reels / Stories
        "square": (1080, 1080),              # 1:1 feed
    }

    def stop_recording(self, path: str, audio: np.ndarray | None = None,
                       sr: int = 44100, audio_t0: float = 0.0,
                       fmt: str = "screen") -> str:
        """Finish capture, retime to exactly 60 fps, scale/crop for the target
        format, encode H.264+AAC with the synced audio slice, write `path`."""
        self._recording = False
        if self._rec_writer is None or self._rec_frames == 0:
            return ""
        self._rec_writer.release()
        self._rec_writer = None
        duration = max(time.time() - self._rec_t0, 1e-3)
        measured_fps = self._rec_frames / duration

        from ..audio.engine import _ffmpeg_exe
        import tempfile

        # retime nominal-60fps container to the real capture rate, then
        # resample to a true constant 60 fps
        vf = [f"setpts=PTS*({self.NOMINAL_FPS}/{measured_fps:.4f})",
              "fps=60"]
        size = self.FORMATS.get(fmt)
        if size:
            w, h = size
            vf.append(f"scale={w}:{h}:force_original_aspect_ratio=increase")
            vf.append(f"crop={w}:{h}")
            vf.append("setsar=1")
        else:
            vf.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")

        cmd = [_ffmpeg_exe(), "-y", "-i", self._rec_tmp]
        tmp_wav = ""
        if audio is not None:
            import soundfile as sf
            a0 = int(audio_t0 * sr)
            a1 = min(len(audio), a0 + int(duration * sr))
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav",
                                                  delete=False).name
            sf.write(tmp_wav, audio[a0:a1], sr)
            cmd += ["-i", tmp_wav]
        cmd += ["-filter:v", ",".join(vf),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
                "-pix_fmt", "yuv420p", "-r", "60"]
        if audio is not None:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += [path]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.unlink(self._rec_tmp)
        except Exception:
            # fall back to the raw capture if encoding failed
            os.replace(self._rec_tmp, path)
        finally:
            if tmp_wav:
                try:
                    os.unlink(tmp_wav)
                except OSError:
                    pass
        return path
