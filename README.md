# NeanderthalV — Audio-Reactive 3D Geometric Visualizer

Cross-platform desktop app (macOS & Raspberry Pi 5) that turns audio files,
live microphone input, or video soundtracks into ten velocity-driven 3D
visualizations, colored by palettes extracted from your photos.

## Quick start

```bash
./setup.sh    # one-time: venv, dependencies, Demucs model download
./run.sh      # launch
```

Both scripts auto-detect macOS vs Raspberry Pi 5 and apply the right
performance profile (60 fps / 2500 particles on macOS, 30 fps / 800 particles
on the Pi). The whole folder is the deployment: copy it to either machine and
run the two scripts.

**macOS single-app build:** `./packaging/build_macos.sh` → `dist/NeanderthalV.app`
(with app icon). Note: the bundle is unsigned — to distribute it beyond your
own machine you'll need to sign & notarize with your Apple Developer ID
(`codesign` + `notarytool`); on your own Mac, right-click → Open bypasses
Gatekeeper.

**Tests:** `pip install -r requirements-dev.txt && python -m pytest tests/`
(36 tests covering analysis, beat grid, anticipation, stems, SP-1 export,
physics, geometry, palettes, presets, point-cloud and text geometry).

## Interface

A dark, tabbed control panel — **Source** (files, mic, playlist, video),
**Mix** (stem faders), **Visuals** (mode + palette, with only the active
mode's own settings shown), **Motion** (analysis & physics), **FX** (bloom,
trails, anticipation, grain) and **Export** — with presets pinned above and
a media-player transport bar (play, scrub, speed, loop, pitch) under the
canvas.

## Features

- **Inputs**: MP3 / WAV / FLAC / OGG audio, live microphone, and MP4 / MOV /
  WebM / AVI video (audio extracted and beat-matched, frames synced to the
  audio clock within one frame).
- **Stems**: Demucs v4 (htdemucs) separation into Vocals / Drums / Bass /
  Other, cached on disk per file, ~30–60 s per 3-minute song on first run and
  instant afterwards. Without Demucs installed (typical on the Pi) a fast
  HPSS/band-split fallback provides pseudo-stems.
- **Analysis @30 Hz**: 2048-sample Hann-window FFT, 7 frequency bands
  (sub-bass → presence), per-stem bands, spectral-flux beat detection, BPM
  estimation, RMS energy, spectral centroid, EMA smoothing (0.8 default).
  Band and RMS values are normalized against an **adaptive ceiling and
  floor**, so whatever the track's loudness window, values span the full
  0–1 range (real music no longer reads as "static"). A per-band transient
  **attack** signal and a low-band **punch** signal fire on every kick/bass
  hit — driving morphs, camera punches and flash impulses even between
  detected beats.
- **Offline beat grid**: when a file is loaded, the whole song is analyzed
  in the background (refined from the drum stem once separation finishes):
  beat times, low-end onsets, BPM, and section boundaries. During playback
  the engine *schedules* beat/punch impulses exactly as the playhead crosses
  grid entries — effects land on the transient instead of a frame late.
  Live microphone input keeps the real-time detector.
- **Playback**: loop toggle, playlist queue (drop several audio files at
  once), and **pitch-preserving speed** — a phase-vocoder stretch renders in
  the background and swaps in seamlessly (varispeed remains the default).
- **Bloom**: real GPU post-processing (FBO bright-pass + gaussian blur +
  additive composite) for cinematic glow; toggle in Settings, auto-disables
  on GL failure.
- **Feedback trails**: the composite pass samples the previous frame,
  slightly zoomed and rotated, so motion smears into decaying echoes — a
  warp-tunnel at high strength, motion blur at low. Trail length follows the
  music and stretches through build-ups. *Trail strength* in Settings.
- **Anticipation**: because the beat grid is computed offline, the visuals
  know a drop is coming. Over the 3 seconds before a detected section
  boundary the color drains toward grey, the frame dims, the camera dollies
  in and the trails stretch; on the downbeat everything detonates — white
  flash, camera snap-back, trail spin. This is the difference between
  reacting to the music and being cut to it. Toggle in Settings.
- **Presets & choreography**: save/load named looks (mode, palette, grain,
  physics), per-mode palette/grain memory, fade-through-black transitions,
  and *Auto-switch at song sections* — the beat grid's verse/drop boundaries
  advance the mode automatically.
- **MIDI control** (optional, for VJing): notes 36–45 select the modes;
  CC1 sensitivity, CC2 damping, CC3 beat impulse, CC4 grain intensity.
- **Image colors**: drop in a photo — k-means extracts 5–8 dominant colors,
  mapped to bands/stems, rotated by spectral centroid, blended
  (overlay/multiply/screen) and eased by audio energy. Click a swatch to
  hand-adjust any extracted color.
- **Velocity physics**: every position, rotation and scale moves through a
  velocity integrator with acceleration toward targets, configurable damping
  (0.70–0.95), beat-triggered velocity impulses, and momentum decay.

## The ten modes (keys 1–9, 0)

1. **Frequency Mesh Icosphere** — faces binned into the 7 bands, radial
   deformation with per-band velocity, palette colors per band.
2. **Polyhedra Harmonic Array** — tetra/cube/octa/icosa orbiting, one per
   stem; orbit & spin speed follow stem energy, beats kick angular velocity.
3. **Particle Swarm** — 2500 particles (800 on Pi) with momentum physics,
   swirl field, radial beat bursts, fading palette-colored trails.
4. **Reactive Fractal** — Julia set whose parameter glides on velocity
   integrators (bass drives angle speed, centroid drives radius), palette LUT
   color cycling.
5. **Waveform 3D Topology** — radial surface with waves propagating outward
   at velocity-controlled speed, injected from the live waveform, height
   layers colored from the palette.
6. **Geometric Kaleidoscope** — central icosahedron plus 6-fold mirrored
   satellites, counter-rotating rings, beat velocity spikes.
7. **Blueprint Constellation** — monochrome generative-art style: a geodesic
   wireframe sphere counter-rotating at the heart of sacred-geometry hairline
   scaffolding, white particle spray with ray streaks, tiny technical
   annotations. Bass drives diagram rotation & sphere breathing, mids spin
   the sphere, beats fire radial bursts and a white flash envelope.
8. **Fiber Nebula** — monochrome smoke-and-sphere composition: a geodesic
   wireframe sphere wrapped in churning particle smoke plumes (curl flow
   field, emission ribbons, motion streaks) above a spinning spiral vortex
   disc. Bass breathes the sphere & pulses the vortex, mids drive plume
   turbulence, treble sparkles, energy drives flow/spin speed, beats fire
   smoke bursts, velocity kicks and flashes.

9. **Depth Point Cloud** — a Kinect-style volumetric reconstruction, after
   the three.js [`webgl_video_kinect`](https://threejs.org/examples/#webgl_video_kinect)
   example: every pixel of a video becomes a 3D point, un-projected by its
   depth in a vertex shader (145k points on Mac, ~30k on the Pi). Extended
   well past the original — see below.

10. **Reactive 3D Text** — your own words rasterized into a 3D point cloud
   and extruded into 7 depth layers with front-to-back shading. Long words
   wrap to two lines so the letters stay large, and the word *sways* within
   a bounded angle rather than spinning freely — a continuously rotating
   word is edge-on and unreadable half the time. The word doubles as a spectrum analyzer: a
   point's horizontal position picks its frequency band, so the left of the
   word answers the bass and the right the highs — each zone in its own
   color and pushed toward the viewer by that band. Kicks blow the letters
   into drifting dust that springs back into legible type. Type your text
   (and tune band depth / beat explode) in the *Reactive 3D Text* panel.

Modes 7–8 are intentionally grayscale (additive glow on black); a matching
**mono** palette is also available for the other modes.

### Depth Point Cloud (mode 9)

The original three.js demo plays one fixed grayscale Kinect clip. This one:

| | three.js original | NeanderthalV |
| --- | --- | --- |
| Source | the bundled depth clip | that same clip (bundled, looping — the default), plus any video, a photo, the live webcam, or the music itself |
| Depth | luminance | luminance, or true **RGBD** side-by-side / top-bottom footage |
| Motion | static params | bass exaggerates depth, kicks fire ripples that travel across the surface |
| Camera | fixed | orbiting, with bass-driven sway and a beat camera-punch |
| Color | raw grayscale | remapped onto your extracted image palette, with real GPU bloom |
| Projection | fixed frustum | **Perspective** slider: 0 = clean relief, 1 = the exact Kinect frustum |

Extra behaviors: loud passages **disintegrate** the figure — points scatter
along their own view rays and reassemble as energy drops; treble adds
per-point sparkle; a depth **cutoff** drops the background so the subject
floats. With nothing loaded it falls back to a scrolling spectrogram
terrain, so the mode is never blank.

**Frequency-band colors**: *Band colors* paints lows / mids / highs across
an axis of the cloud (vertical, depth, horizontal or radial). Each zone is
tinted from the palette, flares with its own band's energy, and is pushed
toward the viewer by it — so a kick lights up and swells the low zone while
the highs stay still. Set it to *Off* for plain palette tinting.

The three.js demo clip itself ships in `assets/kinect.mp4` (fetched from
threejs.org) and is the default source: select mode 9 with nothing loaded
and the original subject appears, looping and reacting to your music. Load
your own video or photo and *Auto* switches to it.

Controls live in the *Depth Point Cloud* panel: source, depth layout,
near/far clipping, point size, cutoff, perspective and band colors. The webcam source is
optional — it needs camera permission for the app running NeanderthalV (macOS:
System Settings ▸ Privacy & Security ▸ Camera).

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| `Space` | Play / pause |
| `1`–`9`, `0` | Select visualization mode (`0` = mode 10) |
| `V` `D` `B` `O` | Toggle Vocals / Drums / Bass / Other stem |
| `F` | Fullscreen |
| `I` | Toggle image-palette colors |
| `Shift+V` | Cycle video display (background → PiP → texture → off) |
| `S` | Screenshot to Desktop |

(The spec assigned `V` to both stem-vocals and video display; video display is
on `Shift+V`.)

## Video display modes

Background (full-bleed behind the 3D scene), picture-in-picture (corner),
texture (video quad rotating inside the 3D scene), or off. Optional chromakey
keys out green screens and re-extracts the color palette from the video every
2 seconds.

## Drum Grain FX

Optional grain overlay that flashes when the drum stem hits (velocity
envelope: drum transients punch it up, damping decays it). Styles: **film**
(fine analog grain), **static** (chunky TV static), **scanlines**
(interference lines + row tears), **burst** (violent full-frame slams).
Style + intensity are in the left panel.

## Export

All exports go to a folder you choose (Export ▸ Folder…, default Desktop):

- **Screenshot** (`S`) — PNG of the canvas.
- **Record** — live capture of the canvas, encoded H.264 + AAC at a constant
  **60 fps** with the synced audio slice. Formats:
  *Screen* (native size), *Instagram Reel* (1080×1920 9:16), *Instagram
  Square* (1080×1080) — both Instagram formats are scale-cropped to cover.
- **Export Full Video (offline 60fps)** — renders *every* frame against the
  audio clock (no dropped frames, bloom and grain included) and encodes the
  whole song at true 60 fps with the current stem mix. Roughly 5× realtime;
  a progress dialog lets you cancel.
- **Export for TE SP-1 Stem Player** — writes the single WAV the
  [solderless.engineering stem loader](https://solderless.engineering/stemloader/help/)
  requires: 24-bit, 48 kHz, 8-channel PCM with the four stereo stems on
  channels 1/2 (vocals), 3/4 (drums), 5/6 (bass), 7/8 (other), one common
  normalization pass (stem balance preserved), and the detected BPM embedded
  in the filename (`Song_124BPM.wav`) so the loader picks up the tempo.
  Requires stem separation to have finished for the loaded song; drag the
  resulting file into the stem loader web app with your SP-1 in boot mode.

## Settings

Smoothing, sensitivity, velocity damping, beat impulse, FFT range, blend mode,
palette (Viridis / Plasma / Turbo / Neon / Mono / Custom / Image), auto-camera, and
velocity debug are all in the left panel and persist to `config.ini`
(`~/Library/Application Support/NeanderthalV` on macOS, `~/.config/neanderthalv` on the Pi).
Stem caches live under the platform cache dir (`~/Library/Caches/NeanderthalV` /
`~/.cache/neanderthalv`).

## Project layout

```
main.py                     entry point
visualizer/
  config.py                 settings + platform profiles (macOS / RPi 5)
  audio/analyzer.py         FFT, bands, beat/BPM, EMA
  audio/stems.py            Demucs v4 + cache + DSP fallback
  audio/engine.py           playback, stem mixing, mic, master clock
  color/palette.py          k-means extraction, blend modes, animation
  physics/velocity.py       velocity/acceleration/damping integrators
  video/player.py           frame-accurate video sync, chromakey
  viz/                      six modes + manager (vispy/OpenGL)
  ui/                       PyQt6 main window + custom widgets
packaging/                  PyInstaller spec + macOS build script
setup.sh / run.sh           deploy scripts (both platforms)
```
