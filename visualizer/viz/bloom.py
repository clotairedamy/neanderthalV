"""Post-processing chain: bloom, feedback trails, desaturation and flash.

    scene -> FBO -> bright pass -> blur (half res) -> composite -> screen

The composite pass also samples the *previous* composited frame, slightly
zoomed and rotated, so every moving point smears into an echo that decays
over time (classic video feedback). History ping-pongs between two textures.

Desaturation and flash live here too: they are how the anticipation system
drains the color out of a build-up and detonates on the drop.

If any GL step fails (weak drivers, e.g. some RPi stacks), the whole chain
disables itself and the canvas keeps drawing normally.
"""
from __future__ import annotations

import numpy as np
from vispy import gloo, scene

_QUAD_VERT = """
attribute vec2 a_pos;
varying vec2 v_uv;
void main() {
    v_uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

_BRIGHT_FRAG = """
uniform sampler2D u_tex;
uniform float u_thresh;
varying vec2 v_uv;
void main() {
    vec3 c = texture2D(u_tex, v_uv).rgb;
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    float k = smoothstep(u_thresh, u_thresh + 0.3, l);
    gl_FragColor = vec4(c * k, 1.0);
}
"""

_BLUR_FRAG = """
uniform sampler2D u_tex;
uniform vec2 u_dir;
varying vec2 v_uv;
void main() {
    float w[5];
    w[0] = 0.227027; w[1] = 0.194595; w[2] = 0.121622;
    w[3] = 0.054054; w[4] = 0.016216;
    vec3 acc = texture2D(u_tex, v_uv).rgb * w[0];
    for (int i = 1; i < 5; i++) {
        vec2 off = u_dir * float(i);
        acc += texture2D(u_tex, v_uv + off).rgb * w[i];
        acc += texture2D(u_tex, v_uv - off).rgb * w[i];
    }
    gl_FragColor = vec4(acc, 1.0);
}
"""

_COMP_FRAG = """
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform sampler2D u_hist;
uniform float u_amt;      // bloom strength
uniform float u_decay;    // trail persistence (0 = no trails)
uniform float u_zoom;     // feedback zoom per frame
uniform float u_rot;      // feedback rotation per frame (radians)
uniform float u_desat;    // 0 = full color, 1 = greyscale
uniform float u_dim;      // multiplies the whole frame
uniform float u_flash;    // additive white
uniform float u_aspect;
varying vec2 v_uv;

void main() {
    vec3 cur = texture2D(u_scene, v_uv).rgb + texture2D(u_bloom, v_uv).rgb * u_amt;

    vec3 col = cur;
    if (u_decay > 0.001) {
        // rotate/zoom about the centre, aspect-corrected so the spin is round
        vec2 c = (v_uv - 0.5) * vec2(u_aspect, 1.0);
        float ca = cos(u_rot), sa = sin(u_rot);
        vec2 r = vec2(c.x * ca - c.y * sa, c.x * sa + c.y * ca) / u_zoom;
        vec2 huv = r / vec2(u_aspect, 1.0) + 0.5;
        if (huv.x >= 0.0 && huv.x <= 1.0 && huv.y >= 0.0 && huv.y <= 1.0) {
            vec3 h = texture2D(u_hist, huv).rgb * u_decay;
            // max() rather than a sum: trails persist and fade without a
            // feedback loop that runs away to white on static bright areas
            col = max(cur, h);
        }
    }

    float lum = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(col, vec3(lum), u_desat) * u_dim + vec3(u_flash);
    gl_FragColor = vec4(col, 1.0);
}
"""

_PRESENT_FRAG = """
uniform sampler2D u_tex;
varying vec2 v_uv;
void main() { gl_FragColor = vec4(texture2D(u_tex, v_uv).rgb, 1.0); }
"""

_QUAD = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.float32)


class BloomCanvas(scene.SceneCanvas):
    def __init__(self, *args, bloom: bool = True, bloom_amount: float = 1.1,
                 bloom_threshold: float = 0.25, trails: bool = True, **kwargs):
        self.bloom_enabled = bloom
        self.bloom_amount = bloom_amount
        self.bloom_threshold = bloom_threshold
        # feedback trails — driven per frame by VizManager
        self.trails_enabled = trails
        self.trail_decay = 0.82
        self.trail_zoom = 1.006
        self.trail_rot = 0.0
        # anticipation
        self.desat = 0.0
        self.dim = 1.0
        self.flash = 0.0
        # advanced once per render tick, so a screenshot mid-tick doesn't
        # double-step the feedback
        self.tick_id = 0
        self._last_tick = -1
        self._fx_failed = False
        self._res_size = None
        super().__init__(*args, **kwargs)
        self.unfreeze()     # vispy freezes attrs; we add FBOs lazily on resize

    # ------------------------------------------------------------ resources

    @property
    def _fx_wanted(self) -> bool:
        return (self.bloom_enabled or self.trails_enabled
                or self.desat > 0.001 or self.flash > 0.001
                or self.dim < 0.999)

    def _ensure_resources(self):
        w, h = self.physical_size
        w, h = max(w, 4), max(h, 4)
        if self._res_size == (w, h):
            return
        hw, hh = max(w // 2, 2), max(h // 2, 2)

        def tex(shape):
            return gloo.Texture2D(shape, interpolation="linear",
                                  wrapping="clamp_to_edge")

        self._scene_tex = tex((h, w, 4))
        self._scene_fbo = gloo.FrameBuffer(self._scene_tex,
                                           gloo.RenderBuffer((h, w)))
        self._tex_a = tex((hh, hw, 4))
        self._fbo_a = gloo.FrameBuffer(self._tex_a)
        self._tex_b = tex((hh, hw, 4))
        self._fbo_b = gloo.FrameBuffer(self._tex_b)
        self._hist_tex = [tex((h, w, 4)), tex((h, w, 4))]
        self._hist_fbo = [gloo.FrameBuffer(t) for t in self._hist_tex]
        self._hist_i = 0
        self._out_tex = tex((h, w, 4))
        self._out_fbo = gloo.FrameBuffer(self._out_tex)

        def prog(frag):
            p = gloo.Program(_QUAD_VERT, frag)
            p["a_pos"] = _QUAD
            return p

        self._p_bright = prog(_BRIGHT_FRAG)
        self._p_blur = prog(_BLUR_FRAG)
        self._p_comp = prog(_COMP_FRAG)
        self._p_present = prog(_PRESENT_FRAG)
        self._res_size = (w, h)     # only mark ready once everything exists

    # ------------------------------------------------------------ pipeline

    def _draw_scene_to_fbo(self):
        w, h = self._res_size
        self.push_fbo(self._scene_fbo, (0, 0), (w, h))
        try:
            self._draw_scene()
        finally:
            self.pop_fbo()

    @staticmethod
    def _post_state():
        """Full-screen passes must REPLACE their target, not blend into it.

        The scene leaves whatever blend mode its visuals used (the particle
        and point-cloud modes use additive), and without resetting it here
        each pass adds onto the previous frame's buffer — the bloom texture
        then ramps to white over a few seconds.
        """
        gloo.set_state(depth_test=False, blend=False, cull_face=False)

    def _run_bloom_chain(self):
        w, h = self._res_size
        hw, hh = max(w // 2, 2), max(h // 2, 2)
        self._post_state()
        with self._fbo_a:
            gloo.set_viewport(0, 0, hw, hh)
            self._p_bright["u_tex"] = self._scene_tex
            self._p_bright["u_thresh"] = self.bloom_threshold
            self._p_bright.draw("triangle_strip")
        with self._fbo_b:
            gloo.set_viewport(0, 0, hw, hh)
            self._p_blur["u_tex"] = self._tex_a
            self._p_blur["u_dir"] = (1.5 / hw, 0.0)
            self._p_blur.draw("triangle_strip")
        with self._fbo_a:
            gloo.set_viewport(0, 0, hw, hh)
            self._p_blur["u_tex"] = self._tex_b
            self._p_blur["u_dir"] = (0.0, 1.5 / hh)
            self._p_blur.draw("triangle_strip")

    def _composite(self, advance: bool):
        """Run the composite pass. Returns (fbo, texture) holding the result.

        `advance` writes into the history ping-pong (stepping the feedback);
        otherwise the result goes to a scratch buffer and history is left
        untouched, so re-compositing the same tick is idempotent.
        """
        w, h = self._res_size
        read_i = self._hist_i
        write_i = 1 - read_i
        target_fbo = self._hist_fbo[write_i] if advance else self._out_fbo
        target_tex = self._hist_tex[write_i] if advance else self._out_tex

        p = self._p_comp
        p["u_scene"] = self._scene_tex
        p["u_bloom"] = self._tex_a
        p["u_hist"] = self._hist_tex[read_i]
        p["u_amt"] = self.bloom_amount if self.bloom_enabled else 0.0
        p["u_decay"] = self.trail_decay if self.trails_enabled else 0.0
        p["u_zoom"] = self.trail_zoom
        p["u_rot"] = self.trail_rot
        p["u_desat"] = self.desat
        p["u_dim"] = self.dim
        p["u_flash"] = self.flash
        p["u_aspect"] = float(w) / float(h)
        self._post_state()
        with target_fbo:
            gloo.set_viewport(0, 0, w, h)
            p.draw("triangle_strip")
        if advance:
            self._hist_i = write_i
        return target_fbo, target_tex

    def _run_chain(self):
        """Full offscreen chain for the current tick. Returns (fbo, tex)."""
        self.set_current()
        self._ensure_resources()
        self._draw_scene_to_fbo()
        if self.bloom_enabled:
            self._run_bloom_chain()
        advance = self.tick_id != self._last_tick
        self._last_tick = self.tick_id
        return self._composite(advance)

    def on_draw(self, event):
        if self._scene is None:
            return
        # CRITICAL: SceneCanvas.update() is a no-op while _update_pending is
        # True, and only the draw handler may clear it. Skipping this freezes
        # the canvas after the first frame.
        self._update_pending = False
        if self._fx_failed or not self._fx_wanted:
            self._draw_scene()
            return
        try:
            _, tex = self._run_chain()
            w, h = self._res_size
            self._post_state()
            gloo.set_viewport(0, 0, w, h)
            self._p_present["u_tex"] = tex
            self._p_present.draw("triangle_strip")
        except Exception:
            self._fx_failed = True
            self.update()

    def render_composited(self) -> np.ndarray:
        """Offscreen grab including bloom and trails (screenshots/recording)."""
        if self._fx_failed or not self._fx_wanted:
            return self.render(alpha=False)
        try:
            fbo, _ = self._run_chain()
            return np.asarray(fbo.read()[:, :, :3])
        except Exception:
            self._fx_failed = True
            return self.render(alpha=False)

    def reset_feedback(self) -> None:
        """Clear trail history (on mode switches, seeks, export start)."""
        if self._res_size is None:
            return
        try:
            for fbo in self._hist_fbo:
                with fbo:
                    gloo.clear(color=(0, 0, 0, 1))
        except Exception:
            pass
