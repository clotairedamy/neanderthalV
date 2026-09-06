"""Mode 13 -- Storm.

A thunderhead at dusk that throws lightning on the drums.

The look and, more usefully, the timing come from measuring real footage
(a storm shot from above at dusk, 86s, 30fps): 59 flashes, median duration
233 ms, median 0.87 s apart, peaking around +230% over the ambient
brightness. Those numbers set the flash envelope here -- a hard attack and
a ~250 ms decay, with a secondary re-strike, because real lightning almost
never flashes just once.

The sky is a fragment shader: domain-warped fBm for the cloud mass, a warm
band along the horizon, and a lit term the flash drives so a strike
illuminates the cloud volume from inside rather than merely drawing a line
over it. The bolts themselves are built on the CPU by midpoint
displacement, which is cheap and gives the right recursive jaggedness.

Photosensitivity: flashes are rate-limited and their amplitude is scaled
down when they would come faster than about 3 Hz, which is also roughly
where the reference footage tops out.
"""
from __future__ import annotations

import numpy as np
from vispy import gloo, scene
from vispy.scene.visuals import create_visual_node
from vispy.visuals import Visual

from ..physics.velocity import VelocityValue
from .base import BaseMode

# measured from the reference clip
FLASH_DECAY = 0.25          # seconds; median flash lasted 233 ms
MIN_FLASH_GAP = 0.30        # s between strikes: ~3 Hz ceiling

SKY_VERT = """
attribute vec2 a_pos;
varying vec2 v_uv;
void main() {
    v_uv = a_pos;
    gl_Position = $transform(vec4(a_pos, 0.0, 1.0));
}
"""

SKY_FRAG = """
varying vec2 v_uv;

uniform float u_time;
uniform float u_flash;        // 0..1 current strike brightness
uniform vec2  u_flash_pos;    // where the bolt struck, in [-1,1]
uniform float u_horizon;      // warm band strength
uniform float u_billow;       // cloud detail gain, rises with energy
uniform vec3  u_glow;         // horizon colour, from the palette

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
               u.y);
}

float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) {
        v += a * vnoise(p);
        p = p * 2.03 + vec2(1.7, 9.2);
        a *= 0.5;
    }
    return v;
}

void main() {
    vec2 uv = v_uv;
    // the horizon sits high: this is shot from above the cloud deck
    float horizon = 0.22;
    float above = smoothstep(horizon - 0.02, horizon + 0.25, uv.y);

    // two layers drifting at different speeds read as depth
    vec2 p = vec2(uv.x * 1.6, uv.y * 2.2);
    vec2 warp = vec2(fbm(p * 0.7 + u_time * 0.02),
                     fbm(p * 0.7 + 5.2 - u_time * 0.015));
    float cloud = fbm(p * 1.5 + warp * (0.8 + 0.5 * u_billow)
                      + vec2(u_time * 0.03, 0.0));
    float cloud2 = fbm(p * 3.1 - warp * 0.6 + vec2(u_time * 0.05, 0.0));
    float density = cloud * 0.7 + cloud2 * 0.3;

    // cloud mass fills the lower frame and thins out into the sky
    float mass = density * (1.0 - above) + density * 0.18 * above;
    mass = clamp(mass * 1.5 - 0.18, 0.0, 1.0);

    // --- base colours: deep teal sky over a slate cloud deck
    vec3 sky = mix(vec3(0.020, 0.045, 0.075), vec3(0.008, 0.016, 0.032),
                   smoothstep(horizon, 1.0, uv.y));
    vec3 deck = mix(vec3(0.018, 0.024, 0.036), vec3(0.130, 0.140, 0.160),
                    smoothstep(0.10, 0.80, mass));
    vec3 col = mix(deck, sky, above);

    // --- the dusk band. Asymmetric on purpose: the glow is sky light, so
    // it sits above the cloud tops and the deck below occludes it, which
    // is what stops it reading as a stripe pasted across the frame.
    float dy = uv.y - horizon;
    float band = dy > 0.0 ? exp(-dy * 4.0) : exp(dy * 11.0);
    col += u_glow * band * u_horizon * (0.30 + 0.45 * mass);

    // --- the strike lights the cloud from inside: a tight radial falloff
    // around the bolt, gated by density so it glows where there is cloud
    // to light rather than washing the whole quadrant
    float d = length((uv - u_flash_pos) * vec2(1.0, 1.3));
    float lit = exp(-d * 3.6) * u_flash;
    col += vec3(0.55, 0.66, 1.0) * lit * (0.18 + 0.85 * mass);
    // a much weaker whole-sky lift, so the frame reads as flashing
    col += vec3(0.26, 0.33, 0.50) * u_flash * 0.10;

    // roll off the highlights instead of clipping them to flat white; the
    // reference flashes peaked around +230% over ambient, not saturation
    col = col / (1.0 + col * 0.55);

    gl_FragColor = vec4(col, 1.0);
}
"""


class StormSkyVisual(Visual):
    def __init__(self):
        Visual.__init__(self, vcode=SKY_VERT, fcode=SKY_FRAG)
        quad = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], np.float32)
        self.shared_program["a_pos"] = gloo.VertexBuffer(quad)
        for k, v in (("u_time", 0.0), ("u_flash", 0.0),
                     ("u_flash_pos", (0.0, 0.3)), ("u_horizon", 1.0),
                     ("u_billow", 0.0), ("u_glow", (1.0, 0.55, 0.18))):
            self.shared_program[k] = v
        self._draw_mode = "triangle_strip"
        self.set_gl_state(depth_test=False, blend=False, cull_face=False)

    def set_uniform(self, name, value):
        self.shared_program[name] = value

    def _prepare_transforms(self, view):
        view.view_program.vert["transform"] = view.get_transform()

    def _prepare_draw(self, view):
        return True

    def _compute_bounds(self, axis, view):
        return (-1.0, 1.0)


StormSky = create_visual_node(StormSkyVisual)


def make_bolt(rng, start, end, jag: float = 0.20, depth: int = 6,
              branch_p: float = 0.45):
    """A lightning path by midpoint displacement, plus forks.

    Repeatedly splitting each segment and pushing the new midpoint sideways
    is what gives lightning its self-similar jaggedness: the same kink
    shape appears at every scale, which a single-scale random walk never
    produces. Offsets are perpendicular to the segment and halve with each
    level, so the path stays headed where it was going.

    Returns a list of (n, 2) polylines: the main channel first, then forks.
    """
    pts = np.array([start, end], np.float64)
    span = np.linalg.norm(np.asarray(end) - np.asarray(start))
    amp = span * jag
    for _ in range(depth):
        seg = pts[1:] - pts[:-1]
        mids = (pts[:-1] + pts[1:]) * 0.5
        # perpendicular of each segment, normalized
        perp = np.stack([-seg[:, 1], seg[:, 0]], 1)
        perp /= np.linalg.norm(perp, axis=1, keepdims=True) + 1e-9
        mids += perp * rng.normal(0.0, amp, (len(mids), 1))
        out = np.empty((len(pts) + len(mids), 2), np.float64)
        out[0::2] = pts
        out[1::2] = mids
        pts = out
        amp *= 0.5

    paths = [pts]
    # forks leave the channel part-way down and die out early, which is
    # what stops the bolt reading as one lonely squiggle
    n = len(pts)
    for i in range(int(rng.integers(1, 4))):
        if rng.random() > branch_p:
            continue
        k = int(rng.integers(n // 5, n * 3 // 4))
        base = pts[k]
        head = pts[min(k + 6, n - 1)] - base
        if not np.any(head):
            continue
        head = head / (np.linalg.norm(head) + 1e-9)
        ang = rng.normal(0.0, 0.6)
        rot = np.array([[np.cos(ang), -np.sin(ang)],
                        [np.sin(ang), np.cos(ang)]])
        tip = base + rot @ head * span * rng.uniform(0.15, 0.4)
        paths.append(make_bolt(rng, base, tip, jag * 0.8, 3, branch_p=0.0)[0])
    return paths


class StormMode(BaseMode):
    name = "Storm"
    camera = "panzoom"
    trail_scale = 0.35
    bloom_scale = 1.35          # the bolt should bloom; that is the drama

    MAX_BOLT_VERTS = 4096

    def build(self):
        self.rng = np.random.default_rng(1)
        r = self.profile.fractal_resolution
        self.extent = r / 2 + 20

        self.sky = StormSky(parent=self.view.scene)
        from vispy.visuals.transforms import STTransform
        self.sky.transform = STTransform(scale=(self.extent, self.extent))
        self.sky.order = 0

        self.bolt = scene.visuals.Line(connect="segments", width=2.0,
                                       parent=self.view.scene)
        self.bolt.set_gl_state("additive", depth_test=False)
        self.bolt.order = 1
        self.visuals = [self.sky, self.bolt]

        d = self.settings.damping
        self.horizon = VelocityValue(1.0, accel=4.0, damping=d)
        self.billow = VelocityValue(0.0, accel=3.0, damping=d)
        self._t = 0.0
        self._last_ft = -1.0
        self._strikes = []          # active bolts: dict(paths, born, power)
        self._last_strike = -99.0
        self._drum_ema = 0.0

    # ----------------------------------------------------------- striking

    def _trigger(self, power: float) -> None:
        """Fire a bolt from the cloud tops, and remember when."""
        rng = self.rng
        x0 = float(rng.uniform(-0.85, 0.85))
        # strikes start in the sky above the deck and drive down into it
        start = (x0, float(rng.uniform(0.75, 1.0)))
        # kept near-vertical: displacement large against the span makes the
        # channel wander sideways and stop reading as a descending strike
        end = (x0 + float(rng.normal(0, 0.10)), float(rng.uniform(0.16, 0.32)))
        self._strikes.append({
            "paths": make_bolt(rng, start, end),
            "born": self._t,
            "power": power,
            "pos": (float(np.clip((start[0] + end[0]) * 0.5, -1, 1)),
                    float(end[1])),
            # real lightning usually re-strikes; a second pulse shortly after
            "restrike": self._t + float(rng.uniform(0.06, 0.16))
            if rng.random() < 0.55 else None,
        })
        self._last_strike = self._t

    def _drum_hit(self, frame, fresh: bool) -> float:
        """How hard the drums just hit, 0..1.

        Prefers the separated drums stem when stems are available, because
        that is what "on the drums" actually means; falls back to the
        low-band transient, which on a full mix is mostly the kick anyway.
        """
        if not fresh:
            return 0.0
        drums = frame.stem_energy.get("drums")
        if drums is not None:
            # a hit is energy above the running average, not loudness itself
            prev = self._drum_ema
            self._drum_ema = 0.82 * prev + 0.18 * drums
            return float(np.clip((drums - prev) * 3.5, 0.0, 1.0))
        return float(np.clip(frame.punch, 0.0, 1.0))

    # --------------------------------------------------------------- frame

    def update(self, frame, dt):
        dt = min(dt, 0.05)
        s = self.settings
        fresh = frame.time != self._last_ft
        self._last_ft = frame.time
        self._t += dt

        bass = float(np.clip(frame.bands[:2].mean() * s.sensitivity, 0, 1.5))

        hit = self._drum_hit(frame, fresh)
        strong = hit > (1.0 - s.storm_sensitivity) or (
            frame.beat and frame.beat_strength > 0.55 and hit > 0.12)
        if strong and (self._t - self._last_strike) >= MIN_FLASH_GAP:
            self._trigger(float(np.clip(0.55 + hit, 0.0, 1.6)))

        # --- flash envelope: hard attack, measured decay, plus re-strikes
        flash = 0.0
        alive = []
        for st in self._strikes:
            age = self._t - st["born"]
            env = float(np.exp(-age / FLASH_DECAY))
            if st["restrike"] is not None and self._t >= st["restrike"]:
                st["born"] = self._t          # second pulse of the same bolt
                st["restrike"] = None
                env = 1.0
            if env > 0.02:
                st["env"] = env
                alive.append(st)
                flash = max(flash, env * st["power"])
        self._strikes = alive[-4:]
        flash = float(np.clip(flash * s.storm_flash, 0.0, 1.0))

        # --- sky
        self.billow.set_target(0.3 + 1.4 * frame.rms)
        self.horizon.set_target(0.55 + 1.1 * bass + 0.5 * frame.rms)
        u = self.sky.set_uniform
        u("u_time", float(self._t * (0.4 + 1.6 * frame.rms) * s.storm_drift))
        u("u_flash", flash)
        u("u_billow", float(np.clip(self.billow.update(dt), 0, 2)))
        u("u_horizon", float(np.clip(self.horizon.update(dt), 0, 2.5)))
        if self._strikes:
            u("u_flash_pos", self._strikes[-1]["pos"])
        # The dusk band is fixed warm amber rather than palette-driven: a
        # green or magenta horizon stops reading as a sky at all. The
        # palette only shifts it a little, so mode-to-mode colour still
        # carries through.
        lut = self.palette.lut(256)
        tint = np.asarray(lut[int(len(lut) * 0.75)], np.float32)
        amber = np.array([1.00, 0.52, 0.16], np.float32)
        u("u_glow", tuple(float(c) for c in (amber * 0.78 + tint * 0.22)))

        # --- bolts
        self._draw_bolts()

    def _draw_bolts(self):
        segs = []
        cols = []
        for st in self._strikes:
            env = st.get("env", 0.0)
            for j, path in enumerate(st["paths"]):
                p = path * self.extent
                a = np.empty((len(p) - 1) * 2, np.int64)
                a[0::2] = np.arange(len(p) - 1)
                a[1::2] = np.arange(1, len(p))
                segs.append(p[a])
                # the main channel is brighter than its forks
                w = 1.0 if j == 0 else 0.45
                c = np.empty((len(a), 4), np.float32)
                c[:, 0], c[:, 1], c[:, 2] = 0.80, 0.88, 1.0
                c[:, 3] = np.clip(env * st["power"] * w, 0, 1)
                cols.append(c)
        if not segs:
            self.bolt.visible = False
            return
        self.bolt.visible = True
        pos = np.concatenate(segs).astype(np.float32)
        col = np.concatenate(cols)
        if len(pos) > self.MAX_BOLT_VERTS:
            pos, col = pos[:self.MAX_BOLT_VERTS], col[:self.MAX_BOLT_VERTS]
        self.bolt.set_data(pos=pos, color=col)

    def velocity_magnitude(self):
        return self.horizon.speed + self.billow.speed
