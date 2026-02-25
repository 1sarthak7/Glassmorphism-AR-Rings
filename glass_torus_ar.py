"""
Hand-Tracked Glass Rings
========================
Real-time AR filter that renders multiple 3D glass torus rings on each finger,
tracked via MediaPipe HandLandmarker. Features chromatic aberration refraction,
rainbow caustics, Fresnel rim glow, gesture-reactive animations, and multiple
stacked rings per finger.

Gestures:
  - Open palm  → rings expand outward and pulse
  - Fist       → rings contract tight and glow hotter
  - Pinch      → pinched finger rings orbit rapidly
  - Peace sign → rings on index+middle get rainbow boost

Tech Stack: OpenCV, MediaPipe, PyOpenGL (glfw), NumPy
Requires:   pip install opencv-python mediapipe PyOpenGL glfw numpy
"""

import cv2
import numpy as np
import mediapipe as mp
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import ctypes
import sys
import math
import os
import time
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# ============================================================================
#  GLSL SHADERS
# ============================================================================

BG_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
"""

BG_FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D bgTexture;
void main() {
    FragColor = texture(bgTexture, TexCoord);
}
"""

RING_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;
out vec4 ClipPos;

void main() {
    vec4 worldPos = model * vec4(aPos, 1.0);
    FragPos = worldPos.xyz;
    Normal = normalize(mat3(transpose(inverse(model))) * aNormal);
    ClipPos = projection * view * worldPos;
    gl_Position = ClipPos;
}
"""

RING_FRAGMENT_SHADER = """
#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec4 ClipPos;

out vec4 FragColor;

uniform sampler2D bgTexture;
uniform vec3 cameraPos;
uniform vec3 lightDir;
uniform float glowIntensity;     // 0.0 = normal, 1.0 = max gesture glow
uniform float rainbowBoost;      // 0.0 = normal, 1.0 = rainbow mode
uniform float pulsePhase;        // animated pulse phase

// -------------------------------------------------------------------------
// Enhanced Glass with Chromatic Aberration + Gesture Reactivity
//
// Chromatic Aberration:
//   R/G/B sampled at slightly different IOR values to simulate dispersion.
//
// Gesture reactivity is controlled by uniforms:
//   glowIntensity — makes the ring edges brighter (fist / pinch)
//   rainbowBoost  — amplifies iridescent caustics (peace sign)
//   pulsePhase    — sinusoidal alpha/scale pulsing (open palm)
// -------------------------------------------------------------------------

void main() {
    vec3 N = normalize(Normal);
    vec3 V = normalize(cameraPos - FragPos);
    vec3 L = normalize(-lightDir);

    vec2 screenUV = (ClipPos.xy / ClipPos.w) * 0.5 + 0.5;

    // --- Chromatic Aberration ---
    float eta_R = 1.0 / 1.47;
    float eta_G = 1.0 / 1.50;
    float eta_B = 1.0 / 1.53;

    vec3 refR = refract(-V, N, eta_R);
    vec3 refG = refract(-V, N, eta_G);
    vec3 refB = refract(-V, N, eta_B);

    float dist = 0.12;
    vec2 uvR = clamp(screenUV + refR.xy * dist, 0.0, 1.0);
    vec2 uvG = clamp(screenUV + refG.xy * dist, 0.0, 1.0);
    vec2 uvB = clamp(screenUV + refB.xy * dist, 0.0, 1.0);

    vec3 refractedColor = vec3(
        texture(bgTexture, uvR).r,
        texture(bgTexture, uvG).g,
        texture(bgTexture, uvB).b
    );

    // --- Fresnel ---
    float cosTheta = max(dot(N, V), 0.0);
    float fresnel = 0.04 + 0.96 * pow(1.0 - cosTheta, 5.0);

    // --- Iridescence / Rainbow Caustics ---
    float angle = dot(N, V);
    float rainbowMix = 0.35 + rainbowBoost * 0.5;  // boost for peace sign
    vec3 iridescence = vec3(
        0.5 + 0.5 * sin(angle * 6.2832 + 0.0),
        0.5 + 0.5 * sin(angle * 6.2832 + 2.094),
        0.5 + 0.5 * sin(angle * 6.2832 + 4.189)
    );
    vec3 color = mix(refractedColor, iridescence, fresnel * rainbowMix);

    // --- Reflection tint ---
    color = mix(color, vec3(0.88, 0.92, 1.0), fresnel * 0.4);

    // --- Specular ---
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 196.0);
    color += vec3(1.0) * spec * (0.8 + glowIntensity * 0.6);

    // --- Rim Glow (boosted by gesture) ---
    float rim = pow(1.0 - cosTheta, 3.0);
    vec3 rimColor = mix(vec3(0.7, 0.8, 1.0), vec3(1.0, 0.6, 0.2), glowIntensity);
    color += rimColor * rim * (0.3 + glowIntensity * 0.7);

    // --- Pulse (open palm breathe effect) ---
    float pulse = 0.5 + 0.5 * sin(pulsePhase);
    float alpha = 0.10 + fresnel * 0.45 + rim * 0.2;
    alpha *= (0.85 + pulse * 0.15);

    FragColor = vec4(color, alpha);
}
"""


# ============================================================================
#  EMA SMOOTHING FILTER
# ============================================================================

class EMAFilter:
    """Exponential Moving Average filter for jitter-free tracking."""
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.value = None

    def update(self, raw):
        if self.value is None:
            self.value = raw
        else:
            self.value = self.alpha * raw + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


# ============================================================================
#  HAND TRACKING + GESTURE MODULE
# ============================================================================

_HAND_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
)

# Landmark indices
#   Tips:  Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
#   PIPs:  Thumb=3, Index=6, Middle=10, Ring=14, Pinky=18
#   DIPs:  Thumb=3, Index=7, Middle=11, Ring=15, Pinky=19
#   MCPs:  Thumb=2, Index=5, Middle=9,  Ring=13, Pinky=17
#   Wrist: 0
TIPS = [4, 8, 12, 16, 20]
PIPS = [3, 6, 10, 14, 18]
DIPS = [3, 7, 11, 15, 19]   # For stacked ring #2
MCPS = [2, 5,  9, 13, 17]

# Gesture enum
GESTURE_NONE  = 0
GESTURE_OPEN  = 1   # Open palm
GESTURE_FIST  = 2   # Closed fist
GESTURE_PINCH = 3   # Thumb + index pinch
GESTURE_PEACE = 4   # Peace / V sign


def _dist(a, b):
    """Euclidean distance between two landmarks."""
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


def _finger_is_open(lms, finger_idx):
    """Check if a finger is extended by comparing tip-to-wrist vs pip-to-wrist."""
    tip = lms[TIPS[finger_idx]]
    pip = lms[PIPS[finger_idx]]
    wrist = lms[0]
    return _dist(tip, wrist) > _dist(pip, wrist)


def detect_gesture(lms):
    """Detect hand gesture from landmarks."""
    open_fingers = [_finger_is_open(lms, i) for i in range(5)]
    n_open = sum(open_fingers)

    # Pinch: thumb tip close to index tip
    thumb_tip = lms[4]
    index_tip = lms[8]
    pinch_dist = _dist(thumb_tip, index_tip)
    wrist_mcp_dist = _dist(lms[0], lms[9])

    if pinch_dist < wrist_mcp_dist * 0.25:
        return GESTURE_PINCH

    # Peace: only index + middle extended
    if (open_fingers[1] and open_fingers[2] and
            not open_fingers[3] and not open_fingers[4]):
        return GESTURE_PEACE

    if n_open >= 4:
        return GESTURE_OPEN

    if n_open <= 1:
        return GESTURE_FIST

    return GESTURE_NONE


class HandTracker:
    """
    Tracks hand(s), returns per-finger ring data and detected gesture.

    Each hand returns:
      - rings: list of (x, y, angle_deg, scale, finger_idx) tuples
               with MULTIPLE rings per finger (stacked at PIP and DIP)
      - gesture: one of GESTURE_* constants
    """
    def __init__(self, smoothing_alpha=0.35, max_hands=2):
        if not os.path.exists(_HAND_MODEL_PATH):
            raise FileNotFoundError(
                f"Hand landmarker model not found: {_HAND_MODEL_PATH}"
            )

        base_options = mp_tasks.BaseOptions(model_asset_path=_HAND_MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._frame_ts = 0
        self.max_hands = max_hands

        # filters[hand][joint_slot] = (fx, fy)
        # 5 PIP slots + 5 DIP slots = 10
        self.filters = [
            [(EMAFilter(smoothing_alpha), EMAFilter(smoothing_alpha))
             for _ in range(10)]
            for _ in range(max_hands)
        ]

    def process(self, frame_rgb):
        """
        Returns list of (rings_list, gesture) per hand.
        Each ring: (x, y, angle_deg, scale, finger_idx)
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._frame_ts += 33

        result = self.landmarker.detect_for_video(mp_image, self._frame_ts)
        hands_out = []

        if result.hand_landmarks:
            for h_idx, hand_lms in enumerate(result.hand_landmarks):
                if h_idx >= self.max_hands:
                    break

                # Hand scale
                wrist = hand_lms[0]
                mid_mcp = hand_lms[9]
                hand_size = _dist(wrist, mid_mcp)

                # Detect gesture
                gesture = detect_gesture(hand_lms)

                rings = []
                for f_idx in range(5):
                    pip_lm = hand_lms[PIPS[f_idx]]
                    dip_lm = hand_lms[DIPS[f_idx]]
                    mcp_lm = hand_lms[MCPS[f_idx]]
                    tip_lm = hand_lms[TIPS[f_idx]]

                    # Finger direction for ring orientation
                    dx = tip_lm.x - mcp_lm.x
                    dy = tip_lm.y - mcp_lm.y
                    angle = math.degrees(math.atan2(dy, dx))

                    ring_scale = hand_size * 1.1

                    # Ring 1: at PIP joint
                    sx1 = self.filters[h_idx][f_idx][0].update(pip_lm.x)
                    sy1 = self.filters[h_idx][f_idx][1].update(pip_lm.y)
                    rings.append((sx1, sy1, angle, ring_scale, f_idx))

                    # Ring 2: at DIP joint (skip for thumb — DIP==PIP)
                    if f_idx > 0:
                        sx2 = self.filters[h_idx][f_idx + 5][0].update(dip_lm.x)
                        sy2 = self.filters[h_idx][f_idx + 5][1].update(dip_lm.y)
                        rings.append((sx2, sy2, angle, ring_scale * 0.85, f_idx))

                hands_out.append((rings, gesture))

        return hands_out

    def close(self):
        self.landmarker.close()


# ============================================================================
#  TORUS GEOMETRY GENERATOR
# ============================================================================

def generate_torus(R=0.5, r=0.15, n_major=48, n_minor=24):
    """
    Parametric torus mesh.
    x = (R + r·cos(v))·cos(u),  y = (R + r·cos(v))·sin(u),  z = r·sin(v)
    Normal = (cos(v)·cos(u), cos(v)·sin(u), sin(v))
    """
    verts, norms, idxs = [], [], []
    for i in range(n_major):
        u = 2.0 * math.pi * i / n_major
        cu, su = math.cos(u), math.sin(u)
        for j in range(n_minor):
            v = 2.0 * math.pi * j / n_minor
            cv, sv = math.cos(v), math.sin(v)
            verts.append([(R + r*cv)*cu, (R + r*cv)*su, r*sv])
            norms.append([cv*cu, cv*su, sv])
    for i in range(n_major):
        ni = (i+1) % n_major
        for j in range(n_minor):
            nj = (j+1) % n_minor
            a, b = i*n_minor+j, ni*n_minor+j
            c, d = ni*n_minor+nj, i*n_minor+nj
            idxs.extend([a, b, c, a, c, d])
    return (np.array(verts, dtype=np.float32),
            np.array(norms, dtype=np.float32),
            np.array(idxs, dtype=np.uint32))


# ============================================================================
#  MATRIX UTILITIES
# ============================================================================

def perspective_matrix(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4,4), dtype=np.float32)
    m[0,0]=f/aspect; m[1,1]=f
    m[2,2]=(far+near)/(near-far); m[2,3]=(2*far*near)/(near-far)
    m[3,2]=-1.0
    return m

def look_at_matrix(eye, center, up):
    e, c, u = [np.array(v, dtype=np.float32) for v in (eye, center, up)]
    f = c-e; f /= np.linalg.norm(f)
    s = np.cross(f, u); s /= np.linalg.norm(s)
    u2 = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0,:3]=s;  m[0,3]=-np.dot(s,e)
    m[1,:3]=u2; m[1,3]=-np.dot(u2,e)
    m[2,:3]=-f; m[2,3]=np.dot(f,e)
    return m

def translation_matrix(tx, ty, tz):
    m = np.eye(4, dtype=np.float32)
    m[0,3]=tx; m[1,3]=ty; m[2,3]=tz
    return m

def scale_matrix(sx, sy, sz):
    m = np.eye(4, dtype=np.float32)
    m[0,0]=sx; m[1,1]=sy; m[2,2]=sz
    return m

def rotation_matrix_x(deg):
    a=math.radians(deg); c,s=math.cos(a),math.sin(a)
    m=np.eye(4,dtype=np.float32)
    m[1,1]=c; m[1,2]=-s; m[2,1]=s; m[2,2]=c
    return m

def rotation_matrix_y(deg):
    a=math.radians(deg); c,s=math.cos(a),math.sin(a)
    m=np.eye(4,dtype=np.float32)
    m[0,0]=c; m[0,2]=s; m[2,0]=-s; m[2,2]=c
    return m

def rotation_matrix_z(deg):
    a=math.radians(deg); c,s=math.cos(a),math.sin(a)
    m=np.eye(4,dtype=np.float32)
    m[0,0]=c; m[0,1]=-s; m[1,0]=s; m[1,1]=c
    return m


# ============================================================================
#  MAIN APPLICATION
# ============================================================================

class GlassRingsApp:
    def __init__(self, width=1280, height=720):
        print("=" * 60)
        print("  Hand-Tracked Glass Rings")
        print("  Gestures: Open Palm | Fist | Pinch | Peace Sign")
        print("  Press ESC to quit")
        print("=" * 60)

        self.width = width
        self.height = height
        self.aspect = width / height
        self.spin_angle = 0.0
        self.pulse_phase = 0.0
        self.time_start = time.time()

        # ── Camera ──
        self.cap = None
        for attempt in range(5):
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if self.cap.isOpened():
                ret, _ = self.cap.read()
                if ret:
                    break
            print(f"  Camera not ready (attempt {attempt+1}/5)...")
            self.cap.release()
            time.sleep(1.5)
        else:
            print("ERROR: Cannot open webcam.")
            sys.exit(1)

        # ── Hand Tracker ──
        self.tracker = HandTracker(smoothing_alpha=0.35, max_hands=2)

        # ── OpenGL ──
        self._init_window()
        self._init_gl()
        print("Initialization complete. Starting render loop...")

    def _init_window(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        self.window = glfw.create_window(
            self.width, self.height, "Glass Rings AR", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

    def _init_gl(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)

        self.bg_shader = compileProgram(
            compileShader(BG_VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(BG_FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
            validate=False)
        self.ring_shader = compileProgram(
            compileShader(RING_VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(RING_FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
            validate=False)

        # BG quad
        quad = np.array([
            -1,-1,0,0, 1,-1,1,0, 1,1,1,1,
            -1,-1,0,0, 1,1,1,1, -1,1,0,1,
        ], dtype=np.float32)
        self.bg_vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(self.bg_vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

        # BG texture
        self.bg_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.bg_tex)
        for p, v in [(GL_TEXTURE_MIN_FILTER, GL_LINEAR),
                      (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
                      (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE),
                      (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)]:
            glTexParameteri(GL_TEXTURE_2D, p, v)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Torus
        v, n, idx = generate_torus(R=0.5, r=0.15, n_major=48, n_minor=24)
        self.torus_n_idx = len(idx)
        data = np.hstack([v, n]).astype(np.float32)
        self.torus_vao = glGenVertexArrays(1)
        vbo2 = glGenBuffers(1)
        ebo = glGenBuffers(1)
        glBindVertexArray(self.torus_vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo2)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

        # Matrices
        self.cam_pos = np.array([0,0,3], dtype=np.float32)
        self.view = look_at_matrix([0,0,3],[0,0,0],[0,1,0])
        self.proj = perspective_matrix(45, self.aspect, 0.1, 100)

        # Uniform locations (ring shader)
        p = self.ring_shader
        self.u = {
            'model':   glGetUniformLocation(p, "model"),
            'view':    glGetUniformLocation(p, "view"),
            'proj':    glGetUniformLocation(p, "projection"),
            'cam':     glGetUniformLocation(p, "cameraPos"),
            'light':   glGetUniformLocation(p, "lightDir"),
            'bgTex':   glGetUniformLocation(p, "bgTexture"),
            'glow':    glGetUniformLocation(p, "glowIntensity"),
            'rainbow': glGetUniformLocation(p, "rainbowBoost"),
            'pulse':   glGetUniformLocation(p, "pulsePhase"),
        }

    # ── Rendering helpers ──

    def _upload_bg(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        flipped = cv2.flip(rgb, 0)
        glBindTexture(GL_TEXTURE_2D, self.bg_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                     flipped.shape[1], flipped.shape[0],
                     0, GL_RGB, GL_UNSIGNED_BYTE, flipped)

    def _draw_bg(self):
        glDisable(GL_DEPTH_TEST)
        glUseProgram(self.bg_shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.bg_tex)
        glUniform1i(glGetUniformLocation(self.bg_shader, "bgTexture"), 0)
        glBindVertexArray(self.bg_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)

    def _draw_ring(self, x, y, finger_angle, ring_scale, spin_offset,
                   glow, rainbow, pulse,
                   extra_spin_x=0.0, extra_spin_y=0.0, extra_spin_z=0.0,
                   scale_mult=1.0):
        """
        Render a single ring at normalized position (x, y).
        extra_spin_x/y/z add gesture-driven rotation on top of base spin.
        scale_mult multiplies the ring scale for gesture-driven size changes.
        """
        spread_x = 5.0
        spread_y = 5.0 / self.aspect
        tx = (x - 0.5) * spread_x
        ty = -(y - 0.5) * spread_y

        actual_scale = ring_scale * scale_mult

        T = translation_matrix(tx, ty, 0.0)
        Rz = rotation_matrix_z(-finger_angle + 90.0 + extra_spin_z)
        Rx_perp = rotation_matrix_x(90.0 + extra_spin_x)
        Rs = rotation_matrix_y(self.spin_angle + spin_offset + extra_spin_y)
        S = scale_matrix(actual_scale, actual_scale, actual_scale)

        model = T @ Rz @ Rx_perp @ Rs @ S

        glUseProgram(self.ring_shader)
        glUniformMatrix4fv(self.u['model'], 1, GL_TRUE, model)
        glUniformMatrix4fv(self.u['view'], 1, GL_TRUE, self.view)
        glUniformMatrix4fv(self.u['proj'], 1, GL_TRUE, self.proj)
        glUniform3fv(self.u['cam'], 1, self.cam_pos)

        ld = np.array([0.5, -1.0, -0.3], dtype=np.float32)
        ld /= np.linalg.norm(ld)
        glUniform3fv(self.u['light'], 1, ld)

        glUniform1f(self.u['glow'], glow)
        glUniform1f(self.u['rainbow'], rainbow)
        glUniform1f(self.u['pulse'], pulse)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.bg_tex)
        glUniform1i(self.u['bgTex'], 0)

        glBindVertexArray(self.torus_vao)
        glDrawElements(GL_TRIANGLES, self.torus_n_idx, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    # ── Main loop ──

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                glfw.set_window_should_close(self.window, True)

            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)  # Mirror for natural feel

            # Track hands
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            hands_data = self.tracker.process(frame_rgb)

            # Upload BG
            self._upload_bg(frame)

            # Render
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self._draw_bg()

            t = time.time() - self.time_start
            self.pulse_phase = t * 4.0  # 4 rad/s pulse

            for (rings, gesture) in hands_data:
                # ── Gesture-reactive parameters ──
                glow = 0.0
                rainbow = 0.0
                extra_sx = 0.0   # extra spin on X axis
                extra_sy = 0.0   # extra spin on Y axis
                extra_sz = 0.0   # extra spin on Z axis
                scale_mult = 1.0

                if gesture == GESTURE_OPEN:
                    # Open palm: rings pulse bigger and float gently
                    pulse_val = 0.5 + 0.5 * math.sin(t * 3.0)
                    glow = 0.1
                    rainbow = 0.2
                    scale_mult = 1.1 + 0.15 * pulse_val  # breathe 1.1→1.25
                    extra_sz = math.sin(t * 2.0) * 15.0  # gentle wobble

                elif gesture == GESTURE_FIST:
                    # Fist: rings compress small and glow hot orange
                    glow = 1.0
                    rainbow = 0.0
                    scale_mult = 0.6 + 0.1 * math.sin(t * 8.0)  # tight pulse
                    extra_sx = t * 120.0  # tumble on X

                elif gesture == GESTURE_PINCH:
                    # Pinch: rings flip and tumble rapidly like spinning coins
                    glow = 0.8
                    rainbow = 0.4
                    scale_mult = 1.0
                    extra_sx = t * 360.0   # full flip on X
                    extra_sy = t * 540.0   # fast spin on Y
                    extra_sz = t * 180.0   # roll on Z

                elif gesture == GESTURE_PEACE:
                    # Peace: rainbow explosion + wobble
                    glow = 0.3
                    rainbow = 1.0
                    scale_mult = 1.2
                    extra_sz = math.sin(t * 5.0) * 30.0  # fast wobble
                    extra_sx = math.cos(t * 3.0) * 20.0

                for ring_idx, (rx, ry, angle, scl, f_idx) in enumerate(rings):
                    # Per-finger overrides
                    r_glow = glow
                    r_rainbow = rainbow
                    r_scale = scale_mult
                    r_esx = extra_sx
                    r_esy = extra_sy
                    r_esz = extra_sz

                    if gesture == GESTURE_PINCH:
                        if f_idx <= 1:
                            # Thumb + index: max tumble + glow
                            r_glow = 1.0
                            r_scale = 1.3
                        else:
                            # Other fingers: gentle float
                            r_esx *= 0.2
                            r_esy *= 0.2
                            r_esz *= 0.2
                            r_glow = 0.2

                    if gesture == GESTURE_PEACE:
                        if f_idx in (1, 2):
                            r_rainbow = 1.0
                            r_scale = 1.4
                        else:
                            r_rainbow = 0.3
                            r_scale = 0.8

                    # Stagger spin offset per ring for variety
                    spin_off = ring_idx * 60.0

                    self._draw_ring(
                        rx, ry, angle, scl, spin_off,
                        r_glow, r_rainbow, self.pulse_phase,
                        extra_spin_x=r_esx,
                        extra_spin_y=r_esy,
                        extra_spin_z=r_esz,
                        scale_mult=r_scale,
                    )

            self.spin_angle += 1.5
            glfw.swap_buffers(self.window)

        self._shutdown()

    def _shutdown(self):
        print("\nShutting down...")
        self.tracker.close()
        self.cap.release()
        glfw.terminate()
        print("Done.")


# ============================================================================
if __name__ == "__main__":
    app = GlassRingsApp(width=1280, height=720)
    app.run()
