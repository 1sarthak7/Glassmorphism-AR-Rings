"""
Cinematic AR Energy Interface — Main Entry Point
=================================================
Integrates all modules: gesture engine, energy state, ring controller,
shader manager, particle system, post-processing pipeline.

Usage:
    python main.py

Controls:
    ESC         — Quit
    Snap        — Cycle shader mode (thumb + middle finger snap)
    Open Palm   — Rings breathe and pulse
    Fist (hold) — Charge energy (release for shockwave)
    Pinch       — Fast spin + glow
    Peace Sign  — Rainbow caustics
    Draw Circle — Spawn portal ring
    Two-hand Pull — Ring deformation
"""

import cv2
import numpy as np
import mediapipe as mp
import glfw
from OpenGL.GL import *
import ctypes
import sys
import time
import os

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from math_utils import (perspective_matrix, look_at_matrix,
                         translation_matrix, scale_matrix)
from gesture_engine import GestureEngine, PIPS
from energy_state import EnergyState
from shader_manager import ShaderManager, SHADER_NAMES
from ring_controller import RingController
from particle_system import ParticleSystem
from post_processing import PostProcessing


# ============================================================================
#  HAND LANDMARKER WRAPPER
# ============================================================================

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
)


class HandLandmarkerWrapper:
    """Thin wrapper around MediaPipe HandLandmarker for the Tasks API."""

    def __init__(self, max_hands=2):
        if not os.path.exists(_MODEL_PATH):
            print(f"ERROR: {_MODEL_PATH} not found")
            sys.exit(1)

        base_options = mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._ts = 0

    def detect(self, frame_rgb):
        """Returns list of hand landmarks (each is a list of 21 landmarks)."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._ts += 33
        result = self.landmarker.detect_for_video(mp_image, self._ts)
        return result.hand_landmarks if result.hand_landmarks else []

    def close(self):
        self.landmarker.close()


# ============================================================================
#  MAIN APPLICATION
# ============================================================================

class CinematicARApp:
    """
    Main application: ties together camera, hand detection, gesture engine,
    energy state, ring rendering, particles, and post-processing.
    """

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.aspect = width / height

        print("=" * 60)
        print("  Cinematic AR Energy Interface")
        print("  6 Shader Modes | Gesture Reactive | Post-Processing")
        print("=" * 60)
        print("  Controls:")
        print("    Snap (thumb+middle) = Cycle shader mode")
        print("    Open Palm = Breathe & pulse")
        print("    Fist hold → Open = Shockwave")
        print("    Pinch = Fast spin + glow")
        print("    Peace = Rainbow caustics")
        print("    Draw circle = Portal ring")
        print("    ESC = Quit")
        print("=" * 60)

        # ── Camera ──
        self.cap = self._init_camera()

        # ── Modules ──
        self.hand_detector = HandLandmarkerWrapper(max_hands=2)
        self.gesture_engine = GestureEngine(max_hands=2)
        self.energy_state = EnergyState()
        self.shader_mgr = ShaderManager()
        self.ring_ctrl = RingController()
        self.particles = ParticleSystem()
        self.post_proc = PostProcessing(width, height)

        # ── OpenGL ──
        self._init_window()
        self._init_gl()

        self.time_start = time.time()
        self._last_mode_name = ""
        self._frame_count = 0
        self._fps_time = time.time()
        print("Initialization complete. Starting render loop...")

    def _init_camera(self):
        cap = None
        for attempt in range(5):
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap
            print(f"  Camera not ready ({attempt + 1}/5)...")
            if cap:
                cap.release()
            time.sleep(1.5)
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    def _init_window(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        self.window = glfw.create_window(
            self.width, self.height, "Cinematic AR Energy Interface", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

    def _init_gl(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)

        # Compile all shaders
        print("Compiling shaders...")
        self.shader_mgr.compile_all()

        # Init ring geometry
        self.ring_ctrl.init_gl()
        self.ring_ctrl.set_aspect(self.aspect)

        # Init particles
        self.particles.init_gl()

        # Init post-processing FBOs
        print("Creating post-processing FBOs...")
        self.post_proc.init_gl()

        # Background texture
        self.bg_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.bg_tex)
        for p, v in [(GL_TEXTURE_MIN_FILTER, GL_LINEAR),
                      (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
                      (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE),
                      (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)]:
            glTexParameteri(GL_TEXTURE_2D, p, v)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Background quad VAO
        quad = np.array([
            -1, -1, 0, 0,  1, -1, 1, 0,  1, 1, 1, 1,
            -1, -1, 0, 0,  1, 1, 1, 1,  -1, 1, 0, 1,
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

        # Camera matrices
        self.cam_pos = np.array([0, 0, 3], dtype=np.float32)
        self.view_mat = look_at_matrix([0, 0, 3], [0, 0, 0], [0, 1, 0])
        self.proj_mat = perspective_matrix(45, self.aspect, 0.1, 100)

    def _upload_bg(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        flipped = cv2.flip(rgb, 0)
        # CRITICAL: reset to texture unit 0 — post-processing leaves unit 1 active
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.bg_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                     flipped.shape[1], flipped.shape[0],
                     0, GL_RGB, GL_UNSIGNED_BYTE, flipped)

    def _draw_bg(self):
        glDisable(GL_DEPTH_TEST)
        self.shader_mgr.use_bg_shader()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.bg_tex)
        glUniform1i(glGetUniformLocation(self.shader_mgr.bg_shader, "bgTexture"), 0)
        glBindVertexArray(self.bg_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)

    # ── Main render loop ──

    def run(self):
        use_post_proc = True
        pp_toggled = False

        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                glfw.set_window_should_close(self.window, True)

            # Toggle post-processing with P key
            p_pressed = glfw.get_key(self.window, glfw.KEY_P) == glfw.PRESS
            if p_pressed and not pp_toggled:
                use_post_proc = not use_post_proc
                print(f"  Post-processing: {'ON' if use_post_proc else 'OFF'}")
                pp_toggled = True
            elif not p_pressed:
                pp_toggled = False

            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)  # Mirror
            t = time.time() - self.time_start

            # ── Hand detection ──
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            hand_landmarks = self.hand_detector.detect(frame_rgb)

            # ── Gesture processing ──
            gesture_result = self.gesture_engine.process(hand_landmarks)

            # ── Energy state update ──
            self.energy_state.update(gesture_result, gesture_result.dt)

            # Apply shader mode from energy state
            self.shader_mgr.current_mode = self.energy_state.shader_mode

            # ── Ring controller update ──
            self.ring_ctrl.update(gesture_result, self.energy_state,
                                  gesture_result.dt, t)

            # Collect ring positions for particles
            ring_positions = [(r.x, r.y) for r in self.ring_ctrl.rings[:20]]

            # ── Particle update ──
            self.particles.update(ring_positions, self.energy_state,
                                   t, gesture_result.dt, self.aspect)

            # ── Reset GL state for clean frame ──
            glActiveTexture(GL_TEXTURE0)

            # ── Upload webcam texture ──
            self._upload_bg(frame)

            # ── Get actual framebuffer size (retina = 2x window) ──
            fb_w, fb_h = glfw.get_framebuffer_size(self.window)

            if use_post_proc:
                # --- FBO path: render scene → FBO → post-process → screen ---
                self.post_proc.begin_scene()
            else:
                # --- Direct path: render straight to screen ---
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                glViewport(0, 0, fb_w, fb_h)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # ── Draw scene ──
            self._draw_bg()

            self.ring_ctrl.draw(
                self.shader_mgr, self.energy_state,
                self.view_mat, self.proj_mat, self.cam_pos,
                self.bg_tex, t
            )

            self.particles.draw(self.view_mat, self.proj_mat)

            if use_post_proc:
                # ── Post-processing → screen ──
                self.post_proc.end_and_apply(self.energy_state, t, fb_w, fb_h)

            # ── HUD: shader mode name ──
            mode_name = self.shader_mgr.current_mode_name
            if mode_name != self._last_mode_name:
                print(f"  Shader mode: {mode_name}")
                self._last_mode_name = mode_name

            # FPS counter
            self._frame_count += 1
            now = time.time()
            if now - self._fps_time >= 3.0:
                fps = self._frame_count / (now - self._fps_time)
                es = self.energy_state
                print(f"  FPS: {fps:.1f}  rainbow={es.rainbow_boost:.3f}  "
                      f"glow={es.glow_intensity:.2f}  state={es.state}")
                self._frame_count = 0
                self._fps_time = now

            glfw.swap_buffers(self.window)

        self._shutdown()

    def _shutdown(self):
        print("\nShutting down...")
        self.hand_detector.close()
        self.cap.release()
        glfw.terminate()
        print("Done.")


# ============================================================================
if __name__ == "__main__":
    app = CinematicARApp(width=1280, height=720)
    app.run()

