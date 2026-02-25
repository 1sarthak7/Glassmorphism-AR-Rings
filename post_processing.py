"""
Post-Processing Pipeline
========================
FBO-based multi-pass: bloom, chromatic aberration, shockwave ripple, vignette.
Uses ping-pong FBOs for efficient multi-pass without stalling.
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import ctypes

# ============================================================================
#  POST-PROCESSING SHADERS
# ============================================================================

FULLSCREEN_VERT = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
"""

# --- BRIGHT PASS (extract bloom-worthy pixels) ---
BRIGHT_PASS_FRAG = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform float uThreshold;

void main() {
    vec3 color = texture(uTexture, TexCoord).rgb;
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    if (brightness > uThreshold) {
        FragColor = vec4(color * (brightness - uThreshold), 1.0);
    } else {
        FragColor = vec4(0.0);
    }
}
"""

# --- GAUSSIAN BLUR (single-axis, called twice for H+V) ---
BLUR_FRAG = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform vec2 uDirection;  // (1/width, 0) or (0, 1/height)
uniform float uRadius;

void main() {
    vec3 color = vec3(0.0);
    float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    color += texture(uTexture, TexCoord).rgb * weights[0];
    for (int i = 1; i < 5; i++) {
        vec2 offset = uDirection * float(i) * uRadius;
        color += texture(uTexture, TexCoord + offset).rgb * weights[i];
        color += texture(uTexture, TexCoord - offset).rgb * weights[i];
    }
    FragColor = vec4(color, 1.0);
}
"""

# --- COMPOSITE (bloom blend + chromatic aberration + vignette + grain + shockwave) ---
COMPOSITE_FRAG = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uBloomStrength;
uniform float uChromaticAmount;
uniform float uVignetteStrength;
uniform float uGrainAmount;
uniform float uTime;
uniform float uShockwaveT;        // 0..1 shockwave progress (0 = off)
uniform vec2 uShockwaveCenter;    // screen-space center

void main() {
    vec2 uv = TexCoord;

    // --- Shockwave ripple ---
    if (uShockwaveT > 0.001 && uShockwaveT < 0.999) {
        vec2 delta = uv - uShockwaveCenter;
        float dist = length(delta);
        float radius = uShockwaveT * 1.5;
        float thickness = 0.08;
        float wave = smoothstep(radius - thickness, radius, dist) *
                     (1.0 - smoothstep(radius, radius + thickness, dist));
        float distortion = wave * 0.03 * (1.0 - uShockwaveT);
        uv += normalize(delta + 0.001) * distortion;
    }

    // --- Chromatic aberration ---
    vec2 center = uv - 0.5;
    float dist = length(center);
    float ca = uChromaticAmount * dist;
    vec3 scene;
    scene.r = texture(uScene, uv + center * ca).r;
    scene.g = texture(uScene, uv).g;
    scene.b = texture(uScene, uv - center * ca).b;

    // --- Bloom blend ---
    vec3 bloom = texture(uBloom, TexCoord).rgb;
    vec3 color = scene + bloom * uBloomStrength;

    // --- Vignette ---
    float vignette = 1.0 - smoothstep(0.4, 1.2, dist * 1.4);
    color *= mix(1.0, vignette, uVignetteStrength);

    // --- Film grain ---
    float grain = fract(sin(dot(uv + uTime * 0.1, vec2(12.9898, 78.233))) * 43758.5453);
    color += (grain - 0.5) * uGrainAmount;

    FragColor = vec4(color, 1.0);
}
"""


# ============================================================================
#  POST-PROCESSING CLASS
# ============================================================================

class PostProcessing:
    """
    FBO-based post-processing pipeline with ping-pong buffers.

    Pipeline order:
    1. Scene rendered to scene FBO
    2. Bright pass → bloom FBO A
    3. Blur H → bloom FBO B
    4. Blur V → bloom FBO A
    5. Composite (scene + bloom + chromatic + vignette + grain + shockwave) → screen
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.enabled = True

    def init_gl(self):
        """Create FBOs, textures, and compile post-processing shaders."""
        # Scene FBO (with depth buffer)
        self.scene_fbo, self.scene_tex = self._create_fbo(with_depth=True)
        # Bloom ping-pong FBOs (half resolution for performance)
        hw, hh = self.width // 2, self.height // 2
        self.bloom_fbo_a, self.bloom_tex_a = self._create_fbo(hw, hh)
        self.bloom_fbo_b, self.bloom_tex_b = self._create_fbo(hw, hh)

        # Compile shaders
        vert = compileShader(FULLSCREEN_VERT, GL_VERTEX_SHADER)
        self.bright_shader = compileProgram(
            vert, compileShader(BRIGHT_PASS_FRAG, GL_FRAGMENT_SHADER),
            validate=False)
        self.blur_shader = compileProgram(
            compileShader(FULLSCREEN_VERT, GL_VERTEX_SHADER),
            compileShader(BLUR_FRAG, GL_FRAGMENT_SHADER),
            validate=False)
        self.composite_shader = compileProgram(
            compileShader(FULLSCREEN_VERT, GL_VERTEX_SHADER),
            compileShader(COMPOSITE_FRAG, GL_FRAGMENT_SHADER),
            validate=False)

        # Fullscreen quad
        quad = np.array([
            -1, -1, 0, 0,  1, -1, 1, 0,  1, 1, 1, 1,
            -1, -1, 0, 0,  1, 1, 1, 1,  -1, 1, 0, 1,
        ], dtype=np.float32)
        self.quad_vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(self.quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

    def _create_fbo(self, w=None, h=None, with_depth=False):
        """Create an FBO with a color texture attachment."""
        w = w or self.width
        h = h or self.height

        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        # Use GL_RGBA / GL_UNSIGNED_BYTE for macOS Metal compatibility
        # (Metal's OpenGL translation layer can silently fail with
        #  GL_RGB16F or non-RGBA formats)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                GL_TEXTURE_2D, tex, 0)

        # Depth renderbuffer for scene FBO
        if with_depth:
            rbo = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, rbo)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                      GL_RENDERBUFFER, rbo)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f"WARNING: FBO incomplete (status {status})")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        return fbo, tex

    def begin_scene(self):
        """Bind scene FBO for rendering."""
        if not self.enabled:
            return
        glBindFramebuffer(GL_FRAMEBUFFER, self.scene_fbo)
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def end_and_apply(self, energy, t, fb_w=None, fb_h=None):
        """Run post-processing passes and render to screen.
        fb_w, fb_h: actual framebuffer size (for retina displays).
        """
        if not self.enabled:
            return

        # Use provided framebuffer size or fall back to logical size
        screen_w = fb_w or self.width
        screen_h = fb_h or self.height

        hw, hh = self.width // 2, self.height // 2

        # *** CRITICAL: Disable depth testing for all 2D post-processing passes.
        # Without this, the fullscreen quad (at z=0, depth=0.5) passes on frame 1
        # (against default depth 1.0) but FAILS on frame 2+ because the depth
        # buffer retains 0.5 from the previous frame → GL_LESS rejects it → black.
        glDisable(GL_DEPTH_TEST)

        # --- Pass 1: Bright pass ---
        glBindFramebuffer(GL_FRAMEBUFFER, self.bloom_fbo_a)
        glViewport(0, 0, hw, hh)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.bright_shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.scene_tex)
        glUniform1i(glGetUniformLocation(self.bright_shader, "uTexture"), 0)
        threshold = max(0.3, 0.8 - energy.glow_intensity * 0.5)
        glUniform1f(glGetUniformLocation(self.bright_shader, "uThreshold"),
                    threshold)
        self._draw_quad()

        # --- Pass 2: Horizontal blur ---
        glBindFramebuffer(GL_FRAMEBUFFER, self.bloom_fbo_b)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.blur_shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.bloom_tex_a)
        glUniform1i(glGetUniformLocation(self.blur_shader, "uTexture"), 0)
        glUniform2f(glGetUniformLocation(self.blur_shader, "uDirection"),
                    1.0 / hw, 0.0)
        glUniform1f(glGetUniformLocation(self.blur_shader, "uRadius"), 2.0)
        self._draw_quad()

        # --- Pass 3: Vertical blur ---
        glBindFramebuffer(GL_FRAMEBUFFER, self.bloom_fbo_a)
        glClear(GL_COLOR_BUFFER_BIT)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.bloom_tex_b)
        glUniform2f(glGetUniformLocation(self.blur_shader, "uDirection"),
                    0.0, 1.0 / hh)
        self._draw_quad()

        # --- Pass 4: Composite to screen ---
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, screen_w, screen_h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.composite_shader)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.scene_tex)
        glUniform1i(glGetUniformLocation(self.composite_shader, "uScene"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.bloom_tex_a)
        glUniform1i(glGetUniformLocation(self.composite_shader, "uBloom"), 1)

        bloom_str = 0.3 + energy.glow_intensity * 0.7
        glUniform1f(glGetUniformLocation(self.composite_shader, "uBloomStrength"),
                    bloom_str)
        glUniform1f(glGetUniformLocation(self.composite_shader, "uChromaticAmount"),
                    0.002 + energy.energy * 0.008)
        glUniform1f(glGetUniformLocation(self.composite_shader, "uVignetteStrength"),
                    0.3 + energy.energy * 0.3)
        glUniform1f(glGetUniformLocation(self.composite_shader, "uGrainAmount"),
                    0.02)
        glUniform1f(glGetUniformLocation(self.composite_shader, "uTime"), t)
        glUniform1f(glGetUniformLocation(self.composite_shader, "uShockwaveT"),
                    energy.shockwave_t)
        glUniform2f(glGetUniformLocation(self.composite_shader, "uShockwaveCenter"),
                    energy.shockwave_center[0], 1.0 - energy.shockwave_center[1])

        self._draw_quad()

        # Re-enable depth testing for next frame's scene rendering
        glEnable(GL_DEPTH_TEST)
        # Reset active texture unit
        glActiveTexture(GL_TEXTURE0)

    def _draw_quad(self):
        glBindVertexArray(self.quad_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
