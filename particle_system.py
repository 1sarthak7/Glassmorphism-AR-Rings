"""
Particle System
===============
GPU-efficient particle orbits, energy arcs, trail echoes.
"""

import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import ctypes

# ============================================================================
#  PARTICLE SHADERS
# ============================================================================

PARTICLE_VERT = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;  // rgba
layout (location = 2) in float aSize;

uniform mat4 view;
uniform mat4 projection;

out vec4 vColor;

void main() {
    gl_Position = projection * view * vec4(aPos, 1.0);
    gl_PointSize = aSize / gl_Position.w * 400.0;
    vColor = aColor;
}
"""

PARTICLE_FRAG = """
#version 330 core
in vec4 vColor;
out vec4 FragColor;

void main() {
    // Soft circle with glow
    vec2 center = gl_PointCoord - 0.5;
    float dist = length(center);
    if (dist > 0.5) discard;
    float alpha = smoothstep(0.5, 0.1, dist);
    FragColor = vec4(vColor.rgb, vColor.a * alpha);
}
"""


class ParticleSystem:
    """
    Manages orbit particles and energy arc particles.
    Updated each frame on CPU (vectorized NumPy), rendered as GL_POINTS.
    """

    MAX_PARTICLES = 800

    def __init__(self):
        self.shader = None
        self.vao = None
        self.vbo = None
        self.n_active = 0

        # Particle data: [x, y, z, r, g, b, a, size] per particle
        self.data = np.zeros((self.MAX_PARTICLES, 8), dtype=np.float32)
        # Particle state
        self.angles = np.zeros(self.MAX_PARTICLES, dtype=np.float32)
        self.radii = np.zeros(self.MAX_PARTICLES, dtype=np.float32)
        self.speeds = np.zeros(self.MAX_PARTICLES, dtype=np.float32)
        self.z_offsets = np.zeros(self.MAX_PARTICLES, dtype=np.float32)

    def init_gl(self):
        """Compile particle shader and create VAO."""
        self.shader = compileProgram(
            compileShader(PARTICLE_VERT, GL_VERTEX_SHADER),
            compileShader(PARTICLE_FRAG, GL_FRAGMENT_SHADER),
            validate=False,
        )

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER,
                     self.MAX_PARTICLES * 8 * 4,
                     None, GL_DYNAMIC_DRAW)

        stride = 8 * 4  # 8 floats * 4 bytes
        # Position (x, y, z)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # Color (r, g, b, a)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        # Size
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(28))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)
        glEnable(GL_PROGRAM_POINT_SIZE)

    def update(self, ring_positions, energy, t, dt, aspect):
        """
        Update particle positions based on ring locations and energy state.

        ring_positions: list of (x, y) normalized positions
        """
        intensity = energy.particle_intensity
        n_rings = len(ring_positions)
        if n_rings == 0 or intensity < 0.01:
            self.n_active = 0
            return

        # Allocate particles per ring
        particles_per_ring = min(
            int(10 + intensity * 40),
            self.MAX_PARTICLES // max(n_rings, 1)
        )
        self.n_active = min(particles_per_ring * n_rings, self.MAX_PARTICLES)

        spread_x = 5.0
        spread_y = 5.0 / aspect
        rainbow_amt = energy.rainbow_boost

        idx = 0
        for ri, (rx, ry) in enumerate(ring_positions):
            if idx >= self.n_active:
                break

            cx = (rx - 0.5) * spread_x
            cy = -(ry - 0.5) * spread_y

            for p in range(particles_per_ring):
                if idx >= self.n_active:
                    break

                # Initialize on first appearance
                if self.speeds[idx] < 0.01:
                    self.angles[idx] = np.random.uniform(0, 2 * math.pi)
                    self.radii[idx] = np.random.uniform(0.3, 0.9)
                    self.speeds[idx] = np.random.uniform(1.0, 4.0)
                    self.z_offsets[idx] = np.random.uniform(-0.3, 0.3)

                # Update orbit
                self.angles[idx] += self.speeds[idx] * dt * (1.0 + energy.energy * 3.0)
                angle = self.angles[idx]
                radius = self.radii[idx] * (0.5 + intensity * 0.5)

                # 3D orbit position
                x = cx + math.cos(angle) * radius
                y = cy + math.sin(angle) * radius * 0.6
                z = self.z_offsets[idx] + math.sin(angle * 2.0 + t) * 0.15

                # --- Color: neutral by default, rainbow only with peace sign ---
                if rainbow_amt > 0.1:
                    # Rainbow mode: colorful particles
                    hue = (angle / (2 * math.pi) + t * 0.1) % 1.0
                    r = 0.5 + 0.5 * math.sin(hue * 6.2832)
                    g = 0.5 + 0.5 * math.sin(hue * 6.2832 + 2.094)
                    b = 0.5 + 0.5 * math.sin(hue * 6.2832 + 4.189)
                    # Blend with white based on rainbow_amt
                    r = 0.8 * (1.0 - rainbow_amt) + r * rainbow_amt
                    g = 0.85 * (1.0 - rainbow_amt) + g * rainbow_amt
                    b = 1.0 * (1.0 - rainbow_amt) + b * rainbow_amt
                else:
                    # Default: clean ice-white/blue
                    r, g, b = 0.8, 0.85, 1.0

                # Energy/glow brightness boost
                brightness = 0.3 + energy.glow_intensity * 0.7
                r *= brightness; g *= brightness; b *= brightness

                alpha = 0.15 + intensity * 0.35
                size = 1.5 + energy.energy * 4.0

                self.data[idx] = [x, y, z, r, g, b, alpha, size]
                idx += 1

    def draw(self, view_mat, proj_mat):
        """Render all active particles."""
        if self.n_active == 0:
            return

        glUseProgram(self.shader)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"),
                           1, GL_TRUE, view_mat)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"),
                           1, GL_TRUE, proj_mat)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                        self.n_active * 8 * 4,
                        self.data[:self.n_active])

        glDepthMask(GL_FALSE)  # Don't write depth for transparent particles
        glDrawArrays(GL_POINTS, 0, self.n_active)
        glDepthMask(GL_TRUE)

        glBindVertexArray(0)
