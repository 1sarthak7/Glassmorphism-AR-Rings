"""
Ring Controller
===============
Multi-ring spawning, nested rings, dynamic deformation, collapse animation.
"""

import math
import numpy as np
from OpenGL.GL import *
import ctypes
from math_utils import (generate_torus, translation_matrix, scale_matrix,
                         rotation_matrix_x, rotation_matrix_y, rotation_matrix_z,
                         fbm_noise, clamp, lerp)
from gesture_engine import PIPS, DIPS, MCPS, TIPS


class Ring:
    """A single ring instance with its own state."""
    def __init__(self, layer=0):
        self.layer = layer          # 0=innermost, higher=outer
        self.x = 0.5
        self.y = 0.5
        self.angle = 0.0           # Finger direction angle
        self.base_scale = 1.0
        self.spin = 0.0            # Current spin angle
        self.spin_speed = 1.5      # deg/frame
        self.alpha_mult = 1.0
        self.extra_rx = 0.0        # Extra rotation on X
        self.extra_ry = 0.0        # Extra rotation on Y
        self.extra_rz = 0.0        # Extra rotation on Z
        self.is_portal = False
        self.lifetime = 0.0


class RingController:
    """
    Manages multiple ring instances, their positions, deformation,
    and rendering.
    """

    def __init__(self):
        self.rings = []
        self.torus_vao = None
        self.torus_n_idx = 0
        self._aspect = 16.0 / 9.0
        # Ring geometry params for nested rings
        self._ring_radii = [
            (0.50, 0.15),   # Layer 0: standard
            (0.70, 0.10),   # Layer 1: bigger, thinner
            (0.90, 0.08),   # Layer 2: even bigger
            (1.10, 0.06),   # Layer 3: large, thin
            (1.30, 0.05),   # Layer 4: outermost, very thin
        ]

    def init_gl(self):
        """Create torus VAO with geometry."""
        verts, norms, indices = generate_torus(R=0.5, r=0.15, n_major=48, n_minor=24)
        self.torus_n_idx = len(indices)

        data = np.hstack([verts, norms]).astype(np.float32)
        self.torus_vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        ebo = glGenBuffers(1)
        glBindVertexArray(self.torus_vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

    def update(self, gesture_result, energy, dt, t):
        """
        Update ring positions from hand data and energy state.
        """
        self.rings.clear()

        for hand in gesture_result.hands:
            lm = hand.smooth_lm
            if lm is None:
                continue

            hand_size = hand.hand_size

            for f_idx in range(5):
                pip = lm[PIPS[f_idx]]
                mcp = lm[MCPS[f_idx]]
                tip = lm[TIPS[f_idx]]

                dx = tip[0] - mcp[0]
                dy = tip[1] - mcp[1]
                angle = math.degrees(math.atan2(dy, dx))

                # Base ring at PIP
                ring = Ring(layer=0)
                ring.x = pip[0]
                ring.y = pip[1]
                ring.angle = angle
                ring.base_scale = hand_size * 1.1 * energy.scale_multiplier
                ring.spin_speed = 1.5 * energy.spin_multiplier
                ring.spin = t * ring.spin_speed * 60.0 + f_idx * 72.0
                ring.extra_rx = 0.0
                ring.extra_ry = 0.0
                ring.extra_rz = 0.0
                self.rings.append(ring)

                # DIP ring (skip for thumb)
                if f_idx > 0:
                    dip = lm[DIPS[f_idx]]
                    ring2 = Ring(layer=1)
                    ring2.x = dip[0]
                    ring2.y = dip[1]
                    ring2.angle = angle
                    ring2.base_scale = hand_size * 0.9 * energy.scale_multiplier
                    ring2.spin = t * ring.spin_speed * 60.0 + f_idx * 72.0 + 180.0
                    ring2.alpha_mult = 0.8
                    self.rings.append(ring2)

                # Nested rings (energy-dependent, layers 2+)
                for layer in range(2, energy.ring_count):
                    nested = Ring(layer=layer)
                    nested.x = pip[0]
                    nested.y = pip[1]
                    nested.angle = angle
                    R_mult = 1.0 + layer * 0.4
                    nested.base_scale = hand_size * 1.1 * R_mult * energy.scale_multiplier
                    nested.spin = t * ring.spin_speed * 40.0 + layer * 60.0
                    nested.alpha_mult = max(0.15, 0.6 - layer * 0.12)
                    # Counter-rotate nested layers
                    nested.extra_ry = layer * 30.0
                    nested.extra_rz = math.sin(t * 2.0 + layer) * 15.0
                    self.rings.append(nested)

        # Apply collapse animation
        if energy.collapse_t > 0.01:
            for ring in self.rings:
                ring.base_scale *= (1.0 - energy.collapse_t * 0.7)
                ring.spin += energy.collapse_t * 360.0 * 3  # Spiral inward

    def draw(self, shader_mgr, energy, view_mat, proj_mat, cam_pos, bg_tex, t):
        """Render all rings."""
        shader_mgr.use_ring_shader()
        prog = shader_mgr.current_ring_shader

        # Shared uniforms
        glUniformMatrix4fv(glGetUniformLocation(prog, "view"), 1, GL_TRUE, view_mat)
        glUniformMatrix4fv(glGetUniformLocation(prog, "projection"), 1, GL_TRUE, proj_mat)
        glUniform3fv(glGetUniformLocation(prog, "cameraPos"), 1, cam_pos)
        ld = np.array([0.5, -1.0, -0.3], dtype=np.float32)
        ld /= np.linalg.norm(ld)
        glUniform3fv(glGetUniformLocation(prog, "lightDir"), 1, ld)
        glUniform1f(glGetUniformLocation(prog, "uTime"), t)
        glUniform1f(glGetUniformLocation(prog, "uEnergy"), energy.energy)
        glUniform1f(glGetUniformLocation(prog, "uGlow"), energy.glow_intensity)
        glUniform1f(glGetUniformLocation(prog, "uRainbow"), energy.rainbow_boost)
        glUniform1f(glGetUniformLocation(prog, "uPulse"), t * 4.0)
        glUniform1f(glGetUniformLocation(prog, "uDeformAmount"), energy.deform_amount)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, bg_tex)
        glUniform1i(glGetUniformLocation(prog, "bgTexture"), 0)

        glBindVertexArray(self.torus_vao)

        spread_x = 5.0
        spread_y = 5.0 / self._aspect

        for ring in self.rings:
            tx = (ring.x - 0.5) * spread_x
            ty = -(ring.y - 0.5) * spread_y

            T = translation_matrix(tx, ty, 0.0)
            Rz = rotation_matrix_z(-ring.angle + 90.0 + ring.extra_rz)
            Rx = rotation_matrix_x(90.0 + ring.extra_rx)
            Ry = rotation_matrix_y(ring.spin + ring.extra_ry)
            S = scale_matrix(ring.base_scale, ring.base_scale, ring.base_scale)

            model = T @ Rz @ Rx @ Ry @ S
            glUniformMatrix4fv(glGetUniformLocation(prog, "model"), 1, GL_TRUE, model)

            # Bend target = ring center in world space
            glUniform3f(glGetUniformLocation(prog, "uBendTarget"), tx, ty, 0.0)

            glDrawElements(GL_TRIANGLES, self.torus_n_idx, GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

    def set_aspect(self, aspect):
        self._aspect = aspect
