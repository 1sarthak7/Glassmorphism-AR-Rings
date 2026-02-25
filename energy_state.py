"""
Energy State Engine
===================
State machine driving all visual parameters based on gesture input.
States: Idle → Charging → Overload → Shockwave, plus Portal and Collapse.
"""

import time
from math_utils import lerp, smoothstep, clamp
from gesture_engine import (G_FIST, G_OPEN_PALM, G_PINCH, G_PEACE,
                             G_SPREAD, G_POINT, G_NONE)

# ============================================================================
#  STATES
# ============================================================================

STATE_IDLE      = 'idle'
STATE_CHARGING  = 'charging'
STATE_OVERLOAD  = 'overload'
STATE_SHOCKWAVE = 'shockwave'
STATE_PORTAL    = 'portal'
STATE_COLLAPSE  = 'collapse'

# Shader mode indices
SHADER_GLASS       = 0
SHADER_LIQUID       = 1
SHADER_PLASMA       = 2
SHADER_HOLOGRAPHIC  = 3
SHADER_DARK_MATTER  = 4
SHADER_SPACETIME    = 5
NUM_SHADER_MODES    = 6


class EnergyState:
    """
    Central state machine that drives all visual parameters.

    Exposes smoothly-interpolated values consumed by the rendering pipeline:
      - energy:          0→1 overall energy level
      - charge:          0→1 charge buildup during fist hold
      - glow_intensity:  0→1 ring edge glow
      - rainbow_boost:   0→1 iridescence strength
      - shake_intensity: 0→1 screen/ring shake
      - ring_count:      target number of visible ring layers (1→5)
      - shader_mode:     which shader program to use
      - shockwave_t:     0→1 shockwave animation progress (0 = inactive)
      - portal_t:        0→1 portal ring animation
      - collapse_t:      0→1 collapse animation
      - spin_multiplier: rotation speed multiplier
      - scale_multiplier: ring scale multiplier
      - deform_amount:   vertex deformation strength
      - particle_intensity: 0→1 particle system intensity
    """

    def __init__(self):
        self.state = STATE_IDLE
        self.energy = 0.0
        self.charge = 0.0
        self.glow_intensity = 0.0
        self.rainbow_boost = 0.0
        self.shake_intensity = 0.0
        self.ring_count = 1
        self.shader_mode = SHADER_GLASS
        self.shockwave_t = 0.0
        self.shockwave_center = (0.5, 0.5)
        self.portal_t = 0.0
        self.portal_center = (0.5, 0.5)
        self.portal_radius = 0.1
        self.collapse_t = 0.0
        self.spin_multiplier = 1.0
        self.scale_multiplier = 1.0
        self.deform_amount = 0.0
        self.particle_intensity = 0.0
        self._charge_start = None
        self._state_time = time.time()

    def update(self, gesture_result, dt):
        """
        Update state machine based on gesture results.
        Call once per frame.
        """
        # Decay values toward idle
        self._decay(dt)

        # Process per-hand static gestures
        for hand in gesture_result.hands:
            self._process_hand_gesture(hand, dt)

        # Process events (temporal gestures)
        for event_type, h_idx, data in gesture_result.events:
            self._process_event(event_type, h_idx, data, gesture_result)

        # Animate active states
        self._animate_states(dt)

        # Clamp all values
        self._clamp_all()

    def _process_hand_gesture(self, hand, dt):
        """React to static per-hand gestures."""
        g = hand.static_gesture

        if g == G_FIST:
            # Charging: build energy while holding fist
            if self.state in (STATE_IDLE, STATE_CHARGING):
                self.state = STATE_CHARGING
                self.charge = min(self.charge + dt * 0.6, 1.0)
                self.energy = lerp(self.energy, self.charge, dt * 3.0)
                self.glow_intensity = lerp(self.glow_intensity,
                                            0.5 + self.charge * 0.5, dt * 4.0)
                self.spin_multiplier = lerp(self.spin_multiplier,
                                             0.3 + self.charge * 0.7, dt * 3.0)
                self.scale_multiplier = lerp(self.scale_multiplier,
                                              0.6 + 0.2 * self.charge, dt * 4.0)
                self.deform_amount = lerp(self.deform_amount,
                                           self.charge * 0.5, dt * 3.0)
                self.shake_intensity = self.charge * 0.4

                # Overload at max charge
                if self.charge >= 0.99:
                    self.state = STATE_OVERLOAD

        elif g == G_OPEN_PALM:
            self.glow_intensity = lerp(self.glow_intensity, 0.15, dt * 3.0)
            self.scale_multiplier = lerp(self.scale_multiplier, 1.2, dt * 3.0)
            self.spin_multiplier = lerp(self.spin_multiplier, 1.0, dt * 2.0)
            self.particle_intensity = lerp(self.particle_intensity, 0.0, dt * 3.0)

        elif g == G_PINCH:
            self.glow_intensity = lerp(self.glow_intensity, 0.7, dt * 5.0)
            self.spin_multiplier = lerp(self.spin_multiplier, 4.0, dt * 5.0)
            self.particle_intensity = lerp(self.particle_intensity, 0.3, dt * 3.0)

        elif g == G_PEACE:
            self.rainbow_boost = lerp(self.rainbow_boost, 1.0, dt * 4.0)
            self.glow_intensity = lerp(self.glow_intensity, 0.3, dt * 3.0)
            self.scale_multiplier = lerp(self.scale_multiplier, 1.3, dt * 3.0)
            self.particle_intensity = lerp(self.particle_intensity, 1.0, dt * 4.0)

        elif g == G_SPREAD:
            # Spread fingers: expand rings, boost particle count
            self.ring_count = min(self.ring_count + 1, 5)
            self.scale_multiplier = lerp(self.scale_multiplier, 1.5, dt * 3.0)
            self.particle_intensity = lerp(self.particle_intensity, 1.0, dt * 2.0)

        elif g == G_POINT:
            self.deform_amount = lerp(self.deform_amount, 0.3, dt * 4.0)

        # Velocity-based modulation
        vel_factor = clamp(hand.wrist_velocity * 2.0, 0.0, 1.0)
        self.particle_intensity = max(self.particle_intensity,
                                       vel_factor * 0.6)

    def _process_event(self, event_type, h_idx, data, gesture_result):
        """React to temporal gesture events."""
        if event_type == 'snap':
            # Cycle shader mode
            self.shader_mode = (self.shader_mode + 1) % NUM_SHADER_MODES
            self.glow_intensity = 1.0  # Flash on switch
            self.shake_intensity = 0.3

        elif event_type == 'fist_release':
            # Shockwave!
            self.state = STATE_SHOCKWAVE
            self.shockwave_t = 0.01  # Start animation
            self.energy = min(data / 3.0, 1.0)
            self.glow_intensity = 1.0
            self.shake_intensity = 0.8
            self.charge = 0.0
            # Center shockwave at hand
            if h_idx >= 0 and h_idx < len(gesture_result.hands):
                hand = gesture_result.hands[h_idx]
                w = hand.smooth_lm[0]
                self.shockwave_center = (w[0], w[1])

        elif event_type == 'circle_draw':
            # Portal spawn
            self.state = STATE_PORTAL
            self.portal_t = 0.01
            cx, cy, r = data
            self.portal_center = (cx, cy)
            self.portal_radius = r
            self.ring_count = min(self.ring_count + 2, 5)
            self.rainbow_boost = 1.0

        elif event_type == 'energy_burst':
            # Energy burst pulse
            intensity = data
            self.glow_intensity = max(self.glow_intensity, intensity)
            self.particle_intensity = 1.0
            self.shake_intensity = max(self.shake_intensity, intensity * 0.5)
            self.spin_multiplier *= 2.0

        elif event_type == 'two_hand':
            gesture, intensity = data
            if gesture == 'two_pull_apart':
                self.scale_multiplier = lerp(self.scale_multiplier,
                                              1.5 + intensity * 0.3, 0.1)
                self.deform_amount = lerp(self.deform_amount, 0.6, 0.1)
            elif gesture == 'two_compress':
                self.state = STATE_COLLAPSE
                self.collapse_t = min(self.collapse_t + 0.02, 1.0)
                self.scale_multiplier = lerp(self.scale_multiplier,
                                              0.3 - intensity * 0.1, 0.1)
                self.glow_intensity = 1.0
            elif gesture == 'two_pinch_torque':
                self.spin_multiplier = lerp(self.spin_multiplier,
                                             3.0 + intensity, 0.1)
                self.rainbow_boost = lerp(self.rainbow_boost, 0.8, 0.1)

    def _animate_states(self, dt):
        """Animate time-based state effects."""
        # Shockwave: expand and fade
        if self.shockwave_t > 0:
            self.shockwave_t = min(self.shockwave_t + dt * 1.5, 1.0)
            if self.shockwave_t >= 1.0:
                self.shockwave_t = 0.0
                if self.state == STATE_SHOCKWAVE:
                    self.state = STATE_IDLE

        # Portal: expand then fade
        if self.portal_t > 0:
            self.portal_t = min(self.portal_t + dt * 0.5, 1.0)
            if self.portal_t >= 1.0:
                self.portal_t = 0.0
                if self.state == STATE_PORTAL:
                    self.state = STATE_IDLE

        # Collapse recovery
        if self.state == STATE_COLLAPSE:
            if self.collapse_t <= 0.01:
                self.state = STATE_IDLE

        # Overload flicker
        if self.state == STATE_OVERLOAD:
            self.shake_intensity = 0.6
            self.glow_intensity = 0.8 + 0.2 * (1.0 if int(time.time() * 10) % 2 == 0 else 0.0)

    def _decay(self, dt):
        """Gradually return values to idle baseline."""
        decay_rate = dt * 2.0

        # Rainbow always decays fast (only peace sign reactivates it)
        self.rainbow_boost = lerp(self.rainbow_boost, 0.0, decay_rate * 5)

        if self.state == STATE_IDLE:
            self.energy = lerp(self.energy, 0.0, decay_rate)
            self.charge = lerp(self.charge, 0.0, decay_rate * 2)
            self.glow_intensity = lerp(self.glow_intensity, 0.05, decay_rate)
            self.shake_intensity = lerp(self.shake_intensity, 0.0, decay_rate * 3)
            self.spin_multiplier = lerp(self.spin_multiplier, 1.0, decay_rate)
            self.scale_multiplier = lerp(self.scale_multiplier, 1.0, decay_rate)
            self.deform_amount = lerp(self.deform_amount, 0.0, decay_rate)
            self.particle_intensity = lerp(self.particle_intensity, 0.0, decay_rate * 3)
            self.collapse_t = lerp(self.collapse_t, 0.0, decay_rate)
            self.ring_count = max(1, self.ring_count)

    def _clamp_all(self):
        self.energy = clamp(self.energy, 0.0, 1.0)
        self.charge = clamp(self.charge, 0.0, 1.0)
        self.glow_intensity = clamp(self.glow_intensity, 0.0, 1.0)
        self.rainbow_boost = clamp(self.rainbow_boost, 0.0, 1.0)
        self.shake_intensity = clamp(self.shake_intensity, 0.0, 1.0)
        self.spin_multiplier = clamp(self.spin_multiplier, 0.1, 10.0)
        self.scale_multiplier = clamp(self.scale_multiplier, 0.1, 3.0)
        self.deform_amount = clamp(self.deform_amount, 0.0, 1.0)
        self.particle_intensity = clamp(self.particle_intensity, 0.0, 1.0)
        self.ring_count = int(clamp(self.ring_count, 1, 5))
