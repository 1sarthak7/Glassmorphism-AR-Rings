"""
Gesture Engine
==============
Static + temporal gesture recognition, velocity tracking, two-hand
interaction detection, Kalman smoothing, confidence filtering.
"""

import math
import time
import numpy as np
from collections import deque
from math_utils import KalmanFilter2D

# ============================================================================
#  LANDMARK INDICES
# ============================================================================

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
DIPS = [THUMB_IP, INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP]
MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

# ============================================================================
#  GESTURE TYPES
# ============================================================================

# Static gestures
G_NONE        = 'none'
G_OPEN_PALM   = 'open_palm'
G_FIST        = 'fist'
G_PINCH       = 'pinch'
G_PEACE       = 'peace'
G_POINT       = 'point'
G_SPREAD      = 'spread'

# Temporal / composite gestures
G_SNAP          = 'snap'
G_CIRCLE_DRAW   = 'circle_draw'
G_FIST_RELEASE  = 'fist_release'        # Charge → shockwave
G_ENERGY_BURST  = 'energy_burst'         # Rapid acceleration

# Two-hand gestures
G_TWO_PULL_APART   = 'two_pull_apart'
G_TWO_COMPRESS     = 'two_compress'
G_TWO_PINCH_TORQUE = 'two_pinch_torque'


# ============================================================================
#  HAND DATA (per-hand state)
# ============================================================================

class HandData:
    """All tracked state for a single hand."""

    def __init__(self):
        # Kalman-smoothed landmark positions
        self.filters = [KalmanFilter2D(process_noise=5e-4, measurement_noise=5e-2)
                        for _ in range(21)]
        self.landmarks = None        # Raw landmarks from MediaPipe
        self.smooth_lm = None        # Smoothed (x, y) for each landmark
        self.hand_size = 0.0         # Wrist-to-middle-MCP distance

        # Gesture state
        self.static_gesture = G_NONE
        self.gesture_confidence = 0.0
        self.finger_open = [False] * 5

        # Velocity tracking
        self.wrist_velocity = 0.0    # pixels/frame
        self.acceleration = 0.0
        self._prev_wrist = None
        self._prev_velocity = 0.0

        # Temporal gesture state
        self.fist_hold_start = None
        self.index_trail = deque(maxlen=60)  # 2 seconds of index tip positions
        self._snap_state = 0         # 0=idle, 1=closed, 2=opened
        self._snap_time = 0.0

    def update_landmarks(self, raw_landmarks):
        """Apply Kalman smoothing to all 21 landmarks."""
        self.landmarks = raw_landmarks
        smoothed = []
        for i, lm in enumerate(raw_landmarks):
            sx, sy = self.filters[i].update(lm.x, lm.y)
            smoothed.append((sx, sy))
        self.smooth_lm = smoothed

        # Hand size
        wx, wy = smoothed[WRIST]
        mx, my = smoothed[MIDDLE_MCP]
        self.hand_size = math.sqrt((mx - wx)**2 + (my - wy)**2)

        # Velocity & acceleration
        wx_cur, wy_cur = smoothed[WRIST]
        if self._prev_wrist is not None:
            dx = wx_cur - self._prev_wrist[0]
            dy = wy_cur - self._prev_wrist[1]
            self.wrist_velocity = math.sqrt(dx*dx + dy*dy) * 30.0  # Normalize to /sec
            self.acceleration = abs(self.wrist_velocity - self._prev_velocity) * 30.0
        self._prev_wrist = (wx_cur, wy_cur)
        self._prev_velocity = self.wrist_velocity

        # Track index tip for circle detection
        self.index_trail.append(smoothed[INDEX_TIP])


# ============================================================================
#  STATIC GESTURE DETECTION
# ============================================================================

def _dist(p1, p2):
    """Euclidean distance between two (x,y) tuples."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def _finger_is_open(lm, finger_idx):
    """Check if finger is extended (tip farther from wrist than pip)."""
    tip = lm[TIPS[finger_idx]]
    pip = lm[PIPS[finger_idx]]
    wrist = lm[WRIST]
    return _dist(tip, wrist) > _dist(pip, wrist)


def detect_static_gesture(hand):
    """
    Detect static hand gesture from smoothed landmarks.
    Returns (gesture_type, confidence).
    """
    lm = hand.smooth_lm
    if lm is None:
        return G_NONE, 0.0

    hand.finger_open = [_finger_is_open(lm, i) for i in range(5)]
    n_open = sum(hand.finger_open)

    # Pinch: thumb tip very close to index tip
    pinch_dist = _dist(lm[THUMB_TIP], lm[INDEX_TIP])
    if hand.hand_size > 0 and pinch_dist < hand.hand_size * 0.25:
        confidence = 1.0 - (pinch_dist / (hand.hand_size * 0.25))
        return G_PINCH, min(confidence, 1.0)

    # Peace: index + middle extended, others closed
    if (hand.finger_open[1] and hand.finger_open[2] and
            not hand.finger_open[3] and not hand.finger_open[4]):
        return G_PEACE, 0.9

    # Point: only index extended
    if (hand.finger_open[1] and not hand.finger_open[2] and
            not hand.finger_open[3] and not hand.finger_open[4]):
        return G_POINT, 0.85

    # Open palm
    if n_open >= 4:
        return G_OPEN_PALM, 0.8 + 0.05 * n_open

    # Spread: all fingers open + wide apart
    if n_open == 5:
        # Check spread by measuring distances between fingertips
        spread_dist = _dist(lm[INDEX_TIP], lm[PINKY_TIP])
        if hand.hand_size > 0 and spread_dist > hand.hand_size * 0.8:
            return G_SPREAD, 0.85

    # Fist
    if n_open <= 1:
        return G_FIST, 0.85

    return G_NONE, 0.3


# ============================================================================
#  TEMPORAL GESTURE DETECTION
# ============================================================================

def detect_snap(hand, now):
    """
    Detect snap gesture: rapid thumb-middle distance change.
    State machine: idle → closed (thumb+middle touch) → opened (rapid separate).
    """
    lm = hand.smooth_lm
    if lm is None:
        return None

    thumb_mid_dist = _dist(lm[THUMB_TIP], lm[MIDDLE_TIP])
    threshold_close = hand.hand_size * 0.15
    threshold_open = hand.hand_size * 0.4

    if hand._snap_state == 0:
        # Idle → detect close
        if thumb_mid_dist < threshold_close:
            hand._snap_state = 1
            hand._snap_time = now
    elif hand._snap_state == 1:
        # Closed → detect open within 200ms
        if thumb_mid_dist > threshold_open:
            elapsed = now - hand._snap_time
            hand._snap_state = 0
            if elapsed < 0.2:
                return G_SNAP
        elif now - hand._snap_time > 0.3:
            hand._snap_state = 0  # Timeout
    return None


def detect_fist_release(hand, now):
    """
    Detect fist hold (> 0.5s) followed by rapid open → shockwave.
    Returns (gesture, hold_duration) or None.
    """
    gesture = hand.static_gesture
    if gesture == G_FIST:
        if hand.fist_hold_start is None:
            hand.fist_hold_start = now
    elif hand.fist_hold_start is not None:
        hold_dur = now - hand.fist_hold_start
        hand.fist_hold_start = None
        if hold_dur > 0.5 and gesture == G_OPEN_PALM:
            return G_FIST_RELEASE, min(hold_dur, 3.0)
    return None


def detect_circle_draw(hand):
    """
    Detect if index fingertip has traced a circle in recent frames.
    Uses least-squares circle fit on the trail buffer.
    Returns (gesture, center_x, center_y, radius) or None.
    """
    trail = hand.index_trail
    if len(trail) < 30:
        return None

    # Use last 45 frames (~1.5s)
    pts = list(trail)[-45:]
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])

    # Least-squares circle fit
    A = np.column_stack([xs, ys, np.ones(len(xs))])
    b = xs**2 + ys**2
    try:
        result = np.linalg.lstsq(A, b, rcond=None)
        c = result[0]
    except np.linalg.LinAlgError:
        return None

    cx, cy = c[0] / 2, c[1] / 2
    radius = math.sqrt(c[2] + cx**2 + cy**2)

    # Check fit quality: all points should be roughly on the circle
    dists = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    residual = np.std(dists - radius)

    # Check coverage: points should wrap around (not just an arc)
    angles = np.arctan2(ys - cy, xs - cx)
    angle_range = np.ptp(angles)

    if residual < radius * 0.25 and angle_range > 4.5 and radius > 0.03:
        hand.index_trail.clear()
        return G_CIRCLE_DRAW, cx, cy, radius

    return None


def detect_energy_burst(hand):
    """Detect rapid hand acceleration → energy burst."""
    if hand.acceleration > 15.0:  # High acceleration threshold
        return G_ENERGY_BURST, min(hand.acceleration / 30.0, 1.0)
    return None


# ============================================================================
#  TWO-HAND INTERACTION
# ============================================================================

class TwoHandDetector:
    """Detects interactions between two hands."""

    def __init__(self):
        self._prev_wrist_dist = None
        self._prev_angle = None

    def detect(self, hand_left, hand_right, dt):
        """
        Returns list of two-hand gesture events.
        """
        events = []
        lm_l = hand_left.smooth_lm
        lm_r = hand_right.smooth_lm
        if lm_l is None or lm_r is None:
            self._prev_wrist_dist = None
            return events

        # Wrist distance
        wrist_dist = _dist(lm_l[WRIST], lm_r[WRIST])

        if self._prev_wrist_dist is not None and dt > 0:
            dist_vel = (wrist_dist - self._prev_wrist_dist) / dt

            # Pull apart: wrist distance increasing fast
            if dist_vel > 0.5:
                events.append((G_TWO_PULL_APART, min(dist_vel, 3.0)))

            # Compress: wrist distance decreasing fast
            if dist_vel < -0.5:
                events.append((G_TWO_COMPRESS, min(abs(dist_vel), 3.0)))

        self._prev_wrist_dist = wrist_dist

        # Pinch torque: both hands pinching + differential rotation
        if (hand_left.static_gesture == G_PINCH and
                hand_right.static_gesture == G_PINCH):
            # Compute angle of each pinch line
            angle_l = math.atan2(
                lm_l[INDEX_TIP][1] - lm_l[THUMB_TIP][1],
                lm_l[INDEX_TIP][0] - lm_l[THUMB_TIP][0]
            )
            angle_r = math.atan2(
                lm_r[INDEX_TIP][1] - lm_r[THUMB_TIP][1],
                lm_r[INDEX_TIP][0] - lm_r[THUMB_TIP][0]
            )
            total_angle = angle_l - angle_r
            if self._prev_angle is not None:
                angular_vel = abs(total_angle - self._prev_angle) / max(dt, 0.001)
                if angular_vel > 0.5:
                    events.append((G_TWO_PINCH_TORQUE, min(angular_vel, 5.0)))
            self._prev_angle = total_angle
        else:
            self._prev_angle = None

        return events


# ============================================================================
#  GESTURE ENGINE (main interface)
# ============================================================================

class GestureEngine:
    """
    Main gesture processing engine.
    Call process() each frame with hand landmarks from MediaPipe.
    Returns GestureResult with all detected gestures.
    """

    def __init__(self, max_hands=2):
        self.max_hands = max_hands
        self.hands = [HandData() for _ in range(max_hands)]
        self.two_hand = TwoHandDetector()
        self.confidence_threshold = 0.65
        self._last_time = time.time()

    def process(self, hand_landmarks_list):
        """
        Process hand landmarks for a single frame.

        Args:
            hand_landmarks_list: list of hand landmark lists from MediaPipe

        Returns:
            GestureResult with per-hand data and events
        """
        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        result = GestureResult()
        active_count = min(len(hand_landmarks_list), self.max_hands)

        for h_idx in range(active_count):
            hand = self.hands[h_idx]
            hand.update_landmarks(hand_landmarks_list[h_idx])

            # Static gesture
            gesture, confidence = detect_static_gesture(hand)
            if confidence >= self.confidence_threshold:
                hand.static_gesture = gesture
                hand.gesture_confidence = confidence
            else:
                hand.static_gesture = G_NONE
                hand.gesture_confidence = confidence

            # Temporal gestures
            snap = detect_snap(hand, now)
            if snap:
                result.events.append(('snap', h_idx, None))

            fist_rel = detect_fist_release(hand, now)
            if fist_rel:
                result.events.append(('fist_release', h_idx, fist_rel[1]))

            circle = detect_circle_draw(hand)
            if circle:
                result.events.append(('circle_draw', h_idx,
                                       (circle[1], circle[2], circle[3])))

            burst = detect_energy_burst(hand)
            if burst:
                result.events.append(('energy_burst', h_idx, burst[1]))

            result.hands.append(hand)

        # Two-hand interactions
        if active_count >= 2:
            two_events = self.two_hand.detect(
                self.hands[0], self.hands[1], dt
            )
            for evt in two_events:
                result.events.append(('two_hand', -1, evt))

        result.dt = dt
        result.n_hands = active_count
        return result


class GestureResult:
    """Result of gesture processing for a single frame."""
    def __init__(self):
        self.hands = []          # List of HandData
        self.events = []         # List of (event_type, hand_idx, data)
        self.dt = 0.0
        self.n_hands = 0
