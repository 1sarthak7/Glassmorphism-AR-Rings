"""
Math Utilities
==============
Foundation module: Kalman filter, matrix math, noise functions, interpolation.
"""

import math
import numpy as np

# ============================================================================
#  KALMAN FILTER (1D, per-axis smoothing)
# ============================================================================

class KalmanFilter1D:
    """
    Simple 1D Kalman filter for landmark smoothing.
    State: [position, velocity]
    
    Provides much better tracking than EMA — predicts position drift and
    corrects with measurement, reducing jitter while maintaining responsiveness.
    """
    def __init__(self, process_noise=1e-3, measurement_noise=1e-1):
        self.x = np.array([0.0, 0.0])  # [position, velocity]
        self.P = np.eye(2) * 1.0       # Covariance
        self.Q = np.eye(2) * process_noise      # Process noise
        self.R = np.array([[measurement_noise]])  # Measurement noise
        self.H = np.array([[1.0, 0.0]])           # Observation matrix
        self.initialized = False

    def predict(self, dt=1.0/30.0):
        F = np.array([[1.0, dt],
                       [0.0, 1.0]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        if not self.initialized:
            self.x[0] = measurement
            self.initialized = True
            return measurement

        self.predict()
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[0]

    @property
    def velocity(self):
        return self.x[1] if self.initialized else 0.0

    def reset(self):
        self.x = np.array([0.0, 0.0])
        self.P = np.eye(2) * 1.0
        self.initialized = False


class KalmanFilter2D:
    """Convenience wrapper for smoothing 2D (x, y) landmarks."""
    def __init__(self, process_noise=1e-3, measurement_noise=1e-1):
        self.kx = KalmanFilter1D(process_noise, measurement_noise)
        self.ky = KalmanFilter1D(process_noise, measurement_noise)

    def update(self, x, y):
        return self.kx.update(x), self.ky.update(y)

    @property
    def velocity(self):
        return math.sqrt(self.kx.velocity**2 + self.ky.velocity**2)

    def reset(self):
        self.kx.reset()
        self.ky.reset()


# ============================================================================
#  MATRIX UTILITIES
# ============================================================================

def perspective_matrix(fov_deg, aspect, near, far):
    """Perspective projection matrix (OpenGL convention, row-major)."""
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at_matrix(eye, center, up):
    """View (look-at) matrix."""
    e, c, u = [np.array(v, dtype=np.float32) for v in (eye, center, up)]
    f = c - e; f /= np.linalg.norm(f)
    s = np.cross(f, u); s /= np.linalg.norm(s)
    u2 = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s;  m[0, 3] = -np.dot(s, e)
    m[1, :3] = u2; m[1, 3] = -np.dot(u2, e)
    m[2, :3] = -f; m[2, 3] = np.dot(f, e)
    return m


def translation_matrix(tx, ty, tz):
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx; m[1, 3] = ty; m[2, 3] = tz
    return m


def scale_matrix(sx, sy, sz):
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = sx; m[1, 1] = sy; m[2, 2] = sz
    return m


def rotation_matrix_x(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[1, 1] = c; m[1, 2] = -s; m[2, 1] = s; m[2, 2] = c
    return m


def rotation_matrix_y(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = c; m[0, 2] = s; m[2, 0] = -s; m[2, 2] = c
    return m


def rotation_matrix_z(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = c; m[0, 1] = -s; m[1, 0] = s; m[1, 1] = c
    return m


# ============================================================================
#  NOISE FUNCTIONS (CPU-side for deformation)
# ============================================================================

def _hash(n):
    """Simple hash for noise generation."""
    return np.mod(np.sin(n) * 43758.5453123, 1.0)


def value_noise_2d(x, y):
    """2D value noise, returns float in [-1, 1]."""
    ix, iy = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - ix, y - iy
    # Smoothstep
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    n00 = _hash(ix + iy * 57.0)
    n10 = _hash(ix + 1.0 + iy * 57.0)
    n01 = _hash(ix + (iy + 1.0) * 57.0)
    n11 = _hash(ix + 1.0 + (iy + 1.0) * 57.0)
    nx0 = n00 * (1.0 - fx) + n10 * fx
    nx1 = n01 * (1.0 - fx) + n11 * fx
    return (nx0 * (1.0 - fy) + nx1 * fy) * 2.0 - 1.0


def fbm_noise(x, y, t, octaves=4, lacunarity=2.0, gain=0.5):
    """
    Fractional Brownian Motion noise.
    Sums multiple octaves of value noise for organic patterns.
    t is used as a Z-axis offset for animation.
    """
    val = 0.0
    amp = 1.0
    freq = 1.0
    for _ in range(octaves):
        val += amp * value_noise_2d(x * freq + t, y * freq + t * 0.7)
        freq *= lacunarity
        amp *= gain
    return val


def fbm_noise_vectorized(xs, ys, t, octaves=3, lacunarity=2.0, gain=0.5):
    """
    Vectorized FBM for batch vertex deformation.
    xs, ys: numpy arrays of coordinates.
    Returns numpy array of noise values.
    """
    val = np.zeros_like(xs, dtype=np.float64)
    amp = 1.0
    freq = 1.0
    for _ in range(octaves):
        n = np.sin((xs * freq + t) * 12.9898 + (ys * freq + t * 0.7) * 78.233)
        n = np.mod(n * 43758.5453123, 1.0) * 2.0 - 1.0
        val += amp * n
        freq *= lacunarity
        amp *= gain
    return val.astype(np.float32)


# ============================================================================
#  TORUS GEOMETRY GENERATOR
# ============================================================================

def generate_torus(R=0.5, r=0.15, n_major=48, n_minor=24):
    """
    Parametric torus. Returns (vertices, normals, indices) as numpy arrays.
    Vertices and normals are (N, 3) float32, indices are (M,) uint32.
    """
    verts, norms = [], []
    for i in range(n_major):
        u = 2.0 * math.pi * i / n_major
        cu, su = math.cos(u), math.sin(u)
        for j in range(n_minor):
            v = 2.0 * math.pi * j / n_minor
            cv, sv = math.cos(v), math.sin(v)
            verts.append([(R + r * cv) * cu, (R + r * cv) * su, r * sv])
            norms.append([cv * cu, cv * su, sv])

    idxs = []
    for i in range(n_major):
        ni = (i + 1) % n_major
        for j in range(n_minor):
            nj = (j + 1) % n_minor
            a, b = i * n_minor + j, ni * n_minor + j
            c, d = ni * n_minor + nj, i * n_minor + nj
            idxs.extend([a, b, c, a, c, d])

    return (np.array(verts, dtype=np.float32),
            np.array(norms, dtype=np.float32),
            np.array(idxs, dtype=np.uint32))


# ============================================================================
#  INTERPOLATION UTILITIES
# ============================================================================

def lerp(a, b, t):
    """Linear interpolation."""
    return a + (b - a) * t


def smoothstep(edge0, edge1, x):
    """Smooth Hermite interpolation."""
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def ease_out_cubic(t):
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_quad(t):
    return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2
