"""Constant-velocity Kalman filter over box state, batched over tracks.

State ``s = (x, y, a, h, vx, vy, va, vh)``: the box as ``(centre x, centre y, w / h, h)`` plus the
velocity of each of those four numbers, in units per frame. A measurement is just the first four,
``z = (x, y, a, h)``. Means are ``(N, 8)`` and covariances ``(N, 8, 8)``, float64.

Noise is proportional to the box height ``h``, so a large (near) box is allowed to move more pixels
per frame than a small (far) one. The weights follow SORT / DeepSORT / ByteTrack.
"""

import numpy as np

from palletvision_track.boxes import Array

# Noise below this height is computed as if the box were this tall: a predicted height can drift
# to zero or below for a track that has been lost for a long time.
_MIN_H = 1e-6


class KalmanFilter:
    """Predict where boxes go next frame and fold in the box a detector then reports.

    ``w_p`` and ``w_v`` set how much position and velocity are trusted relative to box height.
    Smaller is more confident. The aspect-ratio noise is absolute, since ``a`` has no height scale.
    """

    def __init__(self, w_p: float = 1 / 20, w_v: float = 1 / 160, dt: float = 1.0) -> None:
        self.w_p = w_p
        self.w_v = w_v
        # F: next state = F @ state. Identity, plus each position gains dt * its velocity.
        self.F: Array = np.eye(8)
        self.F[np.arange(4), np.arange(4) + 4] = dt
        # H picks the four measured numbers out of the eight-number state.
        self.H: Array = np.eye(4, 8)

        # Standard deviation of each state entry = absolute + scale * h.
        # Process noise Q: how far the true motion can stray from constant velocity in one frame.
        self._q_abs = np.array([0, 0, 1e-2, 0, 0, 0, 1e-5, 0])
        self._q_scale = np.array([w_p, w_p, 0, w_p, w_v, w_v, 0, w_v])
        # Initial uncertainty of a new track: position is fairly known, velocity is not known at all
        self._p0_abs = np.array([0, 0, 1e-2, 0, 0, 0, 1e-5, 0])
        self._p0_scale = np.array([2 * w_p, 2 * w_p, 0, 2 * w_p, 10 * w_v, 10 * w_v, 0, 10 * w_v])
        # Measurement noise R: how far the detector's box can be from the truth.
        self._r_abs = np.array([0, 0, 1e-1, 0])
        self._r_scale = np.array([w_p, w_p, 0, w_p])

    def initiate(self, measurement: np.ndarray) -> tuple[Array, Array]:
        """New track from a box ``(4,)`` or ``(N, 4)`` in ``(x, y, a, h)``: ``(mean, covariance)``.

        Position is the measurement and velocity starts at zero.
        """
        z = np.asarray(measurement, dtype=np.float64).reshape(-1, 4)
        mean = np.concatenate([z, np.zeros_like(z)], axis=1)
        std = self._p0_abs + self._p0_scale * _height(z[:, 3])[:, None]
        return mean, _diag(std**2)

    def predict(self, mean: Array, covariance: Array) -> tuple[Array, Array]:
        """Move every track one frame forward: ``s' = F s`` and ``P' = F P F^T + Q``."""
        std = self._q_abs + self._q_scale * _height(mean[:, 3])[:, None]  # from the current h
        new_mean = mean @ self.F.T
        new_cov = self.F @ covariance @ self.F.T + _diag(std**2)
        return new_mean, new_cov

    def project(self, mean: Array, covariance: Array) -> tuple[Array, Array]:
        """The 8-D state seen as the 4-D box a detector would report, with measurement noise added.

        Returns the expected measurement ``H s`` and its covariance ``S = H P H^T + R``.
        """
        std = self._r_abs + self._r_scale * _height(mean[:, 3])[:, None]
        proj_mean = mean @ self.H.T
        proj_cov = self.H @ covariance @ self.H.T + _diag(std**2)
        return proj_mean, proj_cov

    def update(
        self, mean: Array, covariance: Array, measurement: np.ndarray
    ) -> tuple[Array, Array]:
        """Correct each track with the box matched to it, ``(N, 4)`` in ``(x, y, a, h)``.

        ``K = P H^T S^-1``, ``s = s' + K (z - H s')``, ``P = P' - K S K^T``.
        """
        z = np.asarray(measurement, dtype=np.float64).reshape(-1, 4)
        proj_mean, s = self.project(mean, covariance)
        ph_t = covariance @ self.H.T  # (N, 8, 4)
        # K = ph_t @ S^-1, written as a solve so no inverse is formed. S is symmetric, so
        # S K^T = (P H^T)^T gives K^T directly.
        kalman_t = np.linalg.solve(s, ph_t.transpose(0, 2, 1))  # (N, 4, 8)
        new_mean = mean + np.einsum("ni,nij->nj", z - proj_mean, kalman_t)
        new_cov = covariance - kalman_t.transpose(0, 2, 1) @ s @ kalman_t
        return new_mean, (new_cov + new_cov.transpose(0, 2, 1)) / 2  # drop rounding asymmetry


def _height(h: Array) -> Array:
    return np.maximum(h, _MIN_H)


def _diag(values: Array) -> Array:
    """``(N, k)`` -> ``(N, k, k)`` with each row on a diagonal."""
    out = np.zeros((*values.shape, values.shape[-1]))
    idx = np.arange(values.shape[-1])
    out[..., idx, idx] = values
    return out
