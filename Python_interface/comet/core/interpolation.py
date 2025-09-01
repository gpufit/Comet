import numpy as np
from scipy.interpolate import CubicSpline


def interpolate_drift(center_frames, drift_est, frame_range, method='cubic'):
    """
    Interpolates drift estimates to all frames using specified method.
    Parameters:
    - center_frames: np.ndarray of shape (M,), frames corresponding to drift estimates.
    - drift_est: np.ndarray of shape (M, 3), drift estimates at center frames.
    - frame_range: array-like, frames to interpolate drift estimates to.
    - method: str, interpolation method ('cubic' or 'catmull-rom').
    Returns:
    - drift_interp: np.ndarray of shape (len(frame_range), 3), interpolated drift estimates.
    """
    if method == 'cubic':
        return _interpolate_cubic(center_frames, drift_est, frame_range)
    elif method == 'catmull-rom':
        return _interpolate_catmull_rom(center_frames, drift_est, frame_range)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def _interpolate_cubic(center_frames, drift_est, frame_range):
    drift_x = CubicSpline(center_frames, drift_est[:, 0])(frame_range)
    drift_y = CubicSpline(center_frames, drift_est[:, 1])(frame_range)
    drift_z = CubicSpline(center_frames, drift_est[:, 2])(frame_range)
    return np.vstack([drift_x, drift_y, drift_z]).T


def _interpolate_catmull_rom(center_frames, drift_est, frame_range):
    def catmull_rom_1d(x, y, x_interp):
        result = np.zeros_like(x_interp)
        for i in range(1, len(x) - 2):
            x0, x1, x2, x3 = x[i-1], x[i], x[i+1], x[i+2]
            y0, y1, y2, y3 = y[i-1], y[i], y[i+1], y[i+2]

            mask = (x_interp >= x1) & (x_interp <= x2)
            t = (x_interp[mask] - x1) / (x2 - x1)

            result[mask] = (
                0.5 * (
                    (2 * y1) +
                    (-y0 + y2) * t +
                    (2*y0 - 5*y1 + 4*y2 - y3) * t**2 +
                    (-y0 + 3*y1 - 3*y2 + y3) * t**3
                )
            )
        return result

    x_interp = np.asarray(frame_range)
    x = np.asarray(center_frames)

    if len(x) < 4:
        raise ValueError("Catmull-Rom interpolation requires at least 4 points.")

    drift_x = catmull_rom_1d(x, drift_est[:, 0], x_interp)
    drift_y = catmull_rom_1d(x, drift_est[:, 1], x_interp)
    drift_z = catmull_rom_1d(x, drift_est[:, 2], x_interp)

    return np.vstack([drift_x, drift_y, drift_z]).T
