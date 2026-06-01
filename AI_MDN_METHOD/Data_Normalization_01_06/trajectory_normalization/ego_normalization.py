"""Point-wise ego-coordinate normalization for 3D trajectories."""

from __future__ import annotations

import numpy as np


def normalize_to_ego_coordinates(
    trajectory: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Convert a world-coordinate trajectory into ego-coordinate motion features.

    For each valid point p_k, the previous horizontal movement direction
    p_k - p_{k-1} defines the local ego frame in the x-y plane.

    Convention
    ----------
    - Y_ego is the current movement/forward direction.
    - X_ego is the perpendicular/sideways direction.

    If e_y(k) = [a, b], then e_x(k) = [-b, a].

    The next displacement p_{k+1} - p_k is projected into this local frame:

    - x_ego: sideways horizontal displacement
    - y_ego: forward horizontal displacement
    - delta_z: vertical displacement z_{k+1} - z_k

    The output has shape (N - 2, 3) when all middle points are valid.

    Zero-motion strategy
    --------------------
    If the previous horizontal movement length is smaller than `epsilon`, the
    direction is too small to normalize safely. This implementation reuses the
    last valid forward direction. If no valid direction exists yet, that point
    is skipped. This keeps as much data as possible while avoiding division by
    zero.

    Parameters
    ----------
    trajectory:
        Input NumPy array of shape (N, 3), with columns [x, y, z].
    epsilon:
        Small positive threshold for detecting near-zero horizontal movement.

    Returns
    -------
    np.ndarray
        Ego-coordinate motion features with columns [x_ego, y_ego, delta_z].
    """

    points = np.asarray(trajectory, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected trajectory shape (N, 3), got {points.shape}.")
    if len(points) < 3:
        raise ValueError("At least 3 points are required for ego normalization.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    ego_rows: list[list[float]] = []
    last_valid_forward: np.ndarray | None = None

    for k in range(1, len(points) - 1):
        previous_xy = points[k - 1, :2]
        current_xy = points[k, :2]
        next_xy = points[k + 1, :2]

        d_xy = current_xy - previous_xy
        d_norm = float(np.linalg.norm(d_xy))

        if d_norm > epsilon:
            e_y = d_xy / d_norm
            last_valid_forward = e_y
        elif last_valid_forward is not None:
            e_y = last_valid_forward
        else:
            continue

        a, b = e_y
        e_x = np.array([-b, a], dtype=float)

        delta_xy = next_xy - current_xy
        x_ego = float(np.dot(delta_xy, e_x))
        y_ego = float(np.dot(delta_xy, e_y))
        delta_z = float(points[k + 1, 2] - points[k, 2])

        ego_rows.append([x_ego, y_ego, delta_z])

    return np.asarray(ego_rows, dtype=float).reshape(-1, 3)
