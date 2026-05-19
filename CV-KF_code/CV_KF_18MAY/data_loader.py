"""Data loading and measurement-noise helpers."""

import numpy as np
import pandas as pd


def load_one_trajectory_from_excel(file_path, sheet_name=0, max_points=100):
    """Load one 3D trajectory from an Excel sheet with X, Y, Z columns."""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    gt_positions = df[["X", "Y", "Z"]].to_numpy(dtype=float)

    if max_points is not None:
        gt_positions = gt_positions[:max_points]

    print("Trajectory shape:", gt_positions.shape)
    return gt_positions


def add_measurement_noise(gt_positions, noise_std=0.5, rng=None):
    """Add repeatable Gaussian measurement noise to ground-truth positions."""
    if rng is None:
        rng = np.random.default_rng(42)

    noise = rng.normal(0.0, noise_std, size=gt_positions.shape)
    return gt_positions + noise
