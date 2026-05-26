"""Data loading helpers."""

import pandas as pd


def load_one_trajectory_from_excel(file_path, sheet_name=0, max_points=100):
    """Load one 3D trajectory from an Excel sheet with X, Y, Z columns."""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    positions = df[["X", "Y", "Z"]].to_numpy(dtype=float)

    if max_points is not None:
        positions = positions[:max_points]

    print("Trajectory shape:", positions.shape)
    return positions
