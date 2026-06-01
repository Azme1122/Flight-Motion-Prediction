"""Load raw 3D trajectory data from Excel or CSV files.

This module extracts only the x, y, z columns and returns a clean NumPy array
with shape (N, 3), where each row is one trajectory point:

    [x_k, y_k, z_k]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_trajectory(
    file_path: str | Path,
    *,
    sheet_name: str | int = 0,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    drop_nan: bool = True,
) -> np.ndarray:
    """Load a clean 3D trajectory array from an Excel or CSV file.

    Parameters
    ----------
    file_path:
        Path to the input `.xlsx`, `.xls`, or `.csv` file.
    sheet_name:
        Sheet name or sheet index for Excel files. Ignored for CSV files.
    x_col, y_col, z_col:
        Column names for the world-coordinate x, y, and z values. Matching is
        case-insensitive, so `x_col="x"` will also match a column named `X`.
    drop_nan:
        If True, rows with missing/non-numeric x, y, or z values are removed.
        If False, a helpful ValueError is raised instead.

    Returns
    -------
    np.ndarray
        Clean trajectory array of shape (N, 3).
    """

    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        data = pd.read_excel(path, sheet_name=sheet_name)
    elif suffix == ".csv":
        data = pd.read_csv(path)
    else:
        raise ValueError("Input file must be an Excel file (.xlsx/.xls) or CSV file.")

    selected_columns = _match_xyz_columns(data, x_col=x_col, y_col=y_col, z_col=z_col)
    trajectory_df = data.loc[:, selected_columns].copy()
    trajectory_df.columns = ["x", "y", "z"]

    # Convert values to numeric. Non-numeric entries become NaN and are handled
    # together with missing values below.
    for column in ["x", "y", "z"]:
        trajectory_df[column] = pd.to_numeric(trajectory_df[column], errors="coerce")

    nan_mask = trajectory_df.isna().any(axis=1)
    nan_count = int(nan_mask.sum())
    if nan_count > 0:
        message = f"Found {nan_count} row(s) with NaN or non-numeric x/y/z values."
        if drop_nan:
            print(f"{message} Removing them before normalization.")
            trajectory_df = trajectory_df.loc[~nan_mask]
        else:
            raise ValueError(message)

    if trajectory_df.empty:
        raise ValueError("Trajectory is empty after loading and cleaning.")

    trajectory = trajectory_df.to_numpy(dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError(f"Expected trajectory shape (N, 3), got {trajectory.shape}.")

    return trajectory


def _match_xyz_columns(
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    z_col: str,
) -> list[object]:
    """Find x, y, z columns using case-insensitive matching."""

    available = {str(column).strip().lower(): column for column in data.columns}
    requested = [x_col, y_col, z_col]
    matched_columns = []

    for column_name in requested:
        key = column_name.strip().lower()
        if key not in available:
            raise ValueError(
                f"Required column '{column_name}' was not found. "
                f"Available columns: {list(data.columns)}"
            )
        matched_columns.append(available[key])

    return matched_columns
