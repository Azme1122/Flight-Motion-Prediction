"""Load raw 3D trajectory data from Excel, CSV, or TXT files.

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
    txt_xyz_columns: tuple[int, int, int] | None = None,
    drop_nan: bool = True,
) -> np.ndarray:
    """Load a clean 3D trajectory array from an Excel, CSV, or TXT file.

    Parameters
    ----------
    file_path:
        Path to the input `.xlsx`, `.xls`, or `.csv` file.
    sheet_name:
        Sheet name or sheet index for Excel files. Ignored for CSV files.
    x_col, y_col, z_col:
        Column names for the world-coordinate x, y, and z values. Matching is
        case-insensitive, so `x_col="x"` will also match a column named `X`.
        Used only for Excel/CSV files with named columns.
    txt_xyz_columns:
        Zero-based column indices for TXT files. If None, the loader uses this
        automatic rule: 3-column TXT means `x y z`; 4-or-more-column TXT means
        the last three columns are `x y z`, so an index/time column is ignored.
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
        trajectory_df = _extract_named_xyz(data, x_col=x_col, y_col=y_col, z_col=z_col)
    elif suffix == ".csv":
        data = pd.read_csv(path)
        trajectory_df = _extract_named_xyz(data, x_col=x_col, y_col=y_col, z_col=z_col)
    elif suffix in {".txt", ".dat", ".tsv"}:
        data = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
        trajectory_df = _extract_txt_xyz(data, txt_xyz_columns=txt_xyz_columns)
    else:
        raise ValueError(
            "Input file must be Excel (.xlsx/.xls), CSV (.csv), or TXT (.txt/.dat/.tsv)."
        )

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


def _extract_named_xyz(
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    z_col: str,
) -> pd.DataFrame:
    """Extract named x, y, z columns from Excel/CSV data."""

    selected_columns = _match_xyz_columns(data, x_col=x_col, y_col=y_col, z_col=z_col)
    trajectory_df = data.loc[:, selected_columns].copy()
    trajectory_df.columns = ["x", "y", "z"]
    return trajectory_df


def _extract_txt_xyz(
    data: pd.DataFrame,
    *,
    txt_xyz_columns: tuple[int, int, int] | None,
) -> pd.DataFrame:
    """Extract x, y, z columns from whitespace-separated TXT data."""

    if data.empty:
        raise ValueError("TXT file contains no numeric trajectory rows.")

    column_count = data.shape[1]
    if txt_xyz_columns is None:
        if column_count == 3:
            txt_xyz_columns = (0, 1, 2)
        elif column_count >= 4:
            txt_xyz_columns = (column_count - 3, column_count - 2, column_count - 1)
        else:
            raise ValueError(
                f"TXT file must contain at least 3 numeric columns, got {column_count}."
            )

    if len(txt_xyz_columns) != 3:
        raise ValueError("txt_xyz_columns must contain exactly 3 column indices.")
    if any(column < 0 or column >= column_count for column in txt_xyz_columns):
        raise ValueError(
            f"txt_xyz_columns={txt_xyz_columns} is invalid for {column_count} columns."
        )

    trajectory_df = data.iloc[:, list(txt_xyz_columns)].copy()
    trajectory_df.columns = ["x", "y", "z"]
    return trajectory_df


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
