"""Run full-trajectory ego-coordinate normalization for one dataset.

This script intentionally does not create 8-observation / 12-prediction machine
learning windows. It only saves the full normalized trajectory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_trajectory
from ego_normalization import normalize_to_ego_coordinates
from plot_ego import plot_ego_xy_over_index


DEFAULT_INPUT_FILE = (
    "dataset04.txt"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Normalize raw 3D trajectory data into ego-coordinate features."
    )
    parser.add_argument(
        "--input-file",
        default=DEFAULT_INPUT_FILE,
        help="Path to the input Excel, CSV, or TXT trajectory file.",
    )
    parser.add_argument(
        "--sheet-name",
        default="flight_1",
        help="Excel sheet name or index. Ignored for CSV files.",
    )
    parser.add_argument("--x-col", default="x", help="Name of the x column.")
    parser.add_argument("--y-col", default="y", help="Name of the y column.")
    parser.add_argument("--z-col", default="z", help="Name of the z column.")
    parser.add_argument(
        "--txt-xyz-columns",
        default=None,
        help=(
            "Optional zero-based TXT column indices for x,y,z, e.g. 1,2,3. "
            "If omitted: 3-column TXT uses 0,1,2; 4-column TXT uses 1,2,3."
        ),
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output file base name without _ego. Defaults to sheet/input name.",
    )
    parser.add_argument(
        "--processed-dir",
        default="processed",
        help="Folder where normalized CSV/NPY files will be saved.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Threshold for near-zero horizontal movement.",
    )
    parser.add_argument(
        "--raise-on-nan",
        action="store_true",
        help="Raise an error for NaN rows instead of removing them.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a simple x_ego/y_ego over-index verification plot.",
    )
    return parser.parse_args()


def main() -> None:
    """Load data, normalize it, save CSV/NPY outputs, and print a summary."""

    args = parse_args()
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    trajectory = load_trajectory(
        args.input_file,
        sheet_name=_parse_sheet_name(args.sheet_name),
        x_col=args.x_col,
        y_col=args.y_col,
        z_col=args.z_col,
        txt_xyz_columns=_parse_txt_xyz_columns(args.txt_xyz_columns),
        drop_nan=not args.raise_on_nan,
    )
    ego_trajectory = normalize_to_ego_coordinates(
        trajectory,
        epsilon=args.epsilon,
    )

    output_name = _output_name(args)
    csv_path = processed_dir / f"{output_name}_ego.csv"
    npy_path = processed_dir / f"{output_name}_ego.npy"
    plot_path = processed_dir / f"{output_name}_ego_xy_over_index.png"

    ego_df = pd.DataFrame(ego_trajectory, columns=["x_ego", "y_ego", "delta_z"])
    ego_df.insert(0, "index", np.arange(len(ego_df)))
    ego_df.to_csv(csv_path, index=False)
    np.save(npy_path, ego_trajectory)

    print(f"Number of raw points: {len(trajectory)}")
    print(f"Number of normalized ego points: {len(ego_trajectory)}")

    print("\nFirst few raw points:")
    print(pd.DataFrame(trajectory, columns=["x", "y", "z"]).head().to_string(index=False))

    print("\nFirst few normalized ego points:")
    print(ego_df.head().to_string(index=False))

    print("\nSaved files:")
    print(f"CSV: {csv_path}")
    print(f"NPY: {npy_path}")

    if args.plot:
        plot_ego_xy_over_index(ego_trajectory, plot_path)
        print(f"Plot: {plot_path}")


def _parse_sheet_name(sheet_name: str) -> str | int:
    """Allow either a sheet name like flight_1 or an integer sheet index."""

    try:
        return int(sheet_name)
    except ValueError:
        return sheet_name


def _parse_txt_xyz_columns(value: str | None) -> tuple[int, int, int] | None:
    """Parse a string like '1,2,3' into TXT x/y/z column indices."""

    if value is None:
        return None

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--txt-xyz-columns must contain exactly 3 indices, e.g. 1,2,3.")

    return tuple(int(part) for part in parts)


def _output_name(args: argparse.Namespace) -> str:
    """Choose the base name for output files."""

    if args.output_name:
        return args.output_name

    input_path = Path(args.input_file)
    if input_path.suffix.lower() in {".xlsx", ".xls"} and isinstance(args.sheet_name, str):
        return args.sheet_name

    return input_path.stem


if __name__ == "__main__":
    main()
