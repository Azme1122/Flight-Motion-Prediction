"""Run full-trajectory ego-coordinate normalization for flight_1.

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
    "c1.xlsx"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Normalize raw 3D trajectory data into ego-coordinate features."
    )
    parser.add_argument(
        "--input-file",
        default=DEFAULT_INPUT_FILE,
        help="Path to the input Excel or CSV trajectory file.",
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
        drop_nan=not args.raise_on_nan,
    )
    ego_trajectory = normalize_to_ego_coordinates(
        trajectory,
        epsilon=args.epsilon,
    )

    csv_path = processed_dir / "flight_1_ego.csv"
    npy_path = processed_dir / "flight_1_ego.npy"
    plot_path = processed_dir / "flight_1_ego_xy_over_index.png"

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


if __name__ == "__main__":
    main()
