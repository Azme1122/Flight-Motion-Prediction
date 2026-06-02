"""Normalize all four TXT trajectory datasets into ego-coordinate features.

Each input row is treated as one raw trajectory point. The loader handles:

- 3 columns: x, y, z
- 4 or more columns: index/time is ignored and the last three columns are x, y, z

No machine-learning windows are created here. The full normalized trajectory is
saved for each dataset as CSV and NPY.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_trajectory
from ego_normalization import normalize_to_ego_coordinates
from plot_ego import plot_ego_xy_over_index


DATASET_FILES = [
    "dataset01.txt",
    "dataset02.txt",
    "dataset03.txt",
    "dataset04.txt",
]


def process_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    epsilon: float = 1e-8,
    make_plot: bool = False,
) -> None:
    """Load one TXT dataset, normalize it, and save CSV/NPY outputs."""

    trajectory = load_trajectory(input_path)
    ego_trajectory = normalize_to_ego_coordinates(trajectory, epsilon=epsilon)

    output_base = input_path.stem
    csv_path = output_dir / f"{output_base}_ego.csv"
    npy_path = output_dir / f"{output_base}_ego.npy"
    plot_path = output_dir / f"{output_base}_ego_xy_over_index.png"

    ego_df = pd.DataFrame(ego_trajectory, columns=["x_ego", "y_ego", "delta_z"])
    ego_df.insert(0, "index", np.arange(len(ego_df)))
    ego_df.to_csv(csv_path, index=False)
    np.save(npy_path, ego_trajectory)

    if make_plot:
        plot_ego_xy_over_index(ego_trajectory, plot_path)

    print(f"{input_path.name}")
    print(f"  raw points: {len(trajectory)}")
    print(f"  ego points: {len(ego_trajectory)}")
    print(f"  CSV: {csv_path}")
    print(f"  NPY: {npy_path}")
    if make_plot:
        print(f"  plot: {plot_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Normalize dataset01.txt through dataset04.txt into ego features."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also save x_ego/y_ego over-index plots. Requires matplotlib.",
    )
    return parser.parse_args()


def main() -> None:
    """Process dataset01.txt through dataset04.txt."""

    args = parse_args()

    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "processed_txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_name in DATASET_FILES:
        process_dataset(project_dir / file_name, output_dir, make_plot=args.plot)


if __name__ == "__main__":
    main()
