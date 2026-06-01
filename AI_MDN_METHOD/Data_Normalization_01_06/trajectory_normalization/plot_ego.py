"""Simple verification plots for ego-coordinate motion features."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_ego_xy_over_index(
    ego_trajectory: np.ndarray,
    output_path: str | Path,
    *,
    show: bool = False,
) -> None:
    """Plot x_ego and y_ego over index/time.

    Important: x_ego and y_ego are local displacement features, not absolute
    positions in one fixed coordinate frame. Therefore, this function does not
    plot x_ego vs y_ego as a trajectory path.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required only for plotting. Install it with "
            "`pip install matplotlib`, or run main.py without --plot."
        ) from exc

    ego = np.asarray(ego_trajectory, dtype=float)
    if ego.ndim != 2 or ego.shape[1] != 3:
        raise ValueError(f"Expected ego trajectory shape (M, 3), got {ego.shape}.")

    index = np.arange(len(ego))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(index, ego[:, 0], linewidth=1.5)
    axes[0].set_ylabel("x_ego")
    axes[0].set_title("Sideways local displacement over index")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(index, ego[:, 1], linewidth=1.5)
    axes[1].set_xlabel("index")
    axes[1].set_ylabel("y_ego")
    axes[1].set_title("Forward local displacement over index")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Ego-coordinate local motion features")
    fig.tight_layout()
    fig.savefig(output, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)
