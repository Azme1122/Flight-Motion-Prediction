"""CV-KF visualization helpers."""

from pathlib import Path

from visualization import plot_calibration_curves as _plot_calibration_curves
from visualization import save_table_as_image


def plot_calibration_curves(calibration_curves, horizons_to_plot=None, filename=None, show=False):
    """Plot calibration curves, accepting pathlib paths."""
    filename = str(filename) if filename is not None else None
    return _plot_calibration_curves(
        calibration_curves=calibration_curves,
        horizons_to_plot=horizons_to_plot,
        filename=filename,
        show=show,
    )


def save_summary_table_image(table, filename=None, show=False):
    """Save a summary-table image, accepting pathlib paths."""
    filename = str(filename) if filename is not None else None
    return save_table_as_image(table=table, filename=filename, show=show)


def ensure_dir(path):
    """Create an output directory if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
