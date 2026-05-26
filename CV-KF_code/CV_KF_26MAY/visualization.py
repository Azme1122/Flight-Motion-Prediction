"""Plotting helpers for calibration curves and summary tables."""

import matplotlib.pyplot as plt


def plot_calibration_curves(
    calibration_curves,
    horizons_to_plot=None,
    filename="cv_kf_calibration_plot.png",
    show=True,
):
    """Plot calibration curves for selected prediction horizons."""
    if horizons_to_plot is None:
        horizons_to_plot = list(calibration_curves.keys())

    plt.figure(figsize=(7, 6))

    for h in horizons_to_plot:
        curve = calibration_curves[h]
        plt.plot(
            curve["expected_cls"],
            curve["observed_freq"],
            label=f"t+{h}",
            linewidth=2,
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal")
    plt.xlabel("Expected confidence level")
    plt.ylabel("Observed frequency")
    plt.title("CV-KF Calibration Plot")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)

    if show:
        plt.show()

    plt.close()


def save_table_as_image(table, filename="cv_kf_summary_table.png", show=True):
    """Save a pandas DataFrame summary table as a PNG image."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")

    table_rounded = table.copy()
    table_rounded["CV-KF"] = table_rounded["CV-KF"].apply(lambda x: f"{x:.3f}")

    mpl_table = ax.table(
        cellText=table_rounded.values,
        colLabels=table_rounded.columns,
        cellLoc="center",
        loc="center",
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(11)
    mpl_table.scale(1.2, 1.4)

    plt.title("CV-KF One-Trajectory Sliding-Window Summary")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300)

    if show:
        plt.show()

    plt.close(fig)
