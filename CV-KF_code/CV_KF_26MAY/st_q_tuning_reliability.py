"""Tune prediction Q by making selected reliability curves closest to ideal."""

from config import RELIABILITY_LAST_HORIZON
from tuning_core import run_q_reliability_tuning


RELIABILITY_Q_TUNE_MIN = 0.00001
RELIABILITY_Q_TUNE_MAX = 2.0
RELIABILITY_Q_TUNE_STEP = 0.001


def tune_one_horizon(tune_horizon):
    """Tune one horizon by calibration-curve reliability search."""
    return run_q_reliability_tuning(
        tune_horizon=tune_horizon,
        q_min=RELIABILITY_Q_TUNE_MIN,
        q_max=RELIABILITY_Q_TUNE_MAX,
        q_step=RELIABILITY_Q_TUNE_STEP,
    )


if __name__ == "__main__":
    _, first_best = tune_one_horizon(tune_horizon=1)
    _, last_best = tune_one_horizon(tune_horizon=RELIABILITY_LAST_HORIZON)

    print("\nReliability tuning summary")
    print("--------------------------")
    print(f"t+1 best Q: {first_best['Q_scale']}")
    print(f"t+1 reliability score: {first_best['Reliability_score']:.3f}%")
    print(f"t+{RELIABILITY_LAST_HORIZON} best Q: {last_best['Q_scale']}")
    print(f"t+{RELIABILITY_LAST_HORIZON} reliability score: {last_best['Reliability_score']:.3f}%")
