"""Tune prediction Q by reliability curve across all future prediction points."""

from tuning_core import run_q_reliability_tuning


ALL_RELIABILITY_Q_TUNE_MIN = 0.00001
ALL_RELIABILITY_Q_TUNE_MAX = 2.0
ALL_RELIABILITY_Q_TUNE_STEP = 0.001


if __name__ == "__main__":
    run_q_reliability_tuning(
        tune_horizon=None,
        q_min=ALL_RELIABILITY_Q_TUNE_MIN,
        q_max=ALL_RELIABILITY_Q_TUNE_MAX,
        q_step=ALL_RELIABILITY_Q_TUNE_STEP,
    )
