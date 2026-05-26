"""Tune prediction Q using CL68 coverage across all configured future horizons."""

from tuning_core import run_q_tuning


if __name__ == "__main__":
    run_q_tuning(tune_horizon=None, ci_name="CI68")
