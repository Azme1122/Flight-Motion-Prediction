"""Pseudo-reference trajectory estimation for the self-contained CV-KF workflow."""

import numpy as np
from filterpy.common import kinematic_kf

from kalman_filter_model import extract_position_covariance, extract_position_from_state


INITIAL_P_SCALE = 100.0


def run_cv_kf_over_full_trajectory(measurements, dt, measurement_noise_std, q_scale):
    """
    Run CV-KF over all measured points, one point at a time.

    This creates the pseudo-reference trajectory used later for sliding-window
    evaluation. Each step predicts the next state and then updates with the
    available measured point.
    """
    if len(measurements) < 2:
        raise ValueError("At least two points are needed to initialize velocity.")

    kf = kinematic_kf(dim=3, order=1, dt=dt)

    first_measurement = measurements[0]
    second_measurement = measurements[1]
    initial_velocity = (second_measurement - first_measurement) / dt

    kf.x = np.array([
        [first_measurement[0]],
        [initial_velocity[0]],
        [first_measurement[1]],
        [initial_velocity[1]],
        [first_measurement[2]],
        [initial_velocity[2]],
    ])

    kf.P *= INITIAL_P_SCALE
    kf.R *= measurement_noise_std ** 2
    kf.Q *= q_scale

    estimated_positions = [extract_position_from_state(kf)]
    estimated_covariances = [extract_position_covariance(kf)]

    for measurement in measurements[1:]:
        kf.predict()
        kf.update(measurement)
        estimated_positions.append(extract_position_from_state(kf))
        estimated_covariances.append(extract_position_covariance(kf))

    return np.array(estimated_positions), np.array(estimated_covariances)
