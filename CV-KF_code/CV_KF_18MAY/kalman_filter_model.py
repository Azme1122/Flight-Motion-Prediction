"""Constant-velocity Kalman filter model helpers."""

import numpy as np
from filterpy.common import kinematic_kf


def extract_position_from_state(kf):
    """Extract x, y, z position from FilterPy state order [x, vx, y, vy, z, vz]."""
    return np.array([
        kf.x[0, 0],
        kf.x[2, 0],
        kf.x[4, 0],
    ])


def extract_position_covariance(kf):
    """Extract the 3x3 position covariance from the full Kalman filter covariance."""
    pos_indices = [0, 2, 4]
    return kf.P[np.ix_(pos_indices, pos_indices)]


def run_cv_kf_on_window(obs_measurements, pred_len, dt, measurement_noise_std, q_scale=1.0):
    """
    Run a 3D constant-velocity Kalman filter on one sliding window.

    Observation phase uses the noisy observed points. Prediction phase then predicts
    future points without measurement updates.
    """
    kf = kinematic_kf(dim=3, order=1, dt=dt)

    first_measurement = obs_measurements[0]
    second_measurement = obs_measurements[1]
    initial_velocity = (second_measurement - first_measurement) / dt

    kf.x = np.array([
        [first_measurement[0]],
        [initial_velocity[0]],
        [first_measurement[1]],
        [initial_velocity[1]],
        [first_measurement[2]],
        [initial_velocity[2]],
    ])

    kf.P *= 100.0
    kf.R *= measurement_noise_std ** 2
    kf.Q *= q_scale

    estimated_positions = [extract_position_from_state(kf)]
    estimated_covariances = [extract_position_covariance(kf)]

    for z in obs_measurements[1:]:
        kf.predict()
        kf.update(z)
        estimated_positions.append(extract_position_from_state(kf))
        estimated_covariances.append(extract_position_covariance(kf))

    future_predictions = []
    future_covariances = []

    for _ in range(pred_len):
        kf.predict()
        future_predictions.append(extract_position_from_state(kf))
        future_covariances.append(extract_position_covariance(kf))

    return (
        np.array(estimated_positions),
        np.array(estimated_covariances),
        np.array(future_predictions),
        np.array(future_covariances),
    )
