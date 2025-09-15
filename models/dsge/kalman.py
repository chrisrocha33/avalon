import numpy as np

from .core import (
    get_transition_matrix,
    get_state_noise_covariance,
    get_measurement_noise_covariance,
    compute_observation_components,
)


def run_kalman_filter_and_smoother(data_frame, params, u_star):
    """
    Run forward Kalman filter and backward RTS smoother.

    Args:
        data_frame: pandas DataFrame with columns ['pi', 'i', 'UNRATE']
        params: dict of model parameters
        u_star: natural unemployment rate level

    Returns:
        dict with filtered/smoothed states and covariances
    """
    # Extract observables as numpy arrays to keep this routine library-agnostic
    pi_series = data_frame['pi'].values
    i_series = data_frame['i'].values
    u_series = data_frame['UNRATE'].values

    T = len(pi_series)
    if T == 0:
        raise ValueError('Empty data passed to Kalman filter')

    # Allocate arrays
    s_pred = np.zeros((T, 2))
    P_pred = np.zeros((T, 2, 2))
    s_filt = np.zeros((T, 2))
    P_filt = np.zeros((T, 2, 2))

    # Initial state and covariances
    s_filt[0] = np.array([0.0, params['r_star']])
    P_filt[0] = np.diag([1.0, 0.5])

    # Static pieces
    Q = get_state_noise_covariance(params)
    R = get_measurement_noise_covariance(params)

    # Forward recursion
    for t in range(1, T):
        i_lag = i_series[t - 1]
        pi_lag = pi_series[t - 1]

        # Transition
        A = get_transition_matrix(params)
        trans_u = np.array([-params['b_IS'] * (i_lag - pi_lag), 0.0])

        s_pred[t] = A @ s_filt[t - 1] + trans_u
        P_pred[t] = A @ P_filt[t - 1] @ A.T + Q

        # Observation at time t
        pi_obs = pi_series[t]
        i_lag2 = i_series[t - 1]
        H, d = compute_observation_components(pi_obs, pi_lag, i_lag2, params, u_star)

        z_t = np.array([pi_series[t], i_series[t], u_series[t]])

        # Innovation
        y_t = z_t - (H @ s_pred[t] + d)
        S_t = H @ P_pred[t] @ H.T + R
        K = P_pred[t] @ H.T @ np.linalg.inv(S_t)

        # Update
        s_filt[t] = s_pred[t] + K @ y_t
        P_filt[t] = (np.eye(2) - K @ H) @ P_pred[t]

    # RTS smoother
    s_smooth = s_filt.copy()
    P_smooth = P_filt.copy()

    for t in range(T - 2, -1, -1):
        A = get_transition_matrix(params)
        C = P_filt[t] @ A.T @ np.linalg.inv(P_pred[t + 1])
        s_smooth[t] = s_filt[t] + C @ (s_smooth[t + 1] - s_pred[t + 1])
        P_smooth[t] = P_filt[t] + C @ (P_smooth[t + 1] - P_pred[t + 1]) @ C.T

    return {
        's_filtered': s_filt,
        's_smoothed': s_smooth,
        'P_filtered': P_filt,
        'P_smoothed': P_smooth,
        'u_star': u_star,
    }
