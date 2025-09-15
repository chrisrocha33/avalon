import numpy as np

# Default parameter values (first-pass calibration)
DEFAULT_PARAMS = {
    'beta': 0.5,        # Inflation persistence
    'kappa': 0.15,      # Phillips curve slope
    'sigma': 1.5,       # Risk aversion
    'phi_pi': 1.5,      # Taylor rule inflation response
    'phi_x': 0.5,       # Taylor rule output response
    'rho_i': 0.7,       # Interest rate smoothing
    'ax': 0.6,          # Output gap persistence
    'b_IS': 0.5,        # IS curve slope
    'rho_rn': 0.9,      # Natural rate persistence
    'pi_star': 2.0,     # Inflation target
    'r_star': 0.5,      # Natural real rate
    'lam_okun': 2.0,    # Okun's law coefficient
    # Shock/measurement variances
    'sig_x': 0.50,      # Output gap shock variance
    'sig_rn': 0.10,     # Natural rate shock variance
    'sig_pi': 0.30,     # Inflation measurement noise
    'sig_i': 0.40,      # Interest rate measurement noise
    'sig_u': 0.25       # Unemployment measurement noise
}


def get_transition_matrix(params):
    """Build time-invariant state transition matrix A.
    State vector: [x_t, r^n_t]
    """
    return np.array([
        [params['ax'], params['b_IS']],
        [0.0,           params['rho_rn']]
    ])


def get_state_noise_covariance(params):
    """Process noise covariance Q."""
    return np.diag([params['sig_x'] ** 2, params['sig_rn'] ** 2])


def get_measurement_noise_covariance(params):
    """Measurement noise covariance R for [pi, i, u]."""
    return np.diag([params['sig_pi'] ** 2, params['sig_i'] ** 2, params['sig_u'] ** 2])


def compute_observation_components(pi_obs, pi_lag, i_lag, params, u_star):
    """
    Time-varying observation equation components.
    Returns H (3x2) and d (3,).
    z_t = H*s_t + d_t + v_t
    Order of observables: [pi_t, i_t, u_t]
    """
    d_pi = params['beta'] * pi_lag

    c1_pi = (1 - params['rho_i']) * (1 + params['phi_pi'])
    c2_x = (1 - params['rho_i']) * params['phi_x']
    d_i = (
        params['r_star']
        + (1 - params['rho_i']) * params['pi_star']
        + c1_pi * pi_obs
        + params['rho_i'] * i_lag
    )

    d_u = u_star

    H = np.array([
        [params['kappa'],          0.0],   # Inflation loads on x_t
        [c2_x,                     0.0],   # Policy rate loads on x_t
        [-params['lam_okun'],      0.0]    # Unemployment loads on x_t
    ])

    d = np.array([d_pi, d_i, d_u])
    return H, d


def to_float(value):
    try:
        return float(value)
    except Exception:
        return None
