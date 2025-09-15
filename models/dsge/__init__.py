from typing import Dict, Optional, Tuple

import pandas as pd

from .core import DEFAULT_PARAMS
from .kalman import run_kalman_filter_and_smoother
from .simulation import simulate_policy_shock_loop


class DSGEModel:
    """
    DSGE model wrapper exposing loop-based estimation and simulation.
    """

    def __init__(self, parameters: Optional[Dict[str, float]] = None):
        self.default_params = DEFAULT_PARAMS.copy()
        self.params = parameters.copy() if parameters else self.default_params.copy()
        self.data = None
        self.kalman_results = None

    def set_parameter(self, param_name: str, value: float):
        if param_name not in self.default_params:
            raise ValueError(f"Unknown parameter: {param_name}")
        self.params[param_name] = value

    def update_parameters(self, parameters: Dict[str, float]):
        for k, v in parameters.items():
            self.set_parameter(k, v)

    def estimate(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        if data_frame is None or len(data_frame.index) == 0:
            raise ValueError('No data provided for estimation')

        # Compute natural unemployment (u_star)
        unrate = data_frame['UNRATE']
        if unrate.notna().sum() > 60:
            u_star = unrate.rolling(120, min_periods=60).median().iloc[-1]
        else:
            u_star = 4.5

        # Persist in params for downstream use
        self.params['u_star'] = float(u_star)

        # Run Kalman filter+smoother
        results = run_kalman_filter_and_smoother(data_frame, self.params, float(u_star))

        # Attach to data
        out_df = data_frame.copy()
        out_df['x_gap_est'] = results['s_smoothed'][:, 0]
        out_df['rn_est'] = results['s_smoothed'][:, 1]

        # Persist results
        self.kalman_results = results
        self.data = out_df
        return out_df

    def simulate(self, data_frame: pd.DataFrame, shock_bp: float = 100.0, horizon: int = 36) -> pd.DataFrame:
        if self.kalman_results is None:
            raise ValueError('Model must be estimated before simulation')
        u_star = float(self.kalman_results.get('u_star'))
        return simulate_policy_shock_loop(data_frame, self.params, u_star, shock_bp=shock_bp, horizon=horizon)
