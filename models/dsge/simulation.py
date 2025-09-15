import pandas as pd


def simulate_policy_shock_loop(data_frame, params, u_star, shock_bp=100.0, horizon=36):
    """
    Loop-based simulation of a policy rate shock.

    Args:
        data_frame: pandas DataFrame with columns used for initial conditions
        params: dict of model parameters
        u_star: natural unemployment rate
        shock_bp: basis points shock applied at t=1
        horizon: number of monthly steps to simulate

    Returns:
        pandas DataFrame with columns ['x', 'pi', 'i', 'u', 'rn']
    """
    if data_frame is None or len(data_frame.index) == 0:
        return pd.DataFrame()

    # Initial conditions from last observation
    last = data_frame.iloc[-1]

    x0 = last.get('x_gap_est', 0.0)
    pi0 = last.get('pi', 0.0)
    i0 = last.get('i', 0.0)
    rn0 = last.get('rn_est', params.get('r_star', 0.5))
    u0 = last.get('UNRATE', u_star)

    start_date = data_frame.index[-1] + pd.DateOffset(months=1)
    sim_index = pd.date_range(start=start_date, periods=horizon, freq='MS')
    sim = pd.DataFrame(index=sim_index, columns=['x', 'pi', 'i', 'u', 'rn'])

    sim.iloc[0] = [x0, pi0, i0, u0, rn0]

    t = 1
    while t < horizon:
        i_tm1 = sim.iloc[t - 1]['i']
        pi_tm1 = sim.iloc[t - 1]['pi']
        rn_tm1 = sim.iloc[t - 1]['rn']
        x_tm1 = sim.iloc[t - 1]['x']

        # State evolution
        x_t = (
            params['ax'] * x_tm1
            - params['b_IS'] * (i_tm1 - pi_tm1 - rn_tm1)
        )

        pi_t = params['beta'] * pi_tm1 + params['kappa'] * x_t

        i_star = (
            (1 - params['rho_i'])
            * (
                params['r_star']
                + pi_t
                + params['phi_pi'] * (pi_t - params['pi_star'])
                + params['phi_x'] * x_t
            )
            + params['rho_i'] * i_tm1
        )

        i_t = i_star + (shock_bp / 100.0 if t == 1 else 0.0)

        rn_t = params['rho_rn'] * rn_tm1 + (1 - params['rho_rn']) * params['r_star']
        u_t = u_star - params['lam_okun'] * x_t

        sim.iloc[t] = [x_t, pi_t, i_t, u_t, rn_t]
        t += 1

    return sim
