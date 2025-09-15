import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# Load indices from SQL and run OLS diagnostics similar to data_prep.py
# -----------------------------------------------------------------------------

server = 'localhost'
port = '5432'
database = 'avalon'
username = 'admin'
password = 'password!'
conn_str = f'postgresql+psycopg2://{username}:{password}@{server}:{port}/{database}'
engine = create_engine(conn_str, future=True)

# Load indices
indices_df = pd.read_sql_table('ml_spx_indices', con=engine, index_col='index')
indices_df.index = pd.to_datetime(indices_df.index)

# Load target table from prior pipeline
try:
    full_data = pd.read_sql_table('ml_spx_data', con=engine, index_col='index')
    full_data.index = pd.to_datetime(full_data.index)
    if 'GSPC_log_return_next_period' not in full_data.columns:
        raise KeyError('Target column missing in ml_spx_data')
    target_series = full_data['GSPC_log_return_next_period']
except Exception as e:
    raise RuntimeError(f"Cannot load target from ml_spx_data: {e}")

# Align on common index
common_index = indices_df.index.intersection(target_series.index)
indices_df = indices_df.reindex(common_index)
target_series = target_series.reindex(common_index)

# Basic NaN handling similar to pipeline: keep target NaNs (filter later), ffill indices
indices_df = indices_df.ffill()

y = target_series
X = indices_df

# Use only rows where target is valid
y_valid_mask = y.notna()
y_filtered = y[y_valid_mask]
X_filtered = X[y_valid_mask]

# Require at least 80% non-NaN features per row
feature_completeness = X_filtered.notna().sum(axis=1) / len(X_filtered.columns)
sufficient_features = feature_completeness >= 0.8

if sufficient_features.sum() < 50:
    print('Insufficient data for OLS regression')
else:
    y_final = y_filtered[sufficient_features]
    X_final = X_filtered[sufficient_features]
    X_final = X_final.ffill()

    X_with_const_initial = sm.add_constant(X_final)
    try:
        initial_ols = sm.OLS(y_final, X_with_const_initial).fit()
        print('='*80)
        print('OLS REGRESSION RESULTS WITH INDICES (ALL)')
        print('='*80)
        print(f"Sample size: {len(y_final)}")
        print(f"Features: {X_final.shape[1]}")
        print(f"R-squared: {initial_ols.rsquared:.4f}")
        print(f"Adjusted R-squared: {initial_ols.rsquared_adj:.4f}")
        print(f"F-statistic p-value: {initial_ols.f_pvalue:.4f}")

        var_names = ['const'] + list(X_final.columns)
        var_data = [(var, initial_ols.params.iloc[i], initial_ols.pvalues.iloc[i]) for i, var in enumerate(var_names)]
        var_data_sorted = sorted(var_data, key=lambda x: x[2])
        print('\nVariables sorted by significance:')
        print(f"{'Variable':<40} {'Coefficient':<15} {'P-value':<15} {'Sig':<5}")
        print('-'*85)
        for var, coef, p_val in var_data_sorted:
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            print(f"{var:<40} {coef:<15.6f} {p_val:<15.4f} {sig:<5}")

        # Rolling R-squared diagnostics with valid windowing (avoid obs <= params)
        if len(y_final) >= 120:
            params_in_model = X_final.shape[1] + 1  # +1 for constant
            window_size = max(60, params_in_model + 5)
            if len(y_final) < window_size + 5:
                window_size = max(30, min(len(y_final) - 1, params_in_model + 5))
            rolling_r2 = []
            rolling_dates = []
            for i in range(window_size, len(y_final)):
                y_w = y_final.iloc[i-window_size:i]
                X_w = sm.add_constant(X_final.iloc[i-window_size:i])
                obs_count = y_w.shape[0]
                param_count = X_w.shape[1]
                if obs_count <= param_count:
                    rolling_r2.append(np.nan)
                    rolling_dates.append(y_final.index[i])
                    continue
                try:
                    rank = np.linalg.matrix_rank(X_w.values)
                except Exception:
                    rank = param_count
                if rank < param_count:
                    rolling_r2.append(np.nan)
                    rolling_dates.append(y_final.index[i])
                    continue
                try:
                    tmp = sm.OLS(y_w, X_w).fit()
                    rolling_r2.append(tmp.rsquared)
                    rolling_dates.append(y_final.index[i])
                except:
                    rolling_r2.append(np.nan)
                    rolling_dates.append(y_final.index[i])
            r2_series = pd.Series(rolling_r2, index=rolling_dates)
            print(f"\nRolling R-squared (window={window_size}, params={params_in_model}):")
            print(f"  Mean: {np.nanmean(rolling_r2):.4f}")
            print(f"  Std:  {np.nanstd(rolling_r2):.4f}")
            print(f"  Min:  {np.nanmin(rolling_r2):.4f}")
            print(f"  Max:  {np.nanmax(rolling_r2):.4f}")
            print(f"  Last: {r2_series.iloc[-1]:.4f}")

        # Subperiod stability (3 blocks)
        n_periods = 3
        period_size = len(y_final) // n_periods
        print('\nSubperiod analysis:')
        for k in range(n_periods):
            s = k * period_size
            e = (k + 1) * period_size if k < n_periods - 1 else len(y_final)
            y_sub = y_final.iloc[s:e]
            X_sub = sm.add_constant(X_final.iloc[s:e])
            try:
                sub = sm.OLS(y_sub, X_sub).fit()
                ps = y_final.index[s].strftime('%Y-%m')
                pe = y_final.index[e-1].strftime('%Y-%m')
                print(f"  Period {k+1} ({ps} to {pe}): R2={sub.rsquared:.4f}, AdjR2={sub.rsquared_adj:.4f}, Fp={sub.f_pvalue:.4f}")
            except Exception as ex:
                print(f"  Period {k+1} error: {ex}")
    except Exception as e:
        print(f"Error running OLS regression with indices: {e}")


