import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import json
# Classification imports
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -----------------------------------------------------------------------------
# Load configuration from variables.json
# -----------------------------------------------------------------------------
try:
    with open('variables.json', 'r') as f:
        variables_config = json.load(f)
    print("Variables configuration loaded")
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading variables.json: {e}")
    raise


# -----------------------------------------------------------------------------
# Load indices from SQL and run OLS diagnostics similar to data_prep.py
# -----------------------------------------------------------------------------

# Use centralized config
from config import Config
engine = create_engine(Config.DATABASE['connection_string'], future=True)

# Load indices
indices_df = pd.read_sql_table('ivw_indices', con=engine, index_col='index')
indices_df.index = pd.to_datetime(indices_df.index)
target_series = indices_df['GSPC_log_return_next_period']

# =============================================================================
# COMPREHENSIVE DATA DIAGNOSTICS
# =============================================================================
print("="*120)
print("COMPREHENSIVE DATA DIAGNOSTICS")
print("="*120)

# Basic data shape and structure
print(f"Number of columns: {len(indices_df.columns)}")
print(f"Number of rows: {len(indices_df)}")
print(f"Date range: {indices_df.index.min()} to {indices_df.index.max()}")
print(f"Total number of data points: {indices_df.size}")

# Data types overview
print(f"\nData types distribution:")
dtype_counts = indices_df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"  {dtype}: {count} columns")

# NaN analysis
total_nans = indices_df.isna().sum().sum()
print(f"\nTotal number of NaNs: {total_nans}")
print(f"Percentage of NaNs: {(total_nans / indices_df.size) * 100:.2f}%")

# Columns with NaNs
columns_with_nans = indices_df.columns[indices_df.isna().any()].tolist()
print(f"\nColumns with NaNs ({len(columns_with_nans)} total):")
for col in columns_with_nans:
    nan_count = indices_df[col].isna().sum()
    nan_pct = (nan_count / len(indices_df)) * 100
    print(f"  {col}: {nan_count} NaNs ({nan_pct:.2f}%)")

# Rows with NaNs
rows_with_nans = indices_df[indices_df.isna().any(axis=1)]
print(f"\nRows with NaNs ({len(rows_with_nans)} total):")
if len(rows_with_nans) > 0:
    print(f"First 10 rows with NaNs:")
    for i, (idx, row) in enumerate(rows_with_nans.head(10).iterrows()):
        nan_cols = row.isna()
        nan_col_names = nan_cols[nan_cols].index.tolist()
        print(f"  Row {i+1} ({idx}): {len(nan_col_names)} NaNs in columns: {nan_col_names[:5]}{'...' if len(nan_col_names) > 5 else ''}")
    
    if len(rows_with_nans) > 10:
        print(f"  ... and {len(rows_with_nans) - 10} more rows with NaNs")

# Target variable specific analysis
print(f"\nTarget variable (GSPC_log_return_next_period) analysis:")
if 'GSPC_log_return_next_period' in indices_df.columns:
    target_nans = indices_df['GSPC_log_return_next_period'].isna().sum()
    target_valid = indices_df['GSPC_log_return_next_period'].notna().sum()
    print(f"  Valid observations: {target_valid}")
    print(f"  NaN observations: {target_nans}")
    print(f"  NaN percentage: {(target_nans / len(indices_df)) * 100:.2f}%")
    if target_valid > 0:
        print(f"  Mean: {indices_df['GSPC_log_return_next_period'].mean():.6f}")
        print(f"  Std: {indices_df['GSPC_log_return_next_period'].std():.6f}")
        print(f"  Min: {indices_df['GSPC_log_return_next_period'].min():.6f}")
        print(f"  Max: {indices_df['GSPC_log_return_next_period'].max():.6f}")
else:
    print("  WARNING: Target variable not found in dataset!")

# Feature completeness analysis
print(f"\nFeature completeness analysis:")
feature_completeness = indices_df.notna().sum(axis=1) / len(indices_df.columns)
print(f"  Mean completeness: {feature_completeness.mean():.3f}")
print(f"  Min completeness: {feature_completeness.min():.3f}")
print(f"  Max completeness: {feature_completeness.max():.3f}")
print(f"  Rows with 100% completeness: {(feature_completeness == 1.0).sum()}")
print(f"  Rows with >=80% completeness: {(feature_completeness >= 0.8).sum()}")
print(f"  Rows with >=50% completeness: {(feature_completeness >= 0.5).sum()}")

# Infinite values check
inf_cols = []
for col in indices_df.columns:
    if pd.api.types.is_numeric_dtype(indices_df[col]):
        if np.isinf(indices_df[col]).any():
            inf_count = np.isinf(indices_df[col]).sum()
            inf_cols.append((col, inf_count))

if inf_cols:
    print(f"\nColumns with infinite values ({len(inf_cols)} total):")
    for col, count in inf_cols:
        print(f"  {col}: {count} infinite values")
else:
    print(f"\nNo infinite values found in any columns")

# Memory usage
memory_usage = indices_df.memory_usage(deep=True).sum() / 1024**2  # MB
print(f"\nMemory usage: {memory_usage:.2f} MB")

# Sample of the data
print(f"\nFirst 5 rows of data:")
print(indices_df.head())

print(f"\nLast 5 rows of data:")
print(indices_df.tail())

print("="*120)
print("END OF DATA DIAGNOSTICS")
print("="*120)

# Basic NaN handling similar to pipeline: keep target NaNs (filter later), ffill indices
indices_df = indices_df.ffill()
indices_df = indices_df.replace(np.inf, np.nan)
indices_df = indices_df.replace(-np.inf, np.nan)
indices_df.dropna()

y = target_series
X = indices_df.drop(columns=['GSPC_log_return_next_period'])

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

    # =============================================================================
    # FEATURE SELECTION AND MULTICOLLINEARITY CONTROL
    # =============================================================================
    print("="*120)
    print("FEATURE SELECTION AND MULTICOLLINEARITY CONTROL")
    print("="*120)
    
    # Right-size the predictor set - select key regional indices
    # Choose one composite index per region (returns + momentum + volatility combined)
    selected_features = []
    
    # North America (most important)
    if 'north_america_equities_returns' in X_final.columns:
        selected_features.extend(['north_america_equities_returns', 'north_america_equities_momentum', 'north_america_equities_volatility'])
    
    # Europe
    if 'europe_equities_returns' in X_final.columns:
        selected_features.extend(['europe_equities_returns', 'europe_equities_momentum', 'europe_equities_volatility'])
    
    # Asia
    if 'asia_equities_returns' in X_final.columns:
        selected_features.extend(['asia_equities_returns', 'asia_equities_momentum', 'asia_equities_volatility'])
    
    # Key futures (metals and energy)
    if 'metals_returns' in X_final.columns:
        selected_features.extend(['metals_returns', 'metals_momentum', 'metals_volatility'])
    if 'energy_returns' in X_final.columns:
        selected_features.extend(['energy_returns', 'energy_momentum', 'energy_volatility'])
    
    # Filter to only selected features
    available_features = [f for f in selected_features if f in X_final.columns]
    X_selected = X_final[available_features].copy()
    
    print(f"Selected {len(available_features)} key features from {len(X_final.columns)} total")
    print(f"Selected features: {available_features}")
    
    # Apply VIF filtering to control multicollinearity
    print(f"\nApplying VIF filtering to control multicollinearity...")
    
    # Standardize features for VIF calculation
    X_std = (X_selected - X_selected.mean()) / X_selected.std(ddof=0).replace(0, 1.0)
    X_std_const = sm.add_constant(X_std)
    
    # Calculate VIF for each feature
    vif_threshold = 5.0
    features_to_keep = list(X_selected.columns)
    dropped_features = []
    
    while len(features_to_keep) > 2:
        X_current = X_std[features_to_keep]
        X_current_const = sm.add_constant(X_current)
        
        vif_values = []
        for idx in range(1, X_current_const.shape[1]):
            fname = X_current_const.columns[idx]
            try:
                vif = float(variance_inflation_factor(X_current_const.values, idx))
            except Exception:
                vif = float('inf')
            vif_values.append((fname, vif))
        
        # Find features with VIF above threshold
        high_vif_features = [(fname, vif) for fname, vif in vif_values if vif > vif_threshold]
        
        if not high_vif_features:
            break
        
        # Remove the feature with highest VIF
        high_vif_features.sort(key=lambda x: x[1], reverse=True)
        feature_to_remove, vif_value = high_vif_features[0]
        
        features_to_keep.remove(feature_to_remove)
        dropped_features.append((feature_to_remove, vif_value))
        print(f"  Removed '{feature_to_remove}' (VIF: {vif_value:.2f})")
    
    # Final feature set
    X_final = X_selected[features_to_keep].copy()
    
    print(f"\nVIF FILTERING RESULTS:")
    print(f"  Original features: {len(X_selected.columns)}")
    print(f"  Features after VIF filtering: {len(features_to_keep)}")
    print(f"  Features removed: {len(dropped_features)}")
    print(f"  Final features: {features_to_keep}")
    
    print("="*120)
    print("FEATURE SELECTION COMPLETE")
    print("="*120)

    X_with_const_initial = sm.add_constant(X_final)
    try:
        initial_ols = sm.OLS(y_final, X_with_const_initial).fit()
        print('='*80)
        print('OLS REGRESSION RESULTS WITH INDICES')
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

        # =================================================================
        # MULTICOLINEARITY DIAGNOSTICS AND CORRELATION ANALYSIS
        # =================================================================
        print(f"\n{'='*80}")
        print("MULTICOLINEARITY DIAGNOSTICS AND CORRELATION ANALYSIS")
        print(f"{'='*80}")
        
        # Clean and standardize data for diagnostics
        X_clean = X_final.replace([np.inf, -np.inf], np.nan).ffill()
        X_clean = X_clean.fillna(X_clean.mean())
        
        # Standardize for comparable scales
        X_means = X_clean.mean()
        X_stds = X_clean.std(ddof=0).replace(0, 1.0)
        X_std = (X_clean - X_means) / X_stds
        
        # Condition number
        try:
            cond_number = float(np.linalg.cond(X_std.values))
            print(f"Condition Number (standardized): {cond_number:.2f}")
            if cond_number > 30:
                print("  WARNING: High condition number suggests multicollinearity")
            elif cond_number > 100:
                print("  CRITICAL: Very high condition number - severe multicollinearity")
        except Exception as e:
            print(f"Condition number error: {e}")
        
        # Correlation analysis
        print(f"\nCORRELATION ANALYSIS:")
        corr_matrix = X_std.corr()
        
        # Find highly correlated pairs
        strong_pairs = []
        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                val = corr_matrix.iloc[i, j]
                if np.isfinite(val) and abs(val) >= 0.95:
                    strong_pairs.append((cols[i], cols[j], float(val)))
        
        print(f"Highly correlated pairs (|r| >= 0.95): {len(strong_pairs)}")
        if strong_pairs:
            print("Top 20 highly correlated pairs:")
            for name_a, name_b, r in strong_pairs[:20]:
                print(f"  {name_a:<35} {name_b:<35} r={r: .3f}")
            if len(strong_pairs) > 20:
                print(f"  ... and {len(strong_pairs)-20} more")
        else:
            print("  No highly correlated pairs found")
        
        
        # Correlation heatmap summary
        print(f"\nCORRELATION MATRIX SUMMARY:")
        corr_abs = corr_matrix.abs()
        high_corr_pairs = 0
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                if corr_abs.iloc[i, j] >= 0.8:
                    high_corr_pairs += 1
        
        print(f"  Total feature pairs: {len(cols) * (len(cols) - 1) // 2}")
        print(f"  Pairs with |r| >= 0.8: {high_corr_pairs}")
        print(f"  Pairs with |r| >= 0.9: {sum(1 for i in range(len(cols)) for j in range(i+1, len(cols)) if corr_abs.iloc[i, j] >= 0.9)}")
        print(f"  Pairs with |r| >= 0.95: {len(strong_pairs)}")
        
        # Feature scale report
        print(f"\nFEATURE SCALE REPORT (pre-standardization):")
        scale_rows = []
        for col in X_clean.columns:
            scale_info = {
                'feature': col,
                'mean': float(np.nanmean(X_clean[col].values)),
                'std': float(np.nanstd(X_clean[col].values)),
                'min': float(np.nanmin(X_clean[col].values)),
                'max': float(np.nanmax(X_clean[col].values))
            }
            scale_rows.append(scale_info)
        
        print(f"Total features: {len(scale_rows)}")
        for scale_info in scale_rows[:15]:
            print(f"  {scale_info['feature']:<35} mean={scale_info['mean']:> .4f} std={scale_info['std']:> .4f} min={scale_info['min']:> .4f} max={scale_info['max']:> .4f}")
        if len(scale_rows) > 15:
            print(f"  ... and {len(scale_rows)-15} more")


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
            valid_r2 = [r for r in rolling_r2 if not np.isnan(r)]
            
            print(f"\nRolling R-squared (window={window_size}, params={params_in_model}):")
            if valid_r2:
                print(f"  Mean: {np.mean(valid_r2):.4f}")
                print(f"  Std:  {np.std(valid_r2):.4f}")
                print(f"  Min:  {np.min(valid_r2):.4f}")
                print(f"  Max:  {np.max(valid_r2):.4f}")
                print(f"  Last: {r2_series.iloc[-1]:.4f}")
                print(f"  Valid observations: {len(valid_r2)}/{len(rolling_r2)}")
            else:
                print(f"  No valid rolling R-squared values computed")
                print(f"  (All values are NaN - insufficient data for rolling window)")

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

        # =============================================================================
        # DYNAMIC LINEAR MODEL (DLM) / TIME-VARYING PARAMETER REGRESSION
        # =============================================================================
        print(f"\n{'='*80}")
        print("DYNAMIC LINEAR MODEL (DLM) - TIME-VARYING PARAMETERS")
        print(f"{'='*80}")
        
        # DLM Configuration
        discount_factor = 0.98  # δ = 0.98 (tune between 0.97-0.99)
        ridge_penalty = 0.01   # Ridge penalty for state evolution stability
        
        print(f"DLM Configuration:")
        print(f"  Discount factor (δ): {discount_factor}")
        print(f"  Ridge penalty: {ridge_penalty}")
        print(f"  Features: {len(features_to_keep)}")
        print(f"  Observations: {len(y_final)}")
        
        # Prepare data for DLM
        y_dlm = y_final.values
        X_dlm = X_final.values
        n_obs, n_features = X_dlm.shape
        
        # Initialize DLM state
        # State: β_t (time-varying coefficients)
        beta_t = np.zeros((n_obs, n_features))  # β_t for each time step
        P_t = np.zeros((n_obs, n_features, n_features))  # Covariance matrices
        
        # Initial state (from static OLS)
        beta_0 = initial_ols.params[1:].values  # Exclude constant
        P_0 = np.eye(n_features) * 0.1  # Initial covariance
        
        beta_t[0] = beta_0
        P_t[0] = P_0
        
        # Process noise covariance (Q) - tuned via discount factor
        Q = (1 - discount_factor) / discount_factor * P_0
        
        # Measurement noise (R) - estimated from residuals
        R = np.var(initial_ols.resid)
        
        print(f"  Initial β: {beta_0}")
        print(f"  Process noise (Q): {np.diag(Q)}")
        print(f"  Measurement noise (R): {R:.6f}")
        
        # DLM Forward Pass (Kalman Filter for time-varying parameters)
        print(f"\nRunning DLM forward pass...")
        
        for t in range(1, n_obs):
            # Prediction step: β_t = β_{t-1} + w_t
            beta_pred = beta_t[t-1]
            P_pred = P_t[t-1] + Q
            
            # Add ridge penalty for stability
            P_pred = P_pred + ridge_penalty * np.eye(n_features)
            
            # Update step: y_t = x_t^T β_t + ε_t
            x_t = X_dlm[t]
            y_t = y_dlm[t]
            
            if not np.isnan(y_t) and not np.any(np.isnan(x_t)):
                # Innovation
                y_innov = y_t - np.dot(x_t, beta_pred)
                
                # Innovation covariance
                S = np.dot(x_t, np.dot(P_pred, x_t)) + R
                
                # Kalman gain
                K = np.dot(P_pred, x_t) / S
                
                # State update
                beta_t[t] = beta_pred + K * y_innov
                P_t[t] = P_pred - np.outer(K, x_t) * P_pred
            else:
                # If observation is NaN, use prediction
                beta_t[t] = beta_pred
                P_t[t] = P_pred
        
        # Calculate DLM fitted values and diagnostics
        y_fitted_dlm = np.array([np.dot(X_dlm[t], beta_t[t]) for t in range(n_obs)])
        residuals_dlm = y_dlm - y_fitted_dlm
        
        # DLM R-squared
        ss_res = np.sum(residuals_dlm**2)
        ss_tot = np.sum((y_dlm - np.mean(y_dlm))**2)
        r2_dlm = 1 - (ss_res / ss_tot)
        
        print(f"\nDLM RESULTS:")
        print(f"  R-squared: {r2_dlm:.4f}")
        print(f"  RMSE: {np.sqrt(np.mean(residuals_dlm**2)):.6f}")
        print(f"  Mean absolute error: {np.mean(np.abs(residuals_dlm)):.6f}")
        
        # Parameter evolution analysis
        print(f"\nPARAMETER EVOLUTION ANALYSIS:")
        for i, feature in enumerate(features_to_keep):
            beta_series = beta_t[:, i]
            print(f"  {feature}:")
            print(f"    Initial β: {beta_series[0]:.6f}")
            print(f"    Final β: {beta_series[-1]:.6f}")
            print(f"    Mean β: {np.mean(beta_series):.6f}")
            print(f"    Std β: {np.std(beta_series):.6f}")
            print(f"    Range: [{np.min(beta_series):.6f}, {np.max(beta_series):.6f}]")
        
        # Sanity checks
        print(f"\nDLM SANITY CHECKS:")
        
        # Check for parameter explosion
        max_beta_change = np.max(np.abs(np.diff(beta_t, axis=0)))
        print(f"  Max parameter change per step: {max_beta_change:.6f}")
        if max_beta_change > 1.0:
            print(f"  WARNING: Large parameter changes detected - consider smaller Q or larger ridge penalty")
        else:
            print(f"  ✓ Parameter changes are reasonable")
        
        # Check for NaN/Inf in parameters
        nan_params = np.any(np.isnan(beta_t))
        inf_params = np.any(np.isinf(beta_t))
        print(f"  NaN parameters: {nan_params}")
        print(f"  Inf parameters: {inf_params}")
        if not nan_params and not inf_params:
            print(f"  ✓ No numerical issues in parameters")
        
        # Compare with static model
        print(f"\nMODEL COMPARISON:")
        print(f"  Static OLS R²: {initial_ols.rsquared:.4f}")
        print(f"  DLM R²: {r2_dlm:.4f}")
        print(f"  R² difference: {r2_dlm - initial_ols.rsquared:+.4f}")
        
        if abs(r2_dlm - initial_ols.rsquared) < 0.1:
            print(f"  ✓ DLM performance is consistent with static model")
        else:
            print(f"  WARNING: Large difference between static and dynamic models")
    except Exception as e:
        print(f"Error running OLS regression with indices: {e}")


    # =====================================================================
    # CLASSIFICATION EXPERIMENTS (time-based split; multiple classifiers)
    # =====================================================================
    try:
        print("\n" + "="*80)
        print("CLASSIFICATION EXPERIMENTS")
        print("="*80)

        # Prepare binary target: next-period up move (>0 -> 1 else 0)
        y_bin = (y_final > 0).astype(int)

        # Use all available features
        X_cls = X_clean.copy()

        # Align indexes
        common_idx = y_bin.index.intersection(X_cls.index)
        y_bin = y_bin.loc[common_idx]
        X_cls = X_cls.loc[common_idx]

        # Time-based split (70% train, 30% test)
        n_obs = len(y_bin)
        split_idx = int(n_obs * 0.7)
        train_idx = y_bin.index[:split_idx]
        test_idx = y_bin.index[split_idx:]

        X_train = X_cls.loc[train_idx]
        y_train = y_bin.loc[train_idx]
        X_test = X_cls.loc[test_idx]
        y_test = y_bin.loc[test_idx]

        # Scale features (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train.values), index=X_train.index, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test.values), index=X_test.index, columns=X_test.columns)

        # Define classifiers and whether to use scaled or raw features
        clf_names = [
            'LogisticRegression',
            'SVC_RBF',
            'RandomForest',
            'GradientBoosting'
        ]

        print("\nRunning classifiers:")
        for name in clf_names:
            if name == 'LogisticRegression':
                clf = LogisticRegression(max_iter=200, n_jobs=None)
                X_tr, X_te = X_train_scaled, X_test_scaled
            elif name == 'SVC_RBF':
                clf = SVC(kernel='rbf', probability=True)
                X_tr, X_te = X_train_scaled, X_test_scaled
            elif name == 'RandomForest':
                clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
                X_tr, X_te = X_train, X_test
            elif name == 'GradientBoosting':
                clf = GradientBoostingClassifier(random_state=42)
                X_tr, X_te = X_train, X_test
            else:
                continue

            # Fit
            clf.fit(X_tr, y_train)
            # Predict proba if available
            if hasattr(clf, 'predict_proba'):
                y_prob = clf.predict_proba(X_te)[:, 1]
            else:
                # Fallback using decision_function if available
                if hasattr(clf, 'decision_function'):
                    df_vals = clf.decision_function(X_te)
                    # Map decision scores to [0,1] via logistic for auc comparability
                    y_prob = 1.0 / (1.0 + np.exp(-df_vals))
                else:
                    # No score; approximate with predictions
                    y_prob = None

            y_pred = clf.predict(X_te)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc = roc_auc_score(y_test, y_prob) if y_prob is not None else float('nan')

            print("-"*70)
            print(f"Model: {name}")
            print(f"  Train samples: {len(y_train)}  Test samples: {len(y_test)}  Features: {X_tr.shape[1]}")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall: {rec:.4f}")
            print(f"  F1: {f1:.4f}")
            print(f"  ROC-AUC: {roc:.4f}")

    except Exception as e:
        print(f"Error in classification experiments: {e}")
