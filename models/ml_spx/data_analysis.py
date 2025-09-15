import pandas as pd
import numpy as np
import statsmodels.api as sm
from sqlalchemy import create_engine
import json
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    with open('variables.json', 'r') as f:
        variables_config = json.load(f)
    print("Variables configuration loaded")
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading variables.json: {e}")
    raise
top_features_count = variables_config.get('config', {}).get('top_features', 50)
top_diagnostics_count = 80


print(f"="*120)
print("SQL Connection Starting...")
server = 'localhost'
port = '5432'
database = 'avalon'
username = 'admin'
password = 'password!'

conn_str = f'postgresql+psycopg2://{username}:{password}@{server}:{port}/{database}'
engine = create_engine(conn_str, future=True)   

print(f"="*120)
print("SQL Connection Successful")
print(f"="*120)
print("LOADING DATA...")
print(f"="*120)
print("")
combined_data = pd.read_sql_table('ml_spx_regional_indices', engine,index_col='index')

# Decompress: upcast numeric columns to float64 to restore original precision
try:
    _num_cols = []
    for _c in combined_data.columns:
        if pd.api.types.is_numeric_dtype(combined_data[_c]):
            _num_cols.append(_c)
    if len(_num_cols) > 0:
        combined_data[_num_cols] = combined_data[_num_cols].astype("float64")
        print(f"Decompression: upcasted {len(_num_cols)} numeric columns to float64")
except Exception as _e_dec:
    print(f"Decompression warning (data_analysis): {_e_dec}")
print(combined_data.head())
print(combined_data.columns.tolist())
print("-"*120)
print("INDEX AND DTYPES DEBUG")
print(f"Index name: {combined_data.index.name}")
print(f"Index dtype: {combined_data.index.dtype}")
try:
    print(f"Index start: {combined_data.index.min()}  end: {combined_data.index.max()}")
except Exception as e:
    print(f"Index min/max error: {e}")
print(f"Rows: {len(combined_data)}  Cols: {combined_data.shape[1]}")

# Dtypes distribution
_dtype_counts = combined_data.dtypes.value_counts()
print("Column dtypes distribution:")
for _dt, _cnt in _dtype_counts.items():
    print(f"  {_dt}: {_cnt}")

# Non-numeric columns overview
_non_numeric_cols = []
for _col in combined_data.columns:
    if not pd.api.types.is_numeric_dtype(combined_data[_col]):
        _non_numeric_cols.append(_col)
print(f"Non-numeric columns in combined_data (count {len(_non_numeric_cols)}):")
print(_non_numeric_cols[:25])
if len(_non_numeric_cols) > 25:
    print(f"... and {len(_non_numeric_cols)-25} more")
print(f"="*120)
print("DATA LOADED")
print(f"="*120)



# Run OLS regression for correlation validation
gspc_col = "GSPC_log_return_next_period"

if gspc_col not in combined_data.columns:
    print("Error: Target variable not found")
else:
    y = combined_data[gspc_col]
    X = combined_data.drop(columns=[gspc_col])
    print("-"*120)
    print("TARGET/FEATURES DTYPES DEBUG")
    print(f"Target dtype: {y.dtype}")
    print(f"Y: {y.head()}")
    print(f"Y: {y.tail()}")
    print(f"X: {X.head()}")
    print(f"X: {X.tail()}")
    print(f"X shape: {X.shape}")
    _x_non_numeric = []
    for _c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[_c]):
            _x_non_numeric.append(_c)
    print(f"Non-numeric columns in X (count {len(_x_non_numeric)}):")
    print(_x_non_numeric[:25])
    if len(_x_non_numeric) > 25:
        print(f"... and {len(_x_non_numeric)-25} more")
    # All-NaN columns
    _all_nan_cols = []
    for _c in X.columns:
        if X[_c].notna().sum() == 0:
            _all_nan_cols.append(_c)
    if len(_all_nan_cols) > 0:
        print(f"Columns with all NaN in X (count {len(_all_nan_cols)}):")
        print(_all_nan_cols[:25])
    # Columns with infinite values (check numeric only)
    _inf_cols = []
    for _c in X.columns:
        if pd.api.types.is_numeric_dtype(X[_c]):
            _vals = X[_c].to_numpy()
            if np.isinf(_vals).any():
                _inf_cols.append(_c)
    if len(_inf_cols) > 0:
        print(f"Columns with +/-inf in X (count {len(_inf_cols)}):")
        print(_inf_cols[:25])
    
    # Use boolean masks to assess validity, but filter original numeric data
    y_mask = y.notna()
    X_mask = X.notna()
    print("-"*120)
    print("MASKS DEBUG")
    print(f"y_mask dtype: {y_mask.dtype}  true_count: {int(y_mask.sum())}")
    # X_mask is boolean matrix
    _unique_mask_dtypes = set()
    for _c in X_mask.columns:
        _unique_mask_dtypes.add(str(X_mask[_c].dtype))
        if len(_unique_mask_dtypes) > 1:
            break
    print(f"X_mask dtype(s): {list(_unique_mask_dtypes)}  shape: {X_mask.shape}")
    
    # Check feature completeness (require 80% of features)
    feature_completeness = X_mask.sum(axis=1) / X_mask.shape[1]
    sufficient_rows = y_mask & (feature_completeness >= 0.8)
    print(f"Feature completeness: min {feature_completeness.min():.3f}  max {feature_completeness.max():.3f}")
    print(f"Rows passing completeness >= 0.8: {int(sufficient_rows.sum())}")
    
    if sufficient_rows.sum() < 50:
        print("Insufficient data for OLS regression")
    else:
        # Filter original numeric data, then clean
        y_final = y.loc[sufficient_rows]
        X_final = X.loc[sufficient_rows]
        print("-"*120)
        print("POST-FILTER DEBUG")
        print(f"y_final dtype: {y_final.dtype}  len: {len(y_final)}  missing: {int(y_final.isna().sum())}")
        # Clean features
        X_final = X_final.replace([np.inf, -np.inf], np.nan).ffill()
        _x_final_obj_cols = []
        for _c in X_final.columns:
            if X_final[_c].dtype == 'object':
                _x_final_obj_cols.append(_c)
        if len(_x_final_obj_cols) > 0:
            print(f"WARNING: object dtype columns remain in X_final: {len(_x_final_obj_cols)}")
            print(_x_final_obj_cols[:25])
        
        
        # =================================================================
        # FEATURE PURGING - SELECT TOP 50 MOST SIGNIFICANT FEATURES
        # =================================================================
        print(f"\n{'='*80}")
        print("FEATURE PURGING - SELECTING TOP MOST SIGNIFICANT FEATURES")
        print(f"{'='*80}")
        print(f"Initial features: {X_final.shape[1]}")
        
        # Run initial OLS regression to get feature significance
        # Keep a copy before any subsetting for diagnostics
        X_candidates = X_final.copy()
        X_with_const_initial = sm.add_constant(X_candidates)
        print("-"*120)
        print("X_with_const_initial DTYPES DEBUG")
        _dtype_counts_initial = X_with_const_initial.dtypes.value_counts()
        for _dt, _cnt in _dtype_counts_initial.items():
            print(f"  {_dt}: {_cnt}")
        _obj_cols_initial = []
        for _c in X_with_const_initial.columns:
            if X_with_const_initial[_c].dtype == 'object':
                _obj_cols_initial.append(_c)
        if len(_obj_cols_initial) > 0:
            print(f"Object dtype columns in X_with_const_initial: {len(_obj_cols_initial)}")
            print(_obj_cols_initial[:25])
        
        try:
            initial_ols = sm.OLS(y_final, X_with_const_initial).fit()
            
            # Extract feature significance (excluding constant)
            feature_pvalues = initial_ols.pvalues[1:]  # Exclude constant term
            feature_significance = [(feature, pval) for feature, pval in zip(X_candidates.columns, feature_pvalues)]
            
            # Sort by p-value (most significant first)
            feature_significance_sorted = sorted(feature_significance, key=lambda x: x[1])
            
            # Full ranking list and top-N lists for downstream steps
            ranked_features = [feature for feature, _ in feature_significance_sorted]
            top_features = ranked_features[:top_features_count]
            top80_count = min(top_diagnostics_count, len(ranked_features))
            top80_features = ranked_features[:top80_count]
            
            print(f"Selected top {top_features_count} features based on p-values:")
            for i, (feature, pval) in enumerate(feature_significance_sorted[:top_features_count]):
                significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                print(f"  {i+1:2d}. {feature:<35} p-val: {pval:.6f} {significance}")
            
            # -----------------------------------------------------------------
            # SKIP DEAD-WEIGHT (CORR/MI) → VIF-ONLY PRUNING
            # -----------------------------------------------------------------
            print("-"*120)
            print("Skipping dead-weight pruning (correlation/MI). Using VIF-only pruning.")
            top80_count = min(top_diagnostics_count, len(ranked_features))
            top80_features = ranked_features[:top80_count]
            
            # -----------------------------------------------------------------
            # TOP-80 DIAGNOSTICS: VIF, CONDITION NUMBER, CORRELATION SUMMARY
            # -----------------------------------------------------------------
            print("-"*120)
            print("TOP-80 DIAGNOSTICS (VIF / Condition Number / Correlation Summary)")
            X_top = X_candidates[top80_features].copy()
            # Safety cleaning
            X_top = X_top.replace([np.inf, -np.inf], np.nan)
            # Fill remaining NaNs with column means for diagnostics
            X_top = X_top.fillna(X_top.mean())
            # Standardize for diagnostics to make scales comparable
            _means_top = X_top.mean()
            _stds_top = X_top.std(ddof=0).replace(0, 1.0)
            X_top_std = (X_top - _means_top) / _stds_top

            # Condition number
            try:
                cond_number = float(np.linalg.cond(X_top_std.values))
            except Exception as e:
                cond_number = float('nan')
                print(f"Condition number error: {e}")
            print(f"Condition Number (top-{top80_count}, standardized): {cond_number:.2f}")

            # Correlation summary (as a heatmap-like report of strong pairs)
            corr_matrix = X_top_std.corr()
            strong_pairs = []
            _cols = list(corr_matrix.columns)
            for _i in range(len(_cols)):
                for _j in range(_i+1, len(_cols)):
                    _val = corr_matrix.iloc[_i, _j]
                    if np.isfinite(_val) and abs(_val) >= 0.95:
                        strong_pairs.append((_cols[_i], _cols[_j], float(_val)))
            print(f"Highly correlated pairs (|r| >= 0.95): {len(strong_pairs)}")
            for _name_a, _name_b, _r in strong_pairs[:30]:
                print(f"  {_name_a:<35} {_name_b:<35} r={_r: .3f}")
            if len(strong_pairs) > 30:
                print(f"  ... and {len(strong_pairs)-30} more")

            # VIF computation
            print("Computing VIF for top-80 features...")
            X_top_std_const = sm.add_constant(X_top_std)
            vif_list = []
            for _idx in range(1, X_top_std_const.shape[1]):
                _fname = X_top_std_const.columns[_idx]
                try:
                    _vif = float(variance_inflation_factor(X_top_std_const.values, _idx))
                except Exception:
                    _vif = float('inf')
                vif_list.append((_fname, _vif))
            vif_list_sorted = sorted(vif_list, key=lambda x: x[1], reverse=True)
            print("Top 15 VIFs (descending):")
            for _fname, _v in vif_list_sorted[:15]:
                print(f"  {_fname:<35} VIF: {_v:>8.2f}")

            # -----------------------------------------------------------------
            # PRUNE REDUNDANCIES from top-80 using correlation and VIF
            # -----------------------------------------------------------------
            print("-"*120)
            print("Pruning redundancies from top-80 (VIF threshold only)...")
            _filters_cfg = variables_config.get('config', {}).get('filters', {}) if isinstance(variables_config.get('config', {}), dict) else {}
            vif_threshold = float(_filters_cfg.get('vif_threshold', 15.0))
            print(f"Threshold → VIF <= {vif_threshold:.1f}")
            feature_rank_index = {f: i for i, f in enumerate(ranked_features)}
            pruned_features = list(top80_features)
            drop_reason = {}

            # VIF-based pruning (iterative)
            max_iter = max(50, len(pruned_features) * 2)
            for _iter in range(max_iter):
                if len(pruned_features) <= 2:
                    break
                _X_iter = X_top_std[pruned_features]
                _X_iter_const = sm.add_constant(_X_iter)
                _vifs_iter = []
                for _idx in range(1, _X_iter_const.shape[1]):
                    _fname = _X_iter_const.columns[_idx]
                    try:
                        _v = float(variance_inflation_factor(_X_iter_const.values, _idx))
                    except Exception:
                        _v = float('inf')
                    _vifs_iter.append((_fname, _v))
                # First, drop any non-finite VIFs (inf/NaN) – choose least significant by initial rank
                _non_finite = [(_f, _v) for (_f, _v) in _vifs_iter if not np.isfinite(_v)]
                if _non_finite:
                    _non_finite_sorted = sorted(_non_finite, key=lambda x: feature_rank_index.get(x[0], 10**9), reverse=True)
                    _fname_drop, _vdrop = _non_finite_sorted[0]
                    if _fname_drop in pruned_features:
                        pruned_features.remove(_fname_drop)
                        drop_reason[_fname_drop] = "vif_nonfinite"
                    continue

                # Otherwise, drop the highest finite VIF above threshold
                _vifs_iter_sorted = sorted(_vifs_iter, key=lambda x: x[1], reverse=True)
                _fname_max, _vmax = _vifs_iter_sorted[0]
                if _vmax <= vif_threshold:
                    break
                pruned_features.remove(_fname_max)
                drop_reason[_fname_max] = f"high_vif({_vmax:.1f})"
            print(f"Pruned features count: {len(top80_features) - len(pruned_features)}")
            if drop_reason:
                _preview = list(drop_reason.items())[:20]
                print("Examples of drops:")
                for _f, _why in _preview:
                    print(f"  {_f:<35} -> {_why}")
                if len(drop_reason) > 20:
                    print(f"  ... and {len(drop_reason)-20} more")

            # Choose final selected features from pruned set preserving initial significance order
            selected_features = [f for f in ranked_features if f in pruned_features][:top_features_count]

            # Update X_final to only include final selected features (post-pruning)
            X_final = X_candidates[selected_features]
            print(f"\nFeatures reduced from {X_with_const_initial.shape[1]-1} to {X_final.shape[1]} after pruning")
            
            
        except Exception as e:
            print(f"Error in feature purging: {e}")
            print("Proceeding with all features...")
        
        # Create final design matrix with constant
        # Standardize X for coefficient comparability and provide a scale report
        print("-"*120)
        print("FEATURE SCALE REPORT (pre-standardization)")
        _scale_rows = []
        for _c in X_final.columns:
            _s = {
                'feature': _c,
                'mean': float(np.nanmean(X_final[_c].values)),
                'std': float(np.nanstd(X_final[_c].values)),
                'min': float(np.nanmin(X_final[_c].values)),
                'max': float(np.nanmax(X_final[_c].values))
            }
            _scale_rows.append(_s)
        # Print concise report
        print(f"Total features: {len(_scale_rows)}")
        for _s in _scale_rows[:20]:
            print(f"  {_s['feature']:<35} mean={_s['mean']:> .4f} std={_s['std']:> .4f} min={_s['min']:> .4f} max={_s['max']:> .4f}")
        if len(_scale_rows) > 20:
            print(f"  ... and {len(_scale_rows)-20} more")

        _stds_final = X_final.std(ddof=0).replace(0, 1.0)
        _means_final = X_final.mean()
        X_final_std = (X_final - _means_final) / _stds_final

        X_with_const = sm.add_constant(X_final_std)
        print("-"*120)
        print("FINAL DESIGN MATRIX DTYPES DEBUG")
        _dtype_counts_final = X_with_const.dtypes.value_counts()
        for _dt, _cnt in _dtype_counts_final.items():
            print(f"  {_dt}: {_cnt}")
        _obj_cols_final = []
        for _c in X_with_const.columns:
            if X_with_const[_c].dtype == 'object':
                _obj_cols_final.append(_c)
        if len(_obj_cols_final) > 0:
            print(f"Object dtype columns in X_with_const: {len(_obj_cols_final)}")
            print(_obj_cols_final[:25])
        
        try:
            ols_result = sm.OLS(y_final, X_with_const).fit()
            
            print(f"\n{'='*80}")
            print(f"FINAL OLS REGRESSION RESULTS (TOP {top_features_count} FEATURES)")
            print(f"{'='*80}")
            print(f"Sample size: {len(y_final)} observations")
            print(f"Features: {X_final.shape[1]}")
            print(f"R-squared: {ols_result.rsquared:.4f}")
            print(f"Adjusted R-squared: {ols_result.rsquared_adj:.4f}")
            print(f"F-statistic p-value: {ols_result.f_pvalue:.4f}")
            
            # Show all variables sorted by significance
            var_names = ['const'] + list(X_final.columns)
            var_data = [(var, ols_result.params.iloc[i], ols_result.pvalues.iloc[i]) 
                       for i, var in enumerate(var_names)]
            var_data_sorted = sorted(var_data, key=lambda x: x[2])
            
            print(f"\nAll Variables Sorted by Significance ({len(var_data_sorted)} total):")
            print(f"{'Variable':<40} {'Coefficient':<15} {'P-value':<15} {'Significance':<15}")
            print("-" * 85)
            for var, coef, p_val in var_data_sorted:
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"{var:<40} {coef:<15.6f} {p_val:<15.4f} {significance:<15}")
            
            # =================================================================
            # FIT DIAGNOSTICS
            # =================================================================
            print(f"\n{'='*80}")
            print("FIT DIAGNOSTICS")
            print(f"{'='*80}")
            
            print(f"AIC: {ols_result.aic:.2f}")
            print(f"BIC: {ols_result.bic:.2f}")
            print(f"Log-likelihood: {ols_result.llf:.2f}")
            print(f"Durbin-Watson: {sm.stats.durbin_watson(ols_result.resid):.4f}")
            
            # Rolling R-squared (if enough data)
            # Enforce window >= k + 10, where k = number of predictors (ex-constant)
            k_predictors = X_with_const.shape[1] - 1
            default_window = 36
            window_size = max(default_window, k_predictors + 10)
            if len(y_final) >= window_size:
                rolling_r2 = []
                rolling_dates = []
                print(f"\nROLLING R-SQUARED ({window_size}-period windows; k={k_predictors})")
                
                
                for i in range(window_size, len(y_final)):
                    y_window = y_final.iloc[i-window_size:i]
                    X_window = X_with_const.iloc[i-window_size:i]
                    
                    try:
                        # Skip windows with insufficient residual df
                        df_resid = len(y_window) - X_window.shape[1]
                        if df_resid < 5:
                            rolling_r2.append(np.nan)
                            rolling_dates.append(y_final.index[i])
                            continue
                        temp_model = sm.OLS(y_window, X_window).fit()
                        rolling_r2.append(temp_model.rsquared)
                        rolling_dates.append(y_final.index[i])
                    except:
                        rolling_r2.append(np.nan)
                        rolling_dates.append(y_final.index[i])
                
                rolling_r2_series = pd.Series(rolling_r2, index=rolling_dates)
                print(f"  Mean Rolling R²: {np.nanmean(rolling_r2):.4f}")
                print(f"  Std Rolling R²: {np.nanstd(rolling_r2):.4f}")
                print(f"  Min Rolling R²: {np.nanmin(rolling_r2):.4f}")
                print(f"  Max Rolling R²: {np.nanmax(rolling_r2):.4f}")
                print(f"  Latest Rolling R²: {rolling_r2_series.iloc[-1]:.4f}")
            
            # =================================================================
            # STABILITY DIAGNOSTICS
            # =================================================================
            print(f"\n{'='*80}")
            print("STABILITY DIAGNOSTICS")
            print(f"{'='*80}")
            
            # Subperiod analysis
            n_periods = 3
            period_size = len(y_final) // n_periods
            
            print(f"SUBPERIOD ANALYSIS ({n_periods} periods):")
            subperiod_stats = []
            
            for i in range(n_periods):
                start_idx = i * period_size
                end_idx = (i + 1) * period_size if i < n_periods - 1 else len(y_final)
                
                y_sub = y_final.iloc[start_idx:end_idx]
                X_sub = X_with_const.iloc[start_idx:end_idx]
                
                try:
                    sub_model = sm.OLS(y_sub, X_sub).fit()
                    period_start = y_final.index[start_idx].strftime('%Y-%m')
                    period_end = y_final.index[end_idx-1].strftime('%Y-%m')
                    
                    print(f"  Period {i+1} ({period_start} to {period_end}):")
                    print(f"    R²: {sub_model.rsquared:.4f}")
                    print(f"    Adj R²: {sub_model.rsquared_adj:.4f}")
                    print(f"    F-stat p-val: {sub_model.f_pvalue:.4f}")
                    
                    subperiod_stats.append({
                        'period': i+1,
                        'r2': sub_model.rsquared,
                        'adj_r2': sub_model.rsquared_adj,
                        'f_pvalue': sub_model.f_pvalue
                    })
                    
                except Exception as e:
                    print(f"  Period {i+1}: Error - {e}")
            
            if subperiod_stats:
                r2_stability = np.std([s['r2'] for s in subperiod_stats])
                print(f"  R² Stability (std dev): {r2_stability:.4f}")
                
        except Exception as e:
            print(f"Error running OLS regression: {e}")

