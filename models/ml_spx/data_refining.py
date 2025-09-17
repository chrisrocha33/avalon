import pandas as pd
import numpy as np
import statsmodels.api as sm
from sqlalchemy import create_engine
import json
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_data_from_database(engine):
    """Load z-scores data from database with manifest support"""
    print(f"="*120)
    print("LOADING Z-SCORES DATA")
    print(f"="*120)
    
    combined_data = pd.DataFrame()
    _zs_manifest_name = 'ml_spx_data_manifest'
    _zs_parts_loaded = 0
    
    try:
        _manifest = pd.read_sql_table(_zs_manifest_name, con=engine)
        if not _manifest.empty and 'table_name' in _manifest.columns:
            print(f"Found manifest {_zs_manifest_name} with {len(_manifest)} parts")
            _parts = list(_manifest['table_name'])
            _assembled = []
            for _t in _parts:
                try:
                    _dfp = pd.read_sql_table(_t, con=engine)
                    if 'index' in _dfp.columns:
                        _dfp['index'] = pd.to_datetime(_dfp['index'])
                        _dfp = _dfp.set_index('index')
                    _assembled.append(_dfp)
                    _zs_parts_loaded += 1
                    print(f"  ✓ Loaded {_t} with {len(_dfp.columns)} columns")
                except Exception as _e_p:
                    print(f"  ❌ Failed to load part {_t}: {_e_p}")
            if len(_assembled) > 0:
                combined_data = pd.concat(_assembled, axis=1)
        else:
            print(f"Manifest {_zs_manifest_name} is empty or missing 'table_name' column; falling back")
    except Exception as _e_m:
        print(f"Manifest not found or unreadable ({_zs_manifest_name}): {_e_m}. Falling back to single table...")
    
    if combined_data.empty:
        try:
            _fallback = pd.read_sql_table("ml_spx_zscores", con=engine)
            if 'index' in _fallback.columns:
                _fallback['index'] = pd.to_datetime(_fallback['index'])
                _fallback = _fallback.set_index('index')
            combined_data = _fallback
            print("Loaded fallback table ml_spx_zscores")
        except Exception as _e_f:
            print(f"❌ Could not load z-scores (parts or fallback): {_e_f}")
            raise
    
    print(f"Z-Scores Columns: {len(combined_data.columns)} (from {_zs_parts_loaded} parts if chunked)")
    print(f"Z-Scores Rows: {len(combined_data)}")
    return combined_data

def load_target_variable(engine, combined_data, target_col="GSPC_log_return_next_period"):
    """Load target variable from separate table or combined data"""
    print(f"="*120)
    print("Loading Target Variable...")
    print(f"="*120)
    
    _target_loaded = False
    
    try:
        _tgt = pd.read_sql_table('ml_spx_target', con=engine)
        if 'index' in _tgt.columns:
            _tgt['index'] = pd.to_datetime(_tgt['index'])
            _tgt = _tgt.set_index('index')
        if target_col in _tgt.columns:
            combined_data = combined_data.join(_tgt[[target_col]], how='left')
            _target_loaded = True
            print(f"Successfully added {target_col} from ml_spx_target")
    except Exception as _e_t:
        print(f"Warning: failed to read ml_spx_target: {_e_t}")
    
    if not _target_loaded:
        if target_col in combined_data.columns:
            print(f"Using {target_col} from combined_data")
        else:
            print(f"Warning: {target_col} column not found in any source")
    
    print(f"="*120)
    print("Target Variable Loading Complete")
    print(f"="*120)
    
    return combined_data

def comprehensive_nan_analysis(df, stage_name):
    """Comprehensive NaN analysis and reporting"""
    print(f"="*120)
    print(f"COMPREHENSIVE NaN ANALYSIS - {stage_name.upper()}")
    print(f"="*120)
    
    # Basic data shape and structure
    print(f"Number of columns: {len(df.columns)}")
    print(f"Number of rows: {len(df)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total number of data points: {df.size}")
    
    # Data types overview
    print(f"\nData types distribution:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # NaN analysis
    total_nans = df.isna().sum().sum()
    print(f"\nTotal number of NaNs: {total_nans}")
    print(f"Percentage of NaNs: {(total_nans / df.size) * 100:.2f}%")
    
    # Columns with NaNs
    columns_with_nans = df.columns[df.isna().any()].tolist()
    print(f"\nColumns with NaNs ({len(columns_with_nans)} total):")
    for col in columns_with_nans:
        nan_count = df[col].isna().sum()
        nan_pct = (nan_count / len(df)) * 100
        print(f"  {col}: {nan_count} NaNs ({nan_pct:.2f}%)")
    
    # Rows with NaNs
    rows_with_nans = df[df.isna().any(axis=1)]
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
    target_col = "GSPC_log_return_next_period"
    if target_col in df.columns:
        print(f"\nTarget variable ({target_col}) analysis:")
        target_nans = df[target_col].isna().sum()
        target_valid = df[target_col].notna().sum()
        print(f"  Valid observations: {target_valid}")
        print(f"  NaN observations: {target_nans}")
        print(f"  NaN percentage: {(target_nans / len(df)) * 100:.2f}%")
        if target_valid > 0:
            print(f"  Mean: {df[target_col].mean():.6f}")
            print(f"  Std: {df[target_col].std():.6f}")
            print(f"  Min: {df[target_col].min():.6f}")
            print(f"  Max: {df[target_col].max():.6f}")
    else:
        print(f"\nTarget variable ({target_col}) not found in data")
    
    # Feature completeness analysis
    print(f"\nFeature completeness analysis:")
    feature_completeness = df.notna().sum(axis=1) / len(df.columns)
    print(f"  Mean completeness: {feature_completeness.mean():.3f}")
    print(f"  Min completeness: {feature_completeness.min():.3f}")
    print(f"  Max completeness: {feature_completeness.max():.3f}")
    print(f"  Rows with 100% completeness: {(feature_completeness == 1.0).sum()}")
    print(f"  Rows with >=80% completeness: {(feature_completeness >= 0.8).sum()}")
    print(f"  Rows with >=50% completeness: {(feature_completeness >= 0.5).sum()}")
    
    # Infinite values check
    inf_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if np.isinf(df[col]).any():
                inf_count = np.isinf(df[col]).sum()
                inf_cols.append((col, inf_count))
    
    if inf_cols:
        print(f"\nColumns with infinite values ({len(inf_cols)} total):")
        for col, count in inf_cols:
            print(f"  {col}: {count} infinite values")
    else:
        print(f"\nNo infinite values found in any columns")
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
    print(f"\nMemory usage: {memory_usage:.2f} MB")
    
    print(f"="*120)
    print(f"END OF {stage_name.upper()} NaN ANALYSIS")
    print(f"="*120)

def prepare_data_for_pruning(combined_data, target_col="GSPC_log_return_next_period"):
    """Prepare data for feature pruning with comprehensive validation"""
    print(f"="*120)
    print("PREPARING DATA FOR FEATURE PRUNING")
    print(f"="*120)
    
    if target_col not in combined_data.columns:
        print(f"Error: Target variable {target_col} not found")
        return None, None, None
    
    y = combined_data[target_col]
    X = combined_data.drop(columns=[target_col])
    
    print("-"*120)
    print("TARGET/FEATURES DTYPES DEBUG")
    print(f"Target dtype: {y.dtype}")
    print(f"Y shape: {y.shape}")
    print(f"X shape: {X.shape}")
    
    # Check for non-numeric columns
    _x_non_numeric = []
    for _c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[_c]):
            _x_non_numeric.append(_c)
    print(f"Non-numeric columns in X (count {len(_x_non_numeric)}):")
    if len(_x_non_numeric) > 0:
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
    
    # Columns with infinite values
    _inf_cols = []
    for _c in X.columns:
        if pd.api.types.is_numeric_dtype(X[_c]):
            _vals = X[_c].to_numpy()
            if np.isinf(_vals).any():
                _inf_cols.append(_c)
    if len(_inf_cols) > 0:
        print(f"Columns with +/-inf in X (count {len(_inf_cols)}):")
        print(_inf_cols[:25])
    
    # Use boolean masks to assess validity
    y_mask = y.notna()
    X_mask = X.notna()
    print("-"*120)
    print("MASKS DEBUG")
    print(f"y_mask dtype: {y_mask.dtype}  true_count: {int(y_mask.sum())}")
    print(f"X_mask shape: {X_mask.shape}")
    
    # Check feature completeness (require 80% of features)
    feature_completeness = X_mask.sum(axis=1) / X_mask.shape[1]
    sufficient_rows = y_mask & (feature_completeness >= 0.8)
    print(f"Feature completeness: min {feature_completeness.min():.3f}  max {feature_completeness.max():.3f}")
    print(f"Rows passing completeness >= 0.8: {int(sufficient_rows.sum())}")
    
    if sufficient_rows.sum() < 50:
        print("Insufficient data for feature pruning")
        return None, None, None
    
    # Filter data
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
    
    return y_final, X_final, sufficient_rows

def feature_pruning_ols_based(y_final, X_final, top_features_count=50, drop_bottom_count=0):
    """Perform OLS-based feature pruning and selection"""
    print(f"\n{'='*80}")
    print("FEATURE PURGING - SELECTING TOP MOST SIGNIFICANT FEATURES")
    print(f"{'='*80}")
    print(f"Initial features: {X_final.shape[1]}")
    
    # Run initial OLS regression to get feature significance
    X_candidates = X_final.copy()
    X_with_const_initial = sm.add_constant(X_candidates)
    
    print("-"*120)
    print("X_with_const_initial DTYPES DEBUG")
    _dtype_counts_initial = X_with_const_initial.dtypes.value_counts()
    for _dt, _cnt in _dtype_counts_initial.items():
        print(f"  {_dt}: {_cnt}")
    
    try:
        initial_ols = sm.OLS(y_final, X_with_const_initial).fit()
        
        # Extract feature significance (excluding constant)
        feature_pvalues = initial_ols.pvalues[1:]  # Exclude constant term
        feature_significance = [(feature, pval) for feature, pval in zip(X_candidates.columns, feature_pvalues)]
        
        # Sort by p-value (most significant first)
        feature_significance_sorted = sorted(feature_significance, key=lambda x: x[1])
        
        # Full ranking list - drop bottom N features instead of taking top N
        ranked_features = [feature for feature, _ in feature_significance_sorted]
        
        if drop_bottom_count > 0:
            # Drop bottom N features (least significant)
            features_after_drop = ranked_features[:-drop_bottom_count] if drop_bottom_count < len(ranked_features) else []
            print(f"Dropped {min(drop_bottom_count, len(ranked_features))} least significant features")
            print(f"Remaining features: {len(features_after_drop)}")
        else:
            # Fallback to top N approach
            features_after_drop = ranked_features[:top_features_count]
            print(f"Using top {top_features_count} features")
        
        top_features = features_after_drop
        top80_count = min(50, len(features_after_drop))  # For diagnostics
        top80_features = features_after_drop[:top80_count]
        
        print(f"Selected features based on p-values:")
        display_count = min(20, len(features_after_drop))
        for i, (feature, pval) in enumerate(feature_significance_sorted[:display_count]):
            significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  {i+1:2d}. {feature:<35} p-val: {pval:.6f} {significance}")
        if len(features_after_drop) > display_count:
            print(f"  ... and {len(features_after_drop) - display_count} more features")
        
        return ranked_features, top80_features, top_features
        
    except Exception as e:
        print(f"Error in OLS-based feature pruning: {e}")
        return list(X_final.columns), list(X_final.columns), list(X_final.columns)

def correlation_pruning(X_candidates, ranked_features, top80_features, cluster_corr_threshold=0.90):
    """Perform correlation-based pruning"""
    print("-"*120)
    print("CORRELATION-BASED PRUNING")
    print("-"*120)
    
    # Build correlation matrix for pool (standardized for scale invariance)
    X_pool_std = X_candidates[top80_features].copy()
    X_pool_std = X_pool_std.replace([np.inf, -np.inf], np.nan).fillna(X_pool_std.mean())
    
    # Standardize for diagnostics
    _means_pool = X_pool_std.mean()
    _stds_pool = X_pool_std.std(ddof=0).replace(0, 1.0)
    X_pool_std = (X_pool_std - _means_pool) / _stds_pool
    
    corr_pool = X_pool_std.corr().abs()
    
    # Find highly correlated pairs
    strong_pairs = []
    _cols = list(corr_pool.columns)
    for _i in range(len(_cols)):
        for _j in range(_i+1, len(_cols)):
            _val = corr_pool.iloc[_i, _j]
            if np.isfinite(_val) and abs(_val) >= cluster_corr_threshold:
                strong_pairs.append((_cols[_i], _cols[_j], float(_val)))
    
    print(f"Highly correlated pairs (|r| >= {cluster_corr_threshold}): {len(strong_pairs)}")
    for _name_a, _name_b, _r in strong_pairs[:30]:
        print(f"  {_name_a:<35} {_name_b:<35} r={_r: .3f}")
    if len(strong_pairs) > 30:
        print(f"  ... and {len(strong_pairs)-30} more")
    
    # Drop one of each highly correlated pair (keep more significant)
    feature_rank_index_corr = {f: i for i, f in enumerate(ranked_features)}
    _corr_drop_set = set()
    for _a, _b, _r in strong_pairs:
        if (_a in top80_features) and (_b in top80_features):
            _rank_a = feature_rank_index_corr.get(_a, 10**9)
            _rank_b = feature_rank_index_corr.get(_b, 10**9)
            # Drop the less significant (higher rank index)
            if _rank_a <= _rank_b:
                _corr_drop_set.add(_b)
            else:
                _corr_drop_set.add(_a)
    
    corr_pruned_features = list(_corr_drop_set)
    if len(corr_pruned_features) > 0:
        print("Applying correlation-based pruning:")
        print(f"  Dropping {len(corr_pruned_features)} features (keeping the more significant in each pair)")
        for _f in corr_pruned_features[:20]:
            print(f"    drop: {_f}")
        if len(corr_pruned_features) > 20:
            print(f"    ... and {len(corr_pruned_features)-20} more")
    
    return corr_pruned_features

def vif_analysis(X_candidates, top80_features, top80_count=50):
    """Perform VIF analysis on top features"""
    print("-"*120)
    print(f"VIF ANALYSIS FOR TOP-{top80_count} FEATURES")
    print("-"*120)
    
    X_top = X_candidates[top80_features].copy()
    X_top = X_top.replace([np.inf, -np.inf], np.nan).fillna(X_top.mean())
    
    # Standardize for VIF computation
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
    
    # VIF computation
    print(f"Computing VIF for top-{top80_count} features...")
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
    
    return vif_list_sorted, cond_number

def greedy_forward_selection(y_final, X_candidates, base_feature_pool, ranked_features, 
                           max_features=50, greedy_corr_guard=0.95, cond_number_cap=80.0, 
                           adjr2_epsilon=0.001):
    """Perform greedy forward selection with guardrails"""
    print("-"*120)
    print("GREEDY FORWARD SELECTION")
    print("-"*120)
    
    # Build correlation matrix for pool
    X_pool_std = X_candidates[base_feature_pool].copy()
    X_pool_std = X_pool_std.replace([np.inf, -np.inf], np.nan).fillna(X_pool_std.mean())
    
    # Standardize
    _means_pool = X_pool_std.mean()
    _stds_pool = X_pool_std.std(ddof=0).replace(0, 1.0)
    X_pool_std = (X_pool_std - _means_pool) / _stds_pool
    
    corr_pool = X_pool_std.corr()
    
    # Start with most significant features
    selected = []
    current_adjr2 = -1e9
    
    # Add initial features one by one
    for feat in ranked_features[:min(10, len(ranked_features))]:
        if feat not in base_feature_pool:
            continue
        
        try:
            test_model = sm.OLS(y_final, sm.add_constant(X_candidates[selected + [feat]])).fit()
            test_adjr2 = float(test_model.rsquared_adj)
            if test_adjr2 > current_adjr2:
                selected.append(feat)
                current_adjr2 = test_adjr2
                print(f"  Added initial feature '{feat}'  AdjR²={current_adjr2:.4f}")
        except Exception:
            continue
    
    # Greedy forward selection
    remaining = [f for f in base_feature_pool if f not in selected]
    improved = True
    
    while improved and len(selected) < max_features and len(remaining) > 0:
        improved = False
        best_feat = None
        best_gain = 0.0
        best_adjr2_next = current_adjr2
        
        for feat in remaining:
            # Pairwise correlation guard
            violate_corr = False
            for sf in selected:
                try:
                    if float(corr_pool.loc[feat, sf]) >= greedy_corr_guard:
                        violate_corr = True
                        break
                except Exception:
                    continue
            if violate_corr:
                continue
            
            # Condition number guard
            try:
                X_std_tmp = sm.add_constant(X_pool_std[selected + [feat]])
                cond_number = float(np.linalg.cond(X_std_tmp.values))
                if cond_number > cond_number_cap:
                    continue
            except Exception:
                continue
            
            # Evaluate adjusted R^2 gain
            try:
                model_tmp = sm.OLS(y_final, sm.add_constant(X_candidates[selected + [feat]])).fit()
                adjr2_next = float(model_tmp.rsquared_adj)
                gain = adjr2_next - current_adjr2
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_adjr2_next = adjr2_next
            except Exception:
                continue
        
        if best_feat is not None and best_gain >= adjr2_epsilon:
            selected.append(best_feat)
            remaining.remove(best_feat)
            current_adjr2 = best_adjr2_next
            improved = True
            print(f"  Added '{best_feat}'  ΔAdjR²={best_gain:.4f}  total={len(selected)}  AdjR²={current_adjr2:.4f}")
        else:
            break
    
    print(f"Greedy forward selection complete. Selected features: {len(selected)}  AdjR²={current_adjr2:.4f}")
    return selected

def save_refined_data(X_refined, y_final, engine, table_prefix="ml_spx_refined"):
    """Save refined data to database"""
    print(f"\n{'='*80}")
    print("SAVING REFINED DATA TO DATABASE")
    print(f"{'='*80}")
    
    try:
        # Save features
        X_refined.to_sql(f"{table_prefix}_features", engine, if_exists='replace', index=True, method='multi', chunksize=1000)
        print(f"✅ Saved refined features to {table_prefix}_features")
        
        # Save target
        y_final.to_frame('GSPC_log_return_next_period').to_sql(f"{table_prefix}_target", engine, if_exists='replace', index=True, method='multi', chunksize=1000)
        print(f"✅ Saved refined target to {table_prefix}_target")
        
        # Save feature list
        feature_list_df = pd.DataFrame({
            'feature_name': X_refined.columns,
            'feature_index': range(len(X_refined.columns))
        })
        feature_list_df.to_sql(f"{table_prefix}_feature_list", engine, if_exists='replace', index=False)
        print(f"✅ Saved feature list to {table_prefix}_feature_list")
        
    except Exception as e:
        print(f"❌ Error saving refined data: {e}")

def main():
    """Main function to run data refining pipeline"""
    print(f"="*120)
    print("DATA REFINING PIPELINE")
    print(f"="*120)
    
    # Load configuration
    try:
        with open('variables.json', 'r') as f:
            variables_config = json.load(f)
        print("Variables configuration loaded")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading variables.json: {e}")
        return
    
    # Get configuration parameters
    top_features_count = variables_config.get('config', {}).get('top_features', 50)
    drop_bottom_count = variables_config.get('config', {}).get('drop_bottom_features', 0)
    
    # Database connection
    print("Connecting to database...")
    
    # Use centralized config
    from config import Config
    engine = create_engine(Config.DATABASE['connection_string'], future=True)
    print("Database connection established")
    
    # Load data
    combined_data = load_data_from_database(engine)
    combined_data = load_target_variable(engine, combined_data)
    
    # NaN analysis before refining
    comprehensive_nan_analysis(combined_data, "BEFORE DATA REFINING")
    
    # Decompress numeric columns
    try:
        _num_cols = []
        for _c in combined_data.columns:
            if pd.api.types.is_numeric_dtype(combined_data[_c]):
                _num_cols.append(_c)
        if len(_num_cols) > 0:
            combined_data[_num_cols] = combined_data[_num_cols].astype("float64")
            print(f"Decompression: upcasted {len(_num_cols)} numeric columns to float64")
    except Exception as _e_dec:
        print(f"Decompression warning: {_e_dec}")
    
    # Prepare data for pruning
    y_final, X_final, sufficient_rows = prepare_data_for_pruning(combined_data)
    
    if y_final is None:
        print("❌ Data preparation failed - insufficient data for pruning")
        return
    
    # NaN analysis after data preparation
    comprehensive_nan_analysis(pd.concat([y_final.to_frame(), X_final], axis=1), "AFTER DATA PREPARATION")
    
    # OLS-based feature pruning
    ranked_features, top80_features, top_features = feature_pruning_ols_based(
        y_final, X_final, top_features_count, drop_bottom_count
    )
    
    # Correlation-based pruning
    corr_pruned_features = correlation_pruning(X_final, ranked_features, top80_features)
    
    # VIF analysis
    vif_list, cond_number = vif_analysis(X_final, top80_features)
    
    # Greedy forward selection
    base_feature_pool = [f for f in top80_features if f not in corr_pruned_features]
    selected_features = greedy_forward_selection(
        y_final, X_final, base_feature_pool, ranked_features, 
        max_features=top_features_count
    )
    
    # Create final refined dataset
    X_refined = X_final[selected_features]
    
    print(f"\n{'='*80}")
    print("REFINEMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Original features: {X_final.shape[1]}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Reduction: {X_final.shape[1] - len(selected_features)} features removed")
    print(f"Final dataset shape: {X_refined.shape}")
    
    # NaN analysis of final refined data
    comprehensive_nan_analysis(pd.concat([y_final.to_frame(), X_refined], axis=1), "FINAL REFINED DATA")
    
    # Save refined data
    save_refined_data(X_refined, y_final, engine)
    
    print(f"="*120)
    print("DATA REFINING PIPELINE COMPLETED")
    print(f"="*120)
    
    return X_refined, y_final

if __name__ == "__main__":
    main()
