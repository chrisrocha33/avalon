import pandas as pd
import numpy as np
import statsmodels.api as sm
from sqlalchemy import create_engine
import json
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

try:
    with open('variables.json', 'r') as f:
        variables_config = json.load(f)
    print("Variables configuration loaded")
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading variables.json: {e}")
    raise
top_features_count = variables_config.get('config', {}).get('top_features', 50)
drop_bottom_count = variables_config.get('config', {}).get('drop_bottom_features', 0)
top_diagnostics_count = top_features_count


print(f"="*120)
print("SQL Connection Starting...")

# Use centralized config
from config import Config
engine = create_engine(Config.DATABASE['connection_string'], future=True)

print(f"="*120)
print("SQL Connection Successful")
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
            # Align by index and concatenate columns
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
print(f"="*120)
print("DATA LOADED")
print(f"="*120)

# Define target variable column name
gspc_col = "GSPC_log_return_next_period"

# =============================================================================
# COMPREHENSIVE DATA DIAGNOSTICS FOR Z-SCORES DATA
# =============================================================================
print("="*120)
print("COMPREHENSIVE DATA DIAGNOSTICS - Z-SCORES DATA")
print("="*120)

# Basic data shape and structure
print(f"Number of columns: {len(combined_data.columns)}")
print(f"Number of rows: {len(combined_data)}")
print(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
print(f"Total number of data points: {combined_data.size}")

# Data types overview
print(f"\nData types distribution:")
dtype_counts = combined_data.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"  {dtype}: {count} columns")

# NaN analysis
total_nans = combined_data.isna().sum().sum()
print(f"\nTotal number of NaNs: {total_nans}")
print(f"Percentage of NaNs: {(total_nans / combined_data.size) * 100:.2f}%")

# Columns with NaNs
columns_with_nans = combined_data.columns[combined_data.isna().any()].tolist()
print(f"\nColumns with NaNs ({len(columns_with_nans)} total):")
for col in columns_with_nans:
    nan_count = combined_data[col].isna().sum()
    nan_pct = (nan_count / len(combined_data)) * 100
    print(f"  {col}: {nan_count} NaNs ({nan_pct:.2f}%)")

# Rows with NaNs
rows_with_nans = combined_data[combined_data.isna().any(axis=1)]
print(f"\nRows with NaNs ({len(rows_with_nans)} total):")
if len(rows_with_nans) > 0:
    print(f"First 10 rows with NaNs:")
    for i, (idx, row) in enumerate(rows_with_nans.head(10).iterrows()):
        nan_cols = row.isna()
        nan_col_names = nan_cols[nan_cols].index.tolist()
        print(f"  Row {i+1} ({idx}): {len(nan_col_names)} NaNs in columns: {nan_col_names[:5]}{'...' if len(nan_col_names) > 5 else ''}")
    
    if len(rows_with_nans) > 10:
        print(f"  ... and {len(rows_with_nans) - 10} more rows with NaNs")

# Target variable specific analysis (if present)
if gspc_col in combined_data.columns:
    print(f"\nTarget variable ({gspc_col}) analysis:")
    target_nans = combined_data[gspc_col].isna().sum()
    target_valid = combined_data[gspc_col].notna().sum()
    print(f"  Valid observations: {target_valid}")
    print(f"  NaN observations: {target_nans}")
    print(f"  NaN percentage: {(target_nans / len(combined_data)) * 100:.2f}%")
    if target_valid > 0:
        print(f"  Mean: {combined_data[gspc_col].mean():.6f}")
        print(f"  Std: {combined_data[gspc_col].std():.6f}")
        print(f"  Min: {combined_data[gspc_col].min():.6f}")
        print(f"  Max: {combined_data[gspc_col].max():.6f}")
else:
    print(f"\nTarget variable ({gspc_col}) not found in z-scores data")

# Feature completeness analysis
print(f"\nFeature completeness analysis:")
feature_completeness = combined_data.notna().sum(axis=1) / len(combined_data.columns)
print(f"  Mean completeness: {feature_completeness.mean():.3f}")
print(f"  Min completeness: {feature_completeness.min():.3f}")
print(f"  Max completeness: {feature_completeness.max():.3f}")
print(f"  Rows with 100% completeness: {(feature_completeness == 1.0).sum()}")
print(f"  Rows with >=80% completeness: {(feature_completeness >= 0.8).sum()}")
print(f"  Rows with >=50% completeness: {(feature_completeness >= 0.5).sum()}")

# Infinite values check
inf_cols = []
for col in combined_data.columns:
    if pd.api.types.is_numeric_dtype(combined_data[col]):
        if np.isinf(combined_data[col]).any():
            inf_count = np.isinf(combined_data[col]).sum()
            inf_cols.append((col, inf_count))

if inf_cols:
    print(f"\nColumns with infinite values ({len(inf_cols)} total):")
    for col, count in inf_cols:
        print(f"  {col}: {count} infinite values")
else:
    print(f"\nNo infinite values found in any columns")

# Memory usage
memory_usage = combined_data.memory_usage(deep=True).sum() / 1024**2  # MB
print(f"\nMemory usage: {memory_usage:.2f} MB")

# Sample of the data
print(f"\nFirst 5 rows of z-scores data:")
print(combined_data.head())

print(f"\nLast 5 rows of z-scores data:")
print(combined_data.tail())

# Column names overview
print(f"\nColumn names overview (first 20):")
for i, col in enumerate(combined_data.columns[:20]):
    print(f"  {i+1:2d}. {col}")
if len(combined_data.columns) > 20:
    print(f"  ... and {len(combined_data.columns) - 20} more columns")

print("="*120)
print("END OF Z-SCORES DATA DIAGNOSTICS")
print("="*120)

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



# Try to load target variable from separate table first, then fallback to combined_data
print(f"="*120)
print("Loading Target Variable...")
print(f"="*120)

_target_loaded = False

try:
    _tgt = pd.read_sql_table('ml_spx_target', con=engine)
    if 'index' in _tgt.columns:
        _tgt['index'] = pd.to_datetime(_tgt['index'])
        _tgt = _tgt.set_index('index')
    if gspc_col in _tgt.columns:
        combined_data = combined_data.join(_tgt[[gspc_col]], how='left')
        _target_loaded = True
        print(f"Successfully added {gspc_col} from ml_spx_target")
except Exception as _e_t:
    print(f"Warning: failed to read ml_spx_target: {_e_t}")

if not _target_loaded:
    if gspc_col in combined_data.columns:
        print(f"Using {gspc_col} from combined_data")
    else:
        print(f"Warning: {gspc_col} column not found in any source")

print(f"="*120)
print("Target Variable Loading Complete")
print(f"="*120)

# Run OLS regression for correlation validation
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
            top80_count = min(top_diagnostics_count, len(features_after_drop))
            top80_features = features_after_drop[:top80_count]
            
            print(f"Selected features based on p-values:")
            display_count = min(20, len(features_after_drop))
            for i, (feature, pval) in enumerate(feature_significance_sorted[:display_count]):
                significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                print(f"  {i+1:2d}. {feature:<35} p-val: {pval:.6f} {significance}")
            if len(features_after_drop) > display_count:
                print(f"  ... and {len(features_after_drop) - display_count} more features")
            
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
            print(f"TOP-{top_diagnostics_count} DIAGNOSTICS (VIF / Condition Number / Correlation Summary)")
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

            # After printing the correlation block, drop one of each highly correlated pair (not both)
            # Keep the more significant feature according to the initial ranking (lower index = more significant)
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
                print("Applying correlation-based pruning (|r| >= 0.95):")
                print(f"  Dropping {len(corr_pruned_features)} features (keeping the more significant in each pair)")
                for _f in corr_pruned_features[:20]:
                    print(f"    drop: {_f}")
                if len(corr_pruned_features) > 20:
                    print(f"    ... and {len(corr_pruned_features)-20} more")

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

            # -----------------------------------------------------------------
            # CORRELATION-CLUSTER REPRESENTATIVE SELECTION + GREEDY FORWARD
            # -----------------------------------------------------------------
            print("-"*120)
            print("Correlation-cluster representative selection + Greedy forward selection...")

            # Parameters
            cluster_corr_threshold = 0.90
            greedy_corr_guard = 0.95
            cond_number_cap = 80.0
            adjr2_epsilon = 0.001
            max_features = top_features_count

            # Start from features after earlier correlation pruning if present
            base_feature_pool = [f for f in top80_features if ('corr_pruned_features' not in locals()) or (f not in corr_pruned_features)]
            print(f"Feature pool after pairwise corr prune: {len(base_feature_pool)}")

            # Build correlation matrix for pool (standardized for scale invariance)
            X_pool_std = X_top_std[base_feature_pool].copy()
            corr_pool = X_pool_std.corr().abs()

            # Cluster by threshold (simple adjacency expansion)
            visited = set()
            clusters = []
            for col in base_feature_pool:
                if col in visited:
                    continue
                cluster = [col]
                visited.add(col)
                # collect strongly correlated neighbors
                for other in base_feature_pool:
                    if other in visited:
                        continue
                    if float(corr_pool.loc[col, other]) >= cluster_corr_threshold:
                        cluster.append(other)
                        visited.add(other)
                clusters.append(cluster)

            print(f"Clusters formed (|r| >= {cluster_corr_threshold:.2f}): {len(clusters)}")

            # Choose representative per cluster using univariate adjusted R^2
            X_pool_raw = X_candidates[base_feature_pool].copy()
            reps = []
            optional_candidates = []
            for cl in clusters:
                best_feat = None
                best_adjr2 = -1e9
                # rank cluster members by univariate adjusted R^2
                ranked = []
                for feat in cl:
                    try:
                        model_uni = sm.OLS(y_final, sm.add_constant(X_pool_raw[[feat]])).fit()
                        ranked.append((feat, float(model_uni.rsquared_adj)))
                    except Exception:
                        ranked.append((feat, -1e9))
                ranked.sort(key=lambda x: x[1], reverse=True)
                if len(ranked) > 0:
                    best_feat, best_adjr2 = ranked[0]
                    reps.append(best_feat)
                    # consider a second representative if it adds enough
                    if len(ranked) > 1:
                        for feat, _ in ranked[1:3]:  # only consider top 2 extras
                            optional_candidates.append(feat)

            print(f"Initial representatives selected: {len(reps)}")

            # Optionally add a second rep per cluster if it improves adj R^2 sufficiently
            selected = list(reps)
            if len(selected) > 0:
                try:
                    base_model = sm.OLS(y_final, sm.add_constant(X_pool_raw[selected])).fit()
                    current_adjr2 = float(base_model.rsquared_adj)
                except Exception:
                    current_adjr2 = -1e9
            else:
                current_adjr2 = -1e9

            for feat in optional_candidates:
                if feat in selected:
                    continue
                try:
                    cand_model = sm.OLS(y_final, sm.add_constant(X_pool_raw[selected + [feat]])).fit()
                    delta = float(cand_model.rsquared_adj) - current_adjr2
                    if delta >= adjr2_epsilon:
                        selected.append(feat)
                        current_adjr2 = float(cand_model.rsquared_adj)
                except Exception:
                    continue

            print(f"After optional reps, selected: {len(selected)}")

            # Greedy forward selection with guardrails
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

                    # Condition number guard (on standardized matrix)
                    try:
                        X_std_tmp = sm.add_constant(X_pool_std[selected + [feat]])
                        cond_number = float(np.linalg.cond(X_std_tmp.values))
                        if cond_number > cond_number_cap:
                            continue
                    except Exception:
                        continue

                    # Evaluate adjusted R^2 gain
                    try:
                        model_tmp = sm.OLS(y_final, sm.add_constant(X_pool_raw[selected + [feat]])).fit()
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

            # Choose final selected features (limit to top_features_count)
            selected_features = selected[:top_features_count]

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


        # =====================================================================
        # CLASSIFICATION EXPERIMENTS (time-based split; multiple classifiers)
        # =====================================================================
        try:
            print("\n" + "="*80)
            print("CLASSIFICATION EXPERIMENTS")
            print("="*80)

            # Prepare binary target: next-period up move (>0 -> 1 else 0)
            y_bin = (y_final > 0).astype(int)

            # Use X_final if available; else fallback to original cleaned features
            if 'X_final' in locals() and X_final.shape[1] >= 2:
                X_cls = X_final.copy()
            else:
                # Fallback: use X (pre-pruning) cleaned
                _X_fallback = X.loc[sufficient_rows].replace([np.inf, -np.inf], np.nan).ffill()
                X_cls = _X_fallback.copy()

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
