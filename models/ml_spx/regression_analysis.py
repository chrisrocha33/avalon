import pandas as pd
import numpy as np
import statsmodels.api as sm
from sqlalchemy import create_engine
import json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

def load_refined_data(engine, table_prefix="ml_spx_refined"):
    """Load refined data from database"""
    print(f"="*120)
    print("LOADING REFINED DATA")
    print(f"="*120)
    
    try:
        # Load features
        X_refined = pd.read_sql_table(f"{table_prefix}_features", con=engine)
        if 'index' in X_refined.columns:
            X_refined['index'] = pd.to_datetime(X_refined['index'])
            X_refined = X_refined.set_index('index')
        print(f"✓ Loaded refined features: {X_refined.shape}")
        
        # Load target
        y_refined = pd.read_sql_table(f"{table_prefix}_target", con=engine)
        if 'index' in y_refined.columns:
            y_refined['index'] = pd.to_datetime(y_refined['index'])
            y_refined = y_refined.set_index('index')
        y_refined = y_refined['GSPC_log_return_next_period']
        print(f"✓ Loaded refined target: {y_refined.shape}")
        
        # Align data
        common_idx = X_refined.index.intersection(y_refined.index)
        X_refined = X_refined.loc[common_idx]
        y_refined = y_refined.loc[common_idx]
        
        print(f"✓ Aligned data: {X_refined.shape[0]} observations, {X_refined.shape[1]} features")
        return X_refined, y_refined
        
    except Exception as e:
        print(f"❌ Error loading refined data: {e}")
        print("Falling back to loading from original z-scores data...")
        return load_fallback_data(engine)

def load_fallback_data(engine):
    """Fallback to load original z-scores data if refined data not available"""
    print("Loading fallback data from z-scores...")
    
    # Load z-scores data (similar to original implementation)
    combined_data = pd.DataFrame()
    _zs_manifest_name = 'ml_spx_zscores_manifest'
    
    try:
        _manifest = pd.read_sql_table(_zs_manifest_name, con=engine)
        if not _manifest.empty and 'table_name' in _manifest.columns:
            _parts = list(_manifest['table_name'])
            _assembled = []
            for _t in _parts:
                try:
                    _dfp = pd.read_sql_table(_t, con=engine)
                    if 'index' in _dfp.columns:
                        _dfp['index'] = pd.to_datetime(_dfp['index'])
                        _dfp = _dfp.set_index('index')
                    _assembled.append(_dfp)
                except Exception as _e_p:
                    print(f"  ❌ Failed to load part {_t}: {_e_p}")
            if len(_assembled) > 0:
                combined_data = pd.concat(_assembled, axis=1)
    except Exception as _e_m:
        print(f"Manifest not found: {_e_m}")
    
    if combined_data.empty:
        try:
            combined_data = pd.read_sql_table("ml_spx_zscores", con=engine)
            if 'index' in combined_data.columns:
                combined_data['index'] = pd.to_datetime(combined_data['index'])
                combined_data = combined_data.set_index('index')
        except Exception as _e_f:
            print(f"❌ Could not load z-scores: {_e_f}")
            raise
    
    # Load target variable
    target_col = "GSPC_log_return_next_period"
    try:
        _tgt = pd.read_sql_table('ml_spx_target', con=engine)
        if 'index' in _tgt.columns:
            _tgt['index'] = pd.to_datetime(_tgt['index'])
            _tgt = _tgt.set_index('index')
        if target_col in _tgt.columns:
            combined_data = combined_data.join(_tgt[[target_col]], how='left')
    except Exception as _e_t:
        print(f"Warning: failed to read ml_spx_target: {_e_t}")
    
    if target_col not in combined_data.columns:
        print(f"Error: Target variable {target_col} not found")
        return None, None
    
    y = combined_data[target_col]
    X = combined_data.drop(columns=[target_col])
    
    # Filter for sufficient data
    y_mask = y.notna()
    X_mask = X.notna()
    feature_completeness = X_mask.sum(axis=1) / X_mask.shape[1]
    sufficient_rows = y_mask & (feature_completeness >= 0.8)
    
    if sufficient_rows.sum() < 50:
        print("Insufficient data for analysis")
        return None, None
    
    y_final = y.loc[sufficient_rows]
    X_final = X.loc[sufficient_rows]
    X_final = X_final.replace([np.inf, -np.inf], np.nan).ffill()
    
    return X_final, y_final

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

def run_ols_regression(y_final, X_final, top_features_count=50):
    """Run OLS regression analysis"""
    print(f"\n{'='*80}")
    print(f"OLS REGRESSION ANALYSIS")
    print(f"{'='*80}")
    
    # Limit to top features if specified
    if top_features_count < X_final.shape[1]:
        print(f"Limiting to top {top_features_count} features for analysis")
        # Simple selection - take first N features (in practice, use feature importance)
        X_final = X_final.iloc[:, :top_features_count]
    
    # Standardize features
    _stds_final = X_final.std(ddof=0).replace(0, 1.0)
    _means_final = X_final.mean()
    X_final_std = (X_final - _means_final) / _stds_final
    
    # Create design matrix with constant
    X_with_const = sm.add_constant(X_final_std)
    
    try:
        ols_result = sm.OLS(y_final, X_with_const).fit()
        
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
        
        # Fit diagnostics
        print(f"\n{'='*80}")
        print("FIT DIAGNOSTICS")
        print(f"{'='*80}")
        
        print(f"AIC: {ols_result.aic:.2f}")
        print(f"BIC: {ols_result.bic:.2f}")
        print(f"Log-likelihood: {ols_result.llf:.2f}")
        print(f"Durbin-Watson: {sm.stats.durbin_watson(ols_result.resid):.4f}")
        
        # Rolling R-squared analysis
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
        
        # Stability diagnostics
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
        
        return ols_result
        
    except Exception as e:
        print(f"Error running OLS regression: {e}")
        return None

def run_classification_experiments(y_final, X_final):
    """Run classification experiments"""
    print("\n" + "="*80)
    print("CLASSIFICATION EXPERIMENTS")
    print("="*80)
    
    # Prepare binary target: next-period up move (>0 -> 1 else 0)
    y_bin = (y_final > 0).astype(int)
    
    # Align indexes
    common_idx = y_bin.index.intersection(X_final.index)
    y_bin = y_bin.loc[common_idx]
    X_cls = X_final.loc[common_idx]
    
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
    
    # Define classifiers
    clf_names = [
        'LogisticRegression',
        'SVC_RBF',
        'RandomForest',
        'GradientBoosting'
    ]
    
    print("\nRunning classifiers:")
    results = []
    
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
            if hasattr(clf, 'decision_function'):
                df_vals = clf.decision_function(X_te)
                y_prob = 1.0 / (1.0 + np.exp(-df_vals))
            else:
                y_prob = None
        
        y_pred = clf.predict(X_te)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob) if y_prob is not None else float('nan')
        
        results.append({
            'model': name,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc
        })
        
        print("-"*70)
        print(f"Model: {name}")
        print(f"  Train samples: {len(y_train)}  Test samples: {len(y_test)}  Features: {X_tr.shape[1]}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1: {f1:.4f}")
        print(f"  ROC-AUC: {roc:.4f}")
    
    return results

def main():
    """Main function to run regression analysis"""
    print(f"="*120)
    print("REGRESSION ANALYSIS PIPELINE")
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
    
    # Database connection
    print("Connecting to database...")
    
    # Use centralized config
    from config import Config
    engine = create_engine(Config.DATABASE['connection_string'], future=True)
    print("Database connection established")
    
    # Load data
    X_final, y_final = load_refined_data(engine)
    
    if X_final is None or y_final is None:
        print("❌ Failed to load data for analysis")
        return
    
    # NaN analysis before analysis
    combined_data = pd.concat([y_final.to_frame(), X_final], axis=1)
    comprehensive_nan_analysis(combined_data, "BEFORE REGRESSION ANALYSIS")
    
    # Run OLS regression
    ols_result = run_ols_regression(y_final, X_final, top_features_count)
    
    # Run classification experiments
    classification_results = run_classification_experiments(y_final, X_final)
    
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Data shape: {X_final.shape}")
    print(f"Features analyzed: {X_final.shape[1]}")
    print(f"Observations: {len(y_final)}")
    
    if ols_result is not None:
        print(f"OLS R²: {ols_result.rsquared:.4f}")
        print(f"OLS Adj R²: {ols_result.rsquared_adj:.4f}")
    
    if classification_results:
        best_model = max(classification_results, key=lambda x: x['roc_auc'])
        print(f"Best classification model: {best_model['model']} (ROC-AUC: {best_model['roc_auc']:.4f})")
    
    print(f"="*120)
    print("REGRESSION ANALYSIS PIPELINE COMPLETED")
    print(f"="*120)

if __name__ == "__main__":
    main()