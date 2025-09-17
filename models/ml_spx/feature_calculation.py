import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine

def load_api_data():
    """Load the data prepared by api_data_collection.py"""
    # This would typically load from database or file
    # For now, we'll assume the data is available
    print("Loading data from API collection...")
    # In practice, you would load from database here
    # combined_data = load_from_database()
    return None  # Placeholder - will be replaced with actual data loading

def calculate_market_features(combined_data, variables_config):
    """Calculate technical indicators and features from market data"""
    print(f"="*120)
    print("Computing Market Features from LAGGED Data...")
    print(f"="*120)
    
    # Market data is already lagged at this point
    # Feature engineering will automatically use lagged inputs
    
    # Calculate log returns for all market symbols (data already lagged)
    market_columns = [col for col in combined_data.columns if any(ohlcv in col for ohlcv in ['Open', 'High', 'Low', 'Close', 'Volume'])]
    market_symbols = set()
    for col in market_columns:
        parts = col.split('_')
        if len(parts) >= 2:
            symbol = '_'.join(parts[:-1])
            if 'GSPC' not in symbol:
                market_symbols.add(symbol)
    
    print(f"Found {len(market_symbols)} market symbols to process (all data already lagged)")
    print(f"Note: All feature calculations will use properly lagged market data")
    
    # Create dictionary to store calculated features
    new_features_dict = {}
    
    for symbol in market_symbols:
        print(f"Processing {symbol}...")
        
        # Get OHLCV columns for this symbol
        open_col = f"{symbol}_Open"
        high_col = f"{symbol}_High"
        low_col = f"{symbol}_Low"
        close_col = f"{symbol}_Close"
        volume_col = f"{symbol}_Volume"
        
        # Check if all required columns exist
        required_cols = [open_col, high_col, low_col, close_col, volume_col]
        if not all(col in combined_data.columns for col in required_cols):
            print(f"  ⚠️  Missing OHLCV columns for {symbol}, skipping...")
            continue
        
        # Extract OHLCV data
        open_data = combined_data[open_col]
        high_data = combined_data[high_col]
        low_data = combined_data[low_col]
        close_data = combined_data[close_col]
        volume_data = combined_data[volume_col]
        
        # 1. Log Returns: log(Close / Close.shift(1))
        close_ratio = close_data / close_data.shift(1)
        close_ratio = close_ratio.replace([0, np.inf, -np.inf], np.nan)
        log_return = np.log(close_ratio)
        new_features_dict[f"{symbol}_log_return"] = log_return
        
        # 2. Intraday Return: log(Close / Open)
        intraday_ratio = close_data / open_data
        intraday_ratio = intraday_ratio.replace([0, np.inf, -np.inf], np.nan)
        intraday_return = np.log(intraday_ratio)
        new_features_dict[f"{symbol}_intraday"] = intraday_return
        
        # 3. Overnight Return: log(Open / Close.shift(1))
        overnight_ratio = open_data / close_data.shift(1)
        overnight_ratio = overnight_ratio.replace([0, np.inf, -np.inf], np.nan)
        overnight_return = np.log(overnight_ratio)
        new_features_dict[f"{symbol}_overnight"] = overnight_return
        
        # Rolling returns (configurable)
        _ti_cfg = variables_config.get('config', {}).get('technical_indicators', {}) if isinstance(variables_config.get('config', {}), dict) else {}
        _rr_windows = _ti_cfg.get('rolling_returns', [5, 20])
        for _w in _rr_windows:
            _rr = np.log((close_data / close_data.shift(_w)).replace([0, np.inf, -np.inf], np.nan))
            new_features_dict[f"{symbol}_rollret_{_w}"] = _rr
        
        # MA gap (configurable)
        _ma_gap_w = int(_ti_cfg.get('ma_gap', 20))
        _maw = close_data.rolling(_ma_gap_w, min_periods=2).mean()
        _gapw = (close_data - _maw) / _maw
        new_features_dict[f"{symbol}_ma_gap_{_ma_gap_w}"] = _gapw
        
        # MACD (configurable)
        _macd_cfg = _ti_cfg.get('macd', {})
        _fast = int(_macd_cfg.get('fast', 12))
        _slow = int(_macd_cfg.get('slow', 26))
        _signal = int(_macd_cfg.get('signal', 9))
        _ema_fast = close_data.ewm(span=_fast, adjust=False).mean()
        _ema_slow = close_data.ewm(span=_slow, adjust=False).mean()
        _macd = _ema_fast - _ema_slow
        _macd_signal = _macd.ewm(span=_signal, adjust=False).mean()
        new_features_dict[f"{symbol}_macd"] = _macd
        new_features_dict[f"{symbol}_macd_signal"] = _macd_signal
        
        # RSI (configurable)
        _rsi_w = int(_ti_cfg.get('rsi', 14))
        _diff = close_data.diff()
        _gain = _diff.clip(lower=0)
        _loss = (-_diff).clip(lower=0)
        _avg_gain = _gain.rolling(_rsi_w, min_periods=2).mean()
        _avg_loss = _loss.rolling(_rsi_w, min_periods=2).mean()
        _rs = _avg_gain / _avg_loss.replace(0, np.nan)
        _rsi = 100.0 - (100.0 / (1.0 + _rs))
        new_features_dict[f"{symbol}_rsi_{_rsi_w}"] = _rsi
        
        # Stochastic %K (configurable)
        _stk_w = int(_ti_cfg.get('stochastic_k', 14))
        _ll = low_data.rolling(_stk_w, min_periods=2).min()
        _hh = high_data.rolling(_stk_w, min_periods=2).max()
        _den = (_hh - _ll).replace(0, np.nan)
        _stoch_k = (close_data - _ll) / _den
        new_features_dict[f"{symbol}_stoch_k_{_stk_w}"] = _stoch_k
        
        # ATR (configurable)
        _atr_w = int(_ti_cfg.get('atr', 14))
        _prev_close = close_data.shift(1)
        _tr = pd.concat([
            (high_data - low_data).abs(),
            (high_data - _prev_close).abs(),
            (low_data - _prev_close).abs()
        ], axis=1).max(axis=1)
        _atr = _tr.rolling(_atr_w, min_periods=2).mean()
        new_features_dict[f"{symbol}_atr_{_atr_w}"] = _atr
        
        # Bollinger z-score (configurable)
        _bb_cfg = _ti_cfg.get('bollinger', {})
        _bb_w = int(_bb_cfg.get('window', 20))
        _bb_std = float(_bb_cfg.get('num_std', 2))
        _ma_bb = close_data.rolling(_bb_w, min_periods=2).mean()
        _std_bb = close_data.rolling(_bb_w, min_periods=2).std(ddof=0)
        _bb_z = (close_data - _ma_bb) / _std_bb.replace(0, np.nan)
        new_features_dict[f"{symbol}_bb_z_{_bb_w}"] = _bb_z
        
        # Rolling skewness and kurtosis (configurable)
        _skw_w = int(_ti_cfg.get('skew_kurt_window', 20))
        new_features_dict[f"{symbol}_ret_skew_{_skw_w}"] = log_return.rolling(_skw_w, min_periods=5).skew()
        new_features_dict[f"{symbol}_ret_kurt_{_skw_w}"] = log_return.rolling(_skw_w, min_periods=5).kurt()
        
        # AC(1) over rolling window (configurable)
        _w_ac = int(_ti_cfg.get('ac1_window', 60))
        _r_mean = log_return.rolling(_w_ac, min_periods=10).mean()
        _r_std = log_return.rolling(_w_ac, min_periods=10).std(ddof=0)
        _r_l1 = log_return.shift(1)
        _r_l1_mean = _r_l1.rolling(_w_ac, min_periods=10).mean()
        _r_l1_std = _r_l1.rolling(_w_ac, min_periods=10).std(ddof=0)
        _cov1 = (log_return * _r_l1).rolling(_w_ac, min_periods=10).mean() - (_r_mean * _r_l1_mean)
        _ac1 = _cov1 / (_r_std * _r_l1_std).replace(0, np.nan)
        new_features_dict[f"{symbol}_ac1_{_w_ac}"] = _ac1
        
        print(f"  ✓ Calculated {len([k for k in new_features_dict.keys() if k.startswith(symbol)])} features for {symbol}")
    
    return new_features_dict

def calculate_economic_features(combined_data, variables_config):
    """Calculate log % change for economic indicators"""
    print("Computing log % change for economic indicators (excluding yields, credit spreads, and UNRATE)...")
    
    # Get economic indicator categories
    yields_symbols = []
    credit_spreads_symbols = []
    
    # Determine non-yield, non-credit-spread economic indicators to transform
    non_yield_econ_symbols = []
    for sym in combined_data.columns:
        if (
            sym not in (yields_symbols or [])
            and sym not in (credit_spreads_symbols or [])
            and sym != 'UNRATE'
            and not any(feature in sym for feature in ['log_return', 'intraday', 'overnight', 'log_vol_change', 'log_pct_change'])
            and not any(feature in sym for feature in ['yield_spread', 'GSPC'])
        ):
            non_yield_econ_symbols.append(sym)
    
    econ_change_features = {}
    for sym in non_yield_econ_symbols:
        series = combined_data[sym]
        ratio = series / series.shift(1)
        ratio = ratio.replace([0, np.inf, -np.inf], np.nan)
        econ_change = np.log(ratio)
        econ_change_features[f"{sym}_log_pct_change"] = econ_change
        print(f"  ✓ Created {sym}_log_pct_change")
    
    return econ_change_features

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
    print(f"\nTarget variable (GSPC_log_return_next_period) analysis:")
    if 'GSPC_log_return_next_period' in df.columns:
        target_nans = df['GSPC_log_return_next_period'].isna().sum()
        target_valid = df['GSPC_log_return_next_period'].notna().sum()
        print(f"  Valid observations: {target_valid}")
        print(f"  NaN observations: {target_nans}")
        print(f"  NaN percentage: {(target_nans / len(df)) * 100:.2f}%")
        if target_valid > 0:
            print(f"  Mean: {df['GSPC_log_return_next_period'].mean():.6f}")
            print(f"  Std: {df['GSPC_log_return_next_period'].std():.6f}")
            print(f"  Min: {df['GSPC_log_return_next_period'].min():.6f}")
            print(f"  Max: {df['GSPC_log_return_next_period'].max():.6f}")
    else:
        print("  WARNING: Target variable not found in dataset!")
    
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

def save_to_database(df, engine, base_table_name, manifest_table_name, chunk_size=200):
    """Save DataFrame to database in chunked parts with manifest"""
    print(f"Saving data to database (chunked parts + manifest)...")
    
    # Separate target column
    target_colname = 'GSPC_log_return_next_period'
    if target_colname in df.columns:
        try:
            df[[target_colname]].to_sql(
                'ml_spx_target', engine, if_exists='replace', index=True, method='multi', chunksize=1000
            )
            print("✅ Target saved to database: ml_spx_target")
        except Exception as e:
            print(f"❌ Error saving target table: {e}")
    else:
        print(f"⚠️ Target column '{target_colname}' not found in DataFrame")
    
    # Prepare feature columns (exclude target)
    feature_cols = []
    for col in df.columns:
        if col != target_colname:
            feature_cols.append(col)
    
    # Write parts
    num_cols = len(feature_cols)
    num_parts = (num_cols + chunk_size - 1) // chunk_size if num_cols > 0 else 0
    manifest_rows = []
    
    for part_idx in range(num_parts):
        start = part_idx * chunk_size
        end = min(start + chunk_size, num_cols)
        part_cols = feature_cols[start:end]
        part_df = df[part_cols].copy()
        table_name = f"{base_table_name}{str(part_idx + 1).zfill(3)}"
        try:
            part_df.to_sql(table_name, engine, if_exists='replace', index=True, method='multi', chunksize=1000)
            print(f"  ✅ Saved part {part_idx + 1}/{num_parts}: {table_name} with {len(part_cols)} columns")
        except Exception as e:
            print(f"  ❌ Error saving part {part_idx + 1}: {e}")
            raise
        manifest_rows.append({'table_name': table_name, 'start_col': start, 'end_col': end - 1, 'num_cols': len(part_cols)})
    
    try:
        pd.DataFrame(manifest_rows).to_sql(manifest_table_name, engine, if_exists='replace', index=False)
        print(f"✅ Manifest saved: {manifest_table_name} ({len(manifest_rows)} parts)")
    except Exception as e:
        print(f"❌ Error saving manifest: {e}")

def compute_z_scores(df):
    """Compute z-scores for all columns"""
    print("Computing z-scores for all columns...")
    zscores_dict = {}
    problematic_columns = []
    
    print("Processing columns for z-score calculation...")
    for col in df.columns:
        try:
            col_data = df[col].copy()
            
            if col_data.isna().all():
                print(f"  ⚠️  Column {col}: All NaN values, skipping z-score calculation")
                zscores_dict[col] = col_data
                problematic_columns.append(f"{col} (all NaN)")
                continue
            
            if np.isinf(col_data).any():
                print(f"  ⚠️  Column {col}: Contains infinite values, replacing with NaN")
                col_data = col_data.replace([np.inf, -np.inf], np.nan)
            
            col_std = col_data.std(ddof=0)
            if col_std == 0 or np.isnan(col_std):
                print(f"  ⚠️  Column {col}: Constant values (std={col_std}), setting z-scores to 0")
                zscores_dict[col] = pd.Series(0, index=col_data.index)
                problematic_columns.append(f"{col} (constant values)")
                continue
            
            col_mean = col_data.mean()
            if np.isnan(col_mean):
                print(f"  ⚠️  Column {col}: Mean is NaN, skipping z-score calculation")
                zscores_dict[col] = col_data
                problematic_columns.append(f"{col} (NaN mean)")
                continue
            
            zscores = (col_data - col_mean) / col_std
            zscores_dict[col] = zscores
            
            if zscores.isna().all():
                print(f"  ⚠️  Column {col}: Z-scores are all NaN")
                problematic_columns.append(f"{col} (all NaN z-scores)")
            else:
                print(f"  ✓ Column {col}: Z-scores calculated successfully")
                
        except Exception as col_error:
            print(f"  ❌ Error processing column {col}: {col_error}")
            zscores_dict[col] = df[col]
            problematic_columns.append(f"{col} (error: {str(col_error)[:50]})")
    
    zscores_df = pd.DataFrame(zscores_dict, index=df.index)
    
    if problematic_columns:
        print(f"\n⚠️  Problematic columns ({len(problematic_columns)}):")
        for prob_col in problematic_columns:
            print(f"    {prob_col}")
    else:
        print("✓ All columns processed successfully")
    
    return zscores_df, problematic_columns

def main():
    """Main function to run feature calculation pipeline"""
    print(f"="*120)
    print("FEATURE CALCULATION PIPELINE")
    print(f"="*120)
    
    # Load configuration
    try:
        with open('variables.json', 'r') as f:
            variables_config = json.load(f)
        print("Variables configuration loaded")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading variables.json: {e}")
        return
    
    # Database connection
    print("Connecting to database...")
    
    # Use centralized config
    from config import Config
    engine = create_engine(Config.DATABASE['connection_string'], future=True)
    print("Database connection established")
    
    # Load data from API collection (placeholder - in practice load from database)
    print("Loading data from API collection...")
    # This is where you would load the data from the database
    # For now, we'll create a placeholder
    combined_data = None  # Load from database here
    
    if combined_data is None:
        print("❌ No data loaded from API collection. Please run api_data_collection.py first.")
        return
    
    # NaN analysis before feature calculation
    comprehensive_nan_analysis(combined_data, "BEFORE FEATURE CALCULATION")
    
    # Calculate market features
    market_features = calculate_market_features(combined_data, variables_config)
    
    # Calculate economic features
    economic_features = calculate_economic_features(combined_data, variables_config)
    
    # Combine all features
    all_features = {**market_features, **economic_features}
    
    # Create features dataframe
    print("Creating features dataframe...")
    new_features_data = pd.DataFrame(all_features, index=combined_data.index)
    
    # Add new features to combined_data
    print("Adding calculated features to dataset...")
    combined_data = pd.concat([combined_data, new_features_data], axis=1)
    
    # Remove original OHLCV columns (which were already lagged)
    print("Removing original lagged OHLCV columns...")
    ohlcv_columns = [col for col in combined_data.columns if any(ohlcv in col for ohlcv in ['Open', 'High', 'Low', 'Close', 'Volume'])]
    combined_data = combined_data.drop(columns=ohlcv_columns)
    
    print(f"Replaced lagged OHLCV columns with calculated features")
    
    # Remove original economic indicator columns that were transformed
    econ_cols_to_drop = [col for col in combined_data.columns if col in economic_features.keys()]
    if econ_cols_to_drop:
        combined_data = combined_data.drop(columns=econ_cols_to_drop)
        print(f"✓ Removed {len(econ_cols_to_drop)} original economic indicator columns")
    
    # NaN analysis after feature calculation
    comprehensive_nan_analysis(combined_data, "AFTER FEATURE CALCULATION")
    
    # Final data cleanup
    print("Final data cleanup...")
    combined_data = combined_data.dropna(how='all')
    
    # Forward-fill remaining NaNs for database storage
    print("Forward-filling remaining NaNs for database storage...")
    combined_data = combined_data.ffill()
    
    # Fill any remaining NaNs with 0
    remaining_nans = combined_data.isna().sum().sum()
    if remaining_nans > 0:
        print(f"Filling {remaining_nans} remaining NaNs with 0...")
        combined_data = combined_data.fillna(0)
    
    # Defragment DataFrame
    print("Defragmenting DataFrame...")
    combined_data = combined_data.copy()
    
    # Final NaN analysis
    comprehensive_nan_analysis(combined_data, "FINAL CLEANED DATA")
    
    # Save to database
    save_to_database(combined_data, engine, 'ml_spx_data_p', 'ml_spx_data_manifest')
    
    # Compute and save z-scores
    print("Computing z-scores...")
    zscores_df, problematic_cols = compute_z_scores(combined_data)
    
    # Z-scores NaN analysis
    comprehensive_nan_analysis(zscores_df, "Z-SCORES DATA")
    
    # Save z-scores to database
    save_to_database(zscores_df, engine, 'ml_spx_zscores_p', 'ml_spx_zscores_manifest')
    
    print(f"="*120)
    print("FEATURE CALCULATION PIPELINE COMPLETED")
    print(f"="*120)
    print(f"Final data shape: {combined_data.shape}")
    print(f"Z-scores shape: {zscores_df.shape}")
    print(f"Problematic columns: {len(problematic_cols)}")

if __name__ == "__main__":
    main()

