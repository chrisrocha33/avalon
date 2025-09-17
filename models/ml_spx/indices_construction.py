import yfinance as yf
from full_fred.fred import Fred
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from datetime import datetime
import json
import os
import matplotlib.pyplot as plt

# Apply centralized matplotlib styling
try:
    from utils import apply_dashboard_plot_style
    apply_dashboard_plot_style()
except ImportError:
    pass

def sanitize_symbol(symbol):
    """Convert raw symbol to column prefix format"""
    return symbol.replace('^', '').replace('=', '').replace('.', '_').replace('-', '_')

# -----------------------------------------------------------------------------
# Configuration and setup (mirrors data_prep.py where relevant)
# -----------------------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))

# Use centralized config for FRED API key
from config import Config

# Initialize FRED
fred = None
try:
    os.environ['FRED_API_KEY'] = Config.FRED_API_KEY
    fred = Fred()
except Exception as e:
    print(f"Error initializing FRED API: {e}")
    raise

print(f"="*120)
print("Loading Variables Configuration...")
print(f"="*120)
# Load variables configuration
try:
    with open(os.path.join(script_dir, 'variables.json'), 'r') as f:
        variables_config = json.load(f)
except Exception as e:
    print(f"Error loading variables.json: {e}")
    raise

print(f"="*120)
print("Variables Configuration Loaded")
print(f"="*120)

print(f"="*120)
print("SQL Connection Starting...")
print(f"="*120)
# Use centralized config
from config import Config
engine = create_engine(Config.DATABASE['connection_string'], future=True)

print(f"="*120)
print("SQL Connection Successful")
print(f"="*120)

print(f"="*120)
print("Loading Z-Scores Data (chunked parts if available)...")
print(f"="*120)

zscore_df = pd.DataFrame()
_zs_manifest_name = 'ml_spx_zscores_manifest'
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
            zscore_df = pd.concat(_assembled, axis=1)
    else:
        print(f"Manifest {_zs_manifest_name} is empty or missing 'table_name' column; falling back")
except Exception as _e_m:
    print(f"Manifest not found or unreadable ({_zs_manifest_name}): {_e_m}. Falling back to single table...")

if zscore_df.empty:
    try:
        _fallback = pd.read_sql_table("ml_spx_zscores", con=engine)
        if 'index' in _fallback.columns:
            _fallback['index'] = pd.to_datetime(_fallback['index'])
            _fallback = _fallback.set_index('index')
        zscore_df = _fallback
        print("Loaded fallback table ml_spx_zscores")
    except Exception as _e_f:
        print(f"❌ Could not load z-scores (parts or fallback): {_e_f}")
        raise

# Decompress: upcast numeric columns to float64 to restore original precision
try:
    _num_cols = []
    for _c in zscore_df.columns:
        if pd.api.types.is_numeric_dtype(zscore_df[_c]):
            _num_cols.append(_c)
    if len(_num_cols) > 0:
        zscore_df[_num_cols] = zscore_df[_num_cols].astype("float64")
        print(f"Decompression: upcasted {len(_num_cols)} numeric columns to float64")
except Exception as _e_dec:
    print(f"Decompression warning (indices_construction): {_e_dec}")

print(f"Z-Scores Columns: {len(zscore_df.columns)} (from {_zs_parts_loaded} parts if chunked)")
print(f"Z-Scores Rows: {len(zscore_df)}")
print(f"="*120)

# =============================================================================
# COMPREHENSIVE DATA DIAGNOSTICS FOR Z-SCORES DATA
# =============================================================================
print("="*120)
print("COMPREHENSIVE DATA DIAGNOSTICS - Z-SCORES DATA")
print("="*120)

# Basic data shape and structure
print(f"Number of columns: {len(zscore_df.columns)}")
print(f"Number of rows: {len(zscore_df)}")
print(f"Date range: {zscore_df.index.min()} to {zscore_df.index.max()}")
print(f"Total number of data points: {zscore_df.size}")

# Data types overview
print(f"\nData types distribution:")
dtype_counts = zscore_df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"  {dtype}: {count} columns")

# NaN analysis
total_nans = zscore_df.isna().sum().sum()
print(f"\nTotal number of NaNs: {total_nans}")
print(f"Percentage of NaNs: {(total_nans / zscore_df.size) * 100:.2f}%")

# Columns with NaNs
columns_with_nans = zscore_df.columns[zscore_df.isna().any()].tolist()
print(f"\nColumns with NaNs ({len(columns_with_nans)} total):")
for col in columns_with_nans:
    nan_count = zscore_df[col].isna().sum()
    nan_pct = (nan_count / len(zscore_df)) * 100
    print(f"  {col}: {nan_count} NaNs ({nan_pct:.2f}%)")

# Rows with NaNs
rows_with_nans = zscore_df[zscore_df.isna().any(axis=1)]
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
if 'GSPC_log_return_next_period' in zscore_df.columns:
    print(f"\nTarget variable (GSPC_log_return_next_period) analysis:")
    target_nans = zscore_df['GSPC_log_return_next_period'].isna().sum()
    target_valid = zscore_df['GSPC_log_return_next_period'].notna().sum()
    print(f"  Valid observations: {target_valid}")
    print(f"  NaN observations: {target_nans}")
    print(f"  NaN percentage: {(target_nans / len(zscore_df)) * 100:.2f}%")
    if target_valid > 0:
        print(f"  Mean: {zscore_df['GSPC_log_return_next_period'].mean():.6f}")
        print(f"  Std: {zscore_df['GSPC_log_return_next_period'].std():.6f}")
        print(f"  Min: {zscore_df['GSPC_log_return_next_period'].min():.6f}")
        print(f"  Max: {zscore_df['GSPC_log_return_next_period'].max():.6f}")
else:
    print(f"\nTarget variable (GSPC_log_return_next_period) not found in z-scores data")

# Feature completeness analysis
print(f"\nFeature completeness analysis:")
feature_completeness = zscore_df.notna().sum(axis=1) / len(zscore_df.columns)
print(f"  Mean completeness: {feature_completeness.mean():.3f}")
print(f"  Min completeness: {feature_completeness.min():.3f}")
print(f"  Max completeness: {feature_completeness.max():.3f}")
print(f"  Rows with 100% completeness: {(feature_completeness == 1.0).sum()}")
print(f"  Rows with >=80% completeness: {(feature_completeness >= 0.8).sum()}")
print(f"  Rows with >=50% completeness: {(feature_completeness >= 0.5).sum()}")

# Infinite values check
inf_cols = []
for col in zscore_df.columns:
    if pd.api.types.is_numeric_dtype(zscore_df[col]):
        if np.isinf(zscore_df[col]).any():
            inf_count = np.isinf(zscore_df[col]).sum()
            inf_cols.append((col, inf_count))

if inf_cols:
    print(f"\nColumns with infinite values ({len(inf_cols)} total):")
    for col, count in inf_cols:
        print(f"  {col}: {count} infinite values")
else:
    print(f"\nNo infinite values found in any columns")

# Memory usage
memory_usage = zscore_df.memory_usage(deep=True).sum() / 1024**2  # MB
print(f"\nMemory usage: {memory_usage:.2f} MB")

# Sample of the data
print(f"\nFirst 5 rows of z-scores data:")
print(zscore_df.head())

print(f"\nLast 5 rows of z-scores data:")
print(zscore_df.tail())

# Column names overview
print(f"\nColumn names overview (first 20):")
for i, col in enumerate(zscore_df.columns[:20]):
    print(f"  {i+1:2d}. {col}")
if len(zscore_df.columns) > 20:
    print(f"  ... and {len(zscore_df.columns) - 20} more columns")

print("="*120)
print("END OF Z-SCORES DATA DIAGNOSTICS")
print("="*120)

master_dataframe = pd.DataFrame(index=zscore_df.index)

print(f"="*120)
print("Loading Markets Configuration...")
print(f"="*120)
markets = variables_config.get('markets', {})

# Collect symbols per region (and also flat lists if needed)
region_to_equities = {}
region_to_fx = {}
all_equities = []
all_fx = []

for region_name, region_data in markets.items():
    equities_symbols = []
    fx_symbols = []

    equities_dict = region_data.get('equities', {})
    for symbol in equities_dict:
        equities_symbols.append(sanitize_symbol(symbol))
        all_equities.append(sanitize_symbol(symbol))

    fx_dict = region_data.get('fx', {})
    for symbol in fx_dict:
        fx_symbols.append(sanitize_symbol(symbol))
        all_fx.append(sanitize_symbol(symbol))

    region_to_equities[region_name] = equities_symbols
    region_to_fx[region_name] = fx_symbols

print(f"="*120)
print("Creating Regional Dataframes...")
print(f"="*120)

# Create a dictionary to store all filtered dataframes
filtered_data = {}

# Loop through all regions
for region in region_to_equities.keys():
    # Get equities and FX for this region
    region_equities = region_to_equities[region]
    region_fx = region_to_fx[region]
    
    # Filter and store in dictionary (with empty list handling)
    filtered_data[f'{region}_equities'] = zscore_df.filter(regex='|'.join(region_equities), axis=1) if region_equities else pd.DataFrame()
    filtered_data[f'{region}_fx'] = zscore_df.filter(regex='|'.join(region_fx), axis=1) if region_fx else pd.DataFrame()

print(f"="*120)
print("Regional Dataframes Created")
print(f"="*120)


#features to include
returns = ['log_return','intraday','overnight','rollret']
momentum = ['rsi','macd','macd_signal','stoch_k','ma_gap','ac1']
volatility = ['atr','bbz','skew','kurt']

print(f"="*120)
print("Creating Regional Indices...")
print(f"="*120)

# Create a dictionary to store all regional dataframes
regional_dataframes = {}

# Loop through all regions to create both equities and fx dataframes
for region in region_to_equities.keys():
    print(f"Creating indices for {region}...")
    
    # Create equities dataframe for this region
    if f'{region}_equities' in filtered_data and not filtered_data[f'{region}_equities'].empty:
        regional_dataframes[f'{region}_equities'] = pd.DataFrame(index=zscore_df.index)
        
        # Calculate composite indices for equities
        equities_data = filtered_data[f'{region}_equities']
        regional_dataframes[f'{region}_equities']['returns'] = equities_data.filter(regex='|'.join(returns), axis=1).sum(axis=1) / len(returns)
        regional_dataframes[f'{region}_equities']['momentum'] = equities_data.filter(regex='|'.join(momentum), axis=1).sum(axis=1) / len(momentum)
        regional_dataframes[f'{region}_equities']['volatility'] = equities_data.filter(regex='|'.join(volatility), axis=1).sum(axis=1) / len(volatility)
        
        print(f"  {region}_equities created with {len(regional_dataframes[f'{region}_equities'].columns)} columns")
    else:
        print(f"  {region}_equities: No data available")
    
    # Create fx dataframe for this region
    if f'{region}_fx' in filtered_data and not filtered_data[f'{region}_fx'].empty:
        regional_dataframes[f'{region}_fx'] = pd.DataFrame(index=zscore_df.index)
        
        # Calculate composite indices for fx
        fx_data = filtered_data[f'{region}_fx']
        regional_dataframes[f'{region}_fx']['returns'] = fx_data.filter(regex='|'.join(returns), axis=1).sum(axis=1) / len(returns)
        regional_dataframes[f'{region}_fx']['momentum'] = fx_data.filter(regex='|'.join(momentum), axis=1).sum(axis=1) / len(momentum)
        regional_dataframes[f'{region}_fx']['volatility'] = fx_data.filter(regex='|'.join(volatility), axis=1).sum(axis=1) / len(volatility)
        
        print(f"  {region}_fx created with {len(regional_dataframes[f'{region}_fx'].columns)} columns")
    else:
        print(f"  {region}_fx: No data available")

print(f"="*120)
print("Regional Indices Creation Complete")
print(f"Total dataframes created: {len(regional_dataframes)}")
print(f"="*120)

# Display sample of first dataframe for verification
if regional_dataframes:
    first_key = list(regional_dataframes.keys())[0]
    print(f"Sample of {first_key}:")
    print(regional_dataframes[first_key].tail())

print(regional_dataframes)

print(f"="*120)
print("Creating Combined Regional Indices DataFrame...")
print(f"="*120)

# Create a comprehensive dataframe with all regional indices
combined_regional_indices = pd.DataFrame(index=zscore_df.index)

# Add all regional indices to the combined dataframe
for region in region_to_equities.keys():
    # Add equities indices
    if f'{region}_equities' in regional_dataframes:
        combined_regional_indices[f'{region}_equities_returns'] = regional_dataframes[f'{region}_equities']['returns']
        combined_regional_indices[f'{region}_equities_momentum'] = regional_dataframes[f'{region}_equities']['momentum']
        combined_regional_indices[f'{region}_equities_volatility'] = regional_dataframes[f'{region}_equities']['volatility']
    
    # Add fx indices
    if f'{region}_fx' in regional_dataframes:
        combined_regional_indices[f'{region}_fx_returns'] = regional_dataframes[f'{region}_fx']['returns']
        combined_regional_indices[f'{region}_fx_momentum'] = regional_dataframes[f'{region}_fx']['momentum']
        combined_regional_indices[f'{region}_fx_volatility'] = regional_dataframes[f'{region}_fx']['volatility']

# Add target column from zscore_df
print(f"Adding GSPC_log_return_next_period target column...")
_target_loaded = False
try:
    _tgt = pd.read_sql_table('ml_spx_target', con=engine)
    if 'index' in _tgt.columns:
        _tgt['index'] = pd.to_datetime(_tgt['index'])
        _tgt = _tgt.set_index('index')
    if 'GSPC_log_return_next_period' in _tgt.columns:
        combined_regional_indices = combined_regional_indices.join(_tgt[['GSPC_log_return_next_period']], how='left')
        _target_loaded = True
        print(f"Successfully added GSPC_log_return_next_period from ml_spx_target")
except Exception as _e_t:
    print(f"Warning: failed to read ml_spx_target: {_e_t}")

if not _target_loaded:
    if 'GSPC_log_return_next_period' in zscore_df.columns:
        combined_regional_indices['GSPC_log_return_next_period'] = zscore_df['GSPC_log_return_next_period']
        print(f"Added target from zscore_df fallback")
    else:
        print(f"Warning: GSPC_log_return_next_period column not found in sources")

combined_regional_indices.to_sql('ml_spx_regional_indices', engine, if_exists='replace', index=True)
print(f"Combined Regional Indices DataFrame created with {len(combined_regional_indices.columns)} columns")
print(f"Columns: {list(combined_regional_indices.columns)}")
print(f"Shape: {combined_regional_indices.shape}")
print(f"="*120)

# Display sample of the combined dataframe
print("Sample of Combined Regional Indices:")
print(combined_regional_indices.tail())

master_dataframe = master_dataframe.join(combined_regional_indices)

print(f"="*120)
print("Loading Futures Configuration...")
print(f"="*120)

# Load futures configuration from variables.json
futures = variables_config.get('futures', {})

# Collect symbols per futures category
futures_category_to_symbols = {}

for category_name, category_data in futures.items():
    symbols = []
    for symbol in category_data:
        symbols.append(sanitize_symbol(symbol))
    futures_category_to_symbols[category_name] = symbols

print(f"Futures categories found: {list(futures_category_to_symbols.keys())}")
for category, symbols in futures_category_to_symbols.items():
    print(f"  {category}: {len(symbols)} symbols - {symbols}")

print(f"="*120)
print("Creating Futures Dataframes...")
print(f"="*120)

# Create a dictionary to store all filtered futures dataframes
futures_filtered_data = {}

# Loop through all futures categories
for category in futures_category_to_symbols.keys():
    # Get symbols for this category
    category_symbols = futures_category_to_symbols[category]
    
    # Filter and store in dictionary (with empty list handling)
    futures_filtered_data[f'{category}'] = zscore_df.filter(regex='|'.join(category_symbols), axis=1) if category_symbols else pd.DataFrame()

print(f"="*120)
print("Futures Dataframes Created")
print(f"="*120)

print(f"="*120)
print("Creating Futures Indices...")
print(f"="*120)

# Create a dictionary to store all futures dataframes
futures_dataframes = {}

# Loop through all futures categories to create dataframes
for category in futures_category_to_symbols.keys():
    print(f"Creating indices for {category}...")
    
    # Create dataframe for this futures category
    if f'{category}' in futures_filtered_data and not futures_filtered_data[f'{category}'].empty:
        futures_dataframes[f'{category}'] = pd.DataFrame(index=zscore_df.index)
        
        # Calculate composite indices for futures
        futures_data = futures_filtered_data[f'{category}']
        futures_dataframes[f'{category}']['returns'] = futures_data.filter(regex='|'.join(returns), axis=1).sum(axis=1) / len(returns)
        futures_dataframes[f'{category}']['momentum'] = futures_data.filter(regex='|'.join(momentum), axis=1).sum(axis=1) / len(momentum)
        futures_dataframes[f'{category}']['volatility'] = futures_data.filter(regex='|'.join(volatility), axis=1).sum(axis=1) / len(volatility)
        
        print(f"  {category} created with {len(futures_dataframes[f'{category}'].columns)} columns")
    else:
        print(f"  {category}: No data available")

print(f"="*120)
print("Futures Indices Creation Complete")
print(f"Total futures dataframes created: {len(futures_dataframes)}")
print(f"="*120)

# Display sample of first dataframe for verification
if futures_dataframes:
    first_key = list(futures_dataframes.keys())[0]
    print(f"Sample of {first_key}:")
    print(futures_dataframes[first_key].tail())

print(futures_dataframes)

print(f"="*120)
print("Creating Combined Futures Indices DataFrame...")
print(f"="*120)

# Create a comprehensive dataframe with all futures indices
combined_futures_indices = pd.DataFrame(index=zscore_df.index)

# Add all futures indices to the combined dataframe
for category in futures_category_to_symbols.keys():
    # Add futures indices
    if f'{category}' in futures_dataframes:
        combined_futures_indices[f'{category}_returns'] = futures_dataframes[f'{category}']['returns']
        combined_futures_indices[f'{category}_momentum'] = futures_dataframes[f'{category}']['momentum']
        combined_futures_indices[f'{category}_volatility'] = futures_dataframes[f'{category}']['volatility']

combined_futures_indices.to_sql('ml_spx_futures_indices', engine, if_exists='replace', index=True)
print(f"Combined Futures Indices DataFrame created with {len(combined_futures_indices.columns)} columns")
print(f"Columns: {list(combined_futures_indices.columns)}")
print(f"Shape: {combined_futures_indices.shape}")
print(f"="*120)


print(f"="*120)
print("Creating Master DataFrame...")
print(f"="*120)


master_dataframe = master_dataframe.join(combined_futures_indices)
master_dataframe['GSPC_log_return_next_period'] = zscore_df['GSPC_log_return_next_period']


print(f"="*120)
print(f"Master Df Columns: {len(master_dataframe.columns)}")
print(f"="*120)
print(f"Master Df Columns: {master_dataframe.columns.tolist()}")
print(f"="*120)
print(f"Master Df Rows: {len(master_dataframe)}")

master_dataframe.to_sql('ivw_indices', engine, if_exists='replace', index=True)

plt.figure(figsize=(10, 5))
plt.plot(master_dataframe.tail(6))
plt.show()