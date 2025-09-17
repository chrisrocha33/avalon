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

# Use centralized config for FRED API key
from config import Config

# Initialize FRED API
try:
    os.environ['FRED_API_KEY'] = Config.FRED_API_KEY
    fred = Fred()
    
    # Test API connection
    test_series = fred.get_series_df('GDP', frequency='q', limit=1)
    print("FRED API initialized successfully")
    
except Exception as e:
    print(f"Error initializing FRED API: {e}")
    raise

# Load variables configuration
try:
    with open('variables.json', 'r') as f:
        variables_config = json.load(f)
    print("Variables configuration loaded")
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading variables.json: {e}")
    raise

def validate_config_structure(config):
    """Validate configuration structure and lag values"""
    required_sections = ['markets', 'futures', 'economic_indicators']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in variables.json")
    
    symbol_count = 0
    def count_symbols(data):
        nonlocal symbol_count
        for key, value in data.items():
            if isinstance(value, dict):
                if 'lag' in value:
                    if not isinstance(value['lag'], int):
                        raise ValueError(f"Lag value for symbol '{key}' must be an integer")
                    symbol_count += 1
                else:
                    count_symbols(value)
    
    count_symbols(config)
    print(f"Configuration validated: {symbol_count} symbols found")
    return True

# Validate configuration structure
validate_config_structure(variables_config)

# Load resampling configuration
resampling_freq = variables_config.get('config', {}).get('resampling', 'D')
valid_frequencies = ['D', 'W', 'M', 'Q', 'A']
if resampling_freq not in valid_frequencies:
    raise ValueError(f"Invalid resampling frequency '{resampling_freq}'. Must be one of: {valid_frequencies}")
print(f"Resampling frequency configured: {resampling_freq}")

# Define frequency mapping for pandas resampling
freq_mapping = {
    'D': 'D',      # Daily (no resampling needed)
    'W': 'W-FRI',  # Weekly ending on Friday
    'M': 'ME',     # Monthly end
    'Q': 'Q',      # Quarterly end  
    'A': 'A'       # Annual end
}

def daily(x):
    dta = yf.download(tickers=x,period='max',interval = '1d',auto_adjust=True,prepost=True)
    return dta

def data(x, y):
    """Load FRED data without forward-filling to prevent time leakage"""
    df = fred.get_series_df(x, frequency=y)
    df.value = df.value.replace('.', np.nan)
    df.index = df.date
    df = df.drop(columns=['date', 'realtime_start', 'realtime_end'])
    df.value = df.value.astype('float')
    return df

def sanitize_symbol(symbol):
    """Convert raw symbol to column prefix format"""
    return symbol.replace('^', '').replace('=', '').replace('.', '_').replace('-', '_')

def compare_frequencies(native_freq, target_freq):
    """
    Compare two frequencies and determine if native is faster, slower, or same as target
    """
    freq_hierarchy = {
        'D': 5,   # Daily
        'W': 4,   # Weekly
        'M': 3,   # Monthly
        'Q': 2,   # Quarterly
        'A': 1    # Annual
    }
    
    target_base = target_freq.split('-')[0]  # Handle 'W-FRI' -> 'W'
    
    native_level = freq_hierarchy.get(native_freq.upper(), 0)
    target_level = freq_hierarchy.get(target_base.upper(), 0)
    
    if native_level > target_level:
        return 'faster'
    elif native_level < target_level:
        return 'slower'
    else:
        return 'same'

def resample_fred_data(df, target_freq, native_freq, symbol):
    """
    Intelligently resample FRED data based on native vs target frequency
    """
    if target_freq == 'D':
        return df  # No resampling needed for daily target
    
    freq_comparison = compare_frequencies(native_freq, target_freq)
    
    if freq_comparison == 'same':
        print(f"  {symbol}: Native frequency {native_freq} matches target {target_freq}, no resampling needed")
        return df
    elif freq_comparison == 'faster':
        print(f"  {symbol}: Resampling from {native_freq} to {target_freq} using last value")
        resampled = df.resample(target_freq).last()
        print(f"    Resampled from {df.shape[0]} to {resampled.shape[0]} periods")
        return resampled
    else:  # slower
        print(f"  {symbol}: Native frequency {native_freq} is slower than target {target_freq}")
        print(f"    Will forward-fill during alignment to target grid")
        return df  # Return as-is, forward-fill will happen during alignment

def resample_data(df, freq):
    """
    Resample data to specified frequency (for market data and simple cases)
    """
    if freq == 'D':
        return df  # No resampling needed for daily
    
    print(f"  Resampling data to {freq}")
    
    resampled = df.resample(freq).last()
    
    print(f"    Resampled from {df.shape[0]} to {resampled.shape[0]} periods")
    return resampled

print(f"="*120)
print("SQL Connection Starting...")

# Use centralized config
from config import Config
engine = create_engine(Config.DATABASE['connection_string'], future=True)

print(f"Using database connection from centralized config")   

print(f"="*120)
print("SQL Connection Successful")
print(f"="*120)

# Extract symbols and lag information from JSON configuration
def extract_symbols_from_config(config):
    """Extract all symbols and their lag values from the configuration"""
    symbols = {}
    
    def extract_recursive(data, path=""):
        for key, value in data.items():
            if isinstance(value, dict):
                if 'lag' in value:
                    symbols[key] = value['lag']
                else:
                    extract_recursive(value, f"{path}.{key}" if path else key)
    
    extract_recursive(config)
    return symbols

# Extract all symbols and their lag values
symbol_lag_mapping = extract_symbols_from_config(variables_config)

def get_all_symbols_from_main_category(config, main_category):
    """Extract all symbols from a main category, traversing all subcategories automatically"""
    symbols = []
    
    if main_category not in config:
        return symbols
    
    def extract_symbols_recursive(data):
        result = []
        for key, value in data.items():
            if isinstance(value, dict):
                if 'lag' in value:
                    result.append(key)
                else:
                    result.extend(extract_symbols_recursive(value))
        return result
    
    return extract_symbols_recursive(config[main_category])

# Extract symbols by main category automatically
indices_symbols = get_all_symbols_from_main_category(variables_config, 'markets')
indices_symbols = [s for s in indices_symbols if not str(s).endswith('=X')]

# Extract FX symbols nested under markets.<region>.fx for processing
fx_symbols = []
if 'markets' in variables_config and isinstance(variables_config['markets'], dict):
    for _region_key, _region_val in variables_config['markets'].items():
        if isinstance(_region_val, dict) and 'fx' in _region_val and isinstance(_region_val['fx'], dict):
            for _fx_sym in _region_val['fx'].keys():
                fx_symbols.append(_fx_sym)
    fx_symbols = list(dict.fromkeys(fx_symbols))

futures_symbols = get_all_symbols_from_main_category(variables_config, 'futures')
economic_indicators_symbols = get_all_symbols_from_main_category(variables_config, 'economic_indicators')

# Load frequency metadata for all economic indicators
print(f"="*120)
print("Loading series catalog for configured FRED indicators...")

econ_ids = list(dict.fromkeys(economic_indicators_symbols))

if len(econ_ids) > 0:
    id_list_sql_parts = []
    for sid in econ_ids:
        id_list_sql_parts.append(f"'{sid}'")
    id_list_sql = ", ".join(id_list_sql_parts)
    catalog_query = f"""
    SELECT series_id, frequency_short
    FROM fred_series_catalog
    WHERE series_id IN ({id_list_sql})
"""
else:
    catalog_query = """
    SELECT series_id, frequency_short
    FROM fred_series_catalog
    WHERE 1=0
"""

try:
    catalog_df = pd.read_sql(catalog_query, con=engine)
except Exception as e:
    print(f"❌ Error querying fred_series_catalog: {e}")
    catalog_df = pd.DataFrame(columns=["series_id", "frequency_short"])

# Build fast lookup mapping
series_id_to_freq = {}
for idx in range(len(catalog_df)):
    sid = catalog_df.loc[idx, "series_id"]
    freq = str(catalog_df.loc[idx, "frequency_short"]).upper()
    series_id_to_freq[sid] = freq

print(f"✓ Loaded catalog for {len(series_id_to_freq)} series from 'fred_series_catalog'")
print(f"="*120)

# Create symbol lists
fred_symbols = economic_indicators_symbols
markets_symbols = indices_symbols + futures_symbols + fx_symbols
markets_symbols = list(dict.fromkeys(markets_symbols))

print(f"Loaded {len(symbol_lag_mapping)} symbols with individual lag configurations:")
print(f"  Markets (equities): {len(indices_symbols)}")
print(f"  FX (markets.fx): {len(fx_symbols)}")
print(f"  Futures: {len(futures_symbols)}")
print(f"  Economic Indicators: {len(economic_indicators_symbols)}")

# Create symbol mapping for consistent column naming
symbol_mapping = {}
for symbol in markets_symbols + economic_indicators_symbols:
    symbol_mapping[symbol] = sanitize_symbol(symbol)

print(f"Symbol mapping created for {len(symbol_mapping)} symbols")

# Create master index covering full history
print(f"="*120)
print("Creating Master Index...")
cfg_dates = variables_config.get('config', {}) if isinstance(variables_config.get('config', {}), dict) else {}
start_date_cfg = cfg_dates.get('start_date', '2010-01-01')
end_date_cfg = cfg_dates.get('end_date', 'today')

# Parse start_date
try:
    start_date = pd.to_datetime(start_date_cfg)
except Exception:
    start_date = pd.to_datetime('2000-01-01')

# Parse end_date, support "today"
if isinstance(end_date_cfg, str) and str(end_date_cfg).lower() == 'today':
    end_date = datetime.today()
else:
    try:
        end_date = pd.to_datetime(end_date_cfg)
    except Exception:
        end_date = datetime.today()

print(f"Configured date range: start={start_date} end={end_date}")

if resampling_freq == 'D':
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    master_index = pd.date_range(start=start_date, end=end_date, freq=us_bd)
    print(f"Created US business day index: {len(master_index)} days from {master_index[0]} to {master_index[-1]}")
else:
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    daily_index = pd.date_range(start=start_date, end=end_date, freq=us_bd)
    dummy_series = pd.Series(1, index=daily_index)
    resampled_dummy = dummy_series.resample(freq_mapping[resampling_freq]).last()
    master_index = resampled_dummy.index
    print(f"Created {resampling_freq} frequency index: {len(master_index)} periods from {master_index[0]} to {master_index[-1]}")

# Initialize combined_data with master index
combined_data = pd.DataFrame(index=master_index)
print(f"Combined data initialized with {resampling_freq} frequency index: {combined_data.shape}")

print(f"="*120)
print("Loading S&P 500 Data for Target Creation...")
# Load GSPC separately for target creation only - not as a feature
gspc_data = daily('^GSPC')
if not gspc_data.empty:
    print(f"  ✓ Loaded ^GSPC for target: {gspc_data.shape[0]} rows")
    gspc_close = gspc_data['Close']['^GSPC']
    gspc_close.index = pd.to_datetime(gspc_close.index)
    
    if resampling_freq != 'D':
        print(f"  Resampling GSPC data to {resampling_freq} frequency...")
        gspc_df = pd.DataFrame({'Close': gspc_close})
        gspc_resampled = gspc_df.resample(freq_mapping[resampling_freq]).last()
        gspc_close = gspc_resampled['Close']
        print(f"  ✓ Resampled GSPC: {gspc_df.shape[0]} -> {gspc_resampled.shape[0]} periods")
    
    gspc_aligned = gspc_close.reindex(combined_data.index)
    print(f"  ✓ Aligned GSPC to {resampling_freq} frequency grid")
else:
    print(f"  ❌ Failed to load GSPC data")
    gspc_aligned = None

print(f"="*120)
print("Loading, Resampling, and Lagging Markets Data...")

# Process each market symbol: download, resample, align, and apply lag
markets_aligned_dict = {}
for symbol in markets_symbols:
    if symbol == '^GSPC':  # Skip GSPC as it's handled separately
        print(f"Skipping {symbol} (used only for target creation)")
        continue
        
    print(f"Loading {symbol}...")
    raw_data = daily(symbol)
    
    if raw_data.empty:
        print(f"  ⚠️  No data for {symbol}")
        continue
        
    print(f"  ✓ Loaded {symbol}: {raw_data.shape[0]} rows")
    
    # Flatten MultiIndex columns and create new column names
    df_flat = raw_data.copy()
    df_flat = df_flat.astype('float32')
    prefix = symbol_mapping[symbol]
    new_columns = []
    
    for col in raw_data.columns:
        col_name, ticker = col
        new_col_name = f"{prefix}_{col_name}"
        new_columns.append(new_col_name)
    
    df_flat.columns = new_columns
    df_flat.index = pd.to_datetime(df_flat.index)
    
    # Resample if needed (before lagging)
    if resampling_freq != 'D':
        print(f"  Resampling {symbol} to {resampling_freq} frequency...")
        df_resampled = resample_data(df_flat, freq_mapping[resampling_freq])
        print(f"  ✓ Resampled {symbol}: {df_flat.shape[0]} -> {df_resampled.shape[0]} periods")
    else:
        df_resampled = df_flat
    
    # Align to master index
    df_aligned = df_resampled.reindex(combined_data.index)
    
    # Get lag value from configuration and apply lag
    lag_value = symbol_lag_mapping.get(symbol, 1)
    print(f"  ✓ Applying lag {lag_value} to {symbol}")
    
    # Apply lag to ALL columns for this symbol
    for col in df_aligned.columns:
        df_aligned[col] = df_aligned[col].shift(lag_value)
        markets_aligned_dict[col] = df_aligned[col]
        print(f"    ✓ Lagged {col} by {lag_value} periods")

# Add all lagged market data at once
if markets_aligned_dict:
    markets_aligned_df = pd.DataFrame(markets_aligned_dict, index=combined_data.index)
    combined_data = pd.concat([combined_data, markets_aligned_df], axis=1)

print(f"✓ Markets data loaded, aligned, and lagged")
print(f"="*120)

print(f"="*120)
print("Loading FRED Data...")

# Create dictionary to store FRED data
fred_data = {}

# Loop through each symbol and load data with correct frequency
for symbol in economic_indicators_symbols:
    try:
        frequency = series_id_to_freq.get(symbol)
        
        if frequency is not None:
            print(f"Loading {symbol} with frequency: {frequency}")
            
            try:
                fred_data[symbol] = data(symbol, frequency.lower())
                if not fred_data[symbol].empty:
                    print(f"✓ Loaded {symbol}: {fred_data[symbol].shape[0]} rows")
                else:
                    print(f"⚠️  {symbol} returned empty data")
            except Exception as fred_error:
                print(f"❌ FRED API error for {symbol}: {fred_error}")
                fred_data[symbol] = pd.DataFrame()
        else:
            print(f"⚠️  No metadata found for {symbol}, skipping...")
            fred_data[symbol] = pd.DataFrame()
            
    except Exception as e:
        print(f"❌ General error loading {symbol}: {e}")
        fred_data[symbol] = pd.DataFrame()

print(f"\nFRED data loading completed. Loaded {len([k for k, v in fred_data.items() if not v.empty])} series successfully.")
print(f"="*120)

print("Intelligently Resampling and Lagging FRED Data...")

# Add FRED data to combined_data with intelligent resampling, alignment, and lagging
fred_aligned_dict = {}
for symbol, df in fred_data.items():
    if not df.empty:
        print(f"Processing {symbol}...")
        
        df.index = pd.to_datetime(df.index)
        
        native_frequency = series_id_to_freq.get(symbol)
        if native_frequency is not None:
            native_frequency = native_frequency.upper()
            print(f"  Native frequency: {native_frequency}, Target: {resampling_freq}")
            
            if resampling_freq != 'D':
                df_resampled = resample_fred_data(df, freq_mapping[resampling_freq], native_frequency, symbol)
            else:
                df_resampled = df
        else:
            print(f"  ⚠️ No metadata found for {symbol}, using standard resampling")
            if resampling_freq != 'D':
                df_resampled = resample_data(df, freq_mapping[resampling_freq])
            else:
                df_resampled = df
        
        # Align to master index and forward fill
        aligned_series = df_resampled['value'].reindex(combined_data.index).ffill()
        
        # Get lag value from configuration and apply lag
        lag_value = symbol_lag_mapping.get(symbol, 1)
        lagged_series = aligned_series.shift(lag_value)
        fred_aligned_dict[symbol] = lagged_series
        
        print(f"  ✓ Aligned, forward-filled, and lagged {symbol} by {lag_value} periods")

# Add all lagged FRED data at once
if fred_aligned_dict:
    fred_aligned_df = pd.DataFrame(fred_aligned_dict, index=combined_data.index)
    combined_data = pd.concat([combined_data, fred_aligned_df], axis=1)

print(f"✓ FRED data loaded, aligned, and lagged")
print(f"="*120)

print(f"="*120)
if resampling_freq == 'D':
    print("Creating Target Variable (GSPC Next-Day Log Return)...")
else:
    print(f"Creating Target Variable (GSPC Next-Period Log Return for {resampling_freq} frequency)...")
print(f"="*120)

# Create target variable early, before any feature processing
if gspc_aligned is not None:
    gspc_returns = np.log(gspc_aligned / gspc_aligned.shift(1))
    target_next_period = gspc_returns.shift(-1)
    
    combined_data['GSPC_log_return_next_period'] = target_next_period
    target_name = 'GSPC_log_return_next_period'
    period_desc = {'D': 'day', 'W': 'week', 'M': 'month', 'Q': 'quarter', 'A': 'year'}[resampling_freq]
    
    print(f"✓ Created target variable: {target_name}")
    print(f"  Target represents: log(GSPC_close[t+1] / GSPC_close[t]) for {period_desc}ly data")
    print(f"  Valid target observations: {target_next_period.notna().sum()}")
else:
    print("❌ Cannot create target - GSPC data not available")
    combined_data['GSPC_log_return_next_period'] = np.nan

# Calculate yield spreads and immediately apply lags
print("Calculating and lagging yield spreads...")
yield_spreads_dict = {}

if 'DGS2' in combined_data.columns and 'DGS10' in combined_data.columns:
    spread_2y_10y = combined_data['DGS10'] - combined_data['DGS2']
    lag_value = symbol_lag_mapping.get('yield_spread_2y_10y', 1)
    yield_spreads_dict['yield_spread_2y_10y'] = spread_2y_10y.shift(lag_value)
    print(f"✓ Calculated and lagged 2y-10y spread by {lag_value} periods")
else:
    print("⚠️  Missing DGS2 or DGS10 data for 2y-10y spread")

if 'DTB3' in combined_data.columns and 'DGS2' in combined_data.columns:
    spread_3m_2y = combined_data['DGS2'] - combined_data['DTB3']
    lag_value = symbol_lag_mapping.get('yield_spread_3m_2y', 1)
    yield_spreads_dict['yield_spread_3m_2y'] = spread_3m_2y.shift(lag_value)
    print(f"✓ Calculated and lagged 3m-2y spread by {lag_value} periods")
else:
    print("⚠️  Missing DTB3 or DGS2 data for 3m-2y spread")

# Add lagged yield spreads
if yield_spreads_dict:
    yield_spreads_df = pd.DataFrame(yield_spreads_dict, index=combined_data.index)
    combined_data = pd.concat([combined_data, yield_spreads_df], axis=1)

# Remove original yield columns used in spread calculations
yield_columns_to_remove = ['DGS2', 'DGS10', 'DTB3']
for col in yield_columns_to_remove:
    if col in combined_data.columns:
        combined_data = combined_data.drop(columns=[col])
        print(f"✓ Removed {col} (used in spread calculation)")

print(f"✓ Yield spreads calculated and lagged")

# Filter data from a specific date
date = '2005-01-01'
combined_data = combined_data[combined_data.index >= date]

# Final NaN analysis and reporting
print(f"="*120)
print("API DATA COLLECTION - NaN ANALYSIS REPORT")
print(f"="*120)

total_nans = combined_data.isna().sum().sum()
total_cells = combined_data.shape[0] * combined_data.shape[1]
print(f"Dataframe shape: {combined_data.shape}")
print(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
print(f"Total cells: {total_cells:,}")
print(f"Total NaNs: {total_nans}")
print(f"Data completeness: {((total_cells - total_nans) / total_cells) * 100:.2f}%")

# Columns with NaNs
columns_with_nans = combined_data.columns[combined_data.isna().any()].tolist()
print(f"\nColumns with NaNs ({len(columns_with_nans)} total):")
for col in columns_with_nans:
    nan_count = combined_data[col].isna().sum()
    nan_pct = (nan_count / len(combined_data)) * 100
    print(f"  {col}: {nan_count} NaNs ({nan_pct:.2f}%)")

# Target variable analysis
if 'GSPC_log_return_next_period' in combined_data.columns:
    target_nans = combined_data['GSPC_log_return_next_period'].isna().sum()
    target_valid = combined_data['GSPC_log_return_next_period'].notna().sum()
    print(f"\nTarget variable analysis:")
    print(f"  Valid observations: {target_valid}")
    print(f"  NaN observations: {target_nans}")
    print(f"  NaN percentage: {(target_nans / len(combined_data)) * 100:.2f}%")

print(f"="*120)
print("API DATA COLLECTION COMPLETED")
print(f"="*120)

# Return the prepared data for feature calculation
if __name__ == "__main__":
    print("API data collection completed. Data ready for feature calculation.")
    print(f"Final data shape: {combined_data.shape}")
    print(f"Columns: {list(combined_data.columns)}")

