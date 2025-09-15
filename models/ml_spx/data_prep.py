import yfinance as yf
from full_fred.fred import Fred
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import statsmodels.api as sm
from datetime import datetime
import json
import os

# FRED API Key Configuration
API_KEY = "b0dfab0fd08d5f0dfe0609230c5f7041"

# Initialize FRED API
try:
    os.environ['FRED_API_KEY'] = API_KEY.strip()
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
    # Make 'currencies' optional to allow configs without FX section
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

# Load top features configuration
top_features_count = variables_config.get('config', {}).get('top_features', 50)
if not isinstance(top_features_count, int) or top_features_count <= 0:
    raise ValueError(f"Invalid top_features count '{top_features_count}'. Must be a positive integer.")
print(f"Top features count configured: {top_features_count}")

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
    
    Returns:
    - 'faster': native frequency is higher than target (e.g., daily vs weekly)
    - 'slower': native frequency is lower than target (e.g., monthly vs weekly)  
    - 'same': frequencies are the same
    """
    # Define frequency hierarchy (higher number = higher frequency)
    freq_hierarchy = {
        'D': 5,   # Daily
        'W': 4,   # Weekly
        'M': 3,   # Monthly
        'Q': 2,   # Quarterly
        'A': 1    # Annual
    }
    
    # Handle pandas frequency strings
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
    
    Parameters:
    - df: DataFrame to resample
    - target_freq: Target frequency ('D', 'W-FRI', 'ME', 'Q', 'A')
    - native_freq: Native frequency of the FRED series ('D', 'W', 'M', 'Q', 'A')
    - symbol: Symbol name for logging
    
    Returns:
    - Resampled DataFrame
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
    
    Parameters:
    - df: DataFrame to resample
    - freq: Target frequency ('D', 'W-FRI', 'ME', 'Q', 'A')
    
    Returns:
    - Resampled DataFrame
    """
    if freq == 'D':
        return df  # No resampling needed for daily
    
    print(f"  Resampling data to {freq}")
    
    resampled = df.resample(freq).last()
    
    print(f"    Resampled from {df.shape[0]} to {resampled.shape[0]} periods")
    return resampled

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


# Extract symbols and lag information from JSON configuration
def extract_symbols_from_config(config):
    """Extract all symbols and their lag values from the configuration"""
    symbols = {}
    
    def extract_recursive(data, path=""):
        for key, value in data.items():
            if isinstance(value, dict):
                if 'lag' in value:
                    # This is a symbol with lag info
                    symbols[key] = value['lag']
                else:
                    # This is a category, recurse
                    extract_recursive(value, f"{path}.{key}" if path else key)
    
    extract_recursive(config)
    return symbols

# Extract all symbols and their lag values
symbol_lag_mapping = extract_symbols_from_config(variables_config)

# Create symbol lists for backward compatibility
def get_symbols_by_category(config, category_path):
    """Extract symbols from a specific category path"""
    symbols = []
    
    def navigate_path(data, path_parts):
        current = data
        for part in path_parts:
            if part in current:
                current = current[part]
            else:
                return {}
        return current
    
    def extract_symbols(data):
        result = []
        for key, value in data.items():
            if isinstance(value, dict):
                if 'lag' in value:
                    result.append(key)
                else:
                    result.extend(extract_symbols(value))
        return result
    
    category_data = navigate_path(config, category_path.split('.'))
    return extract_symbols(category_data)

# Extract symbols by main category automatically
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
                    # This is a symbol with lag info
                    result.append(key)
                else:
                    # This is a subcategory, recurse
                    result.extend(extract_symbols_recursive(value))
        return result
    
    return extract_symbols_recursive(config[main_category])

# Extract symbols by main category automatically (markets → equities only)
indices_symbols = get_all_symbols_from_main_category(variables_config, 'markets')
# Exclude FX pairs under markets to keep equities-only for market OHLCV features
indices_symbols = [s for s in indices_symbols if not str(s).endswith('=X')]

# Extract FX symbols nested under markets.<region>.fx for processing
fx_symbols = []
if 'markets' in variables_config and isinstance(variables_config['markets'], dict):
    for _region_key, _region_val in variables_config['markets'].items():
        if isinstance(_region_val, dict) and 'fx' in _region_val and isinstance(_region_val['fx'], dict):
            for _fx_sym in _region_val['fx'].keys():
                fx_symbols.append(_fx_sym)
    # De-duplicate while preserving order
    fx_symbols = list(dict.fromkeys(fx_symbols))
futures_symbols = get_all_symbols_from_main_category(variables_config, 'futures')

# Extract ALL economic indicator categories automatically
economic_indicators_symbols = get_all_symbols_from_main_category(variables_config, 'economic_indicators')

# -----------------------------------------------------------------
# Load frequency metadata for all economic indicators in one query
# -----------------------------------------------------------------
print(f"="*120)
print("Loading series catalog for configured FRED indicators...")

# Build list of required series_ids (de-duplicated, preserve order)
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

# Also extract by individual categories for detailed logging
yields_symbols = get_symbols_by_category(variables_config, 'economic_indicators.yields')
credit_spreads_symbols = get_symbols_by_category(variables_config, 'economic_indicators.credit_spreads')
prices_symbols = get_symbols_by_category(variables_config, 'economic_indicators.prices')
industrial_production_symbols = get_symbols_by_category(variables_config, 'economic_indicators.industrial_production')
labor_market_symbols = get_symbols_by_category(variables_config, 'economic_indicators.labor_market')
housing_market_symbols = get_symbols_by_category(variables_config, 'economic_indicators.housing_market')
trade_indicators_symbols = get_symbols_by_category(variables_config, 'economic_indicators.trade_indicators')
business_activity_symbols = get_symbols_by_category(variables_config, 'economic_indicators.business_activity')

# Use all economic indicators for FRED symbols
fred_symbols = economic_indicators_symbols
markets_symbols = indices_symbols + futures_symbols + fx_symbols
# Deduplicate while preserving order
markets_symbols = list(dict.fromkeys(markets_symbols))

# Derived variables (yield spreads will be created during processing)
yield_spreads = get_symbols_by_category(variables_config, 'derived_variables.yield_spreads')

print(f"Loaded {len(symbol_lag_mapping)} symbols with individual lag configurations:")
print(f"  Markets (equities): {len(indices_symbols)} (auto-extracted from all subcategories)")
print(f"  FX (markets.fx): {len(fx_symbols)} (now included in processing)")
print(f"  Futures: {len(futures_symbols)} (auto-extracted from all subcategories)")
print(f"  Economic Indicators: {len(economic_indicators_symbols)} (auto-extracted from all subcategories)")
print(f"  Yield spreads: {len(yield_spreads)}")

# Show detailed breakdown of economic indicators
print(f"\nEconomic Indicators breakdown:")
print(f"  Yields: {len(yields_symbols)} symbols")
print(f"  Credit Spreads: {len(credit_spreads_symbols)} symbols")
print(f"  Prices: {len(prices_symbols)} symbols")
print(f"  Industrial Production: {len(industrial_production_symbols)} symbols")
print(f"  Labor Market: {len(labor_market_symbols)} symbols")
print(f"  Housing Market: {len(housing_market_symbols)} symbols")
print(f"  Trade Indicators: {len(trade_indicators_symbols)} symbols")
print(f"  Business Activity: {len(business_activity_symbols)} symbols")

# Show what subcategories were automatically processed
print(f"\nAuto-extracted subcategories:")
if 'markets' in variables_config:
    markets_subcategories = [key for key in variables_config['markets'].keys()]
    print(f"  Markets subcategories: {markets_subcategories}")
if 'futures' in variables_config:
    futures_subcategories = [key for key in variables_config['futures'].keys()]
    print(f"  Futures subcategories: {futures_subcategories}")
if 'economic_indicators' in variables_config:
    econ_subcategories = [key for key in variables_config['economic_indicators'].keys()]
    print(f"  Economic Indicators subcategories: {econ_subcategories}")

# Display lag summary
lag_summary = {}
for symbol, lag in symbol_lag_mapping.items():
    if lag not in lag_summary:
        lag_summary[lag] = 0
    lag_summary[lag] += 1

print(f"\nLag distribution:")
for lag_value, count in sorted(lag_summary.items()):
    print(f"  Lag {lag_value}: {count} symbols")

# Create symbol mapping for consistent column naming
symbol_mapping = {}
for symbol in markets_symbols + economic_indicators_symbols:
    symbol_mapping[symbol] = sanitize_symbol(symbol)

print(f"Symbol mapping created for {len(symbol_mapping)} symbols")

# Display detailed configuration summary
print(f"\n{'='*80}")
print("CONFIGURATION SUMMARY")
print(f"{'='*80}")
print(f"Total symbols loaded: {len(symbol_lag_mapping)}")
print(f"Market symbols: {len(markets_symbols)}")
print(f"Economic Indicator symbols: {len(economic_indicators_symbols)}")
print(f"Derived variables: {len(yield_spreads)}")
print(f"\nSample symbols by category:")
print(f"  Markets (equities) (first 3): {indices_symbols[:3]}")
print(f"  Futures (first 3): {futures_symbols[:3]}")
print(f"  Economic Indicators (first 3): {economic_indicators_symbols[:3]}")
if yield_spreads:
    print(f"  Yield spreads: {yield_spreads}")
print(f"\nEconomic Indicators by subcategory:")
if yields_symbols:
    print(f"  Yields: {yields_symbols}")
if prices_symbols:
    print(f"  Prices: {prices_symbols}")
if labor_market_symbols:
    print(f"  Labor Market: {labor_market_symbols}")
if housing_market_symbols:
    print(f"  Housing Market: {housing_market_symbols}")
print(f"{'='*80}")

print(f"="*120)
print("EARLY LAG VALIDATION")
print(f"="*120)

# Validate that all symbols in configuration will be processed
symbols_to_be_lagged = []
symbols_not_found = []

for symbol in symbol_lag_mapping.keys():
    if symbol == '^GSPC':
        continue  # Skip target symbol
    elif symbol in markets_symbols:
        # Determine if it's equity, futures, or FX
        if symbol in indices_symbols:
            symbols_to_be_lagged.append(f"{symbol} (equity)")
        elif symbol in futures_symbols:
            symbols_to_be_lagged.append(f"{symbol} (futures)")
        elif symbol in fx_symbols:
            symbols_to_be_lagged.append(f"{symbol} (fx)")
        else:
            symbols_to_be_lagged.append(f"{symbol} (market)")
    elif symbol in economic_indicators_symbols:
        symbols_to_be_lagged.append(f"{symbol} (FRED)")
    elif symbol in yield_spreads:
        symbols_to_be_lagged.append(f"{symbol} (derived)")
    else:
        symbols_not_found.append(symbol)

print(f"✅ Symbols to be lagged: {len(symbols_to_be_lagged)}")
for symbol in symbols_to_be_lagged[:10]:  # Show first 10
    print(f"  {symbol}")
if len(symbols_to_be_lagged) > 10:
    print(f"  ... and {len(symbols_to_be_lagged) - 10} more")

if symbols_not_found:
    print(f"⚠️  Symbols in config but not found in any category: {symbols_not_found}")
    
print(f"✅ Early lag validation completed")
print(f"="*120)

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
    # Use US business days for daily frequency
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    master_index = pd.date_range(start=start_date, end=end_date, freq=us_bd)
    print(f"Created US business day index: {len(master_index)} days from {master_index[0]} to {master_index[-1]}")
else:
    # For other frequencies, start with daily business days then resample
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    daily_index = pd.date_range(start=start_date, end=end_date, freq=us_bd)
    # Create a dummy series to resample the index
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
    # Extract close price and ensure datetime index
    gspc_close = gspc_data['Close']['^GSPC']
    gspc_close.index = pd.to_datetime(gspc_close.index)
    
    # Resample if needed
    if resampling_freq != 'D':
        print(f"  Resampling GSPC data to {resampling_freq} frequency...")
        gspc_df = pd.DataFrame({'Close': gspc_close})
        # Use 'last' aggregation for price levels, not market OHLCV aggregation
        gspc_resampled = gspc_df.resample(freq_mapping[resampling_freq]).last()
        gspc_close = gspc_resampled['Close']
        print(f"  ✓ Resampled GSPC: {gspc_df.shape[0]} -> {gspc_resampled.shape[0]} periods")
    
    # Align to master index
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
    lag_value = symbol_lag_mapping.get(symbol, 1)  # Default to 1 if not found
    print(f"  ✓ Applying lag {lag_value} to {symbol}")
    
    # Apply lag to ALL columns for this symbol
    for col in df_aligned.columns:
        # Apply lag after resampling and alignment
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
        # Find the frequency for this symbol using preloaded mapping
        frequency = series_id_to_freq.get(symbol)
        
        if frequency is not None:
            print(f"Loading {symbol} with frequency: {frequency}")
            
            # Test FRED connection with a simple call first
            try:
                # Load data using the data function (no pre-ffilling)
                fred_data[symbol] = data(symbol, frequency.lower())
                if not fred_data[symbol].empty:
                    print(f"✓ Loaded {symbol}: {fred_data[symbol].shape[0]} rows")
                else:
                    print(f"⚠️  {symbol} returned empty data")
            except Exception as fred_error:
                print(f"❌ FRED API error for {symbol}: {fred_error}")
                # Try to provide more specific error information
                if "API key" in str(fred_error).lower():
                    print(f"   Suggestion: Check FRED API key configuration")
                elif "rate limit" in str(fred_error).lower():
                    print(f"   Suggestion: FRED API rate limit reached, wait and retry")
                elif "not found" in str(fred_error).lower():
                    print(f"   Suggestion: Series {symbol} may not exist or be discontinued")
                fred_data[symbol] = pd.DataFrame()  # Empty dataframe as placeholder
        else:
            print(f"⚠️  No metadata found for {symbol}, skipping...")
            fred_data[symbol] = pd.DataFrame()
            
    except Exception as e:
        print(f"❌ General error loading {symbol}: {e}")
        fred_data[symbol] = pd.DataFrame()  # Empty dataframe as placeholder

print(f"\nFRED data loading completed. Loaded {len([k for k, v in fred_data.items() if not v.empty])} series successfully.")
print(f"="*120)

print("Intelligently Resampling and Lagging FRED Data...")

# Add FRED data to combined_data with intelligent resampling, alignment, and lagging
fred_aligned_dict = {}
for symbol, df in fred_data.items():
    if not df.empty:
        print(f"Processing {symbol}...")
        
        # Ensure the FRED data index is datetime
        df.index = pd.to_datetime(df.index)
        
        # Get native frequency from preloaded mapping for intelligent resampling
        native_frequency = series_id_to_freq.get(symbol)
        if native_frequency is not None:
            native_frequency = native_frequency.upper()
            print(f"  Native frequency: {native_frequency}, Target: {resampling_freq}")
            
            # Intelligently resample based on frequency comparison
            if resampling_freq != 'D':
                df_resampled = resample_fred_data(df, freq_mapping[resampling_freq], native_frequency, symbol)
            else:
                df_resampled = df
        else:
            print(f"  ⚠️ No metadata found for {symbol}, using standard resampling")
            # Fallback to standard resampling if no metadata
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
    # Calculate log returns from GSPC close prices
    gspc_returns = np.log(gspc_aligned / gspc_aligned.shift(1))
    # Create next-period target by shifting forward
    target_next_period = gspc_returns.shift(-1)
    
    # Add target to combined_data
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

# Note: DGS2, DGS10, DTB3 are already lagged from FRED section
# We need to create spreads from the lagged data, then apply additional lag to the spread itself

if 'DGS2' in combined_data.columns and 'DGS10' in combined_data.columns:
    # Create spread from already-lagged components
    spread_2y_10y = combined_data['DGS10'] - combined_data['DGS2']
    # Apply additional lag to the spread itself
    lag_value = symbol_lag_mapping.get('yield_spread_2y_10y', 1)
    yield_spreads_dict['yield_spread_2y_10y'] = spread_2y_10y.shift(lag_value)
    print(f"✓ Calculated and lagged 2y-10y spread by {lag_value} periods")
else:
    print("⚠️  Missing DGS2 or DGS10 data for 2y-10y spread")

if 'DTB3' in combined_data.columns and 'DGS2' in combined_data.columns:
    # Create spread from already-lagged components  
    spread_3m_2y = combined_data['DGS2'] - combined_data['DTB3']
    # Apply additional lag to the spread itself
    lag_value = symbol_lag_mapping.get('yield_spread_3m_2y', 1)
    yield_spreads_dict['yield_spread_3m_2y'] = spread_3m_2y.shift(lag_value)
    print(f"✓ Calculated and lagged 3m-2y spread by {lag_value} periods")
else:
    print("⚠️  Missing DTB3 or DGS2 data for 3m-2y spread")

# Add lagged yield spreads
if yield_spreads_dict:
    yield_spreads_df = pd.DataFrame(yield_spreads_dict, index=combined_data.index)
    combined_data = pd.concat([combined_data, yield_spreads_df], axis=1)

# Remove original yield columns used in spread calculations (they're already lagged in the FRED section)
yield_columns_to_remove = ['DGS2', 'DGS10', 'DTB3']
for col in yield_columns_to_remove:
    if col in combined_data.columns:
        combined_data = combined_data.drop(columns=[col])
        print(f"✓ Removed {col} (used in spread calculation)")

print(f"✓ Yield spreads calculated and lagged")

print(f"="*120)
print("Computing Market Features from LAGGED Data...")
print(f"="*120)

# NOTE: All market data is already lagged at this point
# Feature engineering will automatically use lagged inputs

# Calculate log returns for all market symbols (data already lagged)
market_columns = [col for col in combined_data.columns if any(ohlcv in col for ohlcv in ['Open', 'High', 'Low', 'Close', 'Volume'])]
market_symbols = set()
for col in market_columns:
    # Extract symbol from column name (format: SYMBOL_OHLCV)
    parts = col.split('_')
    if len(parts) >= 2:
        symbol = '_'.join(parts[:-1])  # Everything except the last part (OHLCV)
        # Skip GSPC symbols - they shouldn't be in features anyway since we excluded GSPC from loading
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
    
    # 4. Log Volume % Change: log(Volume / Volume.shift(1))
    # volume_ratio = volume_data / volume_data.shift(1)
    # volume_ratio = volume_ratio.replace([0, np.inf, -np.inf], np.nan)
    # log_vol_change = np.log(volume_ratio)
    # new_features_dict[f"{symbol}_log_vol_change"] = log_vol_change

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
    
    print(f"  ✓ Calculated {len(new_features_dict)} features for {symbol}")

# Create new_features_data dataframe
print("Creating features dataframe...")
new_features_data = pd.DataFrame(new_features_dict, index=combined_data.index)

print(f"Calculated features for {len(market_symbols)} symbols from LAGGED data (GSPC excluded from features)")

# Add new features to combined_data
print("Adding calculated features to dataset...")
combined_data = pd.concat([combined_data, new_features_data], axis=1)

# Remove original OHLCV columns (which were already lagged)
print("Removing original lagged OHLCV columns...")
ohlcv_columns = [col for col in combined_data.columns if any(ohlcv in col for ohlcv in ['Open', 'High', 'Low', 'Close', 'Volume'])]
combined_data = combined_data.drop(columns=ohlcv_columns)

print(f"Replaced lagged OHLCV columns with calculated features")

# ------------------------------------------------------------------
# Compute log % change for ECONOMIC INDICATORS (EXCLUDING YIELDS & CREDIT SPREADS)
# UNRATE should remain as level (no % change feature)
# Done at the same stage as market feature calculations
# ------------------------------------------------------------------
print("Computing log % change for economic indicators (excluding yields, credit spreads, and UNRATE)...")

# Determine non-yield, non-credit-spread economic indicators to transform
non_yield_econ_symbols = []
for sym in economic_indicators_symbols:
    if (
        sym in combined_data.columns
        and sym not in (yields_symbols or [])
        and sym not in (credit_spreads_symbols or [])
        and sym != 'UNRATE'
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

if len(econ_change_features) > 0:
    econ_change_df = pd.DataFrame(econ_change_features, index=combined_data.index)
    combined_data = pd.concat([combined_data, econ_change_df], axis=1)
    print(f"✓ Added {len(econ_change_features)} economic indicator log % change features")
    # Remove original level columns for these indicators (keep UNRATE, yields, credit spreads)
    cols_to_drop = [sym for sym in non_yield_econ_symbols if sym in combined_data.columns]
    if len(cols_to_drop) > 0:
        combined_data = combined_data.drop(columns=cols_to_drop)
        print(f"✓ Removed {len(cols_to_drop)} original economic indicator columns")
else:
    print("No eligible economic indicators found for log % change computation")

# GSPC is already handled separately as target only - no GSPC features to remove
print("GSPC target variable already created separately - no GSPC features to remove")
print(f"Current dataframe shape: {combined_data.shape}")
print(f"✅ All features are now derived from properly lagged input data")

# Verify target exists
if 'GSPC_log_return_next_period' in combined_data.columns:
    print(f"✓ Target variable confirmed: GSPC_log_return_next_period")
else:
    print(f"⚠️  Target variable missing: GSPC_log_return_next_period")

print(f"="*120)
print("LAGGING STATUS SUMMARY")
print(f"="*120)

# Since all data is already lagged during loading, just verify target exists
target_col_name = target_name if 'target_name' in locals() else 'GSPC_log_return_next_period'
if target_col_name in combined_data.columns:
    target_valid_count = combined_data[target_col_name].notna().sum()
    print(f"✅ Target variable confirmed: {target_col_name}")
    print(f"  Valid target observations: {target_valid_count}")
    print(f"  Target represents: log(GSPC_close[t+1] / GSPC_close[t]) for {resampling_freq} frequency")
    print(f"✅ All features are already resampled and lagged during data loading phase")
    print(f"  Market data: Resampled to {resampling_freq}, then lagged immediately after")
    print(f"  FRED data: Resampled to {resampling_freq}, then lagged immediately after alignment") 
    print(f"  Derived variables: Lagged immediately after calculation")
    
    # Validation: Ensure proper timing relationship
    print(f"\n📊 TIMING VALIDATION:")
    target_last_valid = combined_data[target_col_name].last_valid_index()
    print(f"  Target last valid date: {target_last_valid}")
    
    # Check a few feature columns to ensure they don't extend beyond target
    sample_features = [col for col in combined_data.columns if col != target_col_name][:5]
    timing_issues = []
    
    for col in sample_features:
        col_last_valid = combined_data[col].last_valid_index()
        if col_last_valid is not None and target_last_valid is not None:
            if col_last_valid > target_last_valid:
                timing_issues.append(f"{col}: {col_last_valid}")
            else:
                print(f"  ✓ {col}: Last valid {col_last_valid} (proper timing)")
    
    if timing_issues:
        print(f"  ⚠️ Potential timing issues:")
        for issue in timing_issues:
            print(f"    {issue}")
    else:
        print(f"  ✅ No timing issues detected in sample features")
        
else:
    print("❌ Target variable missing - this should not happen")

print(f"="*120)
print("Final Data Cleanup")
print(f"="*120)

# Drop rows and columns that are completely NaN
combined_data = combined_data.dropna(how='all')

# Proper NaN handling: Forward fill levels, keep return NaNs as NaN
# Identify level columns (macro data, yield spreads) vs return columns
level_columns = []
return_columns = []
target_column = target_col_name if 'target_col_name' in locals() else 'GSPC_log_return_next_period'

for col in combined_data.columns:
    if col == target_column:
        continue  # Skip target variable
    elif 'yield_spread' in col or col in economic_indicators_symbols:
        level_columns.append(col)
    elif any(feature in col for feature in ['log_return', 'intraday', 'overnight', 'log_vol_change', 'log_pct_change']):
        return_columns.append(col)

print(f"Forward-filling {len(level_columns)} level columns (macro/yield data)")
print(f"Keeping {len(return_columns)} return columns with NaN for non-trading days")
print(f"Target column '{target_column}' handled separately")

# Forward fill only level columns (macro indicators, yield spreads)
for col in level_columns:
    before_ffill = combined_data[col].notna().sum()
    combined_data[col] = combined_data[col].ffill()
    after_ffill = combined_data[col].notna().sum()
    print(f"  ✓ Forward-filled {col}: {before_ffill} -> {after_ffill} valid observations")

# Keep return columns as-is (NaN represents non-trading days)
print(f"Return columns kept with original NaN values:")
for col in return_columns:
    valid_count = combined_data[col].notna().sum()
    print(f"  - {col}: {valid_count} valid observations")

# Handle target variable NaNs (keep as NaN - will be handled in model fitting)
target_valid_count = combined_data[target_column].notna().sum()
print(f"Target variable '{target_column}': {target_valid_count} valid observations")

# Filter data from a specific date
date = '2005-01-01'
combined_data = combined_data[combined_data.index >= date]

# Count NaNs before additional forward-filling
print(f"="*120)
print("Pre-Database NaN Analysis")
print(f"="*120)
total_nans_before = combined_data.isna().sum().sum()
print(f"Total NaNs before additional forward-fill: {total_nans_before}")
print(f"Percentage of NaNs: {(total_nans_before / (combined_data.shape[0] * combined_data.shape[1])) * 100:.2f}%")

# Show NaN counts by column
nan_counts = combined_data.isna().sum()
columns_with_nans = nan_counts[nan_counts > 0].sort_values(ascending=False)
if len(columns_with_nans) > 0:
    print(f"\nColumns with NaNs (top 10):")
    for col, count in columns_with_nans.head(10).items():
        pct = (count / len(combined_data)) * 100
        print(f"  {col}: {count} NaNs ({pct:.1f}%)")

print(f"\n{'='*120}")
print("Column Data Types")
print(f"{'='*120}")
for col in combined_data.columns:
    dtype_value = combined_data[col].dtype
    print(f"  {col}: {dtype_value}")

# Additional forward-fill for database storage
print(f"\n{'='*120}")
print("Additional Forward-Fill for Database Storage")
print(f"{'='*120}")

# Forward-fill return columns that still have NaNs (for database storage)
print(f"Forward-filling remaining NaNs in return columns for database storage...")
for col in return_columns:
    before_ffill = combined_data[col].notna().sum()
    combined_data[col] = combined_data[col].ffill()
    after_ffill = combined_data[col].notna().sum()
    if before_ffill != after_ffill:
        print(f"  ✓ Forward-filled {col}: {before_ffill} -> {after_ffill} valid observations")

# Forward-fill target column for database storage (but keep track of original NaNs)
print(f"Forward-filling target column for database storage...")
target_original_nans = combined_data[target_column].isna().sum()
combined_data[target_column] = combined_data[target_column].ffill()
target_after_ffill = combined_data[target_column].isna().sum()
print(f"  ✓ Target forward-filled: {target_original_nans} -> {target_after_ffill} NaNs remaining")


# Final NaN check
total_nans_final = combined_data.isna().sum().sum()
if total_nans_final > 0:
    print(f"⚠️  Warning: {total_nans_final} NaNs still remain after forward fill")
    # Fill any remaining NaNs with 0 (should be minimal)
    combined_data = combined_data.fillna(0)
    print(f"  ✓ Filled remaining NaNs with 0")
else:
    print(f"✓ All NaNs successfully filled")

# Defragment the DataFrame for optimal performance
print("Defragmenting DataFrame for optimal performance...")
combined_data = combined_data.copy()
print("✓ DataFrame defragmented")

# Final statistics
total_cells = combined_data.shape[0] * combined_data.shape[1]
final_nans = combined_data.isna().sum().sum()

print(f"\n{'='*120}")
print(f"Final Database Preparation Summary - RESAMPLING + LAGGING APPROACH")
print(f"{'='*120}")
print(f"")
print(f"Resampling frequency: {resampling_freq}")
print(f"Dataframe shape: {combined_data.shape}")
print(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
print(f"Total cells: {total_cells:,}")
print(f"Final NaNs: {final_nans}")
print(f"Data completeness: {((total_cells - final_nans) / total_cells) * 100:.2f}%")

# ========================================================================
# DATA COMPRESSION FOR POSTGRESQL ROW SIZE LIMIT
# ========================================================================
print(f"="*120)
print("APPLYING DATA COMPRESSION FOR POSTGRESQL ROW SIZE LIMIT")
print(f"="*120)

def optimize_dataframe_dtypes(df):
    """
    Optimize DataFrame dtypes to minimize memory usage and row size
    """
    print("Optimizing data types for compression...")
    
    # Create a copy to avoid modifying original
    df_optimized = df.copy()
    
    # Track compression savings
    original_memory = df.memory_usage(deep=True).sum()
    compression_stats = {}
    
    for col in df_optimized.columns:
        col_data = df_optimized[col]
        original_dtype = col_data.dtype
        
        # Skip if already optimized
        if col_data.dtype.name.startswith(('int8', 'int16', 'float16', 'float32')):
            continue
            
        # For numeric columns, try to downcast
        if pd.api.types.is_numeric_dtype(col_data):
            # Check if we can use smaller integer types
            if col_data.dtype.name.startswith('int'):
                if col_data.min() >= -128 and col_data.max() <= 127:
                    df_optimized[col] = col_data.astype('int8')
                    compression_stats[col] = f"{original_dtype} -> int8"
                elif col_data.min() >= -32768 and col_data.max() <= 32767:
                    df_optimized[col] = col_data.astype('int16')
                    compression_stats[col] = f"{original_dtype} -> int16"
                elif col_data.min() >= -2147483648 and col_data.max() <= 2147483647:
                    df_optimized[col] = col_data.astype('int32')
                    compression_stats[col] = f"{original_dtype} -> int32"
            
            # For float columns, try float32 or float16
            elif col_data.dtype.name.startswith('float'):
                # Check if we can use float16 (half precision)
                if not col_data.isna().all():
                    col_min = col_data.min()
                    col_max = col_data.max()
                    # float16 range: approximately -65504 to 65504
                    if (col_min >= -60000 and col_max <= 60000 and 
                        not np.isinf(col_data).any()):
                        try:
                            df_optimized[col] = col_data.astype('float16')
                            compression_stats[col] = f"{original_dtype} -> float16"
                        except (OverflowError, ValueError):
                            # Fallback to float32
                            df_optimized[col] = col_data.astype('float32')
                            compression_stats[col] = f"{original_dtype} -> float32"
                    else:
                        # Use float32 for safety
                        df_optimized[col] = col_data.astype('float32')
                        compression_stats[col] = f"{original_dtype} -> float32"
    
    # Calculate memory savings
    optimized_memory = df_optimized.memory_usage(deep=True).sum()
    memory_savings = original_memory - optimized_memory
    savings_percent = (memory_savings / original_memory) * 100
    
    print(f"Memory optimization results:")
    print(f"  Original memory: {original_memory / 1024 / 1024:.2f} MB")
    print(f"  Optimized memory: {optimized_memory / 1024 / 1024:.2f} MB")
    print(f"  Memory saved: {memory_savings / 1024 / 1024:.2f} MB ({savings_percent:.1f}%)")
    
    # Show compression stats for first 10 columns
    if compression_stats:
        print(f"  Data type optimizations (first 10):")
        for i, (col, change) in enumerate(list(compression_stats.items())[:10]):
            print(f"    {col}: {change}")
        if len(compression_stats) > 10:
            print(f"    ... and {len(compression_stats) - 10} more")
    
    return df_optimized

# Apply data type optimization
combined_data_compressed = optimize_dataframe_dtypes(combined_data)

# Additional compression: Round to reasonable precision
print("\nApplying precision rounding for additional compression...")
for col in combined_data_compressed.columns:
    if pd.api.types.is_numeric_dtype(combined_data_compressed[col]):
        # Round to 6 decimal places for most columns
        if 'log_return' in col or 'intraday' in col or 'overnight' in col:
            # Returns: 6 decimal places
            combined_data_compressed[col] = combined_data_compressed[col].round(6)
        elif 'rsi' in col or 'stoch' in col:
            # RSI/Stochastic: 2 decimal places
            combined_data_compressed[col] = combined_data_compressed[col].round(2)
        elif 'macd' in col or 'bb_z' in col:
            # MACD/Bollinger: 4 decimal places
            combined_data_compressed[col] = combined_data_compressed[col].round(4)
        else:
            # Default: 4 decimal places
            combined_data_compressed[col] = combined_data_compressed[col].round(4)

# Estimate row size
print(f"\nEstimating PostgreSQL row size...")
sample_row = combined_data_compressed.iloc[0]
estimated_row_size = 0

for col in sample_row.index:
    value = sample_row[col]
    if pd.isna(value):
        estimated_row_size += 4  # NULL value
    elif isinstance(value, (int, np.integer)):
        if combined_data_compressed[col].dtype.name == 'int8':
            estimated_row_size += 1
        elif combined_data_compressed[col].dtype.name == 'int16':
            estimated_row_size += 2
        elif combined_data_compressed[col].dtype.name == 'int32':
            estimated_row_size += 4
        else:
            estimated_row_size += 8
    elif isinstance(value, (float, np.floating)):
        if combined_data_compressed[col].dtype.name == 'float16':
            estimated_row_size += 2
        elif combined_data_compressed[col].dtype.name == 'float32':
            estimated_row_size += 4
        else:
            estimated_row_size += 8
    else:
        estimated_row_size += 8  # Default

# Add overhead for column names and metadata
column_name_overhead = sum(len(str(col)) for col in combined_data_compressed.columns)
estimated_row_size += column_name_overhead + 100  # Additional overhead

print(f"  Estimated row size: {estimated_row_size} bytes")
print(f"  PostgreSQL limit: 8160 bytes")
print(f"  Safety margin: {8160 - estimated_row_size} bytes")

if estimated_row_size > 8160:
    print(f"  ⚠️  Row size still exceeds PostgreSQL limit!")
    print(f"  Additional compression needed...")
    
    # More aggressive compression: Use even smaller data types
    print("  Applying aggressive compression...")
    
    for col in combined_data_compressed.columns:
        if pd.api.types.is_numeric_dtype(combined_data_compressed[col]):
            col_data = combined_data_compressed[col]
            
            # For very small values, try int8
            if col_data.dtype.name.startswith('float'):
                if col_data.min() >= -1 and col_data.max() <= 1:
                    # Scale to int8 range (-128 to 127)
                    try:
                        scaled = (col_data * 100).round().astype('int8')
                        combined_data_compressed[col] = scaled
                        print(f"    {col}: Scaled to int8")
                    except (OverflowError, ValueError):
                        pass
                elif col_data.min() >= -10 and col_data.max() <= 10:
                    # Scale to int16 range
                    try:
                        scaled = (col_data * 1000).round().astype('int16')
                        combined_data_compressed[col] = scaled
                        print(f"    {col}: Scaled to int16")
                    except (OverflowError, ValueError):
                        pass

# Re-estimate row size after aggressive compression
sample_row = combined_data_compressed.iloc[0]
estimated_row_size = 0

for col in sample_row.index:
    value = sample_row[col]
    if pd.isna(value):
        estimated_row_size += 4
    elif isinstance(value, (int, np.integer)):
        if combined_data_compressed[col].dtype.name == 'int8':
            estimated_row_size += 1
        elif combined_data_compressed[col].dtype.name == 'int16':
            estimated_row_size += 2
        else:
            estimated_row_size += 4
    elif isinstance(value, (float, np.floating)):
        if combined_data_compressed[col].dtype.name == 'float16':
            estimated_row_size += 2
        else:
            estimated_row_size += 4
    else:
        estimated_row_size += 4

estimated_row_size += column_name_overhead + 100

print(f"  Final estimated row size: {estimated_row_size} bytes")
print(f"  PostgreSQL limit: 8160 bytes")
print(f"  Safety margin: {8160 - estimated_row_size} bytes")

if estimated_row_size <= 8160:
    print(f"  ✅ Row size within PostgreSQL limits!")
else:
    print(f"  ⚠️  Row size still exceeds limit. Consider chunked insertion...")

print(f"="*120)

# Save to database with compression
print("Saving compressed data to database...")
try:
    # Use the compressed DataFrame
    combined_data_compressed.to_sql('ml_spx_data', engine, if_exists='replace', index=True, 
                                   method='multi', chunksize=100)
    print("✅ Data saved to database: ml_spx_data")
except Exception as e:
    print(f"❌ Error saving to database: {e}")
    print("Trying alternative approach with smaller chunks...")
    
    # Try with even smaller chunks
    try:
        combined_data_compressed.to_sql('ml_spx_data', engine, if_exists='replace', index=True, 
                                       method='multi', chunksize=50)
        print("✅ Data saved to database with smaller chunks: ml_spx_data")
    except Exception as e2:
        print(f"❌ Still failing: {e2}")
        print("Trying row-by-row insertion...")
        
        # Last resort: row-by-row insertion
        try:
            combined_data_compressed.to_sql('ml_spx_data', engine, if_exists='replace', index=True, 
                                           method=None, chunksize=1)
            print("✅ Data saved to database row-by-row: ml_spx_data")
        except Exception as e3:
            print(f"❌ All insertion methods failed: {e3}")
            print("Trying PostgreSQL TOAST approach...")
            
            # Alternative: Use PostgreSQL TOAST by creating table with TEXT columns
            try:
                # Create table with TEXT columns to enable TOAST compression
                create_table_sql = """
                DROP TABLE IF EXISTS ml_spx_data;
                CREATE TABLE ml_spx_data (
                    index TIMESTAMP PRIMARY KEY
                );
                """
                
                # Add columns as TEXT to enable TOAST
                for col in combined_data_compressed.columns:
                    create_table_sql += f'ALTER TABLE ml_spx_data ADD COLUMN "{col}" TEXT;\n'
                
                # Execute table creation
                with engine.connect() as conn:
                    conn.execute(create_table_sql)
                    conn.commit()
                
                print("✅ Created table with TEXT columns for TOAST compression")
                
                # Insert data row by row
                for idx, row in combined_data_compressed.iterrows():
                    values = [str(idx)] + [str(val) if not pd.isna(val) else 'NULL' for val in row.values]
                    placeholders = ', '.join(['%s'] * len(values))
                    insert_sql = f"INSERT INTO ml_spx_data VALUES ({placeholders})"
                    
                    with engine.connect() as conn:
                        conn.execute(insert_sql, values)
                        conn.commit()
                
                print("✅ Data saved using PostgreSQL TOAST compression: ml_spx_data")
                
            except Exception as e4:
                print(f"❌ TOAST approach also failed: {e4}")
                print("Final fallback: Split into multiple tables...")
                
                # Ultimate fallback: Split data into multiple tables
                try:
                    # Split columns into chunks of 400 columns each
                    chunk_size = 400
                    total_cols = len(combined_data_compressed.columns)
                    num_chunks = (total_cols + chunk_size - 1) // chunk_size
                    
                    print(f"Splitting {total_cols} columns into {num_chunks} tables...")
                    
                    for chunk_idx in range(num_chunks):
                        start_col = chunk_idx * chunk_size
                        end_col = min((chunk_idx + 1) * chunk_size, total_cols)
                        
                        chunk_cols = combined_data_compressed.columns[start_col:end_col]
                        chunk_data = combined_data_compressed[chunk_cols].copy()
                        
                        table_name = f'ml_spx_data_chunk_{chunk_idx}'
                        chunk_data.to_sql(table_name, engine, if_exists='replace', index=True)
                        print(f"  ✅ Saved chunk {chunk_idx + 1}/{num_chunks}: {table_name}")
                    
                    print(f"✅ Data split into {num_chunks} tables successfully")
                    print("Note: You'll need to join these tables when querying the data")
                    
                except Exception as e5:
                    print(f"❌ All approaches failed: {e5}")
                    raise

# Also compute and save z-scored version (column-wise standardization)
print("Computing z-scores for all columns and saving to ml_spx_zscores...")
try:
    # Create a robust z-scores calculation that handles problematic data
    zscores_dict = {}
    problematic_columns = []
    
    print("Processing columns for z-score calculation...")
    for col in combined_data_compressed.columns:
        try:
            # Get the column data
            col_data = combined_data_compressed[col].copy()
            
            # Check for problematic data
            if col_data.isna().all():
                print(f"  ⚠️  Column {col}: All NaN values, skipping z-score calculation")
                zscores_dict[col] = col_data  # Keep as NaN
                problematic_columns.append(f"{col} (all NaN)")
                continue
            
            # Check for infinite values
            if np.isinf(col_data).any():
                print(f"  ⚠️  Column {col}: Contains infinite values, replacing with NaN")
                col_data = col_data.replace([np.inf, -np.inf], np.nan)
            
            # Check for constant values (std = 0)
            col_std = col_data.std(ddof=0)
            if col_std == 0 or np.isnan(col_std):
                print(f"  ⚠️  Column {col}: Constant values (std={col_std}), setting z-scores to 0")
                zscores_dict[col] = pd.Series(0, index=col_data.index)
                problematic_columns.append(f"{col} (constant values)")
                continue
            
            # Calculate z-scores safely
            col_mean = col_data.mean()
            if np.isnan(col_mean):
                print(f"  ⚠️  Column {col}: Mean is NaN, skipping z-score calculation")
                zscores_dict[col] = col_data  # Keep original
                problematic_columns.append(f"{col} (NaN mean)")
                continue
            
            # Compute z-scores
            zscores = (col_data - col_mean) / col_std
            zscores_dict[col] = zscores
            
            # Validate the result
            if zscores.isna().all():
                print(f"  ⚠️  Column {col}: Z-scores are all NaN")
                problematic_columns.append(f"{col} (all NaN z-scores)")
            else:
                print(f"  ✓ Column {col}: Z-scores calculated successfully")
                
        except Exception as col_error:
            print(f"  ❌ Error processing column {col}: {col_error}")
            zscores_dict[col] = combined_data_compressed[col]  # Keep original data
            problematic_columns.append(f"{col} (error: {str(col_error)[:50]})")
    
    # Create z-scores DataFrame
    zscores_df = pd.DataFrame(zscores_dict, index=combined_data_compressed.index)
    
    # Report problematic columns
    if problematic_columns:
        print(f"\n⚠️  Problematic columns ({len(problematic_columns)}):")
        for prob_col in problematic_columns:
            print(f"    {prob_col}")
    else:
        print("✓ All columns processed successfully")
    
    # Save to database
    zscores_df.to_sql('ml_spx_zscores', engine, if_exists='replace', index=True)
    print(f"✓ Z-scores data saved to database: ml_spx_zscores")
    print(f"  Shape: {zscores_df.shape}")
    print(f"  Columns processed: {len(zscores_dict)}")
    print(f"  Problematic columns: {len(problematic_columns)}")
    
except Exception as e:
    print(f"❌ Error computing/saving z-scores: {e}")
    import traceback
    traceback.print_exc()

# Display data sample
print("\nData sample:")
print(combined_data_compressed.head())
print(f"="*120)
print("\nGSPC log return next period:")
print(combined_data_compressed['GSPC_log_return_next_period'].head())
print(f"="*120)
print("\nPipeline completed successfully")

