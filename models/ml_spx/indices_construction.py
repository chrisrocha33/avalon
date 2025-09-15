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

def sanitize_symbol(symbol):
    """Convert raw symbol to column prefix format"""
    return symbol.replace('^', '').replace('=', '').replace('.', '_').replace('-', '_')

# -----------------------------------------------------------------------------
# Configuration and setup (mirrors data_prep.py where relevant)
# -----------------------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
API_KEY = "b0dfab0fd08d5f0dfe0609230c5f7041 "

# Initialize FRED
fred = None
try:
    os.environ['FRED_API_KEY'] = API_KEY.strip()
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
# SQL connection
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

print(f"="*120)
print("Loading Z-Scores Data...")
print(f"="*120)
#Loading Zscore data
zscore_df = pd.read_sql_table("ml_spx_zscores", con=engine)

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

print(f"Z-Scores Columns: {len(zscore_df.columns)}")
print(f"Z-Scores Rows: {len(zscore_df)}")
print(f"="*120)

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
if 'GSPC_log_return_next_period' in zscore_df.columns:
    combined_regional_indices['GSPC_log_return_next_period'] = zscore_df['GSPC_log_return_next_period']
    print(f"Successfully added GSPC_log_return_next_period column")
else:
    print(f"Warning: GSPC_log_return_next_period column not found in zscore_df")
    print(f"Available columns with 'GSPC': {[col for col in zscore_df.columns if 'GSPC' in col]}")

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

print(f"="*120)
print(f"Master Df Columns: {len(master_dataframe.columns)}")
print(f"="*120)
print(f"Master Df Columns: {master_dataframe.columns.tolist()}")
print(f"="*120)
print(f"Master Df Rows: {len(master_dataframe)}")

