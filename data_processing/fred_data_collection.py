import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import requests
from scipy import stats
import re
from sklearn.linear_model import LassoCV
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import time
import warnings
import json
import sys
from typing import Iterable, List, Union, Optional, Dict
from full_fred.fred import Fred
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
key_file_path = os.path.join(script_dir, 'key.txt')

# Note: This file focuses exclusively on FRED economic data collection
# Financial market data (stocks, crypto, etc.) is handled separately by financial_data_collection.py

fred = Fred(key_file_path)
fred.set_api_key_file(key_file_path)




def data(x,y):
    a = fred.get_series_df(x,frequency=y)
    a.value = a.value.replace('.',np.nan)
    a.value = a.value.ffill()
    a.index = a.date
    a = a.drop(columns=['date','realtime_start','realtime_end'])
    a.value = a.value.astype('float')
    return a

# Load Series Catalog from JSON file
print(f"\n" + "="*60)
print("LOADING SERIES CATALOG")
print("="*60)

script_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(script_dir, "variables", "fred_series.json")

print(f"Script directory: {script_dir}")
print(f"JSON file path: {json_file_path}")
print(f"File exists: {os.path.exists(json_file_path)}")

try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        series_categories = json.load(f)
    print(f"✓ Successfully loaded series catalog from {json_file_path}")
    print(f"  Categories loaded: {list(series_categories.keys())}")
    total_series = sum(len(series_list) for series_list in series_categories.values())
    print(f"  Total series: {total_series}")
    
    # Show category breakdown
    for category, series_list in series_categories.items():
        if series_list:
            print(f"    {category}: {len(series_list)} series")
        else:
            print(f"    {category}: 0 series (empty)")
            
except FileNotFoundError:
    print(f"✗ Warning: {json_file_path} not found. Using empty catalog.", file=sys.stderr)
    series_categories = {}
except Exception as exc:
    print(f"✗ Warning: Error loading {json_file_path}: {exc}. Using empty catalog.", file=sys.stderr)
    series_categories = {}

# Extract all series IDs from categories to maintain compatibility with existing code
ids = []
for category_series in series_categories.values():
    ids.extend(category_series)

# Initialize empty DataFrames for each frequency
fred_monthly_data = pd.DataFrame()
fred_quarterly_data = pd.DataFrame()
fred_annual_data = pd.DataFrame()

# Initialize category DataFrames
category_dataframes = {}

series_metadata = {}

# First pass: collect metadata and sort by earliest start date
print("Collecting metadata and sorting by data start date...")
series_info_list = []

for series_id in ids:
    try:
        metadata = fred.get_a_series(series_id)
        series_info = metadata['seriess'][0]
        
        series_metadata[series_id] = series_info

        
        # Extract start date and frequency
        start_date = pd.to_datetime(series_info.get('observation_start', '1900-01-01'))
        frequency = series_info.get('frequency_short', '').upper()
        
        series_info_list.append({
            'id': series_id,
            'start_date': start_date,
            'frequency': frequency,
            'title': series_info.get('title', series_id)
        })
        
    except Exception as e:
        print(f"✗ Error fetching metadata for {series_id}: {e}")
        continue

#Create Metadata dataframe
metadata_df = (
    pd.DataFrame.from_dict(series_metadata, orient='index')
      .rename_axis('series_id')   
      .reset_index()              
)

# Sort by earliest start date (oldest first)
series_info_list.sort(key=lambda x: x['start_date'])

print(f"\nSorted {len(series_info_list)} series by start date:")
for i, info in enumerate(series_info_list[:10]):  # Show first 10
    print(f"{i+1:2d}. {info['id']:15s} | {info['start_date'].strftime('%Y-%m-%d')} | {info['frequency']} | {info['title'][:50]}...")
if len(series_info_list) > 10:
    print(f"... and {len(series_info_list) - 10} more series")

# Process each category separately
print(f"\nProcessing data by categories...")

# Define frequency hierarchy with numeric values (lower = faster)
frequency_hierarchy = {
    'D': 1,    # Daily (fastest)
    'W': 2,    # Weekly
    'M': 3,    # Monthly
    'Q': 4,    # Quarterly
    'A': 5     # Annual (slowest)
}

# Frequency to FRED API parameter mapping
fred_api_mapping = {
    'D': 'd',    # Daily
    'W': 'w',    # Weekly  
    'M': 'm',    # Monthly
    'Q': 'q',    # Quarterly
    'A': 'a'     # Annual
}

# Frequency to pandas frequency string mapping
pandas_freq_mapping = {
    'D': 'D',    # Daily
    'W': 'W',    # Weekly
    'M': 'ME',   # Month End
    'Q': 'QE',   # Quarter End
    'A': 'YE'    # Year End
}

# Initialize logging
log_entries = []
log_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
log_entries.append(f"\n{'='*80}")
log_entries.append(f"DATA COLLECTION RUN: {log_timestamp}")
log_entries.append(f"{'='*80}")

for category_name, category_series in series_categories.items():
    print(f"\n--- Processing category: {category_name} ---")
    
    # Collect frequency information for this category
    category_frequencies = {}
    category_metadata = {}
    
    for series_id in category_series:
        if series_id in series_metadata:
            frequency = series_metadata[series_id].get('frequency_short', '').upper()
            category_frequencies[series_id] = frequency
            category_metadata[series_id] = series_metadata[series_id]
        else:
            print(f"  ⚠ Series {series_id} not found in metadata, skipping category")
            continue
    
    if not category_frequencies:
        print(f"  ✗ No valid series found for category {category_name}")
        continue
    
    # Find the fastest frequency available in this category
    available_frequencies = set(category_frequencies.values())
    print(f"  Frequencies found: {list(available_frequencies)}")
    
    # Get the fastest frequency (lowest numeric value)
    fastest_freq = None
    fastest_value = float('inf')
    
    for freq in available_frequencies:
        if freq in frequency_hierarchy:
            freq_value = frequency_hierarchy[freq]
            if freq_value < fastest_value:
                fastest_value = freq_value
                fastest_freq = freq
    
    if fastest_freq is None:
        print(f"  ⚠ No recognized frequencies found, skipping category")
        continue
    
    print(f"  Selected fastest frequency: {fastest_freq} (value: {fastest_value})")
    
    # Download all data for the category
    series_data = {}
    print(f"    Downloading data for {len(category_series)} series...")
    
    for series_id in category_series:
        try:
            frequency = category_frequencies[series_id]
            print(f"      Downloading {series_id} (frequency: {frequency})...")
            
            # Get data at the series' original frequency using frequency hierarchy
            if frequency in fred_api_mapping:
                api_param = fred_api_mapping[frequency]
                data_series = data(series_id, api_param)
                
                # Get frequency label for display
                if frequency == 'D':
                    freq_label = 'daily'
                elif frequency == 'W':
                    freq_label = 'weekly'
                elif frequency == 'M':
                    freq_label = 'monthly'
                elif frequency == 'Q':
                    freq_label = 'quarterly'
                elif frequency == 'A':
                    freq_label = 'annual'
                else:
                    freq_label = frequency.lower()
            else:
                print(f"        ⚠ {series_id:15s} -> Unknown frequency '{frequency}', skipping")
                continue
            
            # Store the data
            if not data_series.empty:
                series_data[series_id] = data_series['value']
                print(f"        ✓ {series_id:15s} -> {freq_label:9s} | {len(data_series)} obs")
            else:
                print(f"        ⚠ {series_id:15s} -> No data available")
                
        except Exception as e:
            print(f"        ✗ {series_id:15s} -> Error: {e}")
            continue
    
    print(f"    Downloaded data for {len(series_data)} out of {len(category_series)} series")
    
    if not series_data:
        print(f"  ⚠ No data available for category {category_name}")
        continue
    
    # Create one DataFrame for the category using the fastest frequency
    category_df = pd.DataFrame()
    
    for series_id, series_values in series_data.items():
        series_freq = category_frequencies[series_id]
        
        # Ensure series_values has a proper DatetimeIndex
        if not isinstance(series_values.index, pd.DatetimeIndex):
            series_values.index = pd.to_datetime(series_values.index)
        
        # Convert all series to the fastest frequency using dynamic logic
        if fastest_freq == series_freq:
            # Already at target frequency, use as is
            category_df[series_id] = series_values
            print(f"        ✓ Added {series_id} as {fastest_freq} (original)")
        else:
            # Need to convert between frequencies
            source_value = frequency_hierarchy[series_freq]
            target_value = frequency_hierarchy[fastest_freq]
            
            if source_value > target_value:
                # Converting from slower to faster frequency (e.g., Q->M, A->M)
                # Use forward fill to expand data
                pandas_freq = pandas_freq_mapping[fastest_freq]
                category_df[series_id] = series_values.reindex(
                    pd.date_range(start=series_values.index.min(), 
                                end=series_values.index.max(), 
                                freq=pandas_freq)
                ).ffill()
                print(f"        ✓ Added {series_id} converted from {series_freq} to {fastest_freq} (forward fill)")
            else:
                # Converting from faster to slower frequency (e.g., M->Q, M->A)
                # Use resampling to aggregate data
                pandas_freq = pandas_freq_mapping[fastest_freq]
                category_df[series_id] = series_values.resample(pandas_freq).last()
                print(f"        ✓ Added {series_id} converted from {series_freq} to {fastest_freq} (resampled)")
    
    # Create the final category DataFrame
    if not category_df.empty:
        # Set index and name
        category_df.index = pd.to_datetime(category_df.index)
        category_df.index.name = 'Date'
        
        # Store in category_dataframes with just the category name
        category_dataframes[category_name] = category_df
        
        print(f"  ✓ Category DataFrame created: {category_df.shape[1]} series, {category_df.shape[0]} observations")
        print(f"    Date range: {category_df.index.min().strftime('%Y-%m-%d')} to {category_df.index.max().strftime('%Y-%m-%d')}")
        print(f"    All series converted to {fastest_freq}-frequency")
        
        # Add to log entries
        columns_list = list(category_df.columns)
        log_entries.append(f"\nCategory: {category_name}")
        log_entries.append(f"  Frequency: {fastest_freq}")
        log_entries.append(f"  Shape: {category_df.shape[1]} columns × {category_df.shape[0]} rows")
        log_entries.append(f"  Date Range: {category_df.index.min().strftime('%Y-%m-%d')} to {category_df.index.max().strftime('%Y-%m-%d')}")
        log_entries.append(f"  Columns: {', '.join(columns_list)}")
        
    else:
        print(f"  ⚠ Category DataFrame is empty - no data added")

# Write log to file
logs_dir = os.path.join(os.path.dirname(script_dir), 'Logs')
os.makedirs(logs_dir, exist_ok=True)
log_file_path = os.path.join(logs_dir, 'data_collection_log.txt')
try:
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write('\n'.join(log_entries))
        log_file.write('\n')
    print(f"\n✓ Log written to: {log_file_path}")
except Exception as e:
    print(f"\n⚠ Error writing log file: {e}")

# Legacy: Keep the old frequency-based DataFrames for backward compatibility
print(f"\n" + "="*60)
print(f"LEGACY FREQUENCY-BASED PROCESSING")
print(f"="*60)

# Second pass: download data in chronological order (legacy approach)
print(f"Downloading data in chronological order (legacy)...")
for i, info in enumerate(series_info_list):
    series_id = info['id']
    frequency = info['frequency']
    
    try:
        # Determine frequency and get data using frequency hierarchy
        if frequency in fred_api_mapping:
            api_param = fred_api_mapping[frequency]
            data_series = data(series_id, api_param)
            
            # Determine target DataFrame and frequency label
            if frequency == 'Q':
                target_df = fred_quarterly_data
                freq_label = 'quarterly'
            elif frequency == 'A':
                target_df = fred_annual_data
                freq_label = 'annual'
            else:
                # Daily, Weekly, Monthly or other frequencies go to monthly DataFrame
                target_df = fred_monthly_data
                freq_label = 'monthly'
        else:
            # Unknown frequency, default to monthly
            data_series = data(series_id, 'm')
            target_df = fred_monthly_data
            freq_label = 'monthly'
        
        # Add the series to the appropriate DataFrame
        if not data_series.empty:
            target_df[series_id] = data_series['value']
            print(f"✓ [{i+1:2d}/{len(series_info_list)}] {series_id:15s} -> {freq_label:9s} | {info['start_date'].strftime('%Y-%m-%d')} | {len(data_series)} obs")
        else:
            print(f"⚠ [{i+1:2d}/{len(series_info_list)}] {series_id:15s} -> No data available")
            
    except Exception as e:
        print(f"✗ [{i+1:2d}/{len(series_info_list)}] {series_id:15s} -> Error: {e}")
        continue

# Set index for all DataFrames
fred_monthly_data.index = pd.to_datetime(fred_monthly_data.index)
fred_quarterly_data.index = pd.to_datetime(fred_quarterly_data.index)
fred_annual_data.index = pd.to_datetime(fred_annual_data.index)

print(f"\n" + "="*60)
print(f"DOWNLOAD COMPLETE")
print(f"="*60)
print(f"Monthly Data:   {fred_monthly_data.shape[1]:3d} series | {fred_monthly_data.shape[0]:4d} observations")
print(f"Quarterly Data: {fred_quarterly_data.shape[1]:3d} series | {fred_quarterly_data.shape[0]:4d} observations") 
print(f"Annual Data:    {fred_annual_data.shape[1]:3d} series | {fred_annual_data.shape[0]:4d} observations")

# Show date ranges for each DataFrame
if not fred_monthly_data.empty:
    print(f"\nMonthly Data Range:   {fred_monthly_data.index.min().strftime('%Y-%m-%d')} to {fred_monthly_data.index.max().strftime('%Y-%m-%d')}")
if not fred_quarterly_data.empty:
    print(f"Quarterly Data Range: {fred_quarterly_data.index.min().strftime('%Y-%m-%d')} to {fred_quarterly_data.index.max().strftime('%Y-%m-%d')}")
if not fred_annual_data.empty:
    print(f"Annual Data Range:    {fred_annual_data.index.min().strftime('%Y-%m-%d')} to {fred_annual_data.index.max().strftime('%Y-%m-%d')}")



# Ensure all DataFrames have index named "Date"
print(f"\n" + "="*60)
print(f"PREPARING FRED DATA FOR DATABASE STORAGE")
print(f"="*60)

# Create a mapping of DataFrame names to DataFrame objects for FRED data
dataframe_mapping = {
    "fred_monthly_data": fred_monthly_data,
    "fred_quarterly_data": fred_quarterly_data,
    "fred_annual_data": fred_annual_data,
}

# Add category DataFrames to the mapping
dataframe_mapping.update(category_dataframes)

# Rename indices for FRED data
if not fred_monthly_data.empty:
    fred_monthly_data.index.name = 'Date'
    print(f"✓ Monthly FRED data index renamed to 'Date'")
if not fred_quarterly_data.empty:
    fred_quarterly_data.index.name = 'Date'
    print(f"✓ Quarterly FRED data index renamed to 'Date'")
if not fred_annual_data.empty:
    fred_annual_data.index.name = 'Date'
    print(f"✓ Annual FRED data index renamed to 'Date'")

# Ensure all category DataFrames have proper Date index
for category_name, category_df in category_dataframes.items():
    if not category_df.empty:
        # Ensure index is datetime and named 'Date'
        if not isinstance(category_df.index, pd.DatetimeIndex):
            category_df.index = pd.to_datetime(category_df.index)
        category_df.index.name = 'Date'
        print(f"✓ {category_name} index renamed to 'Date'")

print(f"All FRED DataFrame indices have been renamed to 'Date'")

# Show index information for FRED DataFrames
print(f"\nFRED Data index information summary:")
df_info_list = [
    ("fred_monthly_data", fred_monthly_data),
    ("fred_quarterly_data", fred_quarterly_data), 
    ("fred_annual_data", fred_annual_data),
]

# Add category DataFrames to the info list
for category_name, category_df in category_dataframes.items():
    df_info_list.append((category_name, category_df))

for name, df in df_info_list:
    if isinstance(df, pd.DataFrame) and not df.empty:
        index_name = df.index.name if df.index.name else "Unnamed"
        index_type = type(df.index).__name__
        print(f"  {name:20s}: Index '{index_name}' ({index_type}) | Shape: {df.shape[1]:3d} cols × {df.shape[0]:6d} rows")
    else:
        print(f"  {name:20s}: No data available")


def collect_fred_data(db_manager=None) -> bool:
    """Store prepared FRED DataFrames to the database using db_manager."""
    try:
        if db_manager is None:
            print("⚠ No database manager provided - skipping database storage")
            return False
        if not db_manager.health_check():
            print("❌ Database connection unhealthy - skipping storage")
            return False

        print(f"\n" + "="*60)
        print(f"STORING DATA TO DATABASE")
        print(f"="*60)

        df_names = [
            "fred_monthly_data", "fred_quarterly_data", "fred_annual_data",
        ]

        # Add category DataFrames to storage list
        category_df_names = [name for name in category_dataframes.keys()]
        df_names.extend(category_df_names)

        print(f"Preparing to store {len(df_names)} DataFrames to database...")

        for i, name in enumerate(df_names, 1):
            try:
                df_obj = dataframe_mapping.get(name)
                
                if isinstance(df_obj, pd.DataFrame):
                    if not df_obj.empty:
                        print(f"[{i:2d}/{len(df_names)}] {name:20s}...", end=" ")
                        
                        # Show column information including index
                        index_name = df_obj.index.name if df_obj.index.name else "Unnamed"
                        columns_info = f"Index: {index_name}, Columns: {list(df_obj.columns)}"
                        print(f"\n    Columns: {columns_info}")
                        print(f"    Storing...", end=" ")
                        
                        # Store the DataFrame via db_manager
                        with db_manager.get_db_connection() as conn:
                            df_obj.to_sql(name, conn, if_exists='replace', index=True)
                        
                        print(f"✓ Stored ({df_obj.shape[1]:3d} columns, {df_obj.shape[0]:6d} rows)")
                    else:
                        print(f"[{i:2d}/{len(df_names)}] {name:20s}... ⚠ Empty DataFrame - skipping")
                else:
                    print(f"[{i:2d}/{len(df_names)}] {name:20s}... ✗ Not a DataFrame - skipping")
                    
            except Exception as e:
                print(f"[{i:2d}/{len(df_names)}] {name:20s}... ✗ Error: {str(e)[:50]}...")
                continue

        # Store category DataFrames separately by their category names
        if category_dataframes:
            print(f"\nStoring category DataFrames (by category name)...")
            for category_name, category_df in category_dataframes.items():
                try:
                    print(f"  {category_name:25s}...", end=" ")
                    with db_manager.get_db_connection() as conn:
                        category_df.to_sql(category_name, conn, if_exists='replace', index=True)
                    print(f"✓ Stored ({category_df.shape[1]:3d} columns, {category_df.shape[0]:6d} rows)")
                except Exception as e:
                    print(f"✗ Error: {str(e)[:50]}...")

        # Store metadata separately
        print(f"\nStoring metadata...", end=" ")
        try:
            with db_manager.get_db_connection() as conn:
                metadata_df.to_sql('metadata_df', conn, if_exists='replace', index=False)
            print(f"✓ Stored ({metadata_df.shape[1]:3d} columns, {metadata_df.shape[0]:6d} rows)")
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}...")

        print(f"\n" + "="*60)
        print(f"DATABASE STORAGE COMPLETE")
        print(f"="*60)
        print(f"All data has been successfully stored via db_manager")

        # Show what tables were created
        all_tables = ["fred_monthly_data", "fred_quarterly_data", "fred_annual_data", "metadata_df"]
        if category_dataframes:
            category_tables = [name for name in category_dataframes.keys()]
            all_tables.extend(category_tables)
        print(f"Tables created: {', '.join(all_tables)}")

        # Show category summary
        if category_dataframes:
            print(f"\nCategory DataFrames created:")
            for category_name, category_df in category_dataframes.items():
                print(f"  {category_name:30s}: {category_df.shape[1]:3d} series, {category_df.shape[0]:6d} rows")
                print(f"    Date range: {category_df.index.min().strftime('%Y-%m-%d')} to {category_df.index.max().strftime('%Y-%m-%d')}")

        return True

    except Exception as e:
        print(f"Error in collect_fred_data: {e}")
        return False



