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

# =============================================================================
# PCA-BASED INDICES CONSTRUCTION (loop-oriented implementation)
# - Builds 26 regression indices using PCA across pre-defined groups
# - Uses same config, FRED key, and SQL connection as other ml_spx scripts
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration and setup
# -----------------------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
key_file_path = os.path.join(script_dir, '..', '..', 'data_processing', 'key.txt')

# Initialize FRED
fred = None
try:
    if not os.path.exists(key_file_path):
        raise FileNotFoundError(f"FRED API key file not found at: {key_file_path}")
    fred = Fred(key_file_path)
    fred.set_api_key_file(key_file_path)
except Exception as e:
    print(f"Error initializing FRED API: {e}")
    raise

# Load variables configuration
try:
    with open(os.path.join(script_dir, 'variables.json'), 'r') as f:
        variables_config = json.load(f)
except Exception as e:
    print(f"Error loading variables.json: {e}")
    raise

# Resampling configuration
resampling_freq = variables_config.get('config', {}).get('resampling', 'M')
freq_mapping = {
    'D': 'D',
    'W': 'W-FRI',
    'M': 'ME',
    'Q': 'Q',
    'A': 'A'
}
if resampling_freq not in freq_mapping:
    raise ValueError(f"Invalid resampling frequency '{resampling_freq}'")

# SQL connection
server = 'localhost'
port = '5432'
database = 'avalon'
username = 'admin'
password = 'password!'
conn_str = f'postgresql+psycopg2://{username}:{password}@{server}:{port}/{database}'
engine = create_engine(conn_str, future=True)

# Load FRED metadata
metadata_df = pd.read_sql_table("metadata_df", con=engine)

# -----------------------------------------------------------------------------
# Helpers kept minimal
# -----------------------------------------------------------------------------

def sanitize_symbol(symbol: str) -> str:
    return symbol.replace('^', '').replace('=', '').replace('.', '_').replace('-', '_')

def compare_frequencies(native_freq: str, target_freq: str) -> str:
    levels = {'D': 5, 'W': 4, 'M': 3, 'Q': 2, 'A': 1}
    target_base = target_freq.split('-')[0]
    n = levels.get(native_freq.upper(), 0)
    t = levels.get(target_base.upper(), 0)
    if n > t:
        return 'faster'
    elif n < t:
        return 'slower'
    return 'same'

# -----------------------------------------------------------------------------
# Build master index
# -----------------------------------------------------------------------------

start_date = '2000-01-01'
end_date = datetime.today()
if resampling_freq == 'D':
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    master_index = pd.date_range(start=start_date, end=end_date, freq=us_bd)
else:
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    daily_index = pd.date_range(start=start_date, end=end_date, freq=us_bd)
    dummy = pd.Series(1, index=daily_index)
    master_index = dummy.resample(freq_mapping[resampling_freq]).last().index

combined_index = pd.DataFrame(index=master_index)

# -----------------------------------------------------------------------------
# Extract symbol sets and lags from variables.json
# -----------------------------------------------------------------------------

symbol_lag_mapping = {}
def _traverse_and_collect(d):
    for k, v in d.items():
        if isinstance(v, dict):
            if 'lag' in v:
                try:
                    symbol_lag_mapping[k] = int(v['lag'])
                except Exception:
                    symbol_lag_mapping[k] = 1
            else:
                _traverse_and_collect(v)
_traverse_and_collect(variables_config)

# Indices by region
indices_regions = {}
if 'indices' in variables_config:
    for region_key, region_val in variables_config['indices'].items():
        cols = []
        if isinstance(region_val, dict):
            for sym, meta in region_val.items():
                if isinstance(meta, dict) and 'lag' in meta:
                    cols.append(sym)
        indices_regions[region_key] = cols

# Currencies
currencies_all = []
if 'currencies' in variables_config:
    for sym, meta in variables_config['currencies'].items():
        if isinstance(meta, dict) and 'lag' in meta:
            currencies_all.append(sym)

# Futures by family
futures_groups = {}
if 'futures' in variables_config:
    for fam, fam_val in variables_config['futures'].items():
        cols = []
        if isinstance(fam_val, dict):
            for sym, meta in fam_val.items():
                if isinstance(meta, dict) and 'lag' in meta:
                    cols.append(sym)
        futures_groups[fam] = cols

# Macro categories
macro_groups = {}
if 'economic_indicators' in variables_config:
    for cat, cat_val in variables_config['economic_indicators'].items():
        cols = []
        if isinstance(cat_val, dict):
            for sym, meta in cat_val.items():
                if isinstance(meta, dict) and 'lag' in meta:
                    cols.append(sym)
        macro_groups[cat] = cols

# Derived spreads names
derived_spreads_spec = []
if 'derived_variables' in variables_config and 'yield_spreads' in variables_config['derived_variables']:
    for sym, meta in variables_config['derived_variables']['yield_spreads'].items():
        if isinstance(meta, dict) and 'lag' in meta:
            derived_spreads_spec.append(sym)

# -----------------------------------------------------------------------------
# Load MARKET data (yfinance): compute log returns per symbol, apply lag
# -----------------------------------------------------------------------------

def daily(symbol):
    return yf.download(tickers=symbol, period='max', interval='1d', auto_adjust=True, prepost=True)

symbol_mapping = {}
market_returns = {}

equity_symbols_needed = []
for s in ['^RUT', '^VIX', '^GSPTSE', '^MXX', '^N100', '^BVSP', '000001.SS', '^N225', '^HSI', '^BSESN', '^KS11']:
    if s in symbol_lag_mapping:
        equity_symbols_needed.append(s)

fx_symbols_needed = [s for s in currencies_all]

futures_symbols_needed = []
for fam, syms in futures_groups.items():
    for s in syms:
        futures_symbols_needed.append(s)

market_symbols = []
for s in equity_symbols_needed + fx_symbols_needed + futures_symbols_needed:
    if s not in market_symbols:
        market_symbols.append(s)

for symbol in market_symbols:
    df_raw = daily(symbol)
    if df_raw is None or df_raw.empty:
        continue
    prefix = sanitize_symbol(symbol)
    symbol_mapping[symbol] = prefix
    df_flat = df_raw.copy()
    new_cols = []
    for col in df_raw.columns:
        if isinstance(col, tuple) and len(col) == 2:
            col_name, _ticker = col
        else:
            col_name = str(col)
        new_cols.append(f"{prefix}_{col_name}")
    df_flat.columns = new_cols
    df_flat.index = pd.to_datetime(df_flat.index)
    if resampling_freq != 'D':
        df_flat = df_flat.resample(freq_mapping[resampling_freq]).last()
    df_aligned = df_flat.reindex(master_index)
    lag_val = symbol_lag_mapping.get(symbol, 1)
    for col in df_aligned.columns:
        df_aligned[col] = df_aligned[col].shift(lag_val)
    close_col = f"{prefix}_Close"
    if close_col in df_aligned.columns:
        series = df_aligned[close_col]
        r = np.log(series / series.shift(1))
        market_returns[symbol] = r

# -----------------------------------------------------------------------------
# Load FRED macro data (levels), intelligent resampling, align, lag
# -----------------------------------------------------------------------------

fred_levels = {}
macro_symbols_needed = []
for cat in ['yields', 'credit_spreads', 'prices', 'industrial_production', 'labor_market', 'housing_market', 'trade_indicators', 'business_activity']:
    if cat in macro_groups:
        for s in macro_groups[cat]:
            if s not in macro_symbols_needed:
                macro_symbols_needed.append(s)

for symbol in macro_symbols_needed:
    try:
        meta = metadata_df[metadata_df['series_id'] == symbol]
        if meta.empty:
            continue
        native = str(meta['frequency_short'].iloc[0]).upper()
        df = fred.get_series_df(symbol, frequency=native.lower())
        if df is None or df.empty:
            continue
        df.value = df.value.replace('.', np.nan)
        df.index = pd.to_datetime(df.date)
        df = df.drop(columns=['date', 'realtime_start', 'realtime_end'])
        df.value = df.value.astype('float')
        if resampling_freq != 'D':
            cmp = compare_frequencies(native, freq_mapping[resampling_freq])
            if cmp == 'faster':
                df = df.resample(freq_mapping[resampling_freq]).last()
            # if slower or same, keep as is; align+ffill below
        aligned = df['value'].reindex(master_index).ffill()
        lag_val = symbol_lag_mapping.get(symbol, 1)
        fred_levels[symbol] = aligned.shift(lag_val)
    except Exception as e:
        print(f"FRED load error for {symbol}: {e}")
        continue

# Derived spreads from levels (after lag)
if ('DGS10' in fred_levels) and ('DGS2' in fred_levels):
    s = fred_levels['DGS10'] - fred_levels['DGS2']
    lag_val = symbol_lag_mapping.get('yield_spread_2y_10y', 1)
    fred_levels['yield_spread_2y_10y'] = s.shift(lag_val)
if ('DGS2' in fred_levels) and ('DTB3' in fred_levels):
    s = fred_levels['DGS2'] - fred_levels['DTB3']
    lag_val = symbol_lag_mapping.get('yield_spread_3m_2y', 1)
    fred_levels['yield_spread_3m_2y'] = s.shift(lag_val)

# -----------------------------------------------------------------------------
# PCA computation utilities (inline, loops preferred)
# -----------------------------------------------------------------------------

def compute_pca_scores_from_dataframe(df_in: pd.DataFrame, num_components: int, index_out: pd.DatetimeIndex) -> pd.DataFrame:
    # Align index
    Xdf = df_in.reindex(index_out)
    # Convert to numpy
    X = Xdf.values.astype(float)
    n_rows, n_cols = X.shape
    if n_cols == 0:
        return pd.DataFrame(index=index_out)
    if n_cols == 1:
        col = X[:, 0]
        # Fill NaNs
        last = np.nan
        for i in range(n_rows):
            if np.isnan(col[i]):
                col[i] = last
            else:
                last = col[i]
        if np.isnan(col[0]):
            col = np.nan_to_num(col, nan=np.nanmean(col))
        m = np.nanmean(col)
        s = np.nanstd(col, ddof=1)
        s = s if s > 0 else 1.0
        z = (col - m) / s
        out = pd.DataFrame({'PC1': z}, index=index_out)
        return out
    # Fill NaNs forward per column
    for j in range(n_cols):
        last = np.nan
        for i in range(n_rows):
            if np.isnan(X[i, j]):
                X[i, j] = last
            else:
                last = X[i, j]
        if np.isnan(X[0, j]):
            # fill remaining leading NaN with column mean
            col_mean = np.nanmean(X[:, j])
            if np.isnan(col_mean):
                col_mean = 0.0
            for i in range(n_rows):
                if np.isnan(X[i, j]):
                    X[i, j] = col_mean
    # Standardize columns
    for j in range(n_cols):
        col = X[:, j]
        m = np.nanmean(col)
        s = np.nanstd(col, ddof=1)
        if not np.isfinite(s) or s <= 0:
            s = 1.0
        for i in range(n_rows):
            X[i, j] = (X[i, j] - m) / s
    # Compute covariance matrix
    # Rows: time, Cols: variables
    # Handle any residual NaNs
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    C = (X.T @ X) / max(n_rows - 1, 1)
    # Eigen decomposition (symmetric)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]
    k = min(num_components, n_cols)
    Vk = V[:, :k]
    scores = X @ Vk
    out = pd.DataFrame(scores, index=index_out, columns=[f"PC{i+1}" for i in range(k)])
    return out

# -----------------------------------------------------------------------------
# Build groups and compute PCA indices
# -----------------------------------------------------------------------------

indices_out = {}

# 1) EQUITIES
na_equities = ['^RUT', '^VIX', '^GSPTSE', '^MXX']
eu_equities = ['^N100']
sa_equities = ['^BVSP']
asia_equities = ['000001.SS', '^N225', '^HSI', '^BSESN', '^KS11']

def _collect_returns(symbols_list):
    cols = []
    names = []
    for s in symbols_list:
        if s in market_returns:
            cols.append(market_returns[s])
            names.append(s)
    if len(cols) == 0:
        return pd.DataFrame(index=master_index)
    df = pd.DataFrame({names[i]: cols[i] for i in range(len(cols))}, index=master_index)
    return df

# NA PC1
df_eq_na = _collect_returns(na_equities)
if df_eq_na.shape[1] > 0:
    pcs = compute_pca_scores_from_dataframe(df_eq_na, 1, master_index)
    indices_out['EQ_NA_PC1'] = pcs['PC1'] if 'PC1' in pcs else pd.Series(np.nan, index=master_index)

# Europe PC1 (single index)
df_eq_eu = _collect_returns(eu_equities)
if df_eq_eu.shape[1] > 0:
    pcs = compute_pca_scores_from_dataframe(df_eq_eu, 1, master_index)
    indices_out['EQ_EU_PC1'] = pcs['PC1'] if 'PC1' in pcs else pd.Series(np.nan, index=master_index)

# South America PC1 (single index)
df_eq_sa = _collect_returns(sa_equities)
if df_eq_sa.shape[1] > 0:
    pcs = compute_pca_scores_from_dataframe(df_eq_sa, 1, master_index)
    indices_out['EQ_SA_PC1'] = pcs['PC1'] if 'PC1' in pcs else pd.Series(np.nan, index=master_index)

# Asia PC1 and PC2
df_eq_asia = _collect_returns(asia_equities)
if df_eq_asia.shape[1] > 0:
    pcs = compute_pca_scores_from_dataframe(df_eq_asia, 2, master_index)
    if 'PC1' in pcs:
        indices_out['EQ_ASIA_PC1'] = pcs['PC1']
    if 'PC2' in pcs:
        indices_out['EQ_ASIA_PC2'] = pcs['PC2']

# 2) CURRENCIES (returns)
df_fx = _collect_returns(fx_symbols_needed)
if df_fx.shape[1] > 0:
    pcs = compute_pca_scores_from_dataframe(df_fx, 3, master_index)
    if 'PC1' in pcs:
        indices_out['FX_PC1'] = pcs['PC1']
    if 'PC2' in pcs:
        indices_out['FX_PC2'] = pcs['PC2']
    if 'PC3' in pcs:
        indices_out['FX_PC3'] = pcs['PC3']

# 3) COMMODITIES FUTURES (returns)
for fam in ['metals', 'energy', 'grains', 'softs', 'livestock']:
    if fam in futures_groups:
        df_fam = _collect_returns(futures_groups[fam])
        if df_fam.shape[1] > 0:
            pcs = compute_pca_scores_from_dataframe(df_fam, 1, master_index)
            name = {
                'metals': 'COM_METALS_PC1',
                'energy': 'COM_ENERGY_PC1',
                'grains': 'COM_GRAINS_PC1',
                'softs': 'COM_SOFTS_PC1',
                'livestock': 'COM_LIVESTOCK_PC1'
            }[fam]
            indices_out[name] = pcs['PC1'] if 'PC1' in pcs else pd.Series(np.nan, index=master_index)

# 4) YIELDS & CREDIT (levels)
# Yields: PC1-3
if 'yields' in macro_groups:
    cols = []
    names = []
    for s in macro_groups['yields']:
        if s in fred_levels:
            cols.append(fred_levels[s])
            names.append(s)
    if len(cols) > 0:
        df_y = pd.DataFrame({names[i]: cols[i] for i in range(len(cols))}, index=master_index)
        pcs = compute_pca_scores_from_dataframe(df_y, 3, master_index)
        if 'PC1' in pcs:
            indices_out['YIELD_PC1'] = pcs['PC1']
        if 'PC2' in pcs:
            indices_out['YIELD_PC2'] = pcs['PC2']
        if 'PC3' in pcs:
            indices_out['YIELD_PC3'] = pcs['PC3']

# Credit Spreads: PC1-2
if 'credit_spreads' in macro_groups:
    cols = []
    names = []
    for s in macro_groups['credit_spreads']:
        if s in fred_levels:
            cols.append(fred_levels[s])
            names.append(s)
    if len(cols) > 0:
        df_c = pd.DataFrame({names[i]: cols[i] for i in range(len(cols))}, index=master_index)
        pcs = compute_pca_scores_from_dataframe(df_c, 2, master_index)
        if 'PC1' in pcs:
            indices_out['CREDIT_PC1'] = pcs['PC1']
        if 'PC2' in pcs:
            indices_out['CREDIT_PC2'] = pcs['PC2']

# 5) MACRO INDICATORS (levels, PC1 only per small group)
macro_name_map = {
    'prices': 'MACRO_PRICES',
    'industrial_production': 'MACRO_IP',
    'labor_market': 'MACRO_LABOR',
    'housing_market': 'MACRO_HOUSING',
    'trade_indicators': 'MACRO_TRADE',
    'business_activity': 'MACRO_BUSINESS'
}
for cat_key, out_name in macro_name_map.items():
    if cat_key in macro_groups:
        cols = []
        names = []
        for s in macro_groups[cat_key]:
            if s in fred_levels:
                cols.append(fred_levels[s])
                names.append(s)
        if len(cols) > 0:
            df_cat = pd.DataFrame({names[i]: cols[i] for i in range(len(cols))}, index=master_index)
            pcs = compute_pca_scores_from_dataframe(df_cat, 1, master_index)
            indices_out[out_name] = pcs['PC1'] if 'PC1' in pcs else pd.Series(np.nan, index=master_index)

# 6) DERIVED VARIABLES (already compressed, keep raw)
if 'yield_spread_2y_10y' in fred_levels:
    indices_out['SPREAD_2Y10Y'] = fred_levels['yield_spread_2y_10y']
if 'yield_spread_3m_2y' in fred_levels:
    indices_out['SPREAD_3M2Y'] = fred_levels['yield_spread_3m_2y']

# -----------------------------------------------------------------------------
# Assemble final DataFrame and save
# -----------------------------------------------------------------------------

pca_indices_df = pd.DataFrame(index=master_index)
for name, ser in indices_out.items():
    pca_indices_df[name] = ser

# Drop all-empty columns
pca_indices_df = pca_indices_df.dropna(axis=1, how='all')

print("PCA Indices constructed:")
print(f"  Columns ({len(list(pca_indices_df.columns))}): {list(pca_indices_df.columns)}")
print(f"  Shape: {pca_indices_df.shape}")
print(pca_indices_df.tail())

# Save to database
pca_indices_df.to_sql('ml_spx_pca_indices', engine, if_exists='replace', index=True)
print("Saved PCA indices to database table: ml_spx_pca_indices")


