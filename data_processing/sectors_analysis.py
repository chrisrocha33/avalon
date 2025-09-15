# ============================================
# Sector Analysis Model - Restructured Version
# - Load data from PostgreSQL
# - Perform comprehensive financial analysis
# - Store all results in SQL tables
# - Save plots as static files
# - Designed to be schedulable from app.py
# ============================================

import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
from sqlalchemy import text
from statsmodels.stats.stattools import jarque_bera
import seaborn as sns
import matplotlib.pyplot as plt
from utils import apply_dashboard_plot_style

apply_dashboard_plot_style()
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from scipy.stats import shapiro, norm
from scipy.signal import periodogram
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import os
from datetime import datetime
import warnings
from arch import arch_model
from sklearn.decomposition import PCA
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import StandardScaler
import networkx as nx

warnings.filterwarnings("ignore")

# =========================
# PANDAS SETTINGS
# =========================
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# =========================
# SQL CONNECTION
# =========================
# Use db_manager passed from caller; no direct engine here

# =========================
# PLOT SAVING SETTINGS
# =========================
PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'static', 'images', 'sector_analysis')
os.makedirs(PLOTS_DIR, exist_ok=True)

# =========================
# GLOBAL VARIABLES
# =========================
df_model = None
sector_cols = None
returns = None
market_col = None
# Attach db_manager globally for plot storage
_dbm = None
RUN_ID = None

# Analysis parameters
roll_win_long = 252
max_lags = 30
alpha_levels = [0.95, 0.99]
# Batch operations
BATCH_SIZE = 1000

# =========================
# HELPER FUNCTIONS
# =========================

# Inline-DB helpers (loop-oriented) used during plot save
def _ensure_sectors_visuals_table():
    global _dbm
    if _dbm is None:
        return
    try:
        with _dbm.get_db_connection() as conn:
            conn.execute(text(
                """
                CREATE TABLE IF NOT EXISTS sectors_visuals (
                    date TIMESTAMP,
                    title TEXT PRIMARY KEY,
                    data TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
                """
            ))
            # Ensure created_at exists for pre-existing tables
            try:
                conn.execute(text(
                    "ALTER TABLE sectors_visuals ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW()"
                ))
            except Exception:
                pass
            conn.commit()
    except Exception as e:
        print(f"    Warning: failed to ensure sectors_visuals table: {e}")


def save_plot(fig, filename, dpi=150):
    """Store matplotlib figure as base64 in DB only (no file write)"""
    fig.tight_layout()
    try:
        import base64
        from io import BytesIO
        global _dbm
        title = filename
        if _dbm is not None:
            _ensure_sectors_visuals_table()
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
            buf.seek(0)
            b64data = base64.b64encode(buf.read()).decode('utf-8')
            now_dt = datetime.utcnow()
            with _dbm.get_db_connection() as conn:
                # Upsert by title and refresh created_at
                conn.execute(text(
                    """
                    INSERT INTO sectors_visuals (date, title, data, created_at)
                    VALUES (:date, :title, :data, :created_at)
                    ON CONFLICT (title) DO UPDATE
                    SET date = EXCLUDED.date,
                        data = EXCLUDED.data,
                        created_at = EXCLUDED.created_at
                    """
                ), {"date": now_dt, "title": title, "data": b64data, "created_at": now_dt})
                conn.commit()
        else:
            print(f"    Warning: DB manager not available, skipping DB store for '{title}'")
    except Exception as e:
        # Log DB storage issues to not break plotting
        print(f"    Warning: failed to store plot '{filename}' in DB: {e}")
    plt.close(fig)
    print(f"    ✓ Stored plot in DB: {filename}")

def clean_col_name(col: str) -> str:
    """Clean column names for analysis"""
    name = re.sub(r'[^0-9A-Za-z]+', '_', str(col))
    name = name.strip('_')
    if not name:
        name = 'X'
    if name[0].isdigit():
        name = f'X_{name}'
    return name

# =========================
# DATA LOADING & PREPARATION
# =========================

def load_data(db_manager=None):
    """Load and prepare data from database"""
    global df_model, sector_cols, returns, market_col
    
    print("Loading data from database...")

    # Define the data sources and columns to load
    data_sources = {
        "fred_monthly_data": ["DTB3", "INDPRO", "PAYEMS", "CPIAUCSL", "DFF", "DGS2", "DGS10",
                              "BAMLH0A0HYM2EY", "BAMLC0A0CMEY"],
        "fred_quarterly_data": ["GDP"],
        "market_data": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "SPY", "^VIX"]
    }

    loaded_frames = {}

    # Load data from each table
    for table_name, columns in data_sources.items():
        try:
            # Build column list for SELECT statement
            cols = ["Date"] + [col for col in columns if col != "Date"]
            select_list = ", ".join([f'"{col}"' for col in cols])
            
            sql = f"SELECT {select_list} FROM {table_name}"
            if db_manager is None:
                raise RuntimeError("db_manager is required for database reads")
            df = db_manager.read_sql_pandas(sql)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
            
            loaded_frames[table_name] = df
            print(f"✓ Loaded {table_name}: {df.shape[0]} rows x {df.shape[1]} cols")
            print(f"    Index: '{df.index.name}' | Columns: {list(df.columns)}")
        except Exception as e:
            print(f"✗ Failed to load {table_name}: {e}")
            loaded_frames[table_name] = pd.DataFrame()

    # Build combined dataframe
    print("Creating combined dataframe...")

    # Start with market data
    market_data = loaded_frames.get("market_data", pd.DataFrame())
    if not market_data.empty:
        # Resample to monthly and calculate returns
        market_monthly = market_data.resample('MS').last()
        market_returns = market_monthly.pct_change()
        df_model = market_returns.copy()
        print(f"✓ Base market monthly: {df_model.shape[1]} series")
    else:
        df_model = pd.DataFrame()
        print("✗ No market data available")

    # Add FRED monthly data
    fred_monthly = loaded_frames.get("fred_monthly_data", pd.DataFrame())
    if not fred_monthly.empty:
        fred_monthly.index = pd.to_datetime(fred_monthly.index)
        
        # Calculate year-over-year changes for economic indicators
        fred_monthly['CPIAUCSL'] = (fred_monthly['CPIAUCSL'] / fred_monthly['CPIAUCSL'].shift(12)) - 1
        fred_monthly['PAYEMS'] = (fred_monthly['PAYEMS'] / fred_monthly['PAYEMS'].shift(12)) - 1
        fred_monthly['INDPRO'] = (fred_monthly['INDPRO'] / fred_monthly['INDPRO'].shift(12)) - 1
        
        # Align with market data
        if not df_model.empty:
            fred_aligned = fred_monthly.reindex(df_model.index)
            df_model = pd.concat([df_model, fred_aligned], axis=1)
            print(f"✓ Added {fred_aligned.shape[1]} FRED monthly series")
        else:
            df_model = fred_monthly

    # Add FRED quarterly data
    fred_quarterly = loaded_frames.get("fred_quarterly_data", pd.DataFrame())
    if not fred_quarterly.empty:
        fred_quarterly.index = pd.to_datetime(fred_quarterly.index)
        
        # Calculate year-over-year change for GDP
        fred_quarterly['GDP'] = (fred_quarterly['GDP'] / fred_quarterly['GDP'].shift(4)) - 1
        
        # Align with existing data using forward fill
        if not df_model.empty:
            fred_q_aligned = fred_quarterly.reindex(df_model.index, method='ffill')
            df_model = pd.concat([df_model, fred_q_aligned], axis=1)
            print(f"✓ Added {fred_q_aligned.shape[1]} FRED quarterly series (ffill to monthly)")
        else:
            df_model = fred_quarterly

    # Clean and prepare final dataframe
    if not df_model.empty:
        # Handle VIX column naming
        if '^VIX' in df_model.columns and 'VIX' not in df_model.columns:
            df_model['VIX'] = df_model['^VIX']

        # Clean column names
        seen = set()
        rename_map = {}
        for col in df_model.columns:
            base = clean_col_name(col)
            new = base
            i = 2
            while new in seen:
                new = f'{base}_{i}'
                i += 1
            seen.add(new)
            rename_map[col] = new

        df_model = df_model.rename(columns=rename_map)
        
        # Fill NaN values with 0 and take last 30 years
        df_model = df_model.fillna(0)
        years = 30
        df_model = df_model.tail(12 * years)
        
        # Prepare sector returns for analysis
        base_sectors = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "SPY"]
        sector_cols = [col for col in base_sectors if col in df_model.columns]
        returns = df_model[sector_cols].copy()
        returns = returns.dropna()
        
        # Set market column for analysis
        market_col = 'SPY' if 'SPY' in returns.columns else (sector_cols[0] if sector_cols else None)
        
        print("✓ Dataframe creation completed")
        print(f"  Final shape: {df_model.shape[0]} rows × {df_model.shape[1]} columns")
        print(f"  Prepared {len(sector_cols)} sector series for analysis")
    else:
        print("✗ No data available for analysis")
        sector_cols = []
        returns = pd.DataFrame()
        market_col = None

    return df_model, sector_cols, returns, market_col

# =========================
# ANALYSIS FUNCTIONS
# =========================

def run_descriptive_analysis(db_manager=None):
    """Run descriptive statistics analysis and store in SQL"""
    print("  Processing descriptive statistics...")
    rows = []
    for col in sector_cols:
        try:
            s = returns[col].dropna()
            if len(s) < 10:
                continue

            mu = s.mean()
            sd = s.std(ddof=1)
            sk = s.skew()
            kt = s.kurtosis()

            try:
                adf_stat, adf_p, *_ = adfuller(s.values, autolag='AIC')
            except Exception:
                adf_p = np.nan

            try:
                kpss_stat, kpss_p, *_ = kpss(s.values, regression='c', nlags='auto')
            except Exception:
                kpss_p = np.nan

            try:
                jb_stat, jb_p, _, _ = jarque_bera(s.values)
            except Exception:
                jb_p = np.nan

            try:
                x = s.values
                if len(x) > 5000:
                    x = np.random.choice(x, 5000, replace=False)
                sh_stat, sh_p = shapiro(x)
            except Exception:
                sh_p = np.nan

            rows.append({
                "col": col, "mu": mu, "sd": sd, "sk": sk, "kt": kt,
                "adf_p": adf_p, "kpss_p": kpss_p, "jb_p": jb_p, "sh_p": sh_p
            })
        except Exception as e:
            print(f"    Error processing {col}: {e}")

    if rows and db_manager is not None:
        sql = """
            INSERT INTO descriptive_stats_data_render 
            (series_name, mean_value, std_value, skewness, kurtosis, adf_p_value, kpss_p_value, jarque_bera_p_value, shapiro_p_value)
            VALUES (:col, :mu, :sd, :sk, :kt, :adf_p, :kpss_p, :jb_p, :sh_p)
        """
        try:
            with db_manager.get_db_connection() as conn:
                tx = conn.begin()
                try:
                    for i in range(0, len(rows), BATCH_SIZE):
                        conn.execute(text(sql), rows[i:i+BATCH_SIZE])
                    tx.commit()
                except Exception as e:
                    tx.rollback()
                    raise e
        except Exception as e:
            print(f"    Error batch inserting descriptive stats: {e}")

    print("  ✓ Descriptive statistics completed and stored in SQL")

def run_correlation_analysis(db_manager=None):
    """Run correlation analysis and store results"""
    print("  Processing correlation analysis...")

    corr = returns.corr()

    if db_manager is None:
        raise RuntimeError("db_manager is required for database writes")

    rows = []
    for i, col1 in enumerate(corr.columns):
        for j, col2 in enumerate(corr.columns):
            rows.append({"col1": col1, "col2": col2, "corr_val": float(corr.iloc[i, j])})

    try:
        with db_manager.get_db_connection() as conn:
            tx = conn.begin()
            try:
                sql = """
                    INSERT INTO correlation_matrix_data_render 
                    (series1, series2, correlation_value)
                    VALUES (:col1, :col2, :corr_val)
                """
                for i in range(0, len(rows), BATCH_SIZE):
                    conn.execute(text(sql), rows[i:i+BATCH_SIZE])
                tx.commit()
            except Exception as e:
                tx.rollback()
                raise e
    except Exception as e:
        print(f"    Error batch inserting correlation matrix: {e}")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid(True)
    sns.heatmap(corr, annot=True, annot_kws={'color': plt.rcParams.get('text.color', '#FFFFFF')}, cmap='coolwarm', center=0, ax=ax)
    plt.title('Sector Returns Correlation Matrix', fontsize=14)
    plt.tight_layout()
    save_plot(fig, 'plot=correlation_heatmap')

    print("  ✓ Correlation analysis completed and stored in SQL")

def run_pca_analysis(db_manager=None):
    """Run PCA analysis and store results"""
    print("  Processing PCA analysis...")

    try:
        returns_clean = returns.dropna()
        if len(returns_clean) < 10:
            print("    Insufficient data for PCA")
            return

        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(returns_clean)

        pca = PCA()
        pca.fit(returns_scaled)

        if db_manager is None:
            raise RuntimeError("db_manager is required for database writes")

        ev_rows = []
        for i, var_ratio in enumerate(pca.explained_variance_ratio_):
            ev_rows.append({"pc_num": i+1, "var_ratio": float(var_ratio)})

        load_rows = []
        for i, pc in enumerate(pca.components_):
            for j, loading in enumerate(pc):
                series_name = returns_clean.columns[j]
                load_rows.append({"series": series_name, "pc_num": i+1, "loading": float(loading)})

        with db_manager.get_db_connection() as conn:
            tx = conn.begin()
            try:
                sql_ev = """
                    INSERT INTO pca_results_data_render 
                    (pc_number, explained_variance_ratio)
                    VALUES (:pc_num, :var_ratio)
                """
                for i in range(0, len(ev_rows), BATCH_SIZE):
                    conn.execute(text(sql_ev), ev_rows[i:i+BATCH_SIZE])

                sql_ld = """
                    INSERT INTO pca_loadings_data_render 
                    (series_name, pc_number, loading_value)
                    VALUES (:series, :pc_num, :loading)
                """
                for i in range(0, len(load_rows), BATCH_SIZE):
                    conn.execute(text(sql_ld), load_rows[i:i+BATCH_SIZE])
                tx.commit()
            except Exception as e:
                tx.rollback()
                raise e

        print("  ✓ PCA analysis completed and stored in SQL")

    except Exception as e:
        print(f"    Error in PCA analysis: {e}")

def run_var_cvar_analysis(db_manager=None):
    """Run VaR/CVaR analysis and store results"""
    print("  Processing VaR/CVaR analysis...")

    rows = []
    for col in sector_cols:
        try:
            s = returns[col].dropna()
            if len(s) < 50:
                continue
            for alpha in alpha_levels:
                var_value = np.percentile(s, (1 - alpha) * 100)
                cvar_value = s[s <= var_value].mean()
                rows.append({"col": col, "alpha": alpha, "var_val": float(var_value), "cvar_val": float(cvar_value)})
        except Exception as e:
            print(f"    Error processing VaR/CVaR for {col}: {e}")

    if rows and db_manager is not None:
        try:
            with db_manager.get_db_connection() as conn:
                tx = conn.begin()
                try:
                    sql = """
                        INSERT INTO var_cvar_data_render 
                        (series_name, alpha_level, var_value, cvar_value)
                        VALUES (:col, :alpha, :var_val, :cvar_val)
                    """
                    for i in range(0, len(rows), BATCH_SIZE):
                        conn.execute(text(sql), rows[i:i+BATCH_SIZE])
                    tx.commit()
                except Exception as e:
                    tx.rollback()
                    raise e
        except Exception as e:
            print(f"    Error batch inserting VaR/CVaR: {e}")

    print("  ✓ VaR/CVaR analysis completed and stored in SQL")

def run_beta_analysis(db_manager=None):
    """Run beta analysis against market and store results"""
    print("  Processing beta analysis...")

    if market_col is None:
        print("    No market column available for beta analysis")
        return

    rows = []
    for col in sector_cols:
        if col == market_col:
            continue
        try:
            y = returns[col].dropna()
            x = returns[market_col].dropna()
            common_idx = y.index.intersection(x.index)
            if len(common_idx) < 50:
                continue
            y_aligned = y.loc[common_idx]
            x_aligned = x.loc[common_idx]
            X = sm.add_constant(x_aligned)
            model = sm.OLS(y_aligned, X).fit()
            rows.append({
                "col": col,
                "alpha": float(model.params[0]),
                "beta": float(model.params[1] if len(model.params) > 1 else np.nan),
                "r_squared": float(model.rsquared)
            })
        except Exception as e:
            print(f"    Error processing beta for {col}: {e}")

    if rows and db_manager is not None:
        try:
            with db_manager.get_db_connection() as conn:
                tx = conn.begin()
                try:
                    sql = """
                        INSERT INTO beta_analysis_data_render 
                        (series_name, alpha, beta, r_squared)
                        VALUES (:col, :alpha, :beta, :r_squared)
                    """
                    for i in range(0, len(rows), BATCH_SIZE):
                        conn.execute(text(sql), rows[i:i+BATCH_SIZE])
                    tx.commit()
                except Exception as e:
                    tx.rollback()
                    raise e
        except Exception as e:
            print(f"    Error batch inserting beta results: {e}")

    print("  ✓ Beta analysis completed and stored in SQL")

def run_ols_regressions(db_manager=None):
    """Run a single OLS regression with SPY as dependent variable and store result"""
    print("  Processing OLS regression (SPY only)...")
    
    try:
        if df_model is None or df_model.empty or 'SPY' not in df_model.columns:
            print("    No SPY column available for OLS")
            return
        
        y_ols = df_model['SPY'].dropna()
        X_ols = df_model.drop(columns=['SPY']).dropna()
        
        # Align data
        common_idx = y_ols.index.intersection(X_ols.index)
        if len(common_idx) < 50:
            print("    Insufficient data for OLS")
            return
            
        y_aligned = y_ols.loc[common_idx]
        X_aligned = X_ols.loc[common_idx]
        
        # Run regression
        ols_result = sm.OLS(y_aligned, sm.add_constant(X_aligned)).fit()
        ols_results_html = ols_result.summary().as_html()
    
        if db_manager is None:
            raise RuntimeError("db_manager is required for database writes")
        with db_manager.get_db_connection() as conn:
            # Clear any previous results so only this one appears
            try:
                conn.execute(text("DELETE FROM ols_regression_data_render"))
            except Exception:
                pass
            conn.execute(text("""
                INSERT INTO ols_regression_data_render 
                (dependent_variable, html_summary)
                VALUES (:dep_var, :html_summary)
            """), {"dep_var": "SPY", "html_summary": ols_results_html})
            conn.commit()
            
    except Exception as e:
        print(f"    Error processing OLS for SPY: {e}")
    
    print("  ✓ OLS regression (SPY) completed and stored in SQL")

def run_rolling_moments():
    """Generate rolling moments plots and save as static files"""
    print("  Processing rolling moments...")
    
    for col in sector_cols:
        try:
            s = returns[col].dropna()
            if len(s) < 100:
                continue
                
            # Rolling mean, std, skew, kurtosis
            rolling_mean = s.rolling(window=roll_win_long).mean()
            rolling_std = s.rolling(window=roll_win_long).std()
            rolling_skew = s.rolling(window=roll_win_long).apply(lambda x: pd.Series(x).skew(), raw=False)
            rolling_kurt = s.rolling(window=roll_win_long).apply(lambda x: pd.Series(x).kurtosis(), raw=False)
            
            # Save individual rolling moment plots
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(rolling_mean.index, rolling_mean.values, label='Rolling Mean')
            ax.set_title(f'Rolling Mean: {col}')
            ax.legend()
            ax.grid(True)
            save_plot(fig, f'plot=rolling_mean|sector={col}')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(rolling_std.index, rolling_std.values, label='Rolling Std')
            ax.set_title(f'Rolling Standard Deviation: {col}')
            ax.legend()
            ax.grid(True)
            save_plot(fig, f'plot=rolling_std|sector={col}')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(rolling_skew.index, rolling_skew.values, label='Rolling Skewness')
            ax.set_title(f'Rolling Skewness: {col}')
            ax.legend()
            ax.grid(True)
            save_plot(fig, f'plot=rolling_skew|sector={col}')
            
        except Exception as e:
            print(f"    Error generating rolling moments plot for {col}: {e}")
    
    print("  ✓ Rolling moments plots completed and saved as static files")

def run_garch_volatility():
    """Generate GARCH volatility plots and save as static files"""
    print("  Processing GARCH volatility...")
    
    for i, col in enumerate(sector_cols[:2]):  # Only first 2 series for template
        try:
            s = returns[col].dropna()
            if len(s) < 100:
                continue
                
            # GARCH analysis
            s_garch = s * 100.0  # Convert to percentage
            am = arch_model(s_garch.values, mean='constant', vol='GARCH', p=1, q=1, dist='normal', rescale=False)
            res = am.fit(disp='off')
            cond_vol = res.conditional_volatility
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(returns.index[-len(cond_vol):], cond_vol)
            ax.set_title(f'GARCH Conditional Volatility: {col}')
            ax.set_ylabel('Volatility')
            ax.grid(True)
            
            filename = f'plot=garch_vol|sector={col}'
            save_plot(fig, filename)
            
        except Exception as e:
            print(f"    Error generating GARCH plot for {col}: {e}")
    
    print("  ✓ GARCH volatility plots completed and saved as static files")

def run_periodogram():
    """Generate periodogram plots and save as static files"""
    print("  Processing periodograms...")
    
    for i, col in enumerate(sector_cols[:2]):  # Only first 2 series for template
        try:
            s = returns[col].dropna()
            if len(s) < 50:
                continue
                
            # Periodogram
            f, Pxx = periodogram(s.values, scaling='density')
            if len(f) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(f[1:], Pxx[1:])
                ax.set_title(f'Periodogram: {col}')
                ax.set_xlabel('Frequency')
                ax.set_ylabel('Power')
                ax.grid(True)
                
                filename = f'plot=periodogram|sector={col}'
                save_plot(fig, filename)
                
        except Exception as e:
            print(f"    Error generating periodogram for {col}: {e}")
    
    print("  ✓ Periodogram plots completed and saved as static files")

def run_regime_switching():
    """Generate regime switching plot and save as static file"""
    print("  Processing regime switching...")
    
    if market_col is not None:
        try:
            y = returns[market_col].dropna()
            if len(y) < 100:
                print("    Insufficient data for regime switching")
                return
                
            mod = MarkovRegression(y.values, k_regimes=2, trend='c', switching_variance=True)
            res = mod.fit(disp=False)
            
            # Fix: Get the correct dimensions for regime probabilities
            # smoothed_marginal_probabilities has shape (n_regimes, n_observations)
            # We want regime 0 probabilities for all time periods
            probs = res.smoothed_marginal_probabilities[0, :]  # Shape: (n_observations,)
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(y.index, y.values, label='Returns', alpha=0.7)
            ax1.set_ylabel('Returns')
            ax1.tick_params(axis='y', labelcolor='b')
            
            ax2 = ax1.twinx()
            ax2.plot(y.index, probs, label='P(Regime 0)', alpha=0.8)
            ax2.set_ylabel('Regime Probability')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_ylim(0, 1)
            
            ax1.set_title(f'Markov Regime Switching: {market_col}')
            ax1.grid(True)
            
            # Add legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            save_plot(fig, f'plot=regime_switching|market={market_col}')
            
        except Exception as e:
            print(f"    Error generating regime switching plot: {e}")
            # Create a placeholder plot if regime switching fails
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.text(0.5, 0.5, 'Regime Switching Analysis\nFailed to Generate', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title('Regime Switching: Analysis Failed')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                save_plot(fig, 'plot=regime_switching')
                print("    ✓ Created placeholder regime switching plot")
            except Exception as e2:
                print(f"    Error creating placeholder plot: {e2}")
    
    print("  ✓ Regime switching plot completed and saved as static file")

def run_clustering_dendrogram():
    """Generate clustering dendrogram and save as static file"""
    print("  Processing clustering dendrogram...")
    
    try:
        # Calculate correlation matrix and distance
        corr = returns.corr()
        dist = np.sqrt(0.5 * (1 - corr.values))
        condensed = squareform(dist, checks=False)
        Z_link = linkage(condensed, method='average')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.grid(True)
        dendrogram(Z_link, labels=sector_cols, leaf_rotation=90, ax=ax)
        ax.set_title('Hierarchical Clustering Dendrogram (Average Linkage)', fontsize=14)
        ax.set_xlabel('Sectors')
        ax.set_ylabel('Distance')
        
        save_plot(fig, 'plot=clustering_dendrogram')
        
    except Exception as e:
        print(f"    Error generating clustering dendrogram: {e}")
    
    print("  ✓ Clustering dendrogram completed and saved as static file")

def run_network_graph():
    """Generate network graph and save as static file"""
    print("  Processing network graph...")
    
    try:
        # Prepare data for network analysis
        X = returns[sector_cols].dropna()
        X_std = StandardScaler(with_mean=True, with_std=True).fit_transform(X.values)
        
        # Graphical Lasso for partial correlations
        gl = GraphicalLassoCV()
        gl.fit(X_std)
        prec = gl.precision_
        D = np.sqrt(np.diag(prec))
        partial_corr = -prec / np.outer(D, D)
        np.fill_diagonal(partial_corr, 1.0)
        thr = 0.05
        
        # Create network
        G = nx.Graph()
        for sector in sector_cols:
            G.add_node(sector)
            
        for i in range(len(sector_cols)):
            for j in range(i+1, len(sector_cols)):
                w = partial_corr[i, j]
                if abs(w) >= thr:
                    G.add_edge(sector_cols[i], sector_cols[j], weight=float(w))
        
        # Draw network
        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.grid(True)
        
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue', 
                              edgecolors='black', linewidths=2, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        
        # Draw edges with varying thickness based on weight
        edge_weights = [1 + 3*abs(G[u][v]['weight']) for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, 
                              edge_color='gray', ax=ax)
        
        ax.set_title('Partial-Correlation Network Graph', fontsize=14)
        ax.axis('off')
        
        save_plot(fig, 'plot=network_graph')
        
    except Exception as e:
        print(f"    Error generating network graph: {e}")
    
    print("  ✓ Network graph completed and saved as static file")

def run_tail_dependence():
    """Generate tail dependence plots and save as static files"""
    print("  Processing tail dependence...")
    
    try:
        # Calculate tail dependence
        U = returns.rank(method='average') / (len(returns) + 1.0)
        U = U[sector_cols].dropna()
        tau_q = 0.05
        
        lower_mat = np.full((len(sector_cols), len(sector_cols)), np.nan)
        upper_mat = np.full((len(sector_cols), len(sector_cols)), np.nan)
        
        for i, a in enumerate(sector_cols):
            for j, b in enumerate(sector_cols):
                if i == j:
                    lower_mat[i, j] = upper_mat[i, j] = 1.0
                else:
                    ua, ub = U[a].values, U[b].values
                    lower_mat[i, j] = (np.sum((ua < tau_q) & (ub < tau_q)) / max(np.sum(ua < tau_q), 1.0))
                    upper_mat[i, j] = (np.sum((ua > 1-tau_q) & (ub > 1-tau_q)) / max(np.sum(ua > 1-tau_q), 1.0))
        
        # Lower tail dependence plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.grid(True)
        im = ax.imshow(lower_mat, vmin=0, vmax=1, aspect='auto', interpolation='nearest', cmap='Reds')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(sector_cols)))
        ax.set_xticklabels(sector_cols, rotation=90)
        ax.set_yticks(range(len(sector_cols)))
        ax.set_yticklabels(sector_cols)
        ax.set_title(f'Lower-Tail Dependence λ_L (q={tau_q})', fontsize=14)
        
        save_plot(fig, 'plot=tail_dependence_lower')
        
        # Upper tail dependence plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.grid(True)
        im = ax.imshow(upper_mat, vmin=0, vmax=1, aspect='auto', interpolation='nearest', cmap='Blues')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(sector_cols)))
        ax.set_xticklabels(sector_cols, rotation=90)
        ax.set_yticks(range(len(sector_cols)))
        ax.set_yticklabels(sector_cols)
        ax.set_title(f'Upper-Tail Dependence λ_U (q={tau_q})', fontsize=14)
        
        save_plot(fig, 'plot=tail_dependence_upper')
        
    except Exception as e:
        print(f"    Error generating tail dependence plots: {e}")
    
    print("  ✓ Tail dependence plots completed and saved as static files")

def run_rolling_beta():
    """Generate rolling beta plots and save as static files"""
    print("  Processing rolling beta...")
    
    if market_col is None:
        print("    No market column available for rolling beta analysis")
        return
    
    try:
        # Prepare data for rolling beta
        ds_base = returns[[market_col] + [c for c in sector_cols if c != market_col]].dropna()
        
        for i, col in enumerate([c for c in sector_cols if c != market_col][:2]):  # Only first 2 for template
            ds = ds_base[[market_col, col]].dropna()
            rb_vals = []
            win = 252
            
            for j in range(win, len(ds) + 1):
                wnd = ds.iloc[j-win:j]
                X = sm.add_constant(wnd[market_col].values)
                res = sm.OLS(wnd[col].values, X).fit()
                rb_vals.append(res.params[1] if len(res.params) > 1 else np.nan)
            
            rb = pd.Series(rb_vals, index=ds.index[win-1:], name=col)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(rb.index, rb.values)
            ax.set_title(f'Rolling Beta: {col} vs {market_col}', fontsize=14)
            ax.set_ylabel('Beta')
            ax.set_xlabel('Date')
            ax.grid(True)
            ax.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Beta = 1')
            ax.axhline(y=0, color='g', linestyle='--', alpha=0.7, label='Beta = 0')
            ax.legend()
            ax.set_ylim(-1.5, 1.5)
            
            filename = f'plot=rolling_beta|sector={col}|market={market_col}'
            save_plot(fig, filename)
            
    except Exception as e:
        print(f"    Error generating rolling beta plots: {e}")
    
    print("  ✓ Rolling beta plots completed and saved as static files")

def run_pca_cumulative_variance():
    """Generate PCA cumulative variance plot and save as static file"""
    print("  Processing PCA cumulative variance...")
    
    try:
        # Prepare data for PCA
        X = returns[sector_cols].dropna()
        X_std = StandardScaler(with_mean=True, with_std=True).fit_transform(X.values)
        
        # Fit PCA
        pca = PCA()
        pca.fit(X_std)
        
        # Calculate cumulative explained variance
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(cum_var) + 1), cum_var)
        ax.set_xlabel('Principal Component Number')
        ax.set_ylabel('Cumulative Explained Variance Ratio')
        ax.set_title('PCA Cumulative Explained Variance', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        # Add percentage labels
        for i, (x, y) in enumerate(zip(range(1, len(cum_var) + 1), cum_var)):
            ax.annotate(f'{y:.1%}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=10)
        
        save_plot(fig, 'plot=pca_cumvar')
        
    except Exception as e:
        print(f"    Error generating PCA cumulative variance plot: {e}")
    
    print("  ✓ PCA cumulative variance plot completed and saved as static file")

def run_correlation_scatter():
    """Generate correlation scatter plot and save as static file"""
    print("  Processing correlation scatter plot...")
    
    try:
        # Create correlation scatter grid
        n_sectors = len(sector_cols)
        n_indicators = len([col for col in df_model.columns if col not in sector_cols])
        
        fig, axes = plt.subplots(n_sectors, n_indicators, figsize=(20, 16))
        fig.suptitle('All Sectors vs All Economic Indicators', fontsize=16, y=0.95)
        
        indicator_cols = [col for col in df_model.columns if col not in sector_cols]
        
        for i, sector in enumerate(sector_cols):
            for j, indicator in enumerate(indicator_cols):
                if n_sectors == 1:
                    ax = axes[j] if n_indicators > 1 else axes
                elif n_indicators == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]
                
                if indicator in df_model.columns:
                    ax.scatter(df_model[indicator], df_model[sector], 
                             alpha=0.6, s=20)
                else:
                    ax.text(0.5, 0.5, f"{indicator} NA", ha='center', va='center')
                
                ax.grid(True)
                if i == n_sectors - 1:
                    ax.set_xlabel(indicator)
                if j == 0:
                    ax.set_ylabel(sector)
                if i == 0:
                    ax.set_title(indicator)
        
        # Legend omitted to allow rcParams palette and avoid fixed colors
        
        save_plot(fig, 'plot=correlation_scatter')
        
    except Exception as e:
        print(f"    Error generating correlation scatter plot: {e}")
    
    print("  ✓ Correlation scatter plot completed and saved as static file")

def run_plot_generation():
    """Generate and save plots as composite grid images (rows=sectors, cols=analyses)"""
    print("  Generating plots (grid mode)...")

    # Ensure styling is applied (idempotent)
    try:
        apply_dashboard_plot_style()
    except Exception:
        pass

    # Grid-mode configuration
    grid_mode = True
    if grid_mode:
        try:
            analysis_order = ['qq', 'acf', 'pacf', 'periodogram', 'garch', 'roll_mean', 'roll_std', 'roll_skew', 'roll_beta']
            max_cols = 9      # columns per page
            max_rows = 11     # sectors per page
            plot_w, plot_h = 3.0, 2.2  # per-subplot size tuned for 11x9 grid
            
            if not sector_cols or returns is None or returns.empty:
                print("    No sector data for grid plots")
                return

            # Paginate columns then rows
            for col_start in range(0, len(analysis_order), max_cols):
                cols_page = analysis_order[col_start:col_start + max_cols]
                for row_start in range(0, len(sector_cols), max_rows):
                    secs_page = sector_cols[row_start:row_start + max_rows]
                    n_rows, n_cols = len(secs_page), len(cols_page)

                    # Create figure and axes grid
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * plot_w, n_rows * plot_h))
                    if n_rows == 1 and n_cols == 1:
                        axes = np.array([[axes]])
                    elif n_rows == 1:
                        axes = axes.reshape(1, -1)
                    elif n_cols == 1:
                        axes = axes.reshape(-1, 1)

                    for i, sector in enumerate(secs_page):
                        if sector not in returns.columns:
                            continue
                        s = returns[sector].dropna()
                        if s.empty:
                            continue
                        for j, kind in enumerate(cols_page):
                            ax = axes[i, j]
                            try:
                                ax.grid(True, alpha=0.3)
                            except Exception:
                                pass

                            # Draw cell by kind
                            try:
                                if kind == 'qq':
                                    if len(s) >= 10:
                                        tq = norm.ppf(np.linspace(0.01, 0.99, len(s)), s.mean(), s.std())
                                        ax.scatter(tq, np.sort(s), s=10, alpha=0.6)
                                elif kind == 'acf':
                                    if len(s) >= 50:
                                        plot_acf(s.values, ax=ax, lags=max_lags, alpha=0.05)
                                elif kind == 'pacf':
                                    if len(s) >= 50:
                                        plot_pacf(s.values, ax=ax, lags=max_lags, alpha=0.05, method='ywm')
                                elif kind == 'periodogram':
                                    if len(s) >= 50:
                                        f, Pxx = periodogram(s.values, scaling='density')
                                        if len(f) > 1:
                                            ax.plot(f[1:], Pxx[1:], lw=1.2)
                                elif kind == 'garch':
                                    if len(s) >= 100:
                                        try:
                                            am = arch_model((s * 100).values, mean='constant', vol='GARCH', p=1, q=1, dist='normal', rescale=False)
                                            res = am.fit(disp='off')
                                            cv = res.conditional_volatility
                                            ax.plot(returns.index[-len(cv):], cv, lw=1)
                                        except Exception:
                                            pass
                                elif kind == 'roll_mean':
                                    if len(s) >= roll_win_long:
                                        ax.plot(s.rolling(roll_win_long).mean(), lw=1)
                                elif kind == 'roll_std':
                                    if len(s) >= roll_win_long:
                                        ax.plot(s.rolling(roll_win_long).std(), lw=1)
                                elif kind == 'roll_skew':
                                    if len(s) >= roll_win_long:
                                        ax.plot(s.rolling(roll_win_long).apply(lambda x: pd.Series(x).skew(), raw=False), lw=1)
                                elif kind == 'roll_beta' and market_col and sector != market_col:
                                    ds = returns[[market_col, sector]].dropna()
                                    if len(ds) >= 2:
                                        win = min(252, len(ds))
                                        rb = []
                                        idx = []
                                        for k in range(win, len(ds) + 1):
                                            wnd = ds.iloc[k - win:k]
                                            try:
                                                params = sm.OLS(wnd[sector].values, sm.add_constant(wnd[market_col].values)).fit().params
                                                rb.append(params[1] if len(params) > 1 else np.nan)
                                            except Exception:
                                                rb.append(np.nan)
                                            idx.append(ds.index[k - 1])
                                        if any(not np.isnan(v) for v in rb):
                                            ax.plot(idx, rb, lw=1)
                                            try:
                                                ax.axhline(1, color='r', ls='--', alpha=0.4)
                                            except Exception:
                                                pass
                            except Exception:
                                # Keep empty cell on failure
                                pass

                            # Titles/labels
                            if i == 0:
                                try:
                                    ax.set_title(kind.replace('_', ' ').title(), fontsize=10)
                                except Exception:
                                    pass
                            if j == 0:
                                try:
                                    ax.set_ylabel(sector, fontsize=9)
                                except Exception:
                                    pass

                    try:
                        fig.tight_layout()
                    except Exception:
                        pass

                    try:
                        run_tag = f"|run={RUN_ID}" if RUN_ID else ""
                    except Exception:
                        run_tag = ""
                    title = f"plot=grid_core{run_tag}|rows={row_start + 1}-{row_start + n_rows}|cols={col_start + 1}-{col_start + n_cols}"
                    save_plot(fig, title, dpi=120)

            print("  ✓ Grid plot generation completed")
            return
        except Exception as e:
            print(f"    Error in grid plot generation: {e}")
            # If grid fails, do not generate fallback singles to avoid duplicates
            return

    # Fallback: legacy basic plot generation
    # Generate QQ plots
    for i, col in enumerate(sector_cols[:3]):  # Only first 3 for template
        try:
            s = returns[col].dropna()
            if len(s) < 10:
                continue
                
            fig, ax = plt.subplots(figsize=(8, 6))
            theoretical_quantiles = norm.ppf(np.linspace(0.01, 0.99, len(s)), s.mean(), s.std())
            ax.scatter(theoretical_quantiles, np.sort(s), alpha=0.6)
            ax.plot([s.min(), s.max()], [s.min(), s.max()], 'r--', lw=2)
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel('Sample Quantiles')
            ax.set_title(f'Q-Q Plot: {col}')
            ax.grid(True, alpha=0.3)
            
            filename = f'plot=qq|sector={col}'
            save_plot(fig, filename)
            
        except Exception as e:
            print(f"    Error generating QQ plot for {col}: {e}")
    
    # Generate ACF/PACF plots
    for i, col in enumerate(sector_cols[:2]):  # Only first 2 for template
        try:
            s = returns[col].dropna()
            if len(s) < 50:
                continue
            
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_acf(s.values, ax=ax, lags=max_lags, alpha=0.05)
            ax.set_title(f'ACF: {col}')
            filename = f'plot=acf|sector={col}'
            save_plot(fig, filename)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_pacf(s.values, ax=ax, lags=max_lags, alpha=0.05, method='ywm')
            ax.set_title(f'PACF: {col}')
            filename = f'plot=pacf|sector={col}'
            save_plot(fig, filename)
            
        except Exception as e:
            print(f"    Error generating ACF/PACF plot for {col}: {e}")
    
    print("  ✓ Basic plot generation completed and saved as static files")

# =========================
# MAIN EXECUTION FUNCTION
# =========================

def run_full_analysis(db_manager=None):
    """Run the complete sector analysis pipeline"""
    print("\n" + "="*60)
    print("🚀 STARTING COMPREHENSIVE SECTOR ANALYSIS")
    print("="*60)

    start_time = datetime.now()
    
    try:
        # Wire db_manager globally for plot storage
        global _dbm
        _dbm = db_manager
        global RUN_ID
        RUN_ID = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        # 1. Load and prepare data
        load_data(db_manager)
        
        # Check if we have data to work with
        if df_model is None or df_model.empty or len(sector_cols) == 0:
            print("❌ No data available for analysis. Exiting.")
            return False
        
        # 2. Run all analyses
        run_descriptive_analysis(db_manager)
        run_correlation_analysis(db_manager)
        run_pca_analysis(db_manager)
        run_var_cvar_analysis(db_manager)
        run_beta_analysis(db_manager)
        run_ols_regressions(db_manager)
        
        # 3. Generate all plots
        run_plot_generation()
        # Removed standalone per-sector plot generations to avoid duplicates
        # run_rolling_moments()
        # run_garch_volatility()
        # run_periodogram()
        run_regime_switching()
        run_clustering_dendrogram()
        run_network_graph()
        run_tail_dependence()
        # run_rolling_beta()
        run_pca_cumulative_variance()
        run_correlation_scatter()
        
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "="*60)
        print("✅ COMPREHENSIVE SECTOR ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"⏱️  Total execution time: {duration}")
        print(f"📊 Data ready for dashboard querying")
        print(f"📈 All plots stored in SQL table: sectors_visuals")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR IN ANALYSIS PIPELINE: {e}")
        return False

# =========================
# SCHEDULER INTERFACE
# =========================

def schedule_analysis(db_manager=None):
    """Function to be called from app.py scheduler"""
    try:
        return run_full_analysis(db_manager)
    except Exception as e:
        print(f"Error in scheduled analysis: {e}")
        return False

# =========================
# DIRECT EXECUTION
# =========================

if __name__ == "__main__":
    run_full_analysis()
