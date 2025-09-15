"""
Macro Economic Analysis Module for Flask Dashboard
==================================================

This module refactors the original Macro.py into a Flask-compatible module with:
- Database manager integration instead of direct SQLAlchemy connections
- Modular analysis functions for different components
- Comprehensive error handling and logging
- Data export and visualization functions
- DSGE model estimation with Kalman filtering
- Policy shock simulation capabilities

Features:
- 2-state DSGE model (output gap, natural real rate)
- Kalman filter estimation and smoothing
- Policy shock impulse response analysis
- Data alignment and transformation utilities
- Export functionality for visualization

Author: Refactored from original Macro.py
Date: 2024
"""

import os
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from contextlib import contextmanager
from models.DSGE import DSGEModel

# Apply centralized matplotlib styling for any plots generated here
try:
    from utils import apply_dashboard_plot_style
    apply_dashboard_plot_style()
except Exception:
    # If utils not available, continue without failing the module import
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class MacroAnalysisError(Exception):
    """Custom exception for macro analysis errors"""
    pass


# DSGEModel is now provided by models.DSGE and imported above.


class MacroAnalysis:
    """
    Main macro analysis class for Flask dashboard integration.
    
    Handles data loading, transformation, DSGE model estimation,
    and policy simulation with database manager integration.
    """
    
    # Table and field definitions (case-sensitive for PostgreSQL)
    TABLES_CONFIG = {
        "us_Labor_Wages": [
            "CES0500000003", "CE16OV", "CIVPART", "CLF16OV", "ICSA", 
            "JTSJOL", "PAYEMS", "UNRATE", "AWHAETP"
        ],
        "us_Yields_Rates": [
            "DFF", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", 
            "DGS20", "DGS30", "DTB3", "DTB4WK", "DCPF3M", "RHORUSQ156N"
        ],
        "us_Credit_Spreads": [
            "BAMLH0A0HYM2EY", "BAMLHYH0A0HYM2TRIV", "BAMLC0A0CMEY", 
            "BAMLC1A0C13YEY", "BAMLC2A0C35YEY", "BAMLC3A0C57YEY", 
            "BAMLC4A0C710YEY", "BAMLC7A0C1015YEY", "BAMLC8A0C15PYEY"
        ],
        "us_Production_Output": [
            "INDPRO", "MNFCTRIMSA", "MNFCTRIRSA", "PETROLEUMD11", 
            "BUSINV", "TTLCONS", "IEABC", "PATENTUSALLTOTAL", "ATLSBUSRGEP"
        ],
        "us_Consumption_Spending": [
            "DSPIC96", "RETAILIMSA", "RSCCAS", "RSXFS", "TOTALSA", 
            "TOTALSL", "PSAVERT"
        ],
        "us_Housing_RealEstate": [
            "CSUSHPINSA", "EXHOSLUSM495S", "FRGSHPUSM649NCIS", 
            "HOSMEDUSM052N", "WSHOMCB"
        ],
        "us_Trade_External": [
            "BOPGSTB"
        ],
        "us_Monetary_Aggregates": [
            "M2", "M2V", "TOTBUSSMSA", "TMBACBW027SBOG", "RESPPANWW"
        ],
        "us_Prices_Inflation": [
            "CPIAUCSL", "PPIACO"
        ],
        "us_GDP_NationalAccounts": [
            "GDP", "GDPC1", "GDI", "A261RX1Q020SBEA"
        ],
        "us_Banking_Debt_Stability": [
            "GFDEBTN", "H8B1058NCBCMG", "QBPBSTAS", "QBPBSTASSCUSTRSC", 
            "QBPQYNTIY", "QBPQYNUMINST"
        ],
        "us_Consumer_Debt": [
            "ACTLISCOUUS", "DRALACBS", "DRCCLACBS", "DTCDISA066MSFRBNY", "TOTBORR"
        ]
    }
    
    def __init__(self, db_manager, output_dir: str = "model_outputs"):
        """
        Initialize macro analysis with database manager.
        
        Args:
            db_manager: Database manager instance for data operations
            output_dir: Directory for saving analysis outputs
        """
        self.db_manager = db_manager
        self.output_dir = output_dir
        self.data = None
        self.model = DSGEModel()
        self.frames = {}
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        logger.info(f"MacroAnalysis initialized with output directory: {self.output_dir}")
    
    @contextmanager
    def _handle_database_errors(self, operation_name: str):
        """Context manager for database operation error handling"""
        try:
            yield
        except Exception as e:
            logger.error(f"Database operation '{operation_name}' failed: {str(e)}")
            raise MacroAnalysisError(f"Database operation failed: {str(e)}")
    
    def load_data_from_database(self) -> bool:
        """
        Load macro data from database using database manager.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting macro data loading from database...")
        
        try:
            # Test database connection
            if not self.db_manager.health_check():
                raise MacroAnalysisError("Database health check failed")
            
            # Load each table with case-sensitive column names
            self.frames = {}
            
            for table_name, columns in self.TABLES_CONFIG.items():
                try:
                    # Build SQL query with quoted identifiers for case sensitivity
                    select_cols = ['"Date"'] + [f'"{col}"' for col in columns]
                    sql = f'SELECT {", ".join(select_cols)} FROM "{table_name}"'
                    
                    with self._handle_database_errors(f"loading table {table_name}"):
                        # Use database manager's pandas read method
                        df_table = self.db_manager.read_sql_pandas(sql)
                        
                        # Set Date as index and sort
                        df_table['Date'] = pd.to_datetime(df_table['Date'])
                        df_table = df_table.set_index('Date').sort_index()
                        
                        self.frames[table_name] = df_table
                        logger.info(f"Loaded {table_name}: {len(df_table)} observations")
                        
                except Exception as e:
                    logger.warning(f"Failed to load {table_name}: {str(e)}")
                    continue
            
            if not self.frames:
                raise MacroAnalysisError("No tables loaded successfully")
            
            logger.info(f"Successfully loaded {len(self.frames)} tables")
            return True
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            return False
    
    def align_and_merge_data(self) -> bool:
        """
        Align data to monthly frequency and merge all tables.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Aligning and merging data to monthly frequency...")
        
        try:
            if not self.frames:
                raise MacroAnalysisError("No data frames available for merging")
            
            # Align to monthly frequency and merge
            all_df = None
            
            for table_name, df_table in self.frames.items():
                # Convert to monthly start frequency, forward fill
                df_monthly = df_table.resample("MS").last().ffill()
                
                if all_df is None:
                    all_df = df_monthly.copy()
                else:
                    all_df = all_df.join(df_monthly, how="outer")
            
            # Sort index and forward fill missing values
            all_df = all_df.sort_index().ffill()
            
            self.data = all_df
            logger.info(f"Data merged successfully: {len(all_df)} observations, {len(all_df.columns)} variables")
            
            return True
            
        except Exception as e:
            logger.error(f"Data alignment failed: {str(e)}")
            return False
    
    def create_core_variables(self) -> bool:
        """
        Create core macroeconomic variables from raw data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Creating core macroeconomic variables...")
        
        try:
            if self.data is None:
                raise MacroAnalysisError("No data available for variable creation")
            
            # Policy rate (short nominal): prefer DFF, fallback DTB3
            if "DFF" in self.data.columns:
                self.data["i"] = self.data["DFF"]
            elif "DTB3" in self.data.columns:
                self.data["i"] = self.data["DTB3"]
            else:
                raise MacroAnalysisError("No short rate found (need DFF or DTB3)")
            
            # Inflation (YoY % from CPI via 12-month log difference)
            if "CPIAUCSL" in self.data.columns:
                self.data["pi"] = np.log(self.data["CPIAUCSL"]).diff(12) * 100.0
            else:
                raise MacroAnalysisError("CPIAUCSL not found for inflation calculation")
            
            # Unemployment rate (already in levels)
            if "UNRATE" in self.data.columns:
                self.data["UNRATE"] = self.data["UNRATE"]
            else:
                raise MacroAnalysisError("UNRATE not found")
            
            # Real activity level: prefer GDPC1, fallback INDPRO
            if "GDPC1" in self.data.columns:
                self.data["YLVL"] = self.data["GDPC1"]
            elif "INDPRO" in self.data.columns:
                self.data["YLVL"] = self.data["INDPRO"]
            else:
                raise MacroAnalysisError("No output level found (GDPC1 or INDPRO)")
            
            # Credit spread: HY-IG preference, fallback term spread
            if ("BAMLH0A0HYM2EY" in self.data.columns and 
                "BAMLC0A0CMEY" in self.data.columns):
                self.data["SPREAD"] = self.data["BAMLH0A0HYM2EY"] - self.data["BAMLC0A0CMEY"]
            elif ("DGS10" in self.data.columns and "DTB3" in self.data.columns):
                self.data["SPREAD"] = self.data["DGS10"] - self.data["DTB3"]
            else:
                # Try to construct from available HY and IG series
                hy_cols = [col for col in ["BAMLH0A0HYM2EY", "BAMLHYH0A0HYM2TRIV"] 
                          if col in self.data.columns]
                ig_cols = [col for col in ["BAMLC0A0CMEY", "BAMLC1A0C13YEY", "BAMLC2A0C35YEY"] 
                          if col in self.data.columns]
                
                if hy_cols and ig_cols:
                    self.data["SPREAD"] = (self.data[hy_cols].mean(axis=1) - 
                                          self.data[ig_cols].mean(axis=1))
                else:
                    self.data["SPREAD"] = 0.0
                    logger.warning("No spread data available, setting to zero")
            
            # Disposable income (optional)
            if "DSPIC96" in self.data.columns:
                self.data["DPI"] = self.data["DSPIC96"]
            
            # Debt service ratio placeholder
            if "HH_DSR" in self.data.columns:
                self.data["DSR"] = self.data["HH_DSR"]
            else:
                self.data["DSR"] = 0.01 * self.data["i"]  # Simple proxy
            
            # Clean up infinite values and ensure core observables exist
            self.data = self.data.replace([np.inf, -np.inf], np.nan)
            self.data = self.data.dropna(subset=["i", "pi", "UNRATE", "YLVL"]).copy()
            
            logger.info(f"Core variables created successfully: {len(self.data)} clean observations")
            return True
            
        except Exception as e:
            logger.error(f"Variable creation failed: {str(e)}")
            return False
    
    def create_transformed_variables(self) -> bool:
        """
        Create transformed variables for DSGE model estimation.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Creating transformed variables...")
        
        try:
            if self.data is None:
                raise MacroAnalysisError("No data available for transformation")
            
            # Real rate proxy (ex-ante): i - pi
            self.data["r_real"] = self.data["i"] - self.data["pi"]
            
            # Log output level
            self.data["y_log"] = np.log(self.data["YLVL"])
            
            # Crude output gap proxy via slow-moving trend
            trend = self.data["y_log"].rolling(120, min_periods=60).mean()
            self.data["x_gap_proxy"] = self.data["y_log"] - trend
            
            logger.info("Transformed variables created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Variable transformation failed: {str(e)}")
            return False
    
    def estimate_dsge_model(self) -> bool:
        """
        Estimate DSGE model using Kalman filtering.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting DSGE model estimation with Kalman filtering...")
        
        try:
            if self.data is None:
                raise MacroAnalysisError("No data available for model estimation")
            
            # Use the extracted DSGE model (library-agnostic) and keep self.data intact
            obs_df = self.data[["pi", "i", "UNRATE"]].copy()
            est_df = self.model.estimate(obs_df)

            # Attach estimates back to the full dataset
            self.data["x_gap_est"] = est_df["x_gap_est"]
            self.data["rn_est"] = est_df["rn_est"]
            
            logger.info("DSGE model estimation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"DSGE model estimation failed: {str(e)}")
            return False
    
    def simulate_policy_shock(self, shock_bp: float = 100.0, horizon: int = 36) -> pd.DataFrame:
        """
        Simulate policy shock impulse responses using the extracted model.
        """
        logger.info(f"Simulating {shock_bp}bp policy shock over {horizon} months...")
        
        try:
            if self.data is None or self.model.kalman_results is None:
                raise MacroAnalysisError("Model must be estimated before simulation")
            
            sim = self.model.simulate(self.data[["x_gap_est", "pi", "i", "UNRATE", "rn_est"]].copy(), shock_bp=shock_bp, horizon=horizon)
            logger.info("Policy shock simulation completed successfully")
            return sim
            
        except Exception as e:
            logger.error(f"Policy shock simulation failed: {str(e)}")
            return pd.DataFrame()
    
    def export_results(self, simulation_results: Optional[pd.DataFrame] = None) -> bool:
        """
        Export analysis results to CSV files.
        
        Args:
            simulation_results: Optional simulation results to export
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Exporting results to {self.output_dir}...")
        
        try:
            if self.data is None:
                raise MacroAnalysisError("No data available for export")
            
            # Export main dataset
            main_vars = ["i", "pi", "UNRATE", "YLVL", "SPREAD", "r_real", 
                        "y_log", "x_gap_proxy", "x_gap_est", "rn_est"]
            main_data = self.data[main_vars].copy()
            main_data.to_csv(f"{self.output_dir}/dsge_main_data.csv")
            
            # Persist main dataset to SQL for HTML snapshot section
            try:
                create_sql_main = """
                CREATE TABLE IF NOT EXISTS macro_dsge_main_data (
                    id SERIAL PRIMARY KEY,
                    date DATE,
                    "i" DOUBLE PRECISION,
                    "pi" DOUBLE PRECISION,
                    "UNRATE" DOUBLE PRECISION,
                    "YLVL" DOUBLE PRECISION,
                    "SPREAD" DOUBLE PRECISION,
                    "r_real" DOUBLE PRECISION,
                    "y_log" DOUBLE PRECISION,
                    "x_gap_proxy" DOUBLE PRECISION,
                    "x_gap_est" DOUBLE PRECISION,
                    "rn_est" DOUBLE PRECISION
                )
                """
                self.db_manager.execute_query(create_sql_main)
                # Replace contents to keep a clean snapshot
                self.db_manager.execute_query("DELETE FROM macro_dsge_main_data")

                idx_list = list(main_data.index)
                t = 0
                while t < len(idx_list):
                    ts = idx_list[t]
                    dt = None
                    try:
                        dt = pd.Timestamp(ts).date()
                    except Exception:
                        try:
                            dt = datetime.fromtimestamp(ts).date()
                        except Exception:
                            dt = None

                    row = main_data.iloc[t]
                    params_insert = {
                        'p_date': dt,
                        'p_i': float(row['i']) if 'i' in main_data.columns and pd.notna(row['i']) else None,
                        'p_pi': float(row['pi']) if 'pi' in main_data.columns and pd.notna(row['pi']) else None,
                        'p_UNRATE': float(row['UNRATE']) if 'UNRATE' in main_data.columns and pd.notna(row['UNRATE']) else None,
                        'p_YLVL': float(row['YLVL']) if 'YLVL' in main_data.columns and pd.notna(row['YLVL']) else None,
                        'p_SPREAD': float(row['SPREAD']) if 'SPREAD' in main_data.columns and pd.notna(row['SPREAD']) else None,
                        'p_r_real': float(row['r_real']) if 'r_real' in main_data.columns and pd.notna(row['r_real']) else None,
                        'p_y_log': float(row['y_log']) if 'y_log' in main_data.columns and pd.notna(row['y_log']) else None,
                        'p_x_gap_proxy': float(row['x_gap_proxy']) if 'x_gap_proxy' in main_data.columns and pd.notna(row['x_gap_proxy']) else None,
                        'p_x_gap_est': float(row['x_gap_est']) if 'x_gap_est' in main_data.columns and pd.notna(row['x_gap_est']) else None,
                        'p_rn_est': float(row['rn_est']) if 'rn_est' in main_data.columns and pd.notna(row['rn_est']) else None,
                    }

                    insert_sql_main = (
                        'INSERT INTO macro_dsge_main_data '
                        '(date, "i", "pi", "UNRATE", "YLVL", "SPREAD", "r_real", "y_log", "x_gap_proxy", "x_gap_est", "rn_est") '
                        'VALUES (:p_date, :p_i, :p_pi, :p_UNRATE, :p_YLVL, :p_SPREAD, :p_r_real, :p_y_log, :p_x_gap_proxy, :p_x_gap_est, :p_rn_est)'
                    )
                    self.db_manager.execute_query(insert_sql_main, params=params_insert)
                    t += 1
            except Exception as e:
                logger.warning(f"Failed to persist macro_dsge_main_data: {str(e)}")
            
            # Export simulation results if provided
            if simulation_results is not None and not simulation_results.empty:
                simulation_results.to_csv(f"{self.output_dir}/dsge_simulation.csv")
            
            # Export model parameters
            params_data = {
                'parameter': list(self.model.params.keys()),
                'value': list(self.model.params.values()),
            }
            # Build descriptions dynamically to match parameters length (include u_star)
            param_desc_map = {
                'beta': 'Inflation persistence',
                'kappa': 'Phillips curve slope',
                'sigma': 'Risk aversion',
                'phi_pi': 'Taylor rule inflation response',
                'phi_x': 'Taylor rule output response',
                'rho_i': 'Interest rate smoothing',
                'ax': 'Output gap persistence',
                'b_IS': 'IS curve slope',
                'rho_rn': 'Natural rate persistence',
                'pi_star': 'Inflation target',
                'r_star': 'Natural real rate',
                'lam_okun': "Okun's law coefficient",
                'sig_x': 'Output gap shock variance',
                'sig_rn': 'Natural rate shock variance',
                'sig_pi': 'Inflation measurement noise',
                'sig_i': 'Interest rate measurement noise',
                'sig_u': 'Unemployment measurement noise',
                'u_star': 'Natural unemployment rate (u*)',
            }
            descriptions = []
            i_desc = 0
            keys_list = params_data['parameter']
            while i_desc < len(keys_list):
                k = keys_list[i_desc]
                try:
                    desc = param_desc_map.get(k, '')
                except Exception:
                    desc = ''
                descriptions.append(desc)
                i_desc += 1
            params_data['description'] = descriptions
            pd.DataFrame(params_data).to_csv(f"{self.output_dir}/dsge_parameters.csv", index=False)

            # Persist parameters to SQL for HTML model inputs
            try:
                create_sql_params = """
                CREATE TABLE IF NOT EXISTS macro_dsge_parameters (
                    id SERIAL PRIMARY KEY,
                    parameter TEXT,
                    value DOUBLE PRECISION,
                    description TEXT
                )
                """
                self.db_manager.execute_query(create_sql_params)
                self.db_manager.execute_query("DELETE FROM macro_dsge_parameters")

                keys_list = params_data['parameter']
                values_list = params_data['value']
                desc_list = params_data.get('description', [])

                i_params = 0
                while i_params < len(keys_list):
                    p_name = ''
                    try:
                        p_name = str(keys_list[i_params])
                    except Exception:
                        p_name = ''

                    p_val = None
                    try:
                        p_val = float(values_list[i_params])
                    except Exception:
                        p_val = None

                    p_desc = ''
                    try:
                        p_desc = str(desc_list[i_params]) if i_params < len(desc_list) else ''
                    except Exception:
                        p_desc = ''

                    self.db_manager.execute_query(
                        "INSERT INTO macro_dsge_parameters (parameter, value, description) VALUES (:p_param, :p_value, :p_desc)",
                        params={'p_param': p_name, 'p_value': p_val, 'p_desc': p_desc}
                    )
                    i_params += 1
            except Exception as e:
                logger.warning(f"Failed to persist macro_dsge_parameters: {str(e)}")
            
            # Export Kalman filter diagnostics if available
            if self.model.kalman_results is not None:
                kalman_data = pd.DataFrame({
                    'date': self.data.index,
                    'x_gap_filtered': self.model.kalman_results['s_filtered'][:, 0],
                    'rn_filtered': self.model.kalman_results['s_filtered'][:, 1],
                    'x_gap_smoothed': self.model.kalman_results['s_smoothed'][:, 0],
                    'rn_smoothed': self.model.kalman_results['s_smoothed'][:, 1],
                    'x_gap_variance': self.model.kalman_results['P_smoothed'][:, 0, 0],
                    'rn_variance': self.model.kalman_results['P_smoothed'][:, 1, 1]
                })
                kalman_data.to_csv(f"{self.output_dir}/dsge_kalman_diagnostics.csv", index=False)
            
            # Export summary statistics
            core_vars = ["i", "pi", "UNRATE", "SPREAD", "x_gap_est", "rn_est"]
            core_data = self.data[core_vars].dropna()
            summary_stats = core_data.agg(['mean', 'std', 'min', 'max', 'count']).round(4)
            summary_stats.to_csv(f"{self.output_dir}/dsge_summary_stats.csv")

            # Persist summary stats to SQL for HTML key statistics (legacy)
            try:
                create_sql_summary = """
                CREATE TABLE IF NOT EXISTS macro_dsge_summary_stats (
                    id SERIAL PRIMARY KEY,
                    stat TEXT,
                    "i" DOUBLE PRECISION,
                    "pi" DOUBLE PRECISION,
                    "UNRATE" DOUBLE PRECISION,
                    "SPREAD" DOUBLE PRECISION,
                    "x_gap_est" DOUBLE PRECISION,
                    "rn_est" DOUBLE PRECISION
                )
                """
                self.db_manager.execute_query(create_sql_summary)
                self.db_manager.execute_query("DELETE FROM macro_dsge_summary_stats")

                core_vars = ["i", "pi", "UNRATE", "SPREAD", "x_gap_est", "rn_est"]
                r_idx = 0
                while r_idx < len(summary_stats.index):
                    stat_name = ''
                    try:
                        stat_name = str(summary_stats.index[r_idx])
                    except Exception:
                        stat_name = ''

                    v_i = None
                    try:
                        v_i = float(summary_stats.loc[summary_stats.index[r_idx], 'i']) if 'i' in summary_stats.columns else None
                    except Exception:
                        v_i = None

                    v_pi = None
                    try:
                        v_pi = float(summary_stats.loc[summary_stats.index[r_idx], 'pi']) if 'pi' in summary_stats.columns else None
                    except Exception:
                        v_pi = None

                    v_unrate = None
                    try:
                        v_unrate = float(summary_stats.loc[summary_stats.index[r_idx], 'UNRATE']) if 'UNRATE' in summary_stats.columns else None
                    except Exception:
                        v_unrate = None

                    v_spread = None
                    try:
                        v_spread = float(summary_stats.loc[summary_stats.index[r_idx], 'SPREAD']) if 'SPREAD' in summary_stats.columns else None
                    except Exception:
                        v_spread = None

                    v_x = None
                    try:
                        v_x = float(summary_stats.loc[summary_stats.index[r_idx], 'x_gap_est']) if 'x_gap_est' in summary_stats.columns else None
                    except Exception:
                        v_x = None

                    v_rn = None
                    try:
                        v_rn = float(summary_stats.loc[summary_stats.index[r_idx], 'rn_est']) if 'rn_est' in summary_stats.columns else None
                    except Exception:
                        v_rn = None

                    insert_sql_summary = (
                        'INSERT INTO macro_dsge_summary_stats '
                        '(stat, "i", "pi", "UNRATE", "SPREAD", "x_gap_est", "rn_est") '
                        'VALUES (:p_stat, :p_i, :p_pi, :p_UNRATE, :p_SPREAD, :p_x_gap_est, :p_rn_est)'
                    )
                    self.db_manager.execute_query(
                        insert_sql_summary,
                        params={
                            'p_stat': stat_name,
                            'p_i': v_i,
                            'p_pi': v_pi,
                            'p_UNRATE': v_unrate,
                            'p_SPREAD': v_spread,
                            'p_x_gap_est': v_x,
                            'p_rn_est': v_rn,
                        }
                    )
                    r_idx += 1
            except Exception as e:
                logger.warning(f"Failed to persist macro_dsge_summary_stats: {str(e)}")
            
            logger.info("Results exported successfully")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            return False
    
    def generate_summary_report(self, simulation_results: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate summary report of analysis results.
        
        Args:
            simulation_results: Optional simulation results to include
            
        Returns:
            Dictionary containing summary information
        """
        try:
            if self.data is None:
                return {'error': 'No data available for summary'}
            
            core_vars = ["i", "pi", "UNRATE", "SPREAD", "x_gap_est", "rn_est"]
            core_data = self.data[core_vars].dropna()
            
            summary = {
                'data_info': {
                    'total_observations': len(self.data),
                    'clean_observations': len(core_data),
                    'date_range': {
                        'start': self.data.index.min().strftime('%Y-%m-%d'),
                        'end': self.data.index.max().strftime('%Y-%m-%d')
                    },
                    'variables_count': len(self.data.columns)
                },
                'model_parameters': self.model.params,
                'latest_estimates': {
                    'output_gap': float(self.data["x_gap_est"].iloc[-1]),
                    'natural_rate': float(self.data["rn_est"].iloc[-1]),
                    'inflation': float(self.data["pi"].iloc[-1]),
                    'policy_rate': float(self.data["i"].iloc[-1]),
                    'unemployment': float(self.data["UNRATE"].iloc[-1])
                }
            }
            
            if simulation_results is not None and not simulation_results.empty:
                summary['simulation_summary'] = {
                    'horizon_months': len(simulation_results),
                    'peak_output_gap_response': float(simulation_results["x"].min()),
                    'peak_inflation_response': float(simulation_results["pi"].min()),
                    'peak_unemployment_response': float(simulation_results["u"].max())
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary report generation failed: {str(e)}")
            return {'error': str(e)}


def run_macro_analysis(db_manager, output_dir: str = "model_outputs") -> Dict[str, Any]:
    """
    Main function to run complete macro analysis pipeline.
    
    Args:
        db_manager: Database manager instance
        output_dir: Output directory for results
        
    Returns:
        Dictionary containing analysis results and summary
    """
    logger.info("Starting complete macro analysis pipeline...")
    
    try:
        # Initialize analysis
        analysis = MacroAnalysis(db_manager, output_dir)
        
        # Load and process data
        if not analysis.load_data_from_database():
            return {'success': False, 'error': 'Data loading failed'}
        
        if not analysis.align_and_merge_data():
            return {'success': False, 'error': 'Data alignment failed'}
        
        if not analysis.create_core_variables():
            return {'success': False, 'error': 'Variable creation failed'}
        
        if not analysis.create_transformed_variables():
            return {'success': False, 'error': 'Variable transformation failed'}
        
        # Estimate DSGE model
        if not analysis.estimate_dsge_model():
            return {'success': False, 'error': 'Model estimation failed'}
        
        # Run policy shock simulation
        simulation = analysis.simulate_policy_shock(shock_bp=100.0, horizon=36)
        
        # Export results
        if not analysis.export_results(simulation):
            return {'success': False, 'error': 'Export failed'}
        
        # Create latest output DF and persist to SQL (loop-based)
        try:
            latest_row = {}
            try:
                last_idx = len(analysis.data.index) - 1
                if last_idx >= 0:
                    last_ts = analysis.data.index[last_idx]
                else:
                    last_ts = pd.Timestamp(datetime.now().date())
            except Exception:
                last_ts = pd.Timestamp(datetime.now().date())
            try:
                latest_row['date'] = last_ts.date()
            except Exception:
                latest_row['date'] = None

            pairs = [('x_gap_est', 'x_gap_est'), ('rn_est', 'rn_est'),
                     ('pi', 'pi'), ('i', 'i'), ('UNRATE', 'unrate')]
            p_i = 0
            while p_i < len(pairs):
                src, out = pairs[p_i]
                val = None
                try:
                    ser = analysis.data[src]
                    j = len(ser) - 1
                    while j >= 0:
                        v = ser.iloc[j]
                        if v is not None and v == v:
                            try:
                                val = float(v)
                            except Exception:
                                val = None
                            break
                        j -= 1
                except Exception:
                    val = None
                latest_row[out] = val
                p_i += 1

            # Parameters → lower-case keys for SQL columns
            for k, v in analysis.model.params.items():
                kk = str(k).lower()
                try:
                    latest_row[kk] = float(v)
                except Exception:
                    latest_row[kk] = None

            # Add u_star if available
            try:
                latest_row['u_star'] = float(analysis.model.kalman_results.get('u_star'))
            except Exception:
                latest_row['u_star'] = None

            # Create table if not exists, then insert
            create_sql = """
            CREATE TABLE IF NOT EXISTS dsge_model_output (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT NOW(),
                date DATE,
                x_gap_est DOUBLE PRECISION,
                rn_est DOUBLE PRECISION,
                pi DOUBLE PRECISION,
                i DOUBLE PRECISION,
                unrate DOUBLE PRECISION,
                beta DOUBLE PRECISION,
                kappa DOUBLE PRECISION,
                sigma DOUBLE PRECISION,
                phi_pi DOUBLE PRECISION,
                phi_x DOUBLE PRECISION,
                rho_i DOUBLE PRECISION,
                ax DOUBLE PRECISION,
                b_is DOUBLE PRECISION,
                rho_rn DOUBLE PRECISION,
                pi_star DOUBLE PRECISION,
                r_star DOUBLE PRECISION,
                lam_okun DOUBLE PRECISION,
                sig_x DOUBLE PRECISION,
                sig_rn DOUBLE PRECISION,
                sig_pi DOUBLE PRECISION,
                sig_i DOUBLE PRECISION,
                sig_u DOUBLE PRECISION,
                u_star DOUBLE PRECISION
            )
            """
            analysis.db_manager.execute_query(create_sql)

            insert_cols = list(latest_row.keys())
            placeholders = []
            t = 0
            while t < len(insert_cols):
                placeholders.append(f":{insert_cols[t]}")
                t += 1
            insert_sql = f"INSERT INTO dsge_model_output ({', '.join(insert_cols)}) VALUES ({', '.join(placeholders)})"
            analysis.db_manager.execute_query(insert_sql, params=latest_row)
        except Exception as e:
            logger.warning(f"Failed to persist latest model output: {str(e)}")
        
        # Generate summary
        summary = analysis.generate_summary_report(simulation)
        
        logger.info("Macro analysis pipeline completed successfully")
        
        return {
            'success': True,
            'summary': summary,
            'output_directory': output_dir,
            'files_created': [
                'dsge_main_data.csv',
                'dsge_simulation.csv',
                'dsge_parameters.csv',
                'dsge_kalman_diagnostics.csv',
                'dsge_summary_stats.csv'
            ]
        }
        
    except Exception as e:
        logger.error(f"Macro analysis pipeline failed: {str(e)}")
        return {'success': False, 'error': str(e)}


def schedule_macro_analysis(db_manager) -> bool:
    """
    Scheduled function for Flask app integration.
    
    Args:
        db_manager: Database manager instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        result = run_macro_analysis(db_manager)
        return result.get('success', False)
    except Exception as e:
        logger.error(f"Scheduled macro analysis failed: {str(e)}")
        return False


if __name__ == "__main__":
    # This section is for testing/debugging only
    # In production, use the functions above with db_manager
    print("Macro Analysis Module - Flask Integration")
    print("Use run_macro_analysis() or schedule_macro_analysis() with db_manager") 