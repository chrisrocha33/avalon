"""
Macro Visualization for Flask Dashboard
======================================

- Loads DSGE macro model CSV outputs from `model_outputs/`
- Generates static plot files under `Dashboard/static/images/macro_analysis/`
- Provides `schedule_macro_analysis(db_manager)` used by Flask `app.py`

Design notes:
- Non-interactive matplotlib backend (Agg) for server-side rendering
- Loop-based plotting logic inside functions, minimal helpers
- Robust error handling: continue on failures and log issues
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import numpy as np

# Configure matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
# Styling
warnings.filterwarnings('ignore')
from utils import apply_dashboard_plot_style
apply_dashboard_plot_style()

# Paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
STATIC_IMG_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'static', 'images', 'macro_analysis'))
MODEL_OUTPUTS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'model_outputs'))

# New imports and globals for DB storage
import base64
from io import BytesIO

_DBM = None  # Will be set by schedule_macro_analysis
_PRODUCED_NAMES = []  # Track stored visual names


def _ensure_dirs() -> None:
    """Ensure static plot directory exists."""
    try:
        if not os.path.exists(STATIC_IMG_DIR):
            os.makedirs(STATIC_IMG_DIR, exist_ok=True)
    except Exception as exc:
        print(f"⚠️  Could not create static images directory: {STATIC_IMG_DIR} | {exc}")


# Helper to ensure the macro_visuals table exists
def _ensure_macro_visuals_table() -> None:
    try:
        if _DBM is None:
            return
        create_sql = (
            """
            CREATE TABLE IF NOT EXISTS macro_visuals (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                image_base64 TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        _DBM.execute_transaction([{'query': create_sql, 'params': {}}])
        print("🗄️ Ensured table exists: macro_visuals")
    except Exception as exc:
        print(f"✗ Failed ensuring table macro_visuals: {exc}")


def _load_dsge_outputs(output_dir: Optional[str] = None) -> Dict[str, Optional[pd.DataFrame]]:
    """Load CSV outputs produced by the macro analysis stage."""
    base = output_dir if output_dir else MODEL_OUTPUTS_DIR
    data: Dict[str, Optional[pd.DataFrame]] = {
        'main_data': None,
        'simulation': None,
        'parameters': None,
        'kalman_diag': None,
        'summary_stats': None,
    }

    files = {
        'main_data': os.path.join(base, 'dsge_main_data.csv'),
        'simulation': os.path.join(base, 'dsge_simulation.csv'),
        'parameters': os.path.join(base, 'dsge_parameters.csv'),
        'kalman_diag': os.path.join(base, 'dsge_kalman_diagnostics.csv'),
        'summary_stats': os.path.join(base, 'dsge_summary_stats.csv'),
    }

    for key in files:
        try:
            if key in ('main_data', 'simulation'):
                df = pd.read_csv(files[key], index_col=0, parse_dates=True)
            else:
                df = pd.read_csv(files[key])
            data[key] = df
            print(f"✓ Loaded {key} from {files[key]}")
        except Exception as exc:
            print(f"✗ Missing or failed to load {key}: {files[key]} | {exc}")
            data[key] = None

    return data


def _save_fig(fig: plt.Figure, filename: str) -> None:
    """Encode figure as base64 and store in DB table macro_visuals."""
    try:
        buffer = BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode('ascii')

        if _DBM is None:
            print(f"    ⚠️  No database manager; skipped storing: {filename}")
            return

        try:
            insert_sql = (
                "INSERT INTO macro_visuals (name, image_base64, created_at) "
                "VALUES (:name, :image_base64, NOW())"
            )
            _DBM.execute_transaction([
                {'query': insert_sql, 'params': {'name': filename, 'image_base64': img_b64}}
            ])
            _PRODUCED_NAMES.append(filename)
            print(f"    ✓ Stored plot in DB: {filename}")
        except Exception as exc:
            print(f"    ✗ Failed to store {filename} in DB: {exc}")
    except Exception as exc:
        print(f"    ✗ Failed to process {filename}: {exc}")


def create_data_overview_plots(ds: Dict[str, Optional[pd.DataFrame]]) -> None:
    main_data = ds.get('main_data')
    if main_data is None or main_data.empty:
        print("⚠️  Data overview: main_data missing/empty")
        return

    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DSGE Model: Data Overview & Quality', fontsize=16, fontweight='bold')

        # 1. Policy & Inflation
        ax1 = axes[0, 0]
        if 'i' in main_data and 'pi' in main_data:
            ax1.plot(main_data.index, main_data['i'], label='Interest Rate (%)', linewidth=2)
            ax1.plot(main_data.index, main_data['pi'], label='Inflation (%)', linewidth=2)
        ax1.set_title('Monetary Policy & Inflation')
        ax1.set_ylabel('Percentage')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Output & Unemployment
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()
        if 'UNRATE' in main_data:
            ax2.plot(main_data.index, main_data['UNRATE'], label='Unemployment Rate (%)', color='red', linewidth=2)
        if 'YLVL' in main_data:
            ax2_twin.plot(main_data.index, main_data['YLVL'], label='Real GDP/Output', color='blue', linewidth=2)
        ax2.set_title('Real Activity & Labor Market')
        ax2.set_ylabel('Unemployment Rate (%)', color='red')
        ax2_twin.set_ylabel('Output Level', color='blue')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # 3. Credit Spreads
        ax3 = axes[1, 0]
        if 'SPREAD' in main_data:
            ax3.plot(main_data.index, main_data['SPREAD'], label='Credit Spread (bp)', color='purple', linewidth=2)
        ax3.set_title('Financial Conditions')
        ax3.set_ylabel('Basis Points')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Data Availability Heatmap
        ax4 = axes[1, 1]
        data_quality = main_data.notna().astype(int)
        sns.heatmap(data_quality.T, cmap='RdYlGn', ax=ax4, cbar_kws={'label': 'Data Available'})
        ax4.set_title('Data Availability')
        ax4.set_xlabel('Time Period')

        _save_fig(fig, 'macro_01_data_overview.png')
    except Exception as exc:
        print(f"✗ Data overview plotting error: {exc}")


def create_model_estimation_plots(ds: Dict[str, Optional[pd.DataFrame]]) -> None:
    main_data = ds.get('main_data')
    params = ds.get('parameters')
    kalman = ds.get('kalman_diag')

    if main_data is None or main_data.empty or params is None or params.empty:
        print("⚠️  Model estimation: required data missing/empty")
        return

    # Prepare variances if available
    x_var_aligned = None
    rn_var_aligned = None
    try:
        if kalman is not None and not kalman.empty and 'date' in kalman:
            kalman['date'] = pd.to_datetime(kalman['date'])
            kalman_idx = kalman.set_index('date')
            if 'x_gap_variance' in kalman_idx:
                x_var_aligned = kalman_idx['x_gap_variance'].reindex(main_data.index)
            if 'rn_variance' in kalman_idx:
                rn_var_aligned = kalman_idx['rn_variance'].reindex(main_data.index)
    except Exception as exc:
        print(f"  ⚠️  Variance alignment failed: {exc}")

    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DSGE Model: Estimation Results', fontsize=16, fontweight='bold')

        # 1. Output Gap Estimation
        ax1 = axes[0, 0]
        if 'x_gap_proxy' in main_data:
            ax1.plot(main_data.index, main_data['x_gap_proxy'], label='Simple Proxy', alpha=0.6, linewidth=1)
        if 'x_gap_est' in main_data:
            ax1.plot(main_data.index, main_data['x_gap_est'], label='Kalman Estimate', linewidth=2, color='red')
            if x_var_aligned is not None:
                low_band = main_data['x_gap_est'] - 2 * np.sqrt(x_var_aligned.fillna(0))
                high_band = main_data['x_gap_est'] + 2 * np.sqrt(x_var_aligned.fillna(0))
                ax1.fill_between(main_data.index, low_band, high_band, alpha=0.2, color='red', label='±2σ Confidence')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_title('Output Gap Estimation')
        ax1.set_ylabel('Output Gap (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Natural Rate Estimation
        ax2 = axes[0, 1]
        if 'rn_est' in main_data:
            ax2.plot(main_data.index, main_data['rn_est'], label='Natural Rate', linewidth=2, color='green')
        try:
            r_star = params.loc[params['parameter'] == 'r_star', 'value'].iloc[0]
            ax2.axhline(y=r_star, color='black', linestyle='--', alpha=0.5, label='r* (Target)')
        except Exception:
            pass
        if rn_var_aligned is not None and 'rn_est' in main_data:
            low_band = main_data['rn_est'] - 2 * np.sqrt(rn_var_aligned.fillna(0))
            high_band = main_data['rn_est'] + 2 * np.sqrt(rn_var_aligned.fillna(0))
            ax2.fill_between(main_data.index, low_band, high_band, alpha=0.2, color='green', label='±2σ Confidence')
        ax2.set_title('Natural Real Rate')
        ax2.set_ylabel('Real Rate (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Model Fit - Inflation
        ax3 = axes[1, 0]
        if 'pi' in main_data:
            ax3.plot(main_data.index, main_data['pi'], label='Actual Inflation', linewidth=2)
            try:
                beta_val = params.loc[params['parameter'] == 'beta', 'value'].iloc[0]
                kappa_val = params.loc[params['parameter'] == 'kappa', 'value'].iloc[0]
                fitted_pi = beta_val * main_data['pi'].shift(1).fillna(method='bfill') + kappa_val * main_data.get('x_gap_est', 0).fillna(0)
                ax3.plot(main_data.index, fitted_pi, label='Model Fitted', linewidth=2, linestyle='--', color='orange')
            except Exception:
                pass
        ax3.set_title('Inflation: Model Fit')
        ax3.set_ylabel('Inflation (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Kalman Filter Diagnostics
        ax4 = axes[1, 1]
        if kalman is not None and not kalman.empty and 'date' in kalman:
            try:
                ax4.plot(pd.to_datetime(kalman['date']), kalman.get('x_gap_variance', pd.Series(dtype=float)), label='Output Gap Variance', linewidth=2)
                ax4.plot(pd.to_datetime(kalman['date']), kalman.get('rn_variance', pd.Series(dtype=float)), label='Natural Rate Variance', linewidth=2)
            except Exception:
                pass
        ax4.set_title('Kalman Filter: State Uncertainty')
        ax4.set_ylabel('Variance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        _save_fig(fig, 'macro_02_model_estimation.png')
    except Exception as exc:
        print(f"✗ Model estimation plotting error: {exc}")


def create_policy_shock_analysis(ds: Dict[str, Optional[pd.DataFrame]]) -> None:
    simulation = ds.get('simulation')
    params = ds.get('parameters')

    if simulation is None or simulation.empty:
        print("⚠️  Policy shock: simulation missing/empty")
        return

    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DSGE Model: +100bp Policy Shock Analysis', fontsize=16, fontweight='bold')

        months = range(len(simulation))

        # 1. Output Gap
        ax1 = axes[0, 0]
        if 'x' in simulation:
            ax1.plot(months, simulation['x'], linewidth=3, color='red', marker='o')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_title('Output Gap Response')
        ax1.set_ylabel('Output Gap (%)')
        ax1.set_xlabel('Months After Shock')
        ax1.grid(True, alpha=0.3)

        # 2. Inflation
        ax2 = axes[0, 1]
        if 'pi' in simulation:
            ax2.plot(months, simulation['pi'], linewidth=3, color='blue', marker='s')
        try:
            pi_star = params.loc[params['parameter'] == 'pi_star', 'value'].iloc[0]
            ax2.axhline(y=pi_star, color='black', linestyle='--', alpha=0.5, label='π* Target')
            ax2.legend()
        except Exception:
            pass
        ax2.set_title('Inflation Response')
        ax2.set_ylabel('Inflation (%)')
        ax2.set_xlabel('Months After Shock')
        ax2.grid(True, alpha=0.3)

        # 3. Interest Rate
        ax3 = axes[1, 0]
        if 'i' in simulation:
            ax3.plot(months, simulation['i'], linewidth=3, color='green', marker='^')
            ax3.axhline(y=simulation['i'].iloc[0], color='black', linestyle='--', alpha=0.5, label='Pre-shock Level')
            ax3.legend()
        ax3.set_title('Interest Rate Response')
        ax3.set_ylabel('Interest Rate (%)')
        ax3.set_xlabel('Months After Shock')
        ax3.grid(True, alpha=0.3)

        # 4. Unemployment
        ax4 = axes[1, 1]
        if 'u' in simulation:
            ax4.plot(months, simulation['u'], linewidth=3, color='purple', marker='d')
        try:
            u_star = params.loc[params['parameter'] == 'u_star', 'value'].iloc[0]
            ax4.axhline(y=u_star, color='black', linestyle='--', alpha=0.5, label='u* Natural Rate')
            ax4.legend()
        except Exception:
            pass
        ax4.set_title('Unemployment Response')
        ax4.set_ylabel('Unemployment Rate (%)')
        ax4.set_xlabel('Months After Shock')
        ax4.grid(True, alpha=0.3)

        _save_fig(fig, 'macro_03_policy_shock.png')
    except Exception as exc:
        print(f"✗ Policy shock plotting error: {exc}")


def create_model_diagnostics(ds: Dict[str, Optional[pd.DataFrame]]) -> None:
    main_data = ds.get('main_data')
    params = ds.get('parameters')
    summary = ds.get('summary_stats')

    if (main_data is None or main_data.empty) and (params is None or params.empty):
        print("⚠️  Model diagnostics: required data missing/empty")
        return

    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DSGE Model: Diagnostics & Validation', fontsize=16, fontweight='bold')

        # 1. Parameter Summary (Horizontal Bar)
        ax1 = axes[0, 0]
        if params is not None and not params.empty and 'parameter' in params and 'value' in params:
            params_plot = params.set_index('parameter')
            bars = ax1.barh(range(len(params_plot)), params_plot['value'])
            ax1.set_yticks(range(len(params_plot)))
            ax1.set_yticklabels(params_plot.index, fontsize=8)
            ax1.set_title('Model Parameters')
            ax1.set_xlabel('Parameter Value')
            k = 0
            while k < len(bars):
                width = bars[k].get_width()
                ax1.text(width + 0.01, bars[k].get_y() + bars[k].get_height() / 2, f'{width:.2f}', ha='left', va='center', fontsize=8)
                k += 1

        # 2. Summary Statistics (mean/std) or compute from main_data
        ax2 = axes[0, 1]
        summary_plot = None
        if summary is not None and not summary.empty and {'mean', 'std'}.issubset(set(summary.columns)):
            try:
                summary_plot = summary[['mean', 'std']].T
            except Exception:
                summary_plot = None
        if summary_plot is None and main_data is not None and not main_data.empty:
            core_vars = ['i', 'pi', 'UNRATE', 'SPREAD', 'x_gap_est', 'rn_est']
            available_vars = [var for var in core_vars if var in main_data.columns]
            if available_vars:
                summary_plot = main_data[available_vars].agg(['mean', 'std']).round(3)
        if summary_plot is not None:
            summary_plot.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Variable Summary Statistics')
        ax2.set_ylabel('Value')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Correlation Matrix
        ax3 = axes[1, 0]
        if main_data is not None and not main_data.empty:
            corr_vars = [v for v in ['i', 'pi', 'UNRATE', 'SPREAD', 'x_gap_est', 'rn_est'] if v in main_data.columns]
            if len(corr_vars) >= 2:
                corr_matrix = main_data[corr_vars].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax3, fmt='.2f')
        ax3.set_title('Variable Correlations')

        # 4. Output Gap Time Series with band
        ax4 = axes[1, 1]
        if main_data is not None and not main_data.empty and 'x_gap_est' in main_data:
            ax4.plot(main_data.index, main_data['x_gap_est'], label='Output Gap', linewidth=2)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.fill_between(main_data.index, main_data['x_gap_est'] - 1, main_data['x_gap_est'] + 1, alpha=0.2, color='gray', label='±1% Band')
            ax4.legend()
        ax4.set_title('Output Gap: Business Cycle Identification')
        ax4.set_ylabel('Output Gap (%)')
        ax4.grid(True, alpha=0.3)

        _save_fig(fig, 'macro_04_model_diagnostics.png')
    except Exception as exc:
        print(f"✗ Diagnostics plotting error: {exc}")


def create_economic_insights(ds: Dict[str, Optional[pd.DataFrame]]) -> None:
    main_data = ds.get('main_data')
    simulation = ds.get('simulation')

    if main_data is None or main_data.empty:
        print("⚠️  Economic insights: main_data missing/empty")
        return

    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DSGE Model: Economic Insights & Policy Analysis', fontsize=16, fontweight='bold')

        # 1. Policy Stance (i - pi - rn_est)
        ax1 = axes[0, 0]
        if {'i', 'pi', 'rn_est'}.issubset(set(main_data.columns)):
            policy_stance = main_data['i'] - main_data['pi'] - main_data['rn_est']
            ax1.plot(main_data.index, policy_stance, linewidth=2, color='red')
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Neutral Policy')
            ax1.fill_between(main_data.index, policy_stance, 0, where=(policy_stance > 0), alpha=0.3, color='red', label='Tight Policy')
            ax1.fill_between(main_data.index, policy_stance, 0, where=(policy_stance < 0), alpha=0.3, color='green', label='Loose Policy')
            ax1.legend()
        ax1.set_title('Monetary Policy Stance')
        ax1.set_ylabel('Policy Rate - Natural Rate (%)')
        ax1.grid(True, alpha=0.3)

        # 2. Business Cycle Phases (scatter of x_gap_est vs pi)
        ax2 = axes[0, 1]
        output_gap = main_data['x_gap_est'] if 'x_gap_est' in main_data else None
        inflation = main_data['pi'] if 'pi' in main_data else None
        if output_gap is not None and inflation is not None:
            phases = pd.DataFrame(index=main_data.index)
            phases['phase'] = 'Normal'
            phases.loc[(output_gap > 0) & (inflation > 2), 'phase'] = 'Overheating'
            phases.loc[(output_gap < 0) & (inflation < 2), 'phase'] = 'Recession'
            phases.loc[(output_gap > 0) & (inflation < 2), 'phase'] = 'Recovery'
            phases.loc[(output_gap < 0) & (inflation > 2), 'phase'] = 'Stagflation'
            colors = {'Normal': 'gray', 'Overheating': 'red', 'Recession': 'blue', 'Recovery': 'green', 'Stagflation': 'orange'}
            for phase in colors:
                mask = phases['phase'] == phase
                if mask.any():
                    ax2.scatter(output_gap[mask], inflation[mask], c=colors[phase], label=phase, alpha=0.7, s=30)
            ax2.axhline(y=2, color='black', linestyle='--', alpha=0.5, label='Inflation Target')
            ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Potential Output')
            ax2.legend()
        ax2.set_title('Business Cycle Phases')
        ax2.set_xlabel('Output Gap (%)')
        ax2.set_ylabel('Inflation (%)')
        ax2.grid(True, alpha=0.3)

        # 3. Composite Risk Index (simple weights)
        ax3 = axes[1, 0]
        try:
            risk_indicators = pd.DataFrame(index=main_data.index)
            risk_indicators['output_risk'] = np.abs(main_data.get('x_gap_est', 0))
            risk_indicators['inflation_risk'] = np.abs(main_data.get('pi', 0) - 2)
            risk_indicators['financial_risk'] = main_data.get('SPREAD', 0) / 100.0
            risk_indicators['composite_risk'] = (
                risk_indicators['output_risk'] * 0.4 +
                risk_indicators['inflation_risk'] * 0.4 +
                risk_indicators['financial_risk'] * 0.2
            )
            ax3.plot(risk_indicators.index, risk_indicators['composite_risk'], linewidth=2, color='purple', label='Composite Risk')
            try:
                high_thr = risk_indicators['composite_risk'].quantile(0.75)
                ax3.axhline(y=high_thr, color='red', linestyle='--', alpha=0.7, label='High Risk Threshold')
            except Exception:
                pass
            ax3.legend()
        except Exception:
            pass
        ax3.set_title('Economic Risk Assessment')
        ax3.set_ylabel('Risk Index')
        ax3.grid(True, alpha=0.3)

        # 4. Cumulative Policy Effects from simulation
        ax4 = axes[1, 1]
        if simulation is not None and not simulation.empty:
            try:
                cumulative_effects = simulation.cumsum()
                if 'x' in cumulative_effects:
                    ax4.plot(range(len(cumulative_effects)), cumulative_effects['x'], label='Cumulative Output Loss', linewidth=2, color='red')
                if 'pi' in cumulative_effects:
                    ax4.plot(range(len(cumulative_effects)), cumulative_effects['pi'], label='Cumulative Inflation Effect', linewidth=2, color='blue')
                ax4.legend()
            except Exception:
                pass
        ax4.set_title('Cumulative Policy Shock Effects')
        ax4.set_xlabel('Months After Shock')
        ax4.set_ylabel('Cumulative Effect')
        ax4.grid(True, alpha=0.3)

        _save_fig(fig, 'macro_05_economic_insights.png')
    except Exception as exc:
        print(f"✗ Economic insights plotting error: {exc}")


def schedule_macro_analysis(db_manager) -> bool:
    """
    Orchestrates macro analysis and visualization for Flask scheduler route.

    Steps:
    1) Ensure directories
    2) Run macro analysis (creates CSVs in model_outputs)
    3) Load outputs
    4) Generate and save plots to static dir
    """
    print("\n" + "=" * 60)
    print("🚀 STARTING MACRO ANALYSIS & VISUALIZATION")
    print("=" * 60)

    # Set DB manager for this run and ensure table
    global _DBM
    _DBM = db_manager
    
    # Reset produced names tracker per run
    global _PRODUCED_NAMES
    _PRODUCED_NAMES = []

    _ensure_dirs()

    # Ensure target DB table exists
    _ensure_macro_visuals_table()

    # 1) Run macro analysis to (re)generate CSV outputs
    analysis_ok = True
    try:
        # Mirror sectors method: import module by file path to avoid package import issues
        import importlib.util
        target_path = os.path.join(THIS_DIR, 'macro_analysis.py')
        spec = importlib.util.spec_from_file_location('macro_analysis', target_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        run_macro_analysis = getattr(mod, 'run_macro_analysis')

        print("🔧 Running core macro analysis to produce CSV outputs...")
        result = run_macro_analysis(db_manager, output_dir=os.path.basename(MODEL_OUTPUTS_DIR))
        # result may be Dict[str, Any] per outline; treat missing as success fallback
        if isinstance(result, dict) and ('success' in result) and (not result.get('success', True)):
            analysis_ok = False
            print("✗ Macro analysis reported failure flag")
        else:
            print("✓ Macro analysis completed")
    except Exception as exc:
        analysis_ok = False
        print(f"✗ Macro analysis execution error: {exc}")

    # 2) Load data regardless; we may still have prior CSVs
    ds = _load_dsge_outputs(MODEL_OUTPUTS_DIR)

    # 2.5) Store outputs into SQL tables (replace on each run)
    try:
        if _DBM is not None:
            # main_data
            try:
                main_df = ds.get('main_data')
                if main_df is not None and not main_df.empty:
                    df = main_df.copy().reset_index()
                    # Rename index column to 'date'
                    idx_name = df.columns[0]
                    if idx_name != 'date':
                        df = df.rename(columns={idx_name: 'date'})
                    _DBM.bulk_insert('macro_dsge_main_data', df.to_dict(orient='records'), if_exists='replace')
                    print("🗄️ Stored table: macro_dsge_main_data")
            except Exception as exc:
                print(f"✗ Failed storing macro_dsge_main_data: {exc}")

            # simulation
            try:
                sim_df = ds.get('simulation')
                if sim_df is not None and not sim_df.empty:
                    df = sim_df.copy().reset_index()
                    idx_name = df.columns[0]
                    # Call the first column 'step' for clarity
                    if idx_name != 'step':
                        df = df.rename(columns={idx_name: 'step'})
                    _DBM.bulk_insert('macro_dsge_simulation', df.to_dict(orient='records'), if_exists='replace')
                    print("🗄️ Stored table: macro_dsge_simulation")
            except Exception as exc:
                print(f"✗ Failed storing macro_dsge_simulation: {exc}")

            # parameters
            try:
                params_df = ds.get('parameters')
                if params_df is not None and not params_df.empty:
                    _DBM.bulk_insert('macro_dsge_parameters', params_df.to_dict(orient='records'), if_exists='replace')
                    print("🗄️ Stored table: macro_dsge_parameters")
            except Exception as exc:
                print(f"✗ Failed storing macro_dsge_parameters: {exc}")

            # kalman diagnostics
            try:
                kalman_df = ds.get('kalman_diag')
                if kalman_df is not None and not kalman_df.empty:
                    _DBM.bulk_insert('macro_dsge_kalman_diagnostics', kalman_df.to_dict(orient='records'), if_exists='replace')
                    print("🗄️ Stored table: macro_dsge_kalman_diagnostics")
            except Exception as exc:
                print(f"✗ Failed storing macro_dsge_kalman_diagnostics: {exc}")

            # summary stats
            try:
                sum_df = ds.get('summary_stats')
                if sum_df is not None and not sum_df.empty:
                    df = sum_df.copy()
                    if 'Unnamed: 0' in df.columns:
                        df = df.rename(columns={'Unnamed: 0': 'stat'})
                    _DBM.bulk_insert('macro_dsge_summary_stats', df.to_dict(orient='records'), if_exists='replace')
                    print("🗄️ Stored table: macro_dsge_summary_stats")
            except Exception as exc:
                print(f"✗ Failed storing macro_dsge_summary_stats: {exc}")
    except Exception as exc:
        print(f"✗ Failed storing outputs to SQL: {exc}")

    # 3) Generate plots
    try:
        print("📊 Creating data overview plots...")
        create_data_overview_plots(ds)
    except Exception as exc:
        print(f"✗ Failed data overview: {exc}")

    try:
        print("📈 Creating model estimation plots...")
        create_model_estimation_plots(ds)
    except Exception as exc:
        print(f"✗ Failed model estimation: {exc}")

    try:
        print("🏛️ Creating policy shock analysis...")
        create_policy_shock_analysis(ds)
    except Exception as exc:
        print(f"✗ Failed policy shock analysis: {exc}")

    try:
        print("🧪 Creating model diagnostics...")
        create_model_diagnostics(ds)
    except Exception as exc:
        print(f"✗ Failed model diagnostics: {exc}")

    try:
        print("💡 Creating economic insights...")
        create_economic_insights(ds)
    except Exception as exc:
        print(f"✗ Failed economic insights: {exc}")

    print("\n" + "=" * 60)
    print("✅ MACRO VISUALIZATION PIPELINE COMPLETE")
    print("=" * 60)
    print("Plots stored as base64 in table: macro_visuals")

    # Return True if analysis ran OK and at least one visual was stored
    try:
        produced = len(_PRODUCED_NAMES)
        return analysis_ok and produced > 0
    except Exception:
        return analysis_ok


if __name__ == '__main__':
    # Simple manual run (without Flask) if desired
    print("Running macro visualization as a script...")
    try:
        # Lazy import to avoid DB requirements when not available
        try:
            from .database import create_database_manager  # type: ignore
        except Exception:
            from Dashboard.database import create_database_manager  # type: ignore
        dbm = create_database_manager()
    except Exception:
        dbm = None
    ok = schedule_macro_analysis(dbm)
    print(f"Done. Success={ok}") 