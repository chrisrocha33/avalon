from datetime import datetime, timedelta
from flask import Blueprint, render_template
from extensions import db_manager
# Add plotting and encoding utilities
import io
import base64
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import apply_dashboard_plot_style

apply_dashboard_plot_style()

yields_bp = Blueprint('yields', __name__, url_prefix='')


@yields_bp.get('/yields_us')
def yields_view():
    # Load a wider window for stable calculations (e.g., ~6 months)
    today_date = datetime.today().date()
    start_date = today_date - timedelta(days=180)

    df = db_manager.read_sql_pandas(
        """
        SELECT "Date","DTB3","DTB4WK","DGS1","DGS2","DGS3","DGS5","DGS7","DGS10","DGS20","DGS30"
        FROM "us_Yields_Rates"
        WHERE "Date"::date >= %(start_date)s
        ORDER BY "Date" ASC
        """,
        params={"start_date": start_date},
    )

    # Ensure DataFrame integrity and types
    if not hasattr(df, 'copy'):
        df = pd.DataFrame()
    num_cols = ['DTB3','DTB4WK','DGS1','DGS2','DGS3','DGS5','DGS7','DGS10','DGS20','DGS30']
    for c in ['Date'] + num_cols:
        if c not in getattr(df, 'columns', []):
            df[c] = pd.NA

    # Date as date (better labels) and numeric coercion
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    # Convenience columns
    df['short3m'] = df['DTB3'].fillna(df['DTB4WK'])

    # Yield curve (latest snapshot)
    labels = ['3m','1y','2y','3y','5y','7y','10y','20y','30y']
    yc_cols = ['short3m','DGS1','DGS2','DGS3','DGS5','DGS7','DGS10','DGS20','DGS30']
    last_valid = df.dropna(subset=yc_cols, how='all').tail(1)
    if not last_valid.empty:
        yc_vals = last_valid.iloc[0][yc_cols].to_numpy(dtype=float)
        title_date = last_valid.iloc[0]['Date']
    else:
        yc_vals = np.array([np.nan] * len(yc_cols))
        title_date = ''

    # Plot yield curve
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.grid(True)
    xs = np.arange(len(labels))
    mask = ~np.isnan(yc_vals)
    if mask.any():
        ax.plot(xs[mask], yc_vals[mask], marker='o')
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_title(f'Yield Curve (latest) {title_date}')
        ax.set_ylabel('%')
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    yield_curve_plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Spreads: 2s10s, 5s30s, 3m10y
    df['spread_2s10s'] = df['DGS10'] - df['DGS2']
    df['spread_5s30s'] = df['DGS30'] - df['DGS5']
    df['spread_3m10y'] = df['DGS10'] - df['short3m']

    latest_2s10_series = df['spread_2s10s'].dropna()
    if len(latest_2s10_series) > 0:
        latest_2s10 = latest_2s10_series.iloc[-1]
        if latest_2s10 > 0.50:
            spreads_regime_flag = 'steep'
        elif latest_2s10 < -0.25:
            spreads_regime_flag = 'inverted'
        else:
            spreads_regime_flag = 'flat'
    else:
        spreads_regime_flag = 'flat'

    # Plot spreads (last ~20)
    sel1 = df[['Date', 'spread_2s10s']].dropna().tail(20)
    sel2 = df[['Date', 'spread_5s30s']].dropna().tail(20)
    sel3 = df[['Date', 'spread_3m10y']].dropna().tail(20)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.grid(True)
    if not sel1.empty:
        ax.plot(sel1['Date'], sel1['spread_2s10s'], label='2s10s')
    if not sel2.empty:
        ax.plot(sel2['Date'], sel2['spread_5s30s'], label='5s30s')
    if not sel3.empty:
        ax.plot(sel3['Date'], sel3['spread_3m10y'], label='3m10y')
    ax.set_title('Yield Curve Spreads (last ~20 trading days)')
    ax.set_ylabel('Spread (bp)')
    ax.legend(loc='best')
    fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    spreads_plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Spreads table (last ~20)
    spreads_table_html = df[['Date', 'spread_2s10s', 'spread_5s30s', 'spread_3m10y']].tail(1).to_html(
        index=False,
        float_format=lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A'
    )

    # PCA-style factors (Level, Slope, Curvature) – heuristic
    df['factor_level'] = df[yc_cols].mean(axis=1)
    df['factor_slope'] = np.where(
        df['DGS30'].notna() & df['short3m'].notna(),
        df['DGS30'] - df['short3m'],
        df['DGS10'] - df['DGS2']
    )
    df['factor_curv'] = 2.0 * df['DGS10'] - (df['DGS2'] + df['DGS30'])

    # Loadings (fixed illustrative)
    maturities = [0.25, 1, 2, 3, 5, 7, 10, 20, 30]
    load_level = [1.0] * len(maturities)
    load_slope = [(-1.0 if m <= 1 else (1.0 if m >= 10 else 0.0)) for m in maturities]
    load_curv = [(2.0 if 4 <= m <= 7 else (-1.0 if (m <= 1 or m >= 20) else 0.5)) for m in maturities]

    # Plot loadings
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.grid(True)
    positions = np.arange(len(labels))
    ax.bar(positions - 0.2, load_level, width=0.2, label='Level')
    ax.bar(positions, load_slope, width=0.2, label='Slope')
    ax.bar(positions + 0.2, load_curv, width=0.2, label='Curvature')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title('Heuristic Factor Loadings')
    ax.legend(loc='best')
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    pca_loadings_plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Plot factor scores (last ~20)
    selL = df[['Date', 'factor_level']].dropna().tail(20)
    selS = df[['Date', 'factor_slope']].dropna().tail(20)
    selC = df[['Date', 'factor_curv']].dropna().tail(20)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.grid(True)
    if not selL.empty:
        ax.plot(selL['Date'], selL['factor_level'], label='Level')
    if not selS.empty:
        ax.plot(selS['Date'], selS['factor_slope'], label='Slope')
    if not selC.empty:
        ax.plot(selC['Date'], selC['factor_curv'], label='Curvature')
    ax.set_title('Factor Scores (last ~20 trading days)')
    ax.legend(loc='best')
    fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    pca_scores_plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Snapshot table for latest factor values
    latest_level = df['factor_level'].dropna().iloc[-1] if df['factor_level'].notna().any() else np.nan
    latest_slope = df['factor_slope'].dropna().iloc[-1] if df['factor_slope'].notna().any() else np.nan
    latest_curv = df['factor_curv'].dropna().iloc[-1] if df['factor_curv'].notna().any() else np.nan
    pca_snapshot_html = '<table><thead><tr><th>Factor</th><th>Value</th></tr></thead><tbody>'
    pca_snapshot_html += f'<tr><td style="text-align:left;">Level</td><td>{(f"{latest_level:.2f}") if pd.notna(latest_level) else "N/A"}</td></tr>'
    pca_snapshot_html += f'<tr><td style="text-align:left;">Slope</td><td>{(f"{latest_slope:.2f}") if pd.notna(latest_slope) else "N/A"}</td></tr>'
    pca_snapshot_html += f'<tr><td style="text-align:left;">Curvature</td><td>{(f"{latest_curv:.2f}") if pd.notna(latest_curv) else "N/A"}</td></tr>'
    pca_snapshot_html += '</tbody></table>'

    # Forwards (1y1y, 5y5y)
    mask_1y1y = df['DGS1'].notna() & df['DGS2'].notna()
    df['f_1y1y'] = np.where(
        mask_1y1y,
        ((1.0 + df['DGS2']/100.0) ** 2 / (1.0 + df['DGS1']/100.0) - 1.0) * 100.0,
        np.nan,
    )
    mask_5y5y = df['DGS5'].notna() & df['DGS10'].notna()
    ratio = np.where(
        mask_5y5y,
        (1.0 + df['DGS10']/100.0) ** 10 / (1.0 + df['DGS5']/100.0) ** 5,
        np.nan,
    )
    df['f_5y5y'] = np.where(ratio > 0, (np.power(ratio, 1.0/5.0) - 1.0) * 100.0, np.nan)

    # Plot forwards (last ~20)
    selF1 = df[['Date', 'f_1y1y']].dropna().tail(20)
    selF2 = df[['Date', 'f_5y5y']].dropna().tail(20)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.grid(True)
    if not selF1.empty:
        ax.plot(selF1['Date'], selF1['f_1y1y'], label='1y1y')
    if not selF2.empty:
        ax.plot(selF2['Date'], selF2['f_5y5y'], label='5y5y')
    ax.set_title('Forward Rates (last ~20 trading days)')
    ax.set_ylabel('%')
    ax.legend(loc='best')
    fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    forwards_plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Forwards table (last ~20)
    forwards_table_html = df[['Date', 'f_1y1y', 'f_5y5y']].tail(5).to_html(
        index=False,
        float_format=lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A'
    )

    # Term Premium proxy: TP_10y ≈ DGS10 - 5y5y
    df['tp_10y'] = df['DGS10'] - df['f_5y5y']

    # Plot term premium (last ~20)
    selTP = df[['Date', 'tp_10y']].dropna().tail(20)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.grid(True)
    if not selTP.empty:
        ax.plot(selTP['Date'], selTP['tp_10y'], label='TP 10y (proxy)')
    ax.set_title('10y Term Premium Proxy (last ~20 trading days)')
    ax.set_ylabel('pp')
    ax.legend(loc='best')
    fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    term_premium_plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Term premium table (last ~20)
    term_premium_table_html = df[['Date', 'DGS10', 'f_5y5y', 'tp_10y']].tail(5).to_html(
        index=False,
        float_format=lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A'
    )

    # Duration / Convexity profiles (latest)
    dur_conv_mats = [0.25, 1, 2, 3, 5, 7, 10, 20, 30]
    dur_vals = dur_conv_mats[:]  # duration ~ maturity (zero-coupon approx)
    conv_vals = [m * m for m in dur_conv_mats]

    # Plot duration/convexity
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.grid(True)
    pos = np.arange(len(dur_conv_mats))
    ax.bar(pos - 0.15, dur_vals, width=0.3, label='Duration (yrs)')
    ax.bar(pos + 0.15, conv_vals, width=0.3, label='Convexity (approx)')
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_title('Duration and Convexity (Zero-coupon approx)')
    ax.legend(loc='best')
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    duration_convexity_plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Duration/Convexity table
    duration_df = pd.DataFrame({
        'Maturity': labels,
        'Duration (yrs)': dur_vals,
        'Convexity (approx)': conv_vals,
    })
    duration_table_html = duration_df.to_html(index=False, float_format=lambda x: f"{x:.2f}")

    return render_template(
        'yields_analysis.html',
        title='US Yields Analysis',
        date=datetime.now().strftime('%Y-%m-%d'),
        active_page='yields',
        # yield curve
        yield_curve_plot_b64=yield_curve_plot_b64,
        # spreads
        spreads_plot_b64=spreads_plot_b64,
        spreads_table_html=spreads_table_html,
        spreads_regime_flag=spreads_regime_flag,
        # PCA
        pca_loadings_plot_b64=pca_loadings_plot_b64,
        pca_scores_plot_b64=pca_scores_plot_b64,
        pca_snapshot_html=pca_snapshot_html,
        # forwards
        forwards_plot_b64=forwards_plot_b64,
        forwards_table_html=forwards_table_html,
        # term premium
        term_premium_plot_b64=term_premium_plot_b64,
        term_premium_table_html=term_premium_table_html,
        # duration/convexity
        duration_convexity_plot_b64=duration_convexity_plot_b64,
        duration_table_html=duration_table_html,
    ) 