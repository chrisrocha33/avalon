from datetime import datetime
from flask import Blueprint, render_template

from extensions import db_manager
from data_processing.macro_visualization import schedule_macro_analysis

macro_bp = Blueprint('macro', __name__, url_prefix='')


@macro_bp.get('/macro')
def macro_view():
    date_str = datetime.now().strftime('%Y-%m-%d')

    # Map template keys to expected plot filenames
    name_map = {
        'data_overview_plot': 'macro_01_data_overview.png',
        'model_estimation_plot': 'macro_02_model_estimation.png',
        'model_diagnostics_plot': 'macro_04_model_diagnostics.png',
        'economic_insights_plot': 'macro_05_economic_insights.png',
    }

    # Fetch latest base64 for each plot and construct data URLs
    context = {
        'title': 'Macro Analysis',
        'date': date_str,
        'data_start_date': '',
        'data_end_date': '',
        'total_observations': '',
        'key_statistics': [],  # legacy, not used in template after update
        'model_parameters': [],  # legacy compatibility
        'model_inputs': [],
        'snapshot_stats': [],
    }
    context['show_shock_analysis'] = False

    for key, fname in name_map.items():
        try:
            rows = db_manager.execute_query(
                "SELECT image_base64 FROM macro_visuals WHERE name = :name ORDER BY created_at DESC LIMIT 1",
                params={'name': fname},
                fetch=True
            )
            if rows:
                # rows[0] is a Row; access first column
                b64 = rows[0][0]
                if b64:
                    context[key] = f"data:image/png;base64,{b64}"
                else:
                    context[key] = None
            else:
                context[key] = None
        except Exception:
            context[key] = None

    # Friendly display labels (optional)
    series_labels = {
        'i': 'Policy Rate',
        'pi': 'Inflation (YoY %)',
        'UNRATE': 'Unemployment Rate',
        'YLVL': 'Output Level',
        'SPREAD': 'Credit Spread (bp)',
        'r_real': 'Real Rate',
        'y_log': 'Log Output',
        'x_gap_proxy': 'Output Gap (Proxy)',
        'x_gap_est': 'Output Gap (Kalman)',
        'rn_est': 'Natural Real Rate',
    }
    param_labels = {
        'phi_pi': 'Taylor Rule π response (phi_pi)',
        'phi_x': 'Taylor Rule x response (phi_x)',
        'rho_i': 'Rate Smoothing (rho_i)',
        'beta': 'Inflation Persistence (beta)',
        'kappa': "Phillips Curve Slope (kappa)",
        'ax': 'Output Gap Persistence (ax)',
        'b_IS': 'IS Curve Slope (b_IS)',
        'rho_rn': 'Natural Rate Persistence (rho_rn)',
        'pi_star': 'Inflation Target π*',
        'r_star': 'Natural Real Rate r*',
        'lam_okun': "Okun's Coefficient",
        'sig_x': 'Shock Var: Output Gap',
        'sig_rn': 'Shock Var: Natural Rate',
        'sig_pi': 'Meas. Noise: Inflation',
        'sig_i': 'Meas. Noise: Interest Rate',
        'sig_u': 'Meas. Noise: Unemployment',
        'sigma': 'Risk Aversion (sigma)',
        'u_star': 'Natural Unemployment Rate (u*)',
    }

    # Helper: assign group for ordering
    def _series_group(name: str) -> int:
        n = name or ''
        if n in ('i', 'r_real'):  # policy-related
            return 0
        if n in ('pi',):  # inflation
            return 1
        if n in ('YLVL', 'y_log', 'x_gap_proxy', 'x_gap_est', 'rn_est'):  # activity
            return 2
        if n in ('UNRATE',):  # labor
            return 3
        if n in ('SPREAD',):  # financial
            return 4
        return 9

    def _param_group(name: str) -> int:
        n = name or ''
        if n in ('phi_pi', 'phi_x', 'rho_i', 'r_star', 'pi_star'):
            return 0  # policy rule
        if n in ('beta', 'kappa'):
            return 1  # inflation block
        if n in ('ax', 'b_IS'):
            return 2  # activity/IS
        if n in ('lam_okun',):
            return 3  # labor
        if n.startswith('sig_') or n in ('rho_rn',):
            return 4  # shocks/variances
        return 9

    # Read SQL tables and build snapshot + dates
    try:
        main_df = db_manager.read_sql_pandas("SELECT * FROM macro_dsge_main_data ORDER BY date ASC")
        if hasattr(main_df, 'empty') and not main_df.empty:
            # Dates/rows
            try:
                context['data_start_date'] = str(main_df['date'].iloc[0])
                context['data_end_date'] = str(main_df['date'].iloc[-1])
                context['total_observations'] = int(len(main_df))
            except Exception:
                pass

            # Compute loop-based stats for each numeric series (excluding 'date')
            cols = []
            try:
                for c in list(main_df.columns):
                    if str(c).lower() != 'date':
                        cols.append(c)
            except Exception:
                cols = []

            snapshot_rows = []
            k = 0
            while k < len(cols):
                col = cols[k]
                # Collect values with numeric coercion
                vals = []
                try:
                    j = 0
                    series_vals = main_df[col]
                    # iterate row-wise
                    while j < len(series_vals):
                        v = series_vals.iloc[j]
                        num = None
                        try:
                            # attempt numeric cast
                            num = float(v)
                        except Exception:
                            num = None
                        if num is not None:
                            vals.append(num)
                        j += 1
                except Exception:
                    vals = []

                count_n = 0
                sum_v = 0.0
                i2 = 0
                while i2 < len(vals):
                    sum_v += vals[i2]
                    count_n += 1
                    i2 += 1

                mean_v = 0.0
                if count_n > 0:
                    mean_v = sum_v / float(count_n)

                # second pass for std (sample, ddof=1)
                ss = 0.0
                i3 = 0
                while i3 < len(vals):
                    diff = vals[i3] - mean_v
                    ss += diff * diff
                    i3 += 1
                std_v = 0.0
                if count_n > 1:
                    std_v = (ss / float(count_n - 1)) ** 0.5

                # Round values
                try:
                    mean_r = round(mean_v, 3)
                except Exception:
                    mean_r = mean_v
                try:
                    std_r = round(std_v, 3)
                except Exception:
                    std_r = std_v

                # Label + grouping
                label = series_labels.get(str(col), str(col))
                snapshot_rows.append({
                    'series': str(col),
                    'label': label,
                    'mean': mean_r,
                    'count': int(count_n),
                    'std': std_r,
                    'group': _series_group(str(col)),
                })

                k += 1

            # Sort by group then label
            try:
                snapshot_rows.sort(key=lambda r: (r.get('group', 9), r.get('label', '')))
            except Exception:
                pass

            context['snapshot_stats'] = snapshot_rows
    except Exception:
        # leave defaults/empty
        pass

    # Parameters → model inputs
    try:
        params_df = db_manager.read_sql_pandas(
            "SELECT parameter, value, COALESCE(description,'') AS description FROM macro_dsge_parameters"
        )
        inputs = []
        if hasattr(params_df, 'empty') and not params_df.empty:
            i = 0
            while i < len(params_df):
                try:
                    p = str(params_df['parameter'].iloc[i])
                except Exception:
                    p = ''
                try:
                    v_raw = params_df['value'].iloc[i]
                    v_num = float(v_raw)
                except Exception:
                    v_num = None
                try:
                    desc = str(params_df['description'].iloc[i])
                except Exception:
                    desc = ''

                # round numeric values for display
                v_fmt = v_num
                try:
                    if v_num is not None:
                        v_fmt = round(float(v_num), 3)
                except Exception:
                    pass

                inputs.append({
                    'name': param_labels.get(p, p),
                    'raw_name': p,
                    'value': v_fmt,
                    'description': desc,
                    'group': _param_group(p),
                })
                i += 1

        # Sort inputs by group then name
        try:
            inputs.sort(key=lambda r: (r.get('group', 9), r.get('name', '')))
        except Exception:
            pass

        context['model_inputs'] = inputs
        # Keep legacy key for any other templates
        context['model_parameters'] = inputs
    except Exception:
        pass

    # (Optional) legacy summary stats table retained if available
    try:
        summary_df = db_manager.read_sql_pandas("SELECT * FROM macro_dsge_summary_stats")
        if hasattr(summary_df, 'empty') and not summary_df.empty:
            stats = []
            cols = []
            m = 0
            while m < len(summary_df.columns):
                c = summary_df.columns[m]
                if str(c).lower() not in ('stat', 'index', 'unnamed: 0'):
                    cols.append(c)
                m += 1
            if 'stat' in summary_df.columns:
                r = 0
                while r < len(summary_df):
                    stat_name = str(summary_df['stat'].iloc[r])
                    c2 = 0
                    while c2 < len(cols):
                        coln = cols[c2]
                        try:
                            val = float(summary_df[coln].iloc[r])
                            stats.append({'name': f"{coln} {stat_name}", 'value': val, 'description': ''})
                        except Exception:
                            pass
                        c2 += 1
                    r += 1
            context['key_statistics'] = stats
    except Exception:
        pass

    # Build concise highlights with pure loops
    highlights = []
    try:
        latest_vals = {}
        wanted = [
            ('pi', 'Inflation (YoY %)'),
            ('i', 'Policy Rate'),
            ('UNRATE', 'Unemployment'),
            ('x_gap_est', 'Output Gap (Kalman)'),
            ('rn_est', 'Natural Real Rate')
        ]

        for col, label in wanted:
            val = None
            try:
                if 'main_df' in locals() and (hasattr(main_df, 'columns') and col in main_df.columns):
                    idx = len(main_df[col]) - 1
                    while idx >= 0:
                        v = main_df[col].iloc[idx]
                        if v is not None and v == v:
                            val = v
                            break
                        idx -= 1
            except Exception:
                val = None

            if val is not None:
                try:
                    val = float(val)
                    val = round(val, 2)
                except Exception:
                    pass

            latest_vals[col] = val
            highlights.append({'label': label, 'value': latest_vals[col]})
    except Exception:
        highlights = []

    context['highlights'] = highlights

    # Latest model output snapshot from SQL
    try:
        latest_rows = []
        df_out = db_manager.read_sql_pandas("SELECT * FROM dsge_model_output ORDER BY created_at DESC LIMIT 1")
        if hasattr(df_out, 'empty') and not df_out.empty:
            rec = df_out.iloc[0]
            ordered = [
                'date', 'x_gap_est', 'rn_est', 'pi', 'i', 'unrate',
                'phi_pi', 'phi_x', 'rho_i', 'beta', 'kappa', 'sigma', 'ax', 'b_is',
                'rho_rn', 'pi_star', 'r_star', 'lam_okun',
                'sig_x', 'sig_rn', 'sig_pi', 'sig_i', 'sig_u', 'u_star'
            ]
            idx = 0
            while idx < len(ordered):
                k = ordered[idx]
                if k in rec.index:
                    v = rec[k]
                    vfmt = None
                    try:
                        if k == 'date':
                            vfmt = str(v)
                        else:
                            vnum = float(v)
                            vfmt = round(vnum, 3)
                    except Exception:
                        try:
                            vfmt = str(v)
                        except Exception:
                            vfmt = None

                    if k in ('x_gap_est', 'rn_est', 'pi', 'i', 'unrate'):
                        key = {'x_gap_est': 'x_gap_est', 'rn_est': 'rn_est', 'pi': 'pi', 'i': 'i', 'unrate': 'UNRATE'}[k]
                        label = series_labels.get(key, key)
                    else:
                        raw_key = 'b_IS' if k == 'b_is' else k
                        label = param_labels.get(raw_key, raw_key)
                    latest_rows.append({'key': k, 'label': label, 'value': vfmt})
                idx += 1
        context['latest_model_output'] = latest_rows
    except Exception:
        context['latest_model_output'] = []

    return render_template('macro_analysis.html', active_page='macro', **context)


@macro_bp.post('/run-macro-analysis')
def run_macro_analysis_endpoint():
    ok = False
    try:
        ok = schedule_macro_analysis(db_manager)
    except Exception:
        ok = False

    message = 'Macro analysis completed successfully' if ok else 'Macro analysis failed or produced no plots'

    return render_template(
        'analysis_complete.html',
        success=ok,
        message=message,
        date=datetime.now().strftime('%Y-%m-%d @ %H:%M'),
        active_page='macro',
    ) 