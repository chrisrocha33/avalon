from datetime import datetime
from flask import Blueprint, render_template

from extensions import db_manager
import importlib.util
import os

sectors_bp = Blueprint('sectors', __name__, url_prefix='')


@sectors_bp.get('/sectors')
def sectors_analysis():
	date_str = datetime.now().strftime('%Y-%m-%d')
	
	# Defaults if data missing
	desc_stats_html = ''
	qqplots_b64 = []
	rolling_imgs_b64 = []
	hill_df_html = ''
	acf_pacf_imgs_b64 = []
	garch_df_html = ''
	garch_imgs_b64 = []
	periodogram_imgs_b64 = []
	corr_html = ''
	correlation_heatmap = ''
	corr_heatmap_b64_extra = ''
	rolling_corr_imgs_b64 = []
	dcc_img_b64 = ''
	ev_html = ''
	pca_cumvar_b64 = ''
	pca_loadings_html = ''
	cluster_dendro_b64 = ''
	network_img_b64 = ''
	varcvar_html = ''
	tail_lower_b64 = ''
	tail_upper_b64 = ''
	beta_html = ''
	rolling_beta_imgs_b64 = []
	vardec_html = ''
	ols_results = {}
	
	data_set_rows = 0
	data_set_columns = []
	data_start_date = ''
	data_end_date = ''
	
	try:
		# Pull descriptive stats table
		df_stats = db_manager.read_sql_pandas(
			"SELECT * FROM descriptive_stats_data_render"
		)
		if not df_stats.empty:
			data_set_columns = df_stats['series_name'].unique().tolist()
			data_set_rows = len(df_stats)
			# Filter to latest date if a date column exists; hide date column in display
			date_col = None
			for c in ('date', 'Date', 'timestamp', 'created_at'):
				if c in df_stats.columns:
					date_col = c
					break
			if date_col:
				# Compute dataset start/end dates for UI
				try:
					data_start_date = str(df_stats[date_col].min())
					data_end_date = str(df_stats[date_col].max())
				except Exception:
					pass
				try:
					latest_val = df_stats[date_col].max()
					df_latest = df_stats[df_stats[date_col] == latest_val].copy()
					df_latest.drop(columns=[date_col], inplace=True)
					desc_stats_html = df_latest.to_html(index=False)
				except Exception:
					desc_stats_html = df_stats.drop(columns=[date_col], errors='ignore').to_html(index=False)
			else:
				desc_stats_html = df_stats.to_html(index=False)
	except Exception:
		pass
	
	try:
		# Pull correlation matrix for HTML
		df_corr = db_manager.read_sql_pandas(
			"SELECT * FROM correlation_matrix_data_render"
		)
		if not df_corr.empty:
			# Pivot to matrix for display
			corr_pivot = df_corr.pivot(index='series1', columns='series2', values='correlation_value')
			corr_html = corr_pivot.to_html()
	except Exception:
		pass
	
	try:
		# PCA explained variance
		df_ev = db_manager.read_sql_pandas(
			"SELECT * FROM pca_results_data_render ORDER BY pc_number"
		)
		if not df_ev.empty:
			ev_html = df_ev.to_html(index=False)
		df_load = db_manager.read_sql_pandas(
			"SELECT * FROM pca_loadings_data_render ORDER BY pc_number, series_name"
		)
		if not df_load.empty:
			pca_loadings_html = df_load.to_html(index=False)
	except Exception:
		pass
	
	try:
		# VaR/CVaR
		df_var = db_manager.read_sql_pandas(
			"SELECT * FROM var_cvar_data_render ORDER BY series_name, alpha_level"
		)
		if not df_var.empty:
			varcvar_html = df_var.to_html(index=False)
	except Exception:
		pass
	
	try:
		# Beta
		df_beta = db_manager.read_sql_pandas(
			"SELECT * FROM beta_analysis_data_render ORDER BY series_name"
		)
		if not df_beta.empty:
			beta_html = df_beta.to_html(index=False)
	except Exception:
		pass
	
	try:
		# OLS results: dependent_variable, html_summary
		df_ols = db_manager.read_sql_pandas(
			"SELECT * FROM ols_regression_data_render"
		)
		if not df_ols.empty and 'dependent_variable' in df_ols.columns:
			for _, row in df_ols.iterrows():
				dep = row.get('dependent_variable')
				html_summary = row.get('html_summary', '')
				if dep:
					ols_results[str(dep)] = html_summary
	except Exception:
		pass
	
	# Load base64 images from DB (created during plot saving)
	vis_map = {}
	try:
		df_vis = db_manager.read_sql_pandas("SELECT title, data, created_at FROM sectors_visuals")
		if not df_vis.empty:
			for _, r in df_vis.iterrows():
				t = r.get('title')
				d = r.get('data')
				if isinstance(t, str) and isinstance(d, str):
					vis_map[t] = d
	except Exception:
		vis_map = {}
	
	# Collect composite grid pages (rows=sectors, cols=analyses)
	sector_grid_pages = []
	try:
		# Parse run ids present in grid titles
		run_to_items = {}
		for k, v in vis_map.items():
			if isinstance(k, str) and k.startswith('plot=grid_core'):
				# extract run id if present: ...|run=YYYY...|
				parts = k.split('|')
				run_part = next((p for p in parts if p.startswith('run=')), '')
				run_id = run_part.split('=')[1] if '=' in run_part else ''
				if run_id not in run_to_items:
					run_to_items[run_id] = []
				try:
					rows_part = next((p for p in parts if p.startswith('rows=')), '')
					cols_part = next((p for p in parts if p.startswith('cols=')), '')
					r_rng = rows_part.split('=')[1] if '=' in rows_part else '0-0'
					c_rng = cols_part.split('=')[1] if '=' in cols_part else '0-0'
					r_start = int(r_rng.split('-')[0])
					c_start = int(c_rng.split('-')[0])
				except Exception:
					r_start, c_start = 0, 0
				run_to_items[run_id].append((r_start, c_start, v))
		
		selected_items = []
		if run_to_items:
			# choose latest non-empty run_id by numeric/lex order; if only empty keys, keep empty
			candidate_ids = [rid for rid in run_to_items.keys() if rid]
			if candidate_ids:
				latest_run = sorted(candidate_ids)[-1]
				selected_items = run_to_items.get(latest_run, [])
			else:
				# all entries lack run id; fall back to aggregating all (legacy behavior)
				for items in run_to_items.values():
					selected_items.extend(items)
			selected_items.sort(key=lambda t: (t[0], t[1]))
			sector_grid_pages = [x[2] for x in selected_items]
		else:
			sector_grid_pages = []
	except Exception:
		sector_grid_pages = []
	
	# Map singles using new deterministic titles
	correlation_heatmap = vis_map.get('plot=correlation_heatmap', '')
	corr_heatmap_b64_extra = vis_map.get('plot=correlation_heatmap', '')
	pca_cumvar_b64 = vis_map.get('plot=pca_cumvar', '')
	cluster_dendro_b64 = vis_map.get('plot=clustering_dendrogram', '')
	network_img_b64 = vis_map.get('plot=network_graph', '')
	tail_lower_b64 = vis_map.get('plot=tail_dependence_lower', '')
	tail_upper_b64 = vis_map.get('plot=tail_dependence_upper', '')
	# Regime switching (market-specific title): pick any available
	regime_img_b64 = ''
	for k, v in vis_map.items():
		if isinstance(k, str) and k.startswith('plot=regime_switching'):
			regime_img_b64 = v
			break
	correlation_scatter = vis_map.get('plot=correlation_scatter', '')

	# If composite grid pages exist, prefer them and skip assembling per-sector lists
	if sector_grid_pages:
		return render_template(
			'sectors_analysis.html',
			date=date_str,
			data_start_date=data_start_date,
			data_end_date=data_end_date,
			desc_stats_html=desc_stats_html,
			correlation_heatmap=correlation_heatmap,
			ols_results=ols_results,
			sector_grid_pages=sector_grid_pages,
			active_page='sectors',
		)
	
	# Derive sector list
	sector_list = []
	try:
		base_sectors = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "SPY"]
		# Prefer from stats columns
		for s in data_set_columns:
			if isinstance(s, str) and s in base_sectors:
				sector_list.append(s)
		# Fallback: parse from vis_map titles
		if not sector_list:
			for t in vis_map.keys():
				try:
					if isinstance(t, str) and t.startswith('plot=qq|sector='):
						sec = t.split('plot=qq|sector=')[-1]
						if sec and sec not in sector_list:
							sector_list.append(sec)
				except Exception:
					pass
	except Exception:
		sector_list = []
	
	# Build lists from sector-based keys
	for sec in sector_list:
		# QQ plots
		v = vis_map.get(f'plot=qq|sector={sec}')
		if v:
			qqplots_b64.append(v)
		# ACF/PACF
		v1 = vis_map.get(f'plot=acf|sector={sec}')
		v2 = vis_map.get(f'plot=pacf|sector={sec}')
		if v1:
			acf_pacf_imgs_b64.append(v1)
		if v2:
			acf_pacf_imgs_b64.append(v2)
		# Periodogram
		v3 = vis_map.get(f'plot=periodogram|sector={sec}')
		if v3:
			periodogram_imgs_b64.append(v3)
		# Rolling stats
		for fn in (f'plot=rolling_mean|sector={sec}', f'plot=rolling_std|sector={sec}', f'plot=rolling_skew|sector={sec}'):
			rv = vis_map.get(fn)
			if rv:
				rolling_imgs_b64.append(rv)
		# GARCH
		gv = vis_map.get(f'plot=garch_vol|sector={sec}')
		if gv:
			garch_imgs_b64.append(gv)
		# Rolling beta (market-specific title): pick any available for this sector
		rb = ''
		for k, v_rb in vis_map.items():
			if isinstance(k, str) and k.startswith(f'plot=rolling_beta|sector={sec}'):
				rb = v_rb
				break
		if rb:
			rolling_beta_imgs_b64.append(rb)
	
	return render_template(
		'sectors_analysis.html',
		date=date_str,
		data_start_date=data_start_date,
		data_end_date=data_end_date,
		desc_stats_html=desc_stats_html,
		correlation_heatmap=correlation_heatmap,
		ols_results=ols_results,
		sector_grid_pages=sector_grid_pages,
		active_page='sectors',
	) 


@sectors_bp.post('/run-analysis')
def run_sectors_analysis_endpoint():
	ok = False
	try:
		# Deterministic import of the edited sectors_analysis.py by file path
		base_dir = os.path.dirname(os.path.dirname(__file__))
		target_path = os.path.join(base_dir, 'data_processing', 'sectors_analysis.py')
		spec = importlib.util.spec_from_file_location('sectors_analysis', target_path)
		mod = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(mod)
		ok = bool(mod.run_full_analysis(db_manager))
	except Exception:
		ok = False
	
	message = 'Sector analysis completed successfully' if ok else 'Sector analysis failed or produced no outputs'
	return render_template(
		'analysis_complete.html',
		success=ok,
		message=message,
		date=datetime.now().strftime('%Y-%m-%d @ %H:%M'),
		active_page='sectors',
	) 