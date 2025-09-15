-- Avalon database initial schema
-- Creates stable tables referenced across the application

-- ==========================
-- Sector analysis result tables
-- Source: DB/Sector Analysis Tables Creation.sql
-- Converted to IF NOT EXISTS to be idempotent
-- ==========================

CREATE TABLE IF NOT EXISTS descriptive_stats_data_render (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    mean_value FLOAT,
    std_value FLOAT,
    skewness FLOAT,
    kurtosis FLOAT,
    adf_p_value FLOAT,
    kpss_p_value FLOAT,
    jarque_bera_p_value FLOAT,
    shapiro_p_value FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hill_estimator_data_render (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    tail_index_positive FLOAT,
    tail_index_negative FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS garch_parameters_data_render (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    omega FLOAT,
    alpha1 FLOAT,
    beta1 FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS correlation_matrix_data_render (
    id SERIAL PRIMARY KEY,
    series1 VARCHAR(50),
    series2 VARCHAR(50),
    correlation_value FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pca_results_data_render (
    id SERIAL PRIMARY KEY,
    pc_number INTEGER,
    explained_variance_ratio FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pca_loadings_data_render (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    pc_number INTEGER,
    loading_value FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS var_cvar_data_render (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    alpha_level FLOAT,
    var_value FLOAT,
    cvar_value FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS beta_analysis_data_render (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    alpha FLOAT,
    beta FLOAT,
    r_squared FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS variance_decomposition_data_render (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    total_variance FLOAT,
    systematic_variance FLOAT,
    idiosyncratic_variance FLOAT,
    r_squared FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ols_regression_data_render (
    id SERIAL PRIMARY KEY,
    dependent_variable VARCHAR(50),
    html_summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS correlation_heatmap_plot_rendered (
    id SERIAL PRIMARY KEY,
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS correlation_scatter_plot_rendered (
    id SERIAL PRIMARY KEY,
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS qq_plots_plot_rendered (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rolling_moments_plot_rendered (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    moment_type VARCHAR(50),
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS acf_pacf_plot_rendered (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    plot_type VARCHAR(20),
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS garch_volatility_plot_rendered (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS periodogram_plot_rendered (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS regime_switching_plot_rendered (
    id SERIAL PRIMARY KEY,
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS clustering_dendrogram_plot_rendered (
    id SERIAL PRIMARY KEY,
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS network_graph_plot_rendered (
    id SERIAL PRIMARY KEY,
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tail_dependence_plot_rendered (
    id SERIAL PRIMARY KEY,
    dependence_type VARCHAR(20),
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rolling_beta_plot_rendered (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ==========================
-- Macro analysis tables
-- Source: data_processing/macro_analysis.py, data_processing/macro_visualization.py
-- ==========================

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
);

CREATE TABLE IF NOT EXISTS macro_dsge_parameters (
    id SERIAL PRIMARY KEY,
    parameter TEXT,
    value DOUBLE PRECISION,
    description TEXT
);

CREATE TABLE IF NOT EXISTS macro_dsge_summary_stats (
    id SERIAL PRIMARY KEY,
    stat TEXT,
    "i" DOUBLE PRECISION,
    "pi" DOUBLE PRECISION,
    "UNRATE" DOUBLE PRECISION,
    "SPREAD" DOUBLE PRECISION,
    "x_gap_est" DOUBLE PRECISION,
    "rn_est" DOUBLE PRECISION
);

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
);

CREATE TABLE IF NOT EXISTS macro_visuals (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    image_base64 TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ==========================
-- Sectors visuals table
-- Source: data_processing/sectors_analysis.py
-- ==========================

CREATE TABLE IF NOT EXISTS sectors_visuals (
    date TIMESTAMP,
    title TEXT PRIMARY KEY,
    data TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);


-- ==========================
-- Market and analytics tables used by quick_report.py
-- ==========================

CREATE TABLE IF NOT EXISTS stock_data (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(16) NOT NULL,
    date DATE NOT NULL,
    open_price DOUBLE PRECISION,
    high_price DOUBLE PRECISION,
    low_price DOUBLE PRECISION,
    close_price DOUBLE PRECISION,
    volume BIGINT,
    adj_close DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (ticker, date)
);

CREATE TABLE IF NOT EXISTS technical_indicators (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(16) NOT NULL,
    date TIMESTAMP NOT NULL,
    close_price DOUBLE PRECISION,
    rsi DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    bollinger_upper DOUBLE PRECISION,
    bollinger_lower DOUBLE PRECISION,
    bollinger_middle DOUBLE PRECISION,
    williams_r DOUBLE PRECISION,
    stoch_k DOUBLE PRECISION,
    stoch_d DOUBLE PRECISION,
    atr DOUBLE PRECISION,
    cci DOUBLE PRECISION,
    volume BIGINT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (ticker, date)
);

CREATE TABLE IF NOT EXISTS risk_analysis (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(16) NOT NULL,
    date TIMESTAMP NOT NULL,
    var_95_1d DOUBLE PRECISION,
    var_95_5d DOUBLE PRECISION,
    var_95_20d DOUBLE PRECISION,
    cvar_95_1d DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    sortino_ratio DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (ticker, date)
);

CREATE TABLE IF NOT EXISTS sentiment_portfolio_analysis (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(16) NOT NULL,
    date TIMESTAMP NOT NULL,
    news_sentiment DOUBLE PRECISION,
    social_sentiment DOUBLE PRECISION,
    analyst_rating DOUBLE PRECISION,
    price_target DOUBLE PRECISION,
    confidence_score DOUBLE PRECISION,
    position_size DOUBLE PRECISION,
    entry_strategy TEXT,
    stop_loss DOUBLE PRECISION,
    take_profit DOUBLE PRECISION,
    holding_period TEXT,
    risk_level TEXT,
    sector_allocation TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (ticker, date)
);

CREATE TABLE IF NOT EXISTS fundamental_data (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(16) NOT NULL,
    statement_type TEXT NOT NULL,
    metric TEXT NOT NULL,
    date DATE NOT NULL,
    value DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (ticker, statement_type, metric, date)
);

CREATE TABLE IF NOT EXISTS options_data (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(16) NOT NULL,
    option_type TEXT NOT NULL,
    strike_price DOUBLE PRECISION,
    last_price DOUBLE PRECISION,
    bid_price DOUBLE PRECISION,
    ask_price DOUBLE PRECISION,
    volume BIGINT,
    open_interest BIGINT,
    implied_volatility DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW()
);


-- ==========================
-- FRED catalog tables (used by data_processing/fred_data_catalog.py and quick queries)
-- ==========================

CREATE TABLE IF NOT EXISTS fred_series_catalog (
    series_id TEXT PRIMARY KEY,
    title TEXT,
    last_updated TIMESTAMP,
    observation_start DATE,
    frequency_short TEXT,
    category_names TEXT
);

CREATE TABLE IF NOT EXISTS fred_categories (
    category_id INTEGER PRIMARY KEY,
    name TEXT,
    parent_id INTEGER
);

CREATE TABLE IF NOT EXISTS fred_series_categories (
    series_id TEXT,
    category_id INTEGER,
    PRIMARY KEY (series_id, category_id)
);


-- ==========================
-- ML SPX tables (schemas are dynamic; seed minimal structures)
-- ==========================

CREATE TABLE IF NOT EXISTS ml_spx_data (
    "index" TIMESTAMP PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS ml_spx_regional_indices (
    "index" TIMESTAMP PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS ml_spx_futures_indices (
    "index" TIMESTAMP PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS ml_spx_pca_indices (
    "index" TIMESTAMP PRIMARY KEY
);


