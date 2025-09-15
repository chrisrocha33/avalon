-- Descriptive statistics table
CREATE TABLE descriptive_stats_data_render (
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

-- Hill estimator table
CREATE TABLE hill_estimator_data_render (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    tail_index_positive FLOAT,
    tail_index_negative FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- GARCH parameters table
CREATE TABLE garch_parameters_data_render (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    omega FLOAT,
    alpha1 FLOAT,
    beta1 FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Correlation matrix table
CREATE TABLE correlation_matrix_data_render (
    id SERIAL PRIMARY KEY,
    series1 VARCHAR(50),
    series2 VARCHAR(50),
    correlation_value FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- PCA results table
CREATE TABLE pca_results_data_render (
    id SERIAL PRIMARY KEY,
    pc_number INTEGER,
    explained_variance_ratio FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- PCA loadings table
CREATE TABLE pca_loadings_data_render (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    pc_number INTEGER,
    loading_value FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- VaR/CVaR table
CREATE TABLE var_cvar_data_render (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    alpha_level FLOAT,
    var_value FLOAT,
    cvar_value FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Beta analysis table
CREATE TABLE beta_analysis_data_render (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    alpha FLOAT,
    beta FLOAT,
    r_squared FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Variance decomposition table
CREATE TABLE variance_decomposition_data_render (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    total_variance FLOAT,
    systematic_variance FLOAT,
    idiosyncratic_variance FLOAT,
    r_squared FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- OLS regression results table
CREATE TABLE ols_regression_data_render (
    id SERIAL PRIMARY KEY,
    dependent_variable VARCHAR(50),
    html_summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Plot storage tables
CREATE TABLE correlation_heatmap_plot_rendered (
    id SERIAL PRIMARY KEY,
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE correlation_scatter_plot_rendered (
    id SERIAL PRIMARY KEY,
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE qq_plots_plot_rendered (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE rolling_moments_plot_rendered (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    moment_type VARCHAR(50),
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE acf_pacf_plot_rendered (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    plot_type VARCHAR(20),
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE garch_volatility_plot_rendered (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE periodogram_plot_rendered (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE regime_switching_plot_rendered (
    id SERIAL PRIMARY KEY,
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE clustering_dendrogram_plot_rendered (
    id SERIAL PRIMARY KEY,
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE network_graph_plot_rendered (
    id SERIAL PRIMARY KEY,
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE tail_dependence_plot_rendered (
    id SERIAL PRIMARY KEY,
    dependence_type VARCHAR(20),
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE rolling_beta_plot_rendered (
    id SERIAL PRIMARY KEY,
    series_name VARCHAR(50),
    plot_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);