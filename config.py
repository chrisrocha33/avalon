import os


class Config:
    """Minimal application configuration.

    TODO: Adjust DATABASE connection_string, pool sizes, and scheduler timings
    to match the project deployment environment.
    """

    DEBUG = bool(int(os.getenv("FLASK_DEBUG", "1")))
    TESTING = False

    # Request limits (prevents very large payloads / URLs)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    MAX_CONTENT_PATH = 2048

    # Database configuration for DatabaseManager
    DATABASE = {
        # TODO: Replace with the correct DSN for your environment
        "connection_string": os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://avalon_user:bon.jovi33@127.0.0.1:5432/avalon",
        ),
        "pool_size": int(os.getenv("DB_POOL_SIZE", "20")),
        "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "30")),
        "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", "3600")),
        "pool_pre_ping": True,
        "application_name": os.getenv("DB_APP_NAME", "avalon_dashboard"),
    }

    # FRED API Configuration
    FRED_API_KEY = os.getenv("FRED_API_KEY", "dc6a021796123eddb26c049a7bccd312")

    # Scheduler timings (24h clock)
    # TODO: Adjust schedule to your desired run times
    SCHEDULER_RUN_TIMES = [
        (6, 0),   # 06:00
        (12, 0),  # 12:00
        (18, 0),  # 18:00
        (0, 0),   # 00:00
    ]

    # Centralized Matplotlib styling (bright colors, pitch-black background, grid on)
    MATPLOTLIB_CONFIG = {
        'style': 'default',
        'figure_size': (12, 8),
        'font_size': 10,
        'dpi': 100,
        'lines_linewidth': 2.0,
        'axes_grid': True,
        'grid_alpha': 0.4,
        'grid_color': '#333333',
        'grid_linestyle': '--',
        'background_color': '#000000',  # pitch black
        'foreground_color': '#FFFFFF',  # white text/axes
        'palette': [
            '#00FFFF',  # Aqua
            '#FF00FF',  # Magenta
            '#FFFF00',  # Yellow
            '#00FF00',  # Lime
            '#FF4500',  # OrangeRed
            '#1E90FF',  # DodgerBlue
            '#FFD700',  # Gold
            '#7CFC00',  # LawnGreen
            '#FF1493',  # DeepPink
            '#00CED1',  # DarkTurquoise
        ],
    } 