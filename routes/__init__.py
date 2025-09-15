from flask import Flask

# Import blueprints locally to avoid circulars at module import time

def register_blueprints(app: Flask) -> None:
    """Register all app blueprints in order. Safe to call multiple times.

    Order:
    - home
    - sectors
    - macro
    - yields
    - quick_report
    - errors (last)
    """
    # Helper to idempotently register
    def _register(bp):
        name = bp.name
        if name not in app.blueprints:
            app.register_blueprint(bp)

    # Import inside function
    from .home import home_bp
    from .sectors import sectors_bp
    from .macro import macro_bp
    from .yields import yields_bp
    from .quick_report import quick_report_bp
    from .errors import errors_bp

    _register(home_bp)
    _register(sectors_bp)
    _register(macro_bp)
    _register(yields_bp)
    _register(quick_report_bp)
    _register(errors_bp) 