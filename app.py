import atexit

from flask import Flask

from config import Config
from extensions import scheduler, db_manager  # singletons
from routes import register_blueprints
from jobs.scheduler import (
    setup_enhanced_scheduler,
    check_scheduler_status,
    cleanup_resources,
)


def create_app() -> Flask:
    """Create and configure the Flask application.

    Minimal bootstrap: load config, register blueprints.
    """
    app = Flask(__name__)
    app.config.from_object(Config)

    # Register blueprints (idempotent; currently a no-op)
    register_blueprints(app)

    return app


app = create_app()


@atexit.register
def _on_exit():
    """Ensure resources are cleaned up on process exit."""
    cleanup_resources()


if __name__ == "__main__":
    # Optional: lightweight health check (non-fatal)
    try:
        db_manager.health_check()
    except Exception:
        pass

    # Setup scheduler jobs and start
    # TODO: Adjust timings in Config.SCHEDULER_RUN_TIMES for your deployment
    setup_enhanced_scheduler()

    # Optional status print
    try:
        print(check_scheduler_status())
    except Exception:
        pass

    # TODO: Adjust host/port and debug for your environment
    app.run(debug=Config.DEBUG, host="0.0.0.0", port=5000, use_reloader=False)
