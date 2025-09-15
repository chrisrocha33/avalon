from datetime import datetime
from flask import Blueprint, render_template, request, jsonify

# Import only from extensions per constraints
from extensions import db_manager

home_bp = Blueprint('home', __name__)


@home_bp.before_app_request
def log_request_info():
    try:
        method = request.method
        path = request.path
    except Exception:
        pass


@home_bp.get("/")
def index():
    # Minimal context expected by template
    return render_template("home.html", title="Econ/Fin Dashboard", active_page='home')


@home_bp.get('/api/last-updates')
def api_last_updates():
    sectors_ts = ""
    macro_ts = ""
    try:
        # Sectors latest
        try:
            rows = db_manager.execute_query(
                "SELECT MAX(created_at) FROM sectors_visuals",
                params={},
                fetch=True
            )
            if rows and rows[0][0] is not None:
                try:
                    sectors_ts = rows[0][0].isoformat()
                except Exception:
                    sectors_ts = str(rows[0][0])
        except Exception:
            sectors_ts = ""

        # Macro latest
        try:
            rows2 = db_manager.execute_query(
                "SELECT MAX(created_at) FROM macro_visuals",
                params={},
                fetch=True
            )
            if rows2 and rows2[0][0] is not None:
                try:
                    macro_ts = rows2[0][0].isoformat()
                except Exception:
                    macro_ts = str(rows2[0][0])
        except Exception:
            macro_ts = ""
    except Exception:
        sectors_ts = sectors_ts or ""
        macro_ts = macro_ts or ""

    return jsonify({
        "sectors_latest_at": sectors_ts or "",
        "macro_latest_at": macro_ts or ""
    }) 