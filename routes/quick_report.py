from flask import Blueprint, request

from extensions import db_manager
from quick_report import generate_quick_report

quick_report_bp = Blueprint('quick_report', __name__, url_prefix='')


@quick_report_bp.get('/quick_report')
def quick_report_view():
    ticker = request.args.get('ticker', '').strip()
    if not ticker:
        return "Missing 'ticker' parameter", 400
    return generate_quick_report(ticker, db_manager) 