from apscheduler.schedulers.background import BackgroundScheduler

from config import Config
from database import DatabaseManager

# Centralized singletons
scheduler = BackgroundScheduler()

db_manager = DatabaseManager(Config.DATABASE)


__all__ = ["scheduler", "db_manager"] 