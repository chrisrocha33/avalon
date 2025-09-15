from apscheduler.triggers.cron import CronTrigger

from config import Config
from extensions import scheduler, db_manager
from services.data_collection import run_data_collections


def setup_enhanced_scheduler():
    """Setup scheduler jobs idempotently.

    Removes existing jobs and re-adds desired ones. Safe to call multiple times.
    """
    try:
        # Clear any existing jobs
        scheduler.remove_all_jobs()

        # Immediate one-off run after app starts
        try:
            scheduler.add_job(
                func=run_data_collections,
                trigger="date",
                id="initial_data_collection",
                name="Initial Data Collection on App Start",
            )
        except Exception:
            # If an existing job clashes, remove and re-add
            try:
                scheduler.remove_job("initial_data_collection")
            except Exception:
                pass
            scheduler.add_job(
                func=run_data_collections,
                trigger="date",
                id="initial_data_collection",
                name="Initial Data Collection on App Start",
            )

        # Daily scheduled runs
        for i, (hour, minute) in enumerate(Config.SCHEDULER_RUN_TIMES, 1):
            job_id = f"daily_data_collection_{i}"
            try:
                scheduler.remove_job(job_id)
            except Exception:
                pass
            scheduler.add_job(
                func=run_data_collections,
                trigger=CronTrigger(hour=hour, minute=minute),
                id=job_id,
                name=f"Daily Data Collection {i} at {hour:02d}:{minute:02d}",
            )

        if not scheduler.running:
            scheduler.start()
        return True

    except Exception:
        return False


def check_scheduler_status():
    """Return a simple status string for diagnostics."""
    if scheduler.running:
        jobs = scheduler.get_jobs()
        return f"Scheduler RUNNING with {len(jobs)} jobs"
    return "Scheduler STOPPED"


def cleanup_resources():
    """Cleanup connections and scheduler on app shutdown."""
    try:
        # Close DB manager connections
        try:
            db_manager.close()
        except Exception:
            pass

        # Shutdown scheduler
        if scheduler.running:
            scheduler.shutdown()
    except Exception:
        pass 