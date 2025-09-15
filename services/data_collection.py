from datetime import datetime

from extensions import db_manager

# NOTE: This module only provides thin wrappers; business logic remains in existing modules.


def collect_banxico_data():
    try:
        import data_processing.banxico_data_collection as mod
        if hasattr(mod, "collect_banxico_data"):
            return bool(mod.collect_banxico_data(db_manager))
    except Exception:
        return False
    return False


def collect_financial_data():
    try:
        import data_processing.financial_data_collection as mod
        if hasattr(mod, "collect_financial_data"):
            return bool(mod.collect_financial_data(db_manager))
    except Exception:
        return False
    return False


def collect_fred_data():
    try:
        import data_processing.fred_data_collection as mod
        if hasattr(mod, "collect_fred_data"):
            return bool(mod.collect_fred_data(db_manager))
    except Exception:
        return False
    return False


def collect_macro_data():
    try:
        from data_processing.macro_visualization import schedule_macro_analysis
        return bool(schedule_macro_analysis(db_manager))
    except Exception:
        return False


def run_data_collections():
    """Run all data collection tasks sequentially; returns a summary dict.

    TODO: Add logging and persistence as needed by the project.
    """
    start_time = datetime.now()

    results = {
        "banxico": {"status": "pending", "error": None, "duration": None},
        "financial": {"status": "pending", "error": None, "duration": None},
        "fred": {"status": "pending", "error": None, "duration": None},
        "macro": {"status": "pending", "error": None, "duration": None},
    }

    # Banxico
    try:
        t0 = datetime.now()
        ok = collect_banxico_data()
        results["banxico"]["status"] = "success" if ok else "failed"
        results["banxico"]["duration"] = datetime.now() - t0
    except Exception as e:
        results["banxico"]["status"] = "failed"
        results["banxico"]["error"] = str(e)

    # Financial
    try:
        t0 = datetime.now()
        ok = collect_financial_data()
        results["financial"]["status"] = "success" if ok else "failed"
        results["financial"]["duration"] = datetime.now() - t0
    except Exception as e:
        results["financial"]["status"] = "failed"
        results["financial"]["error"] = str(e)

    # FRED
    try:
        t0 = datetime.now()
        ok = collect_fred_data()
        results["fred"]["status"] = "success" if ok else "failed"
        results["fred"]["duration"] = datetime.now() - t0
    except Exception as e:
        results["fred"]["status"] = "failed"
        results["fred"]["error"] = str(e)

    # Macro
    try:
        t0 = datetime.now()
        ok = collect_macro_data()
        results["macro"]["status"] = "success" if ok else "failed"
        results["macro"]["duration"] = datetime.now() - t0
    except Exception as e:
        results["macro"]["status"] = "failed"
        results["macro"]["error"] = str(e)

    results["total_duration"] = datetime.now() - start_time
    return results


# Parallel orchestrator (keeps identical return format)
# Uses thread pool to run independent collectors concurrently

def run_data_collections_parallel(max_workers: int = 4):
    start_time = datetime.now()

    results = {
        "banxico": {"status": "pending", "error": None, "duration": None},
        "financial": {"status": "pending", "error": None, "duration": None},
        "fred": {"status": "pending", "error": None, "duration": None},
        "macro": {"status": "pending", "error": None, "duration": None},
    }

    tasks = {
        "banxico": collect_banxico_data,
        "financial": collect_financial_data,
        "fred": collect_fred_data,
        "macro": collect_macro_data,
    }

    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print("Starting parallel data collections...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for name in tasks:
                print(f"[submit] {name}...")
                future = executor.submit(lambda fn, key: (key, datetime.now(), fn()), tasks[name], name)
                future_map[future] = name
            for future in as_completed(future_map):
                key, t0, ok = future.result()
                results[key]["status"] = "success" if ok else "failed"
                results[key]["duration"] = datetime.now() - t0
                print(f"[done] {key}: {results[key]['status']} ({results[key]['duration']})")
    except Exception as e:
        # Catastrophic failure: mark all remaining as failed with error
        for name in results:
            if results[name]["status"] == "pending":
                results[name]["status"] = "failed"
                results[name]["error"] = str(e)

    results["total_duration"] = datetime.now() - start_time
    return results


# Smart orchestrator: prefer parallel, fallback to sequential for failures

def run_data_collections_smart(max_workers: int = 4):
    print("Running SMART data collections (parallel with fallback)...")
    # If DB unhealthy, avoid parallel to reduce contention
    try:
        if hasattr(db_manager, "health_check") and not db_manager.health_check():
            print("DB health check failed -> running sequential fallback")
            return run_data_collections()
    except Exception:
        pass

    res = run_data_collections_parallel(max_workers=max_workers)

    # Any failed items? retry sequentially
    failed = [k for k, v in res.items() if isinstance(v, dict) and v.get("status") == "failed"]
    if failed:
        print(f"Retrying failed tasks sequentially: {failed}")
        for name in failed:
            try:
                t0 = datetime.now()
                if name == "banxico":
                    ok = collect_banxico_data()
                elif name == "financial":
                    ok = collect_financial_data()
                elif name == "fred":
                    ok = collect_fred_data()
                elif name == "macro":
                    ok = collect_macro_data()
                else:
                    ok = False
                res[name]["status"] = "success" if ok else "failed"
                res[name]["duration"] = datetime.now() - t0
            except Exception as e:
                res[name]["status"] = "failed"
                res[name]["error"] = str(e)
    return res 