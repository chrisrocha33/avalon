# fred_catalog_builder.py
# Build a fast, browseable table of FRED series IDs with metadata + category structure,
# then store it into SQL via SQLAlchemy. Adds:
# - Threaded category scans
# - Global token-bucket rate limiter (stay under RPM)
# - Keep-alive Session + retries with backoff and Retry-After
# - Stable pagination (order_by=series_id)
# - Optional incremental mode via series/updates
#
# Requirements:
#   pip install requests pandas sqlalchemy python-dateutil
#
# Notes:
# - Set FRED_API_KEY env var or leave the fallback below.
# - Adjust WORKERS and MAX_RPM as needed; 16/100 is safe.
# - Provide a configured SQLAlchemy engine object (below or injected).

import os
import time
import json
import threading
import pandas as pd
import requests
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import create_engine

# --------------------------
# CONFIG
# --------------------------
API_KEY = os.getenv("FRED_API_KEY", "dc6a021796123eddb26c049a7bccd312").strip()
BASE = "https://api.stlouisfed.org/fred"
FILE_TYPE = "json"

# Performance knobs
WORKERS = 16              # parallel category workers (8–32 typical)
MAX_RPM = 100             # keep headroom under ~120 req/min
PAGE_LIMIT = 1000         # FRED max for many endpoints
SLEEP_JITTER = 0.02       # small jitter to desynchronize threads
INCREMENTAL_DAYS = None   # e.g. 7 for recent updates only; None for full crawl

# Optional local engine (replace with your own if you prefer to inject)
server = 'localhost'
port = '5432'
database = 'avalon'
username = 'admin'
password = 'password!'
conn_str = f'postgresql+psycopg2://{username}:{password}@{server}:{port}/{database}'
engine = create_engine(conn_str, future=True)

# --------------------------
# HTTP SESSION WITH RETRIES
# --------------------------
session = requests.Session()
session.headers.update({"User-Agent": "fred-catalog-builder-fast/1.1"})
retries = Retry(
    total=6,
    connect=6,
    read=6,
    backoff_factor=1.25,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET",)
)
adapter = HTTPAdapter(max_retries=retries, pool_connections=WORKERS * 2, pool_maxsize=WORKERS * 4)
session.mount("https://", adapter)

# --------------------------
# GLOBAL TOKEN-BUCKET RATE LIMITER
# --------------------------
tokens_lock = threading.Lock()
tokens = MAX_RPM
last_refill = time.monotonic()
rate_per_sec = MAX_RPM / 60.0

def acquire_token():
    global tokens, last_refill
    while True:
        with tokens_lock:
            now = time.monotonic()
            to_add = (now - last_refill) * rate_per_sec
            if to_add >= 1.0:
                tokens = min(MAX_RPM, tokens + int(to_add))
                last_refill = now
            if tokens > 0:
                tokens -= 1
                break
        time.sleep(0.01)

# --------------------------
# GET WRAPPER
# --------------------------
def fred_get(path, params):
    acquire_token()
    p = dict(params or {})
    p["api_key"] = API_KEY
    p["file_type"] = FILE_TYPE
    if SLEEP_JITTER:
        time.sleep(SLEEP_JITTER)
    r = session.get(f"{BASE}/{path}", params=p, timeout=60)
    if r.status_code == 429:
        ra = r.headers.get("Retry-After")
        try:
            time.sleep(float(ra)) if ra else time.sleep(2.0)
        except Exception:
            time.sleep(2.0)
        r = session.get(f"{BASE}/{path}", params=p, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r.json()

# --------------------------
# 1) COLLECT FULL CATEGORY TREE (BFS from root category_id=0)
# --------------------------
print(f"{'='*80}")
print("STEP 1: COLLECTING FULL CATEGORY TREE")
print(f"{'='*80}")
print("Starting BFS traversal from root category (ID: 0)...")

categories = {}           # category_id -> {category_id, name, parent_id}
parent_map = {}           # category_id -> parent_id
queue = [0]
seen = set()
categories_processed = 0

while queue:
    cid = queue.pop(0)
    if cid in seen:
        continue

    categories_processed += 1
    print(f"Processing category {cid} ({categories_processed} processed, {len(queue)} in queue)...")

    meta = fred_get("category", {"category_id": cid})
    cat = meta.get("categories", [{}])[0] if meta.get("categories") else {}
    cat_name_str = cat.get("name", "Unknown")
    print(f"  ✓ Category: {cat_name_str}")

    categories[cid] = {
        "category_id": cid,
        "name": cat.get("name"),
        "parent_id": cat.get("parent_id")
    }
    parent_map[cid] = cat.get("parent_id")

    kids = fred_get("category/children", {"category_id": cid})
    kids_list = kids.get("categories", []) or []
    children_count = 0

    for k in kids_list:
        kid_id = k.get("id")
        if kid_id is None:
            continue
        if kid_id not in categories:
            categories[kid_id] = {
                "category_id": kid_id,
                "name": k.get("name"),
                "parent_id": cid
            }
            parent_map[kid_id] = cid
        if kid_id not in seen:
            queue.append(kid_id)
            children_count += 1

    print(f"  ✓ Found {children_count} new children, added to queue" if children_count > 0 else "  ✓ No new children found")
    seen.add(cid)

# Category name map for later
cat_name = {k: v.get("name") for k, v in categories.items()}

print(f"\n✓ Category tree collection complete!")
print(f"Total categories discovered: {len(categories)}")
print(f"Categories processed: {categories_processed}")

# --------------------------
# SHARED STRUCTURES FOR SERIES
# --------------------------
series_meta = {}          # series_id -> metadata dict (first time seen)
series_to_cats = {}       # series_id -> set(category_ids)
meta_lock = threading.Lock()
map_lock  = threading.Lock()

# --------------------------
# CATEGORY WORKER (parallel)
# --------------------------
def fetch_category_series(category_id):
    got = 0
    offset = 0
    page_count = 0
    while True:
        page_count += 1
        data = fred_get(
            "category/series",
            {
                "category_id": category_id,
                "limit": PAGE_LIMIT,
                "offset": offset,
                "order_by": "series_id",
                "sort_order": "asc"
            }
        )
        sers = data.get("seriess", []) or []
        if not sers:
            break

        for s in sers:
            sid = s.get("id")
            if not sid:
                continue
            if sid not in series_meta:
                with meta_lock:
                    if sid not in series_meta:
                        series_meta[sid] = {
                            "series_id": sid,
                            "title": s.get("title"),
                            "frequency": s.get("frequency"),
                            "frequency_short": s.get("frequency_short"),
                            "units": s.get("units"),
                            "units_short": s.get("units_short"),
                            "seasonal_adjustment": s.get("seasonal_adjustment"),
                            "seasonal_adjustment_short": s.get("seasonal_adjustment_short"),
                            "last_updated": s.get("last_updated"),
                            "observation_start": s.get("observation_start"),
                            "observation_end": s.get("observation_end"),
                            "popularity": s.get("popularity"),
                            "notes": s.get("notes"),
                            "copyright": s.get("copyright")
                        }
            with map_lock:
                if sid not in series_to_cats:
                    series_to_cats[sid] = set()
                series_to_cats[sid].add(category_id)

        got += len(sers)
        if len(sers) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
    return category_id, got, page_count

# --------------------------
# 2) TRAVERSE CATEGORIES (FULL OR INCREMENTAL)
# --------------------------
print(f"\n{'='*80}")
print("STEP 2: GATHERING SERIES METADATA")
print(f"{'='*80}")

all_cat_ids = list(categories.keys())
if INCREMENTAL_DAYS is None:
    print(f"Starting parallel scan of {len(all_cat_ids)} categories with {WORKERS} workers...")
    done = 0
    total_found = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(fetch_category_series, cid) for cid in all_cat_ids]
        for f in as_completed(futures):
            cid, got, pages = f.result()
            done += 1
            total_found += got
            if done % 25 == 0 or done == len(all_cat_ids):
                print(f"[progress] categories scanned: {done}/{len(all_cat_ids)}, series pages fetched (latest): {pages}, total series refs so far: {total_found:,}")
else:
    # Incremental: fetch updated series in a time window, then map their categories
    end = datetime.utcnow()
    start = end - timedelta(days=int(INCREMENTAL_DAYS))
    end_str = end.strftime("%Y%m%d%H%M")
    start_str = start.strftime("%Y%m%d%H%M")
    print(f"Incremental mode: updates from {start_str} to {end_str}")

    updated = []
    offset = 0
    while True:
        d = fred_get("series/updates", {"start_time": start_str, "end_time": end_str, "limit": PAGE_LIMIT, "offset": offset})
        sers = d.get("seriess", []) or []
        if not sers:
            break
        updated.extend(sers)
        if len(sers) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
    print(f"Updated series count: {len(updated):,}")

    for s in updated:
        sid = s.get("id")
        if not sid:
            continue
        info = fred_get("series", {"series_id": sid})
        arr = info.get("seriess", [])
        if arr:
            s0 = arr[0]
            with meta_lock:
                series_meta[sid] = {
                    "series_id": sid,
                    "title": s0.get("title"),
                    "frequency": s0.get("frequency"),
                    "frequency_short": s0.get("frequency_short"),
                    "units": s0.get("units"),
                    "units_short": s0.get("units_short"),
                    "seasonal_adjustment": s0.get("seasonal_adjustment"),
                    "seasonal_adjustment_short": s0.get("seasonal_adjustment_short"),
                    "last_updated": s0.get("last_updated"),
                    "observation_start": s0.get("observation_start"),
                    "observation_end": s0.get("observation_end"),
                    "popularity": s0.get("popularity"),
                    "notes": s0.get("notes"),
                    "copyright": s0.get("copyright")
                }
        sc = fred_get("series/categories", {"series_id": sid})
        cats = sc.get("categories", []) or []
        with map_lock:
            if sid not in series_to_cats:
                series_to_cats[sid] = set()
            for c in cats:
                cid = c.get("id")
                if cid is not None:
                    series_to_cats[sid].add(cid)

# --------------------------
# 3) BUILD BROWSEABLE DATAFRAME
# --------------------------
print(f"\n{'='*80}")
print("STEP 3: BUILDING BROWSEABLE DATAFRAME")
print(f"{'='*80}")
print(f"Processing {len(series_meta)} unique series into dataframe...")

rows = []
i = 0
for sid, meta in series_meta.items():
    cids = sorted(list(series_to_cats.get(sid, [])))
    cnames = [cat_name.get(c) for c in cids]
    rec = dict(meta)
    rec["category_ids"] = json.dumps(cids, ensure_ascii=False)
    rec["category_names"] = json.dumps(cnames, ensure_ascii=False)
    rows.append(rec)
    i += 1
    if i % 10000 == 0:
        print(f"  Processed {i}/{len(series_meta)} series ({(i/len(series_meta)*100):.1f}%)")

df_catalog = pd.DataFrame(rows)
if not df_catalog.empty:
    df_catalog = df_catalog.sort_values(["popularity", "series_id"], ascending=[False, True], kind="stable")

total_series_ids = df_catalog["series_id"].nunique() if not df_catalog.empty else 0
print(f"✓ Total unique FRED series discovered: {total_series_ids:,}")

# --------------------------
# 4) REFERENCE TABLES
# --------------------------
print(f"\n{'='*80}")
print("STEP 4: CREATING REFERENCE TABLES")
print(f"{'='*80}")

df_categories = pd.DataFrame.from_dict(categories, orient="index").sort_values("category_id").reset_index(drop=True)

m2m_rows = []
for sid, cset in series_to_cats.items():
    for cid in cset:
        m2m_rows.append({"series_id": sid, "category_id": cid, "category_name": cat_name.get(cid)})
df_series_categories = pd.DataFrame(m2m_rows)

print(f"Tables ready -> catalog: {df_catalog.shape}, categories: {df_categories.shape}, map: {df_series_categories.shape}")

# --------------------------
# 5) WRITE TO DATABASE
# --------------------------
print(f"\n{'='*80}")
print("STEP 5: WRITING TO DATABASE")
print(f"{'='*80}")

try:
    engine  # noqa: F821
    print("✓ Database engine found")
except NameError:
    raise RuntimeError("Please define a SQLAlchemy `engine` before running the to_sql() calls.")

df_catalog.to_sql("fred_series_catalog", engine, if_exists="replace", index=False)
print(f"✓ fred_series_catalog written ({df_catalog.shape[0]} rows)")
df_categories.to_sql("fred_categories", engine, if_exists="replace", index=False)
print(f"✓ fred_categories written ({df_categories.shape[0]} rows)")
df_series_categories.to_sql("fred_series_categories", engine, if_exists="replace", index=False)
print(f"✓ fred_series_categories written ({df_series_categories.shape[0]} rows)")

print(f"\n{'='*80}")
print("FRED CATALOG BUILD COMPLETE!")
print(f"{'='*80}")
print(f"📊 SUMMARY STATISTICS:")
print(f"   • Categories discovered: {len(categories):,}")
print(f"   • Unique series found: {total_series_ids:,}")
print(f"   • Series-category mappings: {len(m2m_rows):,}")
print(f"   • Database tables created: 3")
print(f"{'='*80}")
print("Tables written: fred_series_catalog, fred_categories, fred_series_categories.")
