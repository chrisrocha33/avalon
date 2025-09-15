"""
Banxico Data Collection Script
==============================

This script collects economic data from Banco de México's (Banxico) 
Economic Information System (SIE) REST API using a loop-based approach.

The script:
1. Loads series identifiers from a JSON catalog
2. Fetches metadata for each series (title, frequency, date range)
3. Retrieves historical observations for each series
4. Aligns data to a common frequency and creates a master DataFrame
5. Stores results to a PostgreSQL database using the database manager

API Endpoints:
- Metadata: /series/{id} - Returns series information
- Data: /series/{id}/datos/{start}/{end} - Returns historical observations

Note: Requires a valid API token from Banxico. The API returns dates in 
DD/MM/YYYY format and uses "N/E" for missing values.
"""

import datetime as _dt
import json as _json
import sys as _sys
from typing import Dict, List, Optional

import pandas as _pd
import requests as _requests


# Configuration
# Replace with your actual Banxico SIE API token
token = "cc5ef66447da0f7fb79726c59a5b85072bde4f914b40bf1483e3e8e70f322c34"


def collect_banxico_data(db_manager=None):
    """
    Collect Banxico data and store to database using database manager.
    
    Args:
        db_manager: Database manager instance for database operations
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\n{'='*60}")
        print("🚀 STARTING BANXICO DATA COLLECTION")
        print(f"{'='*60}")
        
        if db_manager:
            print(f"🔗 Using database manager: {type(db_manager).__name__}")
            if db_manager.health_check():
                print("✅ Database connection healthy")
            else:
                print("⚠️ Database connection unhealthy")
                return False
        else:
            print("ℹ️ No database manager provided - running in standalone mode")
        
        # Run the data collection
        return _run_banxico_collection(db_manager)
        
    except Exception as e:
        print(f"❌ Error in collect_banxico_data: {str(e)}")
        return False


def _run_banxico_collection(db_manager=None):
    """Internal function to run the Banxico data collection process."""
    
    # Helper Functions
    def parse_banxico_date(date_str):
        """Convert Banxico date format (DD/MM/YYYY) to datetime.date"""
        if not date_str:
            return None
        try:
            return _dt.datetime.strptime(date_str, "%d/%m/%Y").date()
        except ValueError:
            try:
                # Fallback to ISO format if available
                return _dt.datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                return None


    # Frequency mapping for ranking and pandas offset conversion
    frequency_map = {
        "Diaria": ("d", 1),      # Daily
        "Semanal": ("w", 2),     # Weekly
        "Quincenal": ("w", 2),   # Fortnightly
        "Mensual": ("m", 3),     # Monthly
        "Trimestral": ("q", 4),  # Quarterly
        "Cuatrimestral": ("q", 4), # Four-monthly
        "Semestral": ("q", 4),   # Semi-annual
        "Anual": ("a", 5),       # Annual
        "": ("a", 5),            # Default fallback
        None: ("a", 5),
    }


    # Load Series Catalog
    print(f"\n" + "="*60)
    print("LOADING SERIES CATALOG")
    print("="*60)

    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(script_dir, "variables", "banxico_series.json")

    print(f"Script directory: {script_dir}")
    print(f"JSON file path: {json_file_path}")
    print(f"File exists: {os.path.exists(json_file_path)}")

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            series_catalog: Dict[str, List[str]] = _json.load(f)
        print(f"✓ Successfully loaded series catalog from {json_file_path}")
        print(f"  Categories loaded: {list(series_catalog.keys())}")
        total_series = sum(len(series_list) for series_list in series_catalog.values())
        print(f"  Total series: {total_series}")
        
        # Show category breakdown
        for category, series_list in series_catalog.items():
            if series_list:
                print(f"    {category}: {len(series_list)} series")
            else:
                print(f"    {category}: 0 series (empty)")
                
    except FileNotFoundError:
        print(f"✗ Warning: {json_file_path} not found. Using empty catalog.", file=_sys.stderr)
        series_catalog: Dict[str, List[str]] = {}
    except Exception as exc:
        print(f"✗ Warning: Error loading {json_file_path}: {exc}. Using empty catalog.", file=_sys.stderr)
        series_catalog: Dict[str, List[str]] = {}


    # Build Series List
    print(f"\n" + "="*60)
    print("BUILDING SERIES LIST")
    print("="*60)

    series_list: List[Dict[str, str]] = []
    for category, ids in series_catalog.items():
        if ids:
            for sid in ids:
                series_list.append({"category": category, "id": sid})
            print(f"  {category}: {len(ids)} series added")
        else:
            print(f"  {category}: 0 series (skipped)")

    print(f"✓ Total series to process: {len(series_list)}")
    if series_list:
        print(f"  First 5 series: {[f'{s['category']}:{s['id']}' for s in series_list[:5]]}")
        if len(series_list) > 5:
            print(f"  ... and {len(series_list) - 5} more")


    # API Configuration
    base_meta_url = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/{id}"
    base_data_url = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/{id}/datos/{start}/{end}"

    # Date range for data extraction
    today = _dt.datetime.now().date()
    start_date = "1900-01-01"
    end_date = today.strftime("%Y-%m-%d")


    # Data Collection Containers
    metadata_rows: List[Dict[str, object]] = []
    data_dict: Dict[str, _pd.Series] = {}


    # Process Each Series
    print(f"\n" + "="*60)
    print(f"PROCESSING {len(series_list)} SERIES")
    print("="*60)

    for i, item in enumerate(series_list, 1):
        sid = item["id"]
        category = item["category"]
        
        print(f"\n[{i:2d}/{len(series_list)}] Processing {sid} ({category})")
        print(f"  {'─' * 50}")

        # Fetch Metadata
        print(f"    📊 Fetching metadata...")
        meta_url = base_meta_url.format(id=sid)
        meta_params = {"mediaType": "json", "token": token, "locale": "en"}
        
        try:
            meta_resp = _requests.get(meta_url, params=meta_params, timeout=30)
            meta_resp.raise_for_status()
            meta_json = meta_resp.json()
            print(f"      ✓ Metadata fetched successfully")
            
            # Extract metadata from API response
            series_meta = meta_json.get("bmx", {}).get("series", [{}])[0]
            
            meta = {
                "idSerie": series_meta.get("idSerie", sid),
                "titulo": series_meta.get("titulo"),
                "frecuencia": series_meta.get("periodicidad"),
                "fechaInicio": series_meta.get("fechaInicio"),
                "fechaFin": series_meta.get("fechaFin"),
                "ultimaActualizacion": None,  # Not provided by API
                "siguienteActualizacion": None,  # Not provided by API
            }
            
            print(f"      Title: {meta['titulo']}")
            print(f"      Frequency: {meta['frecuencia']}")
            print(f"      Date range: {meta['fechaInicio']} to {meta['fechaFin']}")
            
        except Exception as exc:
            print(f"      ✗ Warning: failed to fetch metadata for {sid}: {exc}", file=_sys.stderr)
            meta = {
                "idSerie": sid,
                "titulo": None,
                "frecuencia": None,
                "fechaInicio": None,
                "fechaFin": None,
                "ultimaActualizacion": None,
                "siguienteActualizacion": None,
            }

        # Map frequency to code and rank
        freq_text = meta.get("frecuencia")
        code, rank = frequency_map.get(freq_text, ("a", 5))
        print(f"      Frequency mapping: '{freq_text}' → {code} (rank: {rank})")

        # Store metadata
        metadata_row = {
            "category": category,
            "id": meta["idSerie"],
            "name": meta["titulo"],
            "frequency": code,
            "freq_text": freq_text,
            "freq_rank": rank,
            "last_update": meta["ultimaActualizacion"],
            "next_update": meta["siguienteActualizacion"],
        }
        metadata_rows.append(metadata_row)
        print(f"      ✓ Metadata row added: {category} | {meta['idSerie']} | {meta['titulo']}")

        # Fetch Data (only if we have valid metadata)
        if meta["titulo"]:
            print(f"    📈 Fetching data...")
            data_url = base_data_url.format(id=sid, start=start_date, end=end_date)
            data_params = {"mediaType": "json", "token": token, "locale": "en"}
            
            try:
                data_resp = _requests.get(data_url, params=data_params, timeout=30)
                data_resp.raise_for_status()
                data_json = data_resp.json()
                print(f"      ✓ Data fetched successfully")
                
                # Find the correct series in the response (API can return multiple)
                series_data = []
                for series in data_json.get("bmx", {}).get("series", []):
                    if series.get("idSerie") == sid:
                        series_data = series.get("datos", [])
                        break
                
                print(f"      Raw observations: {len(series_data)} data points")
                
                # Process observations
                dates: List[_dt.date] = []
                values: List[float] = []
                valid_obs = 0
                skipped_obs = 0
                
                for obs in series_data:
                    date_str = obs.get("fecha")
                    value_str = obs.get("dato")
                    
                    # Skip missing values
                    if value_str == "N/E":
                        skipped_obs += 1
                        continue
                    
                    # Parse date (Banxico uses DD/MM/YYYY format)
                    date = parse_banxico_date(date_str)
                    if date is None:
                        skipped_obs += 1
                        continue
                    
                    # Parse value
                    try:
                        value = float(value_str.replace(",", ""))
                        dates.append(date)
                        values.append(value)
                        valid_obs += 1
                    except (ValueError, TypeError):
                        skipped_obs += 1
                        continue
                
                print(f"      Valid observations: {valid_obs}, Skipped: {skipped_obs}")
                
                # Create pandas Series
                if dates:
                    s = _pd.Series(values, index=_pd.to_datetime(dates), name=sid)
                    data_dict[sid] = s
                    print(f"      ✓ Data Series created: {len(s)} observations")
                    print(f"        Date range: {s.index.min().strftime('%Y-%m-%d')} to {s.index.max().strftime('%Y-%m-%d')}")
                    print(f"        Value range: {s.min():.4f} to {s.max():.4f}")
                else:
                    print(f"      ⚠ No valid data points found")
                    
            except Exception as exc:
                print(f"      ✗ Warning: failed to fetch data for {sid}: {exc}", file=_sys.stderr)
        else:
            print(f"    ⚠ Skipping data fetch - no valid title in metadata")


    # Create Metadata DataFrame
    print(f"\n" + "="*60)
    print("CREATING METADATA DATAFRAME")
    print("="*60)

    metadata_df = _pd.DataFrame(metadata_rows)
    print(f"✓ Metadata DataFrame created: {metadata_df.shape[1]} columns × {metadata_df.shape[0]} rows")

    if not metadata_df.empty:
        print(f"\nMetadata summary:")
        print(f"  Categories: {metadata_df['category'].nunique()}")
        print(f"  Total series: {len(metadata_df)}")
        print(f"  Frequency distribution:")
        freq_counts = metadata_df['frequency'].value_counts()
        for freq, count in freq_counts.items():
            print(f"    {freq}: {count} series")


    # Determine Target Frequency
    print(f"\n" + "="*60)
    print("DETERMINING TARGET FREQUENCY")
    print("="*60)

    if len(metadata_df) > 0:
        target_rank = metadata_df["freq_rank"].min()
        print(f"✓ Target frequency rank: {target_rank}")
        print(f"  Available frequency ranks: {sorted(metadata_df['freq_rank'].unique())}")
    else:
        target_rank = 5
        print(f"⚠ No metadata available, using default rank: {target_rank}")

    # Map rank to pandas offset
    rank_to_offset = {1: "D", 2: "W", 3: "MS", 4: "QS", 5: "AS"}
    target_offset = rank_to_offset.get(target_rank, "AS")
    print(f"  Target offset: {target_offset}")

    # Build Master Date Index
    master_index = _pd.Index([])
    if data_dict:
        print(f"  Building master date index...")
        min_start = min((s.index.min() for s in data_dict.values()))
        max_end = max((s.index.max() for s in data_dict.values()))
        print(f"    Overall date range: {min_start.strftime('%Y-%m-%d')} to {max_end.strftime('%Y-%m-%d')}")
        
        master_index = _pd.date_range(
            start=min_start, end=max_end, freq=target_offset, name="Date"
        )
        print(f"    ✓ Master index created: {len(master_index)} dates")
    else:
        print(f"  ⚠ No data available, master index will be empty")


    # Align and Backfill Data
    print(f"\n" + "="*60)
    print("ALIGNING AND BACKFILLING DATA")
    print("="*60)

    aligned_series_list: List[_pd.Series] = []
    print(f"Processing {len(data_dict)} series for alignment...")

    for i, (sid, s) in enumerate(data_dict.items(), 1):
        print(f"  [{i:2d}/{len(data_dict)}] Aligning {sid}...")
        
        if master_index.empty:
            aligned_series_list.append(s)
            print(f"    ✓ Added original series (no master index)")
        else:
            # Reindex to master index and fill missing values
            original_length = len(s)
            s_aligned = s.reindex(master_index)
            s_filled = s_aligned.ffill().bfill()
            aligned_series = s_filled.rename(sid)
            aligned_series_list.append(aligned_series)
            
            nan_before = s_aligned.isna().sum()
            nan_after = aligned_series.isna().sum()
            
            print(f"    ✓ Aligned to master index: {original_length} → {len(aligned_series)} observations")
            print(f"      NaN values: {nan_before} → {nan_after} after filling")


    # Create Final DataFrame
    print(f"\n" + "="*60)
    print("CREATING FINAL DATAFRAME")
    print("="*60)

    if aligned_series_list:
        print(f"Concatenating {len(aligned_series_list)} aligned series...")
        data_df = _pd.concat(aligned_series_list, axis=1)
        print(f"✓ Final DataFrame created: {data_df.shape[1]} columns × {data_df.shape[0]} rows")
        
        print(f"  Columns: {list(data_df.columns)}")
        print(f"  Date range: {data_df.index.min().strftime('%Y-%m-%d')} to {data_df.index.max().strftime('%Y-%m-%d')}")
        
        # Data quality summary
        total_cells = data_df.shape[0] * data_df.shape[1]
        non_null_cells = data_df.count().sum()
        null_cells = total_cells - non_null_cells
        print(f"  Data quality: {non_null_cells:,} non-null values, {null_cells:,} null values")
        print(f"  Completeness: {(non_null_cells/total_cells)*100:.1f}%")
        
    else:
        data_df = _pd.DataFrame(index=master_index)
        print(f"⚠ No aligned series available, created empty DataFrame")

    data_df.index.name = "Date"
    print(f"✓ Index named: '{data_df.index.name}'")


    # Database Storage
    print("\n" + "="*60)
    print("SETTING UP DATABASE CONNECTION")
    print("="*60)

    if db_manager is not None:
        print("✓ Using provided database manager")
        if db_manager.health_check():
            print("✅ Database connection healthy")
        else:
            print("❌ Database connection unhealthy - skipping database storage")
            db_manager = None
    else:
        print("⚠ No database manager provided - database storage will be skipped.")


    # Store Data to Database
    if db_manager is not None:
        print(f"\n" + "="*60)
        print("STORING DATA TO DATABASE")
        print("="*60)
        
        dataframes_to_store = {}
        
        if not metadata_df.empty:
            dataframes_to_store['banxico_metadata'] = metadata_df
            print("✓ Metadata DataFrame ready for storage")
        
        if not data_df.empty:
            dataframes_to_store['banxico_data'] = data_df
            print("✓ Main data DataFrame ready for storage")
        
        if dataframes_to_store:
            print(f"\nStoring {len(dataframes_to_store)} DataFrames to database...")
            
            for i, (table_name, df) in enumerate(dataframes_to_store.items(), 1):
                try:
                    print(f"[{i:2d}/{len(dataframes_to_store)}] {table_name:20s}...", end=" ")
                    
                    index_name = df.index.name if df.index.name else "Unnamed"
                    print(f"\n    Index: {index_name}, Shape: {df.shape[1]:3d} columns × {df.shape[0]:6d} rows")
                    print(f"    Storing...", end=" ")
                    
                    # Use database manager connection only
                    with db_manager.get_db_connection() as conn:
                        df.to_sql(table_name, conn, if_exists='replace', index=True)
                    
                    print(f"✓ Stored successfully")
                    
                except Exception as e:
                    print(f"✗ Error: {str(e)[:50]}...")
                    continue
            
            print(f"\n" + "="*60)
            print("DATABASE STORAGE COMPLETE")
            print("="*60)
            print(f"✓ Data stored using database manager")
            
            stored_tables = list(dataframes_to_store.keys())
            print(f"Tables created: {', '.join(stored_tables)}")
            
        else:
            print("No DataFrames available for storage")
    else:
        print("\nDatabase storage skipped - no connection available")


    # Display Results
    print("\n" + "="*60)
    print("DATA PREVIEW")
    print("="*60)

    print("Metadata (first 10 rows):")
    print(metadata_df.head(10).to_string(index=False))

    print("\nCombined data (first 10 rows):")
    print(data_df.head(10))

    return True  # Return success status

# Main execution block
if __name__ == "__main__":
    # Run the collection without database manager (standalone mode)
    success = collect_banxico_data()
    if success:
        print("\n✅ Banxico data collection completed successfully!")
    else:
        print("\n❌ Banxico data collection failed!")
        _sys.exit(1)
