import pandas as pd
import numpy as np
import time
import os
import json
import yfinance as yf
from datetime import datetime


def collect_financial_data(db_manager=None) -> bool:
    try:
        # Load the JSON metadata
        print("Loading symbols metadata from JSON...")
        json_file_path = os.path.join(os.path.dirname(__file__), 'variables', 'symbols_metadata.json')
        
        try:
            with open(json_file_path, 'r') as f:
                symbols_metadata = json.load(f)
            print("✓ Symbols metadata loaded successfully")
        except FileNotFoundError:
            print(f"✗ JSON file not found at: {json_file_path}")
            print("Please ensure 'symbols_metadata.json' exists in the same directory")
            return False
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing JSON: {e}")
            return False
        
        # Extract all symbols from the JSON using loop-based approach
        print("Extracting symbols from metadata...")
        all_symbols = []
        to_process = [symbols_metadata]
        
        # Process each item in the queue using loops
        i = 0
        while i < len(to_process):
            current_data = to_process[i]
            
            # If it's a data object with symbol/name, extract the symbol
            if isinstance(current_data, dict) and 'symbol' in current_data and 'name' in current_data:
                all_symbols.append(current_data['symbol'])
            
            # If it's a list, add all items to the processing queue
            elif isinstance(current_data, list):
                j = 0
                while j < len(current_data):
                    to_process.append(current_data[j])
                    j += 1
            
            # If it's a dict, add all values to the processing queue
            elif isinstance(current_data, dict):
                for key in current_data:
                    to_process.append(current_data[key])
            
            i += 1
        
        print(f"✓ Extracted {len(all_symbols)} symbols from metadata")

        # Download data for all symbols using loop-based approach
        all_data = {}
        failed_downloads = []
        successful_downloads = []
        
        # Categorize symbols for separate dataframes
        sector_symbols = []
        futures_symbols = []
        crypto_symbols = []
        global_market_symbols = []

        # Build O(1) lookup sets for categorization
        sector_lookup = set()
        for item in symbols_metadata.get('Sectors', []):
            try:
                sector_lookup.add(item['symbol'])
            except Exception:
                pass
        crypto_lookup = set()
        for item in symbols_metadata.get('Crypto', []):
            try:
                crypto_lookup.add(item['symbol'])
            except Exception:
                pass
        futures_lookup = set()
        futures = symbols_metadata.get('Futures', {})
        for cat in futures:
            items = futures[cat]
            j = 0
            while j < len(items):
                try:
                    futures_lookup.add(items[j]['symbol'])
                except Exception:
                    pass
                j += 1
        global_lookup = set()
        gm = symbols_metadata.get('Global Markets', {})
        for region in gm:
            countries = gm[region]
            if isinstance(countries, dict):
                for country in countries:
                    items = countries[country]
                    if isinstance(items, list):
                        j = 0
                        while j < len(items):
                            try:
                                global_lookup.add(items[j]['symbol'])
                            except Exception:
                                pass
                            j += 1

        print(f"Starting download of {len(all_symbols)} symbols...")
        start_time = time.time()

        # Batch download using yfinance with threads; fallback to chunked if needed
        data = None
        try:
            data = yf.download(
                tickers=all_symbols,
                period='max',
                interval='1d',
                auto_adjust=True,
                prepost=True,
                group_by='ticker',
                threads=True,
                progress=True
            )
        except Exception as e:
            print(f"Batch download failed ({str(e)[:80]}). Falling back to chunked downloads...")
            # Chunked fallback
            chunk_size = 50
            chunks = [all_symbols[k:k+chunk_size] for k in range(0, len(all_symbols), chunk_size)]
            frames = []
            for ci, chunk in enumerate(chunks, 1):
                try:
                    print(f"  Chunk {ci}/{len(chunks)}: {len(chunk)} symbols")
                    dpart = yf.download(
                        tickers=chunk,
                        period='max',
                        interval='1d',
                        auto_adjust=True,
                        prepost=True,
                        group_by='ticker',
                        threads=True,
                        progress=False
                    )
                    frames.append(dpart)
                except Exception as e2:
                    print(f"  ✗ Chunk {ci} failed: {str(e2)[:80]}")
            if frames:
                try:
                    # Align columns
                    data = pd.concat(frames, axis=1)
                except Exception:
                    pass

        # Populate all_data and categorize
        if data is None or data.empty:
            # As a last resort, try each symbol quickly but without deep per-symbol categorization printouts
            print("Batch data empty; attempting minimal per-symbol fallback...")
            i = 0
            while i < len(all_symbols):
                ticker = all_symbols[i]
                try:
                    dta = yf.download(ticker, period='max', interval='1d', auto_adjust=True, prepost=True, progress=False)
                    if not dta.empty:
                        series = dta['Close'] if 'Close' in dta.columns else dta.iloc[:, 0]
                        series = series.astype('float64')
                        all_data[ticker] = series
                        successful_downloads.append(ticker)
                    else:
                        failed_downloads.append(ticker)
                except Exception:
                    failed_downloads.append(ticker)
                i += 1
        else:
            # Multi-ticker result handling
            if isinstance(data.columns, pd.MultiIndex):
                # Iterate symbols and extract close
                i = 0
                while i < len(all_symbols):
                    ticker = all_symbols[i]
                    try:
                        cols = data.columns.get_level_values(0)
                        if ticker in cols:
                            sub = data[ticker]
                            if 'Close' in sub.columns:
                                series = sub['Close']
                            elif 'Adj Close' in sub.columns:
                                series = sub['Adj Close']
                            else:
                                # take first numeric column
                                series = sub.select_dtypes(include=[np.number]).iloc[:, 0]
                            series = series.dropna().astype('float64')
                            if not series.empty:
                                all_data[ticker] = series
                                successful_downloads.append(ticker)
                            else:
                                failed_downloads.append(ticker)
                        else:
                            failed_downloads.append(ticker)
                    except Exception:
                        failed_downloads.append(ticker)
                    i += 1
            else:
                # Single-ticker case
                try:
                    if 'Close' in data.columns:
                        series = data['Close']
                    elif 'Adj Close' in data.columns:
                        series = data['Adj Close']
                    else:
                        series = data.select_dtypes(include=[np.number]).iloc[:, 0]
                    series = series.dropna().astype('float64')
                    if len(all_symbols) == 1:
                        all_data[all_symbols[0]] = series
                        successful_downloads.append(all_symbols[0])
                except Exception:
                    pass

            # Categorize using lookup sets
            k = 0
            while k < len(successful_downloads):
                sym = successful_downloads[k]
                if sym in sector_lookup:
                    sector_symbols.append(sym)
                elif sym in crypto_lookup:
                    crypto_symbols.append(sym)
                elif sym in futures_lookup:
                    futures_symbols.append(sym)
                elif sym in global_lookup:
                    global_market_symbols.append(sym)
                else:
                    global_market_symbols.append(sym)
                k += 1

        # Timing and summary
        elapsed_time = time.time() - start_time
        print(f"\n" + "="*60)
        print(f"DOWNLOAD COMPLETE")
        print(f"="*60)
        print(f"Successful: {len(successful_downloads)}/{len(all_symbols)} symbols")
        print(f"Failed:    {len(failed_downloads)}/{len(all_symbols)} symbols")
        print(f"Sectors:   {len(sector_symbols)} symbols")
        print(f"Futures:   {len(futures_symbols)} symbols")
        print(f"Crypto:    {len(crypto_symbols)} symbols")
        print(f"Global:    {len(global_market_symbols)} symbols")
        print(f"Time taken: {elapsed_time:.1f} seconds")
        print(f"Average:   {elapsed_time/len(all_symbols):.2f} seconds per symbol")

        # Create separate DataFrames using loop-based approach
        print(f"\n" + "="*60)
        print(f"CREATING CATEGORIZED DATAFRAMES")
        print(f"="*60)
        
        # Sectors DataFrame
        if sector_symbols:
            sectors_data = {}
            i = 0
            while i < len(sector_symbols):
                symbol = sector_symbols[i]
                if symbol in all_data:
                    sectors_data[symbol] = all_data[symbol]
                i += 1
            
            sectors_df = pd.DataFrame(sectors_data)
            sectors_df.index.name = 'Date'
            print(f"✓ Sectors data shape: {sectors_df.shape}")
        else:
            print("⚠ No sector data available!")
            sectors_df = pd.DataFrame()
        
        # Futures DataFrame
        if futures_symbols:
            futures_data = {}
            i = 0
            while i < len(futures_symbols):
                symbol = futures_symbols[i]
                if symbol in all_data:
                    futures_data[symbol] = all_data[symbol]
                i += 1
            
            futures_df = pd.DataFrame(futures_data)
            futures_df.index.name = 'Date'
            print(f"✓ Futures data shape: {futures_df.shape}")
        else:
            print("⚠ No futures data available!")
            futures_df = pd.DataFrame()

        # Crypto DataFrame
        if crypto_symbols:
            crypto_data = {}
            i = 0
            while i < len(crypto_symbols):
                symbol = crypto_symbols[i]
                if symbol in all_data:
                    crypto_data[symbol] = all_data[symbol]
                i += 1
            
            crypto_df = pd.DataFrame(crypto_data)
            crypto_df.index.name = 'Date'
            print(f"✓ Crypto data shape: {crypto_df.shape}")
        else:
            print("⚠ No crypto data available!")
            crypto_df = pd.DataFrame()

        # Global Markets DataFrame
        if global_market_symbols:
            global_data = {}
            i = 0
            while i < len(global_market_symbols):
                symbol = global_market_symbols[i]
                if symbol in all_data:
                    global_data[symbol] = all_data[symbol]
                i += 1
            
            global_markets_df = pd.DataFrame(global_data)
            global_markets_df.index.name = 'Date'
            print(f"✓ Global markets data shape: {global_markets_df.shape}")
        else:
            print("⚠ No global market data available!")
            global_markets_df = pd.DataFrame()

        # Create markets metadata DataFrame using loop-based approach
        print(f"\n" + "="*60)
        print(f"CREATING MARKETS METADATA")
        print(f"="*60)
        
        metadata_rows = []
        
        # Process sectors
        i = 0
        while i < len(symbols_metadata.get('Sectors', [])):
            sector = symbols_metadata['Sectors'][i]
            symbol = sector['symbol']
            if symbol in all_data and all_data[symbol] is not None:
                data = all_data[symbol]
                metadata_rows.append({
                    'symbol': symbol,
                    'name': sector['name'],
                    'category': 'Sectors',
                    'subcategory': 'Sector ETFs',
                    'region': 'United States',
                    'country': 'United States',
                    'earliest_date': data.index.min(),
                    'latest_date': data.index.max(),
                    'total_rows': len(data),
                    'last_updated': datetime.now()
                })
            i += 1
        
        # Process futures
        futures = symbols_metadata.get('Futures', {})
        for category in futures:
            items = futures[category]
            i = 0
            while i < len(items):
                item = items[i]
                symbol = item['symbol']
                if symbol in all_data and all_data[symbol] is not None:
                    data = all_data[symbol]
                    metadata_rows.append({
                        'symbol': symbol,
                        'name': item['name'],
                        'category': 'Futures',
                        'subcategory': category,
                        'region': 'United States',
                        'country': 'United States',
                        'earliest_date': data.index.min(),
                        'latest_date': data.index.max(),
                        'total_rows': len(data),
                        'last_updated': datetime.now()
                    })
                i += 1
        
        # Process crypto
        i = 0
        while i < len(symbols_metadata.get('Crypto', [])):
            crypto = symbols_metadata['Crypto'][i]
            symbol = crypto['symbol']
            if symbol in all_data and all_data[symbol] is not None:
                data = all_data[symbol]
                metadata_rows.append({
                    'symbol': symbol,
                    'name': crypto['name'],
                    'category': 'Crypto',
                    'subcategory': 'Cryptocurrency',
                    'region': 'Global',
                    'country': 'Global',
                    'earliest_date': data.index.min(),
                    'latest_date': data.index.max(),
                    'total_rows': len(data),
                    'last_updated': datetime.now()
                })
            i += 1
        
        # Process global markets
        global_markets = symbols_metadata.get('Global Markets', {})
        for region in global_markets:
            countries = global_markets[region]
            if isinstance(countries, dict):
                for country in countries:
                    items = countries[country]
                    if isinstance(items, list):
                        i = 0
                        while i < len(items):
                            item = items[i]
                            symbol = item['symbol']
                            if symbol in all_data and all_data[symbol] is not None:
                                data = all_data[symbol]
                                metadata_rows.append({
                                    'symbol': symbol,
                                    'name': item['name'],
                                    'category': 'Global Markets',
                                    'subcategory': region,
                                    'region': region,
                                    'country': country,
                                    'earliest_date': data.index.min(),
                                    'latest_date': data.index.max(),
                                    'total_rows': len(data),
                                    'last_updated': datetime.now()
                                })
                            i += 1
        
        markets_metadata_df = pd.DataFrame(metadata_rows)
        print(f"✓ Markets metadata shape: {markets_metadata_df.shape}")
        
        if not markets_metadata_df.empty:
            print("Sample metadata entries:")
            print(markets_metadata_df.head(3).to_string())

        # Database storage section using db_manager
        if db_manager is None or not hasattr(db_manager, 'get_db_connection'):
            print("⚠ No database manager provided - skipping database storage")
        else:
            try:
                print(f"\n" + "="*60)
                print(f"STORING FINANCIAL DATA TO DATABASE")
                print(f"="*60)
                
                # Create a mapping of DataFrame names to DataFrame objects
                dataframe_mapping = {
                    "sectors_df": sectors_df,
                    "futures_df": futures_df,
                    "crypto_df": crypto_df,
                    "global_markets_df": global_markets_df,
                    "markets_metadata_df": markets_metadata_df
                }

                df_names = ["sectors_df", "futures_df", "crypto_df", "global_markets_df", "markets_metadata_df"]

                print(f"Preparing to store {len(df_names)} DataFrames to database...")

                i = 0
                while i < len(df_names):
                    name = df_names[i]
                    try:
                        df_obj = dataframe_mapping.get(name)
                        
                        if isinstance(df_obj, pd.DataFrame):
                            if not df_obj.empty:
                                print(f"[{i+1:2d}/{len(df_names)}] {name:20s}...", end=" ")
                                
                                # Show column information including index
                                index_name = df_obj.index.name if df_obj.index.name else "Unnamed"
                                columns_info = f"Index: {index_name}, Columns: {list(df_obj.columns)}"
                                print(f"\n    Columns: {columns_info}")
                                print(f"    Storing...", end=" ")
                                
                                # Store the DataFrame via db_manager
                                with db_manager.get_db_connection() as conn:
                                    df_obj.to_sql(name, conn, if_exists='replace', index=True)
                                
                                print(f"✓ Stored ({df_obj.shape[1]:3d} columns, {df_obj.shape[0]:6d} rows)")
                            else:
                                print(f"[{i+1:2d}/{len(df_names)}] {name:20s}... ⚠ Empty DataFrame - skipping")
                        else:
                            print(f"[{i+1:2d}/{len(df_names)}] {name:20s}... ✗ Not a DataFrame - skipping")
                            
                    except Exception as e:
                        print(f"[{i+1:2d}/{len(df_names)}] {name:20s}... ✗ Error: {str(e)[:50]}...")
                    
                    i += 1

                print(f"\n" + "="*60)
                print(f"FINANCIAL DATA STORAGE COMPLETE")
                print(f"="*60)
                print(f"All financial data has been successfully stored to the database")

                # Show what tables were created
                all_tables = ["sectors_df", "futures_df", "crypto_df", "global_markets_df", "markets_metadata_df"]
                print(f"Tables created: {', '.join(all_tables)}")

            except Exception as e:
                print(f"Database storage error: {str(e)}")

        return True

    except Exception as e:
        print(f"Data collection error: {str(e)}")
        return False


if __name__ == "__main__":
    collect_financial_data()
