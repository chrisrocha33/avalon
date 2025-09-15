import pandas as pd
import numpy as np
from sqlalchemy import text
import yfinance as yf
from datetime import datetime
import concurrent.futures
import time
import random

# URLs (official NASDAQ Trader symbol directories)
url_nasdaq = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
url_other  = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

def collect_equities_metadata(db_manager=None) -> bool:
    # ---- Load NASDAQ-listed
    market = pd.read_csv(url_nasdaq, sep="|", dtype=str)
    
    print(market.columns,market.index)
    print("="*60)
    print(market.head(5))
    
    # Initialize an empty list to store company metadata
    market_meta = []
    
    def get_ticker_info(symbol, row_data):
        """Get ticker information for a single symbol"""
        try:
            # Add small random delay to avoid overwhelming the API
            time.sleep(random.uniform(0.1, 0.3))
            
            ticker_data = yf.Ticker(symbol)
            
            # Start with the original NASDAQ data
            company_info = {
                'Symbol': symbol,
                'Security_Name': row_data.get('Security Name', 'N/A'),
                'Market_Category': row_data.get('Market Category', 'N/A'),
                'Test_Issue': row_data.get('Test Issue', 'N/A'),
                'Financial_Status': row_data.get('Financial Status', 'N/A'),
                'Round_Lot_Size': row_data.get('Round Lot Size', 'N/A'),
                'ETF': row_data.get('ETF', 'N/A'),
                'NextShares': row_data.get('NextShares', 'N/A')
            }
            
            # Helper function to safely convert data to string
            def safe_convert(data):
                if data is None:
                    return 'N/A'
                elif isinstance(data, (pd.DataFrame, pd.Series)):
                    return data.to_json() if not data.empty else 'N/A'
                elif isinstance(data, (list, dict)):
                    return str(data) if data else 'N/A'
                else:
                    return str(data)
            
            # Extract info data
            info = ticker_data.info
            company_info.update({
                'Company_Name': info.get('longName', 'N/A'),
                'Short_Name': info.get('shortName', 'N/A'),
                'Industry': info.get('industry', 'N/A'),
                'Sector': info.get('sector', 'N/A'),
                'Description': info.get('longBusinessSummary', 'N/A'),
                'IPO_Date': info.get('firstTradeDateMilliseconds', 'N/A'),
                'Market_Cap': info.get('marketCap', 'N/A'),
                'Country': info.get('country', 'N/A'),
                'Website': info.get('website', 'N/A'),
                'Employees': info.get('fullTimeEmployees', 'N/A'),
                'Exchange': info.get('exchange', 'N/A'),
                'Currency': info.get('currency', 'N/A'),
                'Book_Value': info.get('bookValue', 'N/A'),
                'Price_to_Book': info.get('priceToBook', 'N/A'),
                'Profit_Margins': info.get('profitMargins', 'N/A'),
                'Operating_Margins': info.get('operatingMargins', 'N/A'),
                'Current_Price': info.get('currentPrice', 'N/A'),
                'Volume': info.get('volume', 'N/A'),
                'Avg_Volume': info.get('averageVolume', 'N/A')
            })
            
            # Extract all other yfinance attributes
            try:
                company_info['Actions'] = safe_convert(ticker_data.actions)
            except:
                company_info['Actions'] = 'N/A'
                
            try:
                company_info['Analyst_Price_Targets'] = safe_convert(ticker_data.analyst_price_targets)
            except:
                company_info['Analyst_Price_Targets'] = 'N/A'
                
            try:
                company_info['Balance_Sheet'] = safe_convert(ticker_data.balance_sheet)
            except:
                company_info['Balance_Sheet'] = 'N/A'
                
            try:
                company_info['Calendar'] = safe_convert(ticker_data.calendar)
            except:
                company_info['Calendar'] = 'N/A'
                
            try:
                company_info['Capital_Gains'] = safe_convert(ticker_data.capital_gains)
            except:
                company_info['Capital_Gains'] = 'N/A'
                
            try:
                company_info['Cash_Flow'] = safe_convert(ticker_data.cash_flow)
            except:
                company_info['Cash_Flow'] = 'N/A'
                
            try:
                company_info['Dividends'] = safe_convert(ticker_data.dividends)
            except:
                company_info['Dividends'] = 'N/A'
                
            try:
                company_info['Earnings'] = safe_convert(ticker_data.earnings)
            except:
                company_info['Earnings'] = 'N/A'
                
            try:
                company_info['Earnings_Dates'] = safe_convert(ticker_data.earnings_dates)
            except:
                company_info['Earnings_Dates'] = 'N/A'
                
            try:
                company_info['Earnings_Estimate'] = safe_convert(ticker_data.earnings_estimate)
            except:
                company_info['Earnings_Estimate'] = 'N/A'
                
            try:
                company_info['Earnings_History'] = safe_convert(ticker_data.earnings_history)
            except:
                company_info['Earnings_History'] = 'N/A'
                
            try:
                company_info['EPS_Revisions'] = safe_convert(ticker_data.eps_revisions)
            except:
                company_info['EPS_Revisions'] = 'N/A'
                
            try:
                company_info['EPS_Trend'] = safe_convert(ticker_data.eps_trend)
            except:
                company_info['EPS_Trend'] = 'N/A'
                
            try:
                company_info['Fast_Info'] = safe_convert(ticker_data.fast_info)
            except:
                company_info['Fast_Info'] = 'N/A'
                
            try:
                company_info['Financials'] = safe_convert(ticker_data.financials)
            except:
                company_info['Financials'] = 'N/A'
                
            try:
                company_info['Funds_Data'] = safe_convert(ticker_data.funds_data)
            except:
                company_info['Funds_Data'] = 'N/A'
                
            try:
                company_info['Growth_Estimates'] = safe_convert(ticker_data.growth_estimates)
            except:
                company_info['Growth_Estimates'] = 'N/A'
                
            try:
                company_info['History_Metadata'] = safe_convert(ticker_data.history_metadata)
            except:
                company_info['History_Metadata'] = 'N/A'
                
            try:
                company_info['Income_Stmt'] = safe_convert(ticker_data.income_stmt)
            except:
                company_info['Income_Stmt'] = 'N/A'
                
            try:
                company_info['Insider_Purchases'] = safe_convert(ticker_data.insider_purchases)
            except:
                company_info['Insider_Purchases'] = 'N/A'
                
            try:
                company_info['Insider_Roster_Holders'] = safe_convert(ticker_data.insider_roster_holders)
            except:
                company_info['Insider_Roster_Holders'] = 'N/A'
                
            try:
                company_info['Insider_Transactions'] = safe_convert(ticker_data.insider_transactions)
            except:
                company_info['Insider_Transactions'] = 'N/A'
                
            try:
                company_info['Institutional_Holders'] = safe_convert(ticker_data.institutional_holders)
            except:
                company_info['Institutional_Holders'] = 'N/A'
                
            try:
                company_info['ISIN'] = safe_convert(ticker_data.isin)
            except:
                company_info['ISIN'] = 'N/A'
                
            try:
                company_info['Major_Holders'] = safe_convert(ticker_data.major_holders)
            except:
                company_info['Major_Holders'] = 'N/A'
                
            try:
                company_info['Mutualfund_Holders'] = safe_convert(ticker_data.mutualfund_holders)
            except:
                company_info['Mutualfund_Holders'] = 'N/A'
                
            try:
                company_info['News'] = safe_convert(ticker_data.news)
            except:
                company_info['News'] = 'N/A'
                
            try:
                company_info['Options'] = safe_convert(ticker_data.options)
            except:
                company_info['Options'] = 'N/A'
                
            try:
                company_info['Quarterly_Balance_Sheet'] = safe_convert(ticker_data.quarterly_balance_sheet)
            except:
                company_info['Quarterly_Balance_Sheet'] = 'N/A'
                
            try:
                company_info['Quarterly_Cash_Flow'] = safe_convert(ticker_data.quarterly_cash_flow)
            except:
                company_info['Quarterly_Cash_Flow'] = 'N/A'
                
            try:
                company_info['Quarterly_Earnings'] = safe_convert(ticker_data.quarterly_earnings)
            except:
                company_info['Quarterly_Earnings'] = 'N/A'
                
            try:
                company_info['Quarterly_Financials'] = safe_convert(ticker_data.quarterly_financials)
            except:
                company_info['Quarterly_Financials'] = 'N/A'
                
            try:
                company_info['Quarterly_Income_Stmt'] = safe_convert(ticker_data.quarterly_income_stmt)
            except:
                company_info['Quarterly_Income_Stmt'] = 'N/A'
                
            try:
                company_info['Recommendations'] = safe_convert(ticker_data.recommendations)
            except:
                company_info['Recommendations'] = 'N/A'
                
            try:
                company_info['Recommendations_Summary'] = safe_convert(ticker_data.recommendations_summary)
            except:
                company_info['Recommendations_Summary'] = 'N/A'
                
            try:
                company_info['Revenue_Estimate'] = safe_convert(ticker_data.revenue_estimate)
            except:
                company_info['Revenue_Estimate'] = 'N/A'
                
            try:
                company_info['SEC_Filings'] = safe_convert(ticker_data.sec_filings)
            except:
                company_info['SEC_Filings'] = 'N/A'
                
            try:
                company_info['Shares'] = safe_convert(ticker_data.shares)
            except:
                company_info['Shares'] = 'N/A'
                
            try:
                company_info['Splits'] = safe_convert(ticker_data.splits)
            except:
                company_info['Splits'] = 'N/A'
                
            try:
                company_info['Sustainability'] = safe_convert(ticker_data.sustainability)
            except:
                company_info['Sustainability'] = 'N/A'
                
            try:
                company_info['TTM_Cash_Flow'] = safe_convert(ticker_data.ttm_cash_flow)
            except:
                company_info['TTM_Cash_Flow'] = 'N/A'
                
            try:
                company_info['TTM_Financials'] = safe_convert(ticker_data.ttm_financials)
            except:
                company_info['TTM_Financials'] = 'N/A'
                
            try:
                company_info['TTM_Income_Stmt'] = safe_convert(ticker_data.ttm_income_stmt)
            except:
                company_info['TTM_Income_Stmt'] = 'N/A'
                
            try:
                company_info['Upgrades_Downgrades'] = safe_convert(ticker_data.upgrades_downgrades)
            except:
                company_info['Upgrades_Downgrades'] = 'N/A'
            
            # Convert IPO date from milliseconds to readable date if available
            if company_info['IPO_Date'] != 'N/A' and company_info['IPO_Date'] is not None:
                try:
                    ipo_date = datetime.fromtimestamp(company_info['IPO_Date'] / 1000)
                    company_info['IPO_Date'] = ipo_date.strftime('%Y-%m-%d')
                except:
                    company_info['IPO_Date'] = 'N/A'
            
            return company_info
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            # Add error entry to maintain index alignment with original NASDAQ data
            error_dict = {
                'Symbol': symbol,
                'Security_Name': row_data.get('Security Name', 'N/A'),
                'Market_Category': row_data.get('Market Category', 'N/A'),
                'Test_Issue': row_data.get('Test Issue', 'N/A'),
                'Financial_Status': row_data.get('Financial Status', 'N/A'),
                'Round_Lot_Size': row_data.get('Round Lot Size', 'N/A'),
                'ETF': row_data.get('ETF', 'N/A'),
                'NextShares': row_data.get('NextShares', 'N/A'),
                'Company_Name': 'ERROR',
                'Short_Name': 'ERROR',
                'Industry': 'ERROR',
                'Sector': 'ERROR',
                'Description': f'Error: {str(e)}',
                'IPO_Date': 'ERROR',
                'Market_Cap': 'ERROR',
                'Country': 'ERROR',
                'Website': 'ERROR',
                'Employees': 'ERROR',
                'Exchange': 'ERROR',
                'Currency': 'ERROR',
                'Book_Value': 'ERROR',
                'Price_to_Book': 'ERROR',
                'Profit_Margins': 'ERROR',
                'Operating_Margins': 'ERROR',
                'Current_Price': 'ERROR',
                'Volume': 'ERROR',
                'Avg_Volume': 'ERROR'
            }
        
            # Add all the new fields as ERROR
            additional_fields = [
            'Actions', 'Analyst_Price_Targets', 'Balance_Sheet', 'Calendar', 'Capital_Gains',
            'Cash_Flow', 'Dividends', 'Earnings', 'Earnings_Dates', 'Earnings_Estimate',
            'Earnings_History', 'EPS_Revisions', 'EPS_Trend', 'Fast_Info', 'Financials',
            'Funds_Data', 'Growth_Estimates', 'History_Metadata', 'Income_Stmt',
            'Insider_Purchases', 'Insider_Roster_Holders', 'Insider_Transactions',
            'Institutional_Holders', 'ISIN', 'Major_Holders', 'Mutualfund_Holders',
            'News', 'Options', 'Quarterly_Balance_Sheet', 'Quarterly_Cash_Flow',
            'Quarterly_Earnings', 'Quarterly_Financials', 'Quarterly_Income_Stmt',
            'Recommendations', 'Recommendations_Summary', 'Revenue_Estimate',
            'SEC_Filings', 'Shares', 'Splits', 'Sustainability', 'TTM_Cash_Flow',
            'TTM_Financials', 'TTM_Income_Stmt', 'Upgrades_Downgrades'
            ]
        
            for field in additional_fields:
                error_dict[field] = 'ERROR'
                
            return error_dict
    
    # Replace the sequential loop with concurrent processing
    print("Starting concurrent data collection...")
    start_time = time.time()
    
    # Use ThreadPoolExecutor for concurrent processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks
        future_to_row = {executor.submit(get_ticker_info, row['Symbol'], row): row for index, row in market.iterrows()}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_row):
            result = future.result()
            market_meta.append(result)
            
            # Print progress every 50 companies
            if len(market_meta) % 50 == 0:
                elapsed = time.time() - start_time
                rate = len(market_meta) / elapsed
                print(f"Processed {len(market_meta)} companies... Rate: {rate:.2f} companies/sec")
    
    elapsed_time = time.time() - start_time
    print(f"Data collection completed in {elapsed_time:.2f} seconds")
    
    # Convert to DataFrame
    market_meta_df = pd.DataFrame(market_meta)
    
    # Display results
    print(f"\nTotal companies processed: {len(market_meta_df)}")
    print(f"Successful extractions: {len(market_meta_df[market_meta_df['Company_Name'] != 'ERROR'])}")
    print(f"Errors: {len(market_meta_df[market_meta_df['Company_Name'] == 'ERROR'])}")
    
    # Show first few rows
    print("\nFirst 5 companies:")
    print(market_meta_df.head())
    
    # Show summary statistics
    print("\nSummary by Sector:")
    print(market_meta_df['Sector'].value_counts().head(10))
    
    # Create table name with timestamp to avoid conflicts
    table_name = "market_metadata"
    
    # Save DataFrame to SQL table via db_manager
    if db_manager is None or not db_manager.health_check():
        print("⚠ No database manager or unhealthy - skipping database write")
    else:
        try:
            with db_manager.get_db_connection() as conn:
                market_meta_df.to_sql(
                    name=table_name,
                    con=conn,
                    if_exists='replace',
                    index=False,
                    method='multi',
                    chunksize=1000
                )
            print(f"\nData successfully saved to SQL table: {table_name}")
            print(f"Table contains {len(market_meta_df)} rows and {len(market_meta_df.columns)} columns")
            
            # Optional: Verify the data was saved by checking row count
            try:
                with db_manager.get_db_connection() as conn:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = result.scalar()
                    print(f"Verified SQL table row count: {row_count}")
            except Exception as ve:
                print(f"Verification query failed: {ve}")
        except Exception as e:
            print(f"Database storage error: {str(e)}")
    
    tickers = market_meta_df['Symbol'].dropna().unique().tolist()
    
    # Filter out error symbols before downloading for better performance
    valid_tickers = market_meta_df[market_meta_df['Company_Name'] != 'ERROR']['Symbol'].dropna().unique().tolist()
    
    print(f"Downloading price data for {len(valid_tickers)} valid tickers...")
    start_time = time.time()
    
    # 2) Download data with optimized parameters
    data = yf.download(
        valid_tickers,
        period="5y",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        prepost=False,
        threads=20,
        progress=True
    )
    
    elapsed_time = time.time() - start_time
    print(f"Price data download completed in {elapsed_time:.2f} seconds")
    
    # Fix the column flattening
    data_flat = data.copy()
    data_flat.columns = [f"{t}_{p}" for (t, p) in data_flat.columns]
    data_flat.reset_index(inplace=True)
    
    # Fix data types before saving to SQL
    for col in data_flat.columns:
        if col.endswith('_Volume'):
            data_flat[col] = pd.to_numeric(data_flat[col], errors='coerce').fillna(0).astype('Int64')
        elif col.endswith(('_Open', '_High', '_Low', '_Close')):
            data_flat[col] = pd.to_numeric(data_flat[col], errors='coerce').fillna(0).astype('float64')
    
    data_flat.to_csv('data_flat.csv', index=False)
    return True

if __name__ == '__main__':
    collect_equities_metadata()