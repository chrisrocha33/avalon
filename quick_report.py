# ============================================
# Quick Report Module
# - Handles stock analysis and report generation
# - Separated from main app.py for better organization
# - Uses full dataset for calculations but limits plots to last 2 years
# ============================================

# Standard library imports
import io
import base64
import warnings
from datetime import datetime

# Third-party imports
import pandas as pd
import numpy as np
from flask import render_template
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt
import seaborn as sns
from utils import apply_dashboard_plot_style

# Suppress warnings
warnings.filterwarnings("ignore")
apply_dashboard_plot_style()

def generate_quick_report(ticker, db_manager=None):
    """Generate quick stock analysis report for a given ticker"""
    try:
        # Check for required dependencies
        try:
            import yfinance as yf
        except ImportError:
            return "Error: yfinance package not installed. Please install it with: pip install yfinance", 500
        
        print(f"📊 Generating quick report for {ticker.upper()}...")
        
        # Log database manager status
        if db_manager:
            print(f"🔗 Using database manager: {type(db_manager).__name__}")
            if db_manager.health_check():
                print("✅ Database connection healthy")
            else:
                print("⚠️ Database connection unhealthy")
        else:
            print("ℹ️ No database manager provided - running in standalone mode")
        
        # Try to get historical data from database first
        stock_data = None
        if db_manager and db_manager.health_check():
            try:
                stock_data = get_stock_data_from_db(ticker, db_manager)
                if stock_data is not None and not stock_data.empty:
                    print(f"📊 Retrieved {len(stock_data)} records from database for {ticker.upper()}")
                    # Check if data is recent (within last 24 hours)
                    latest_date = stock_data.index.max()
                    if pd.Timestamp.now() - latest_date < pd.Timedelta(hours=24):
                        print(f"✅ Database data is recent (latest: {latest_date.strftime('%Y-%m-%d')})")
                    else:
                        print(f"⚠️ Database data is stale (latest: {latest_date.strftime('%Y-%m-%d')}), will download fresh data")
                        stock_data = None
                else:
                    print(f"ℹ️ No data found in database for {ticker.upper()}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to retrieve data from database: {str(e)}")
                stock_data = None
        
        # Download stock data if not available from database
        if stock_data is None or stock_data.empty:
            print(f"📥 Downloading fresh data from Yahoo Finance for {ticker.upper()}...")
            stock_ticker = yf.Ticker(ticker.upper())
            stock_data = yf.download(ticker.upper(), period='max', interval='1d', auto_adjust=True)
            if stock_data.empty:
                return f"Error: No data found for ticker {ticker.upper()}", 404
        else:
            print(f"📊 Using cached data from database for {ticker.upper()}")
            stock_ticker = yf.Ticker(ticker.upper())  # Still need ticker object for options/fundamentals
            
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)
        
        # Save stock data to database if database manager is provided
        if db_manager:
            try:
                save_stock_data_to_db(stock_data, ticker, db_manager)
                print(f"💾 Stock data saved to database for {ticker.upper()}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to save stock data to database: {str(e)}")
        
        # Download benchmark data (SPY)
        benchmark = yf.download('SPY', period='max', interval='1d', auto_adjust=True)['Close'].pct_change().dropna()
        
        # Calculate daily returns
        stock_data['daily_return'] = stock_data['Close'].pct_change()
        
        # Calculate technical indicators efficiently
        rolling_windows = [20, 60, 100, 200]
        
        # Moving averages and EMAs
        for i in rolling_windows:
            stock_data[f'{i}_ma'] = stock_data['Close'].rolling(window=i).mean()
            stock_data[f'{i}_ema'] = stock_data['Close'].ewm(span=i, adjust=False).mean()
            stock_data[f'{i}_std'] = stock_data['daily_return'].rolling(window=i).std()
            stock_data[f'{i}_beta'] = stock_data['daily_return'].rolling(i).cov(benchmark) / benchmark.rolling(i).var()
        
        # MACD
        stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
        stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
        stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
        stock_data['MACD_signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
        stock_data['MACD_histogram'] = stock_data['MACD'] - stock_data['MACD_signal']
        
        # RSI
        delta = stock_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        stock_data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        stock_data['BB_middle'] = stock_data['Close'].rolling(window=20).mean()
        bb_std = stock_data['Close'].rolling(window=20).std()
        stock_data['BB_upper'] = stock_data['BB_middle'] + (bb_std * 2)
        stock_data['BB_lower'] = stock_data['BB_middle'] - (bb_std * 2)
        stock_data['BB_width'] = (stock_data['BB_upper'] - stock_data['BB_lower']) / stock_data['BB_middle']
        stock_data['BB_position'] = (stock_data['Close'] - stock_data['BB_lower']) / (stock_data['BB_upper'] - stock_data['BB_lower'])
        
        # Volatility indicators
        for i in rolling_windows:
            stock_data[f'{i}_volatility'] = stock_data['daily_return'].rolling(window=i).std() * np.sqrt(252)
        
        # Support and Resistance
        stock_data['Support_20'] = stock_data['Low'].rolling(window=20).min()
        stock_data['Resistance_20'] = stock_data['High'].rolling(window=20).max()
        
        # Calculate advanced technical indicators
        stock_data = calculate_advanced_metrics(stock_data)
        
        # Save technical indicators to database if database manager is provided
        if db_manager:
            try:
                save_technical_indicators_to_db(stock_data, ticker, db_manager)
                print(f"💾 Technical indicators saved to database for {ticker.upper()}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to save technical indicators to database: {str(e)}")
        
        # Generate risk analysis
        risk_metrics = generate_risk_analysis(stock_data)
        
        # Save risk analysis to database if database manager is provided
        if db_manager:
            try:
                save_risk_analysis_to_db(risk_metrics, ticker, db_manager)
                print(f"💾 Risk analysis saved to database for {ticker.upper()}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to save risk analysis to database: {str(e)}")
        
        # Analyze market sentiment
        sentiment_data = analyze_market_sentiment(ticker)
        
        # Generate portfolio recommendations
        portfolio_recs = generate_portfolio_recommendations(ticker)
        
        # Save sentiment and portfolio analysis to database if database manager is provided
        if db_manager:
            try:
                save_sentiment_portfolio_to_db(sentiment_data, portfolio_recs, ticker, db_manager)
                print(f"💾 Sentiment and portfolio analysis saved to database for {ticker.upper()}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to save sentiment and portfolio analysis to database: {str(e)}")
        
        # Get options data
        try:
            call_options = stock_ticker.option_chain().calls
            put_options = stock_ticker.option_chain().puts
            has_options = True
            
            # Save options data to database if database manager is provided
            if db_manager:
                try:
                    save_options_data_to_db(call_options, put_options, ticker, db_manager)
                    print(f"💾 Options data saved to database for {ticker.upper()}")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to save options data to database: {str(e)}")
                    
        except:
            call_options = pd.DataFrame()
            put_options = pd.DataFrame()
            has_options = False
        
        # Get fundamental data
        # Note: yfinance returns data with most recent dates as FIRST column (index 0)
        try:
            balance_sheet = stock_ticker.balance_sheet
            cash_flow = stock_ticker.cash_flow
            income_statement = stock_ticker.income_stmt
            has_fundamentals = True
            
            # Save fundamental data to database if database manager is provided
            if db_manager:
                try:
                    save_fundamental_data_to_db(balance_sheet, cash_flow, income_statement, ticker, db_manager)
                    print(f"💾 Fundamental data saved to database for {ticker.upper()}")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to save fundamental data to database: {str(e)}")
                    
        except:
            balance_sheet = pd.DataFrame()
            cash_flow = pd.DataFrame()
            income_statement = pd.DataFrame()
            has_fundamentals = False
        
        # Generate plots efficiently
        plots = {}
        
        # Limit data to last 2 years for plotting (keep full data for calculations)
        plot_data = stock_data.tail(504)  # ~2 years (252 trading days per year)
        
        # 1. Price and Moving Averages
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(plot_data.index, plot_data['Close'], label='Close Price', linewidth=1)
        for ma in ['20_ma', '60_ma', '100_ma', '200_ma']:
            ax.plot(plot_data.index, plot_data[ma], label=ma, alpha=0.7)
        ax.set_title(f'{ticker.upper()} - Price and Moving Averages (Last 2 Years)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plots['price_ma'] = fig_to_base64(fig)
        plt.close(fig)
        
        # 2. Technical Indicators Grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # RSI
        axes[0,0].plot(plot_data.index, plot_data['RSI'], color='purple', alpha=0.7)
        axes[0,0].axhline(y=70, color='r', linestyle='--', alpha=0.5)
        axes[0,0].axhline(y=30, color='g', linestyle='--', alpha=0.5)
        axes[0,0].set_title('RSI (Last 2 Years)')
        axes[0,0].set_ylim(0, 100)
        axes[0,0].grid(True, alpha=0.3)
        
        # MACD
        axes[0,1].plot(plot_data.index, plot_data['MACD'], label='MACD', alpha=0.7)
        axes[0,1].plot(plot_data.index, plot_data['MACD_signal'], label='Signal', alpha=0.7)
        axes[0,1].bar(plot_data.index, plot_data['MACD_histogram'], label='Histogram', alpha=0.5)
        axes[0,1].set_title('MACD (Last 2 Years)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Bollinger Bands
        axes[1,0].plot(plot_data.index, plot_data['Close'], label='Close', alpha=0.7)
        axes[1,0].plot(plot_data.index, plot_data['BB_upper'], label='Upper BB', alpha=0.7)
        axes[1,0].plot(plot_data.index, plot_data['BB_middle'], label='Middle BB', alpha=0.7)
        axes[1,0].plot(plot_data.index, plot_data['BB_lower'], label='Lower BB', alpha=0.7)
        axes[1,0].fill_between(plot_data.index, plot_data['BB_upper'], plot_data['BB_lower'], alpha=0.1)
        axes[1,0].set_title('Bollinger Bands (Last 2 Years)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Volatility
        for vol in ['20_volatility', '60_volatility', '100_volatility', '200_volatility']:
            axes[1,1].plot(plot_data.index, plot_data[vol], label=vol, alpha=0.7)
        axes[1,1].set_title('Volatility (Annualized) - Last 2 Years')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plots['technical_indicators'] = fig_to_base64(fig)
        plt.close(fig)
        
        # 3. Advanced Technical Indicators
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Williams %R
        axes[0,0].plot(plot_data.index, plot_data['Williams_R'], color='orange', alpha=0.7)
        axes[0,0].axhline(y=-20, color='r', linestyle='--', alpha=0.5)
        axes[0,0].axhline(y=-80, color='g', linestyle='--', alpha=0.5)
        axes[0,0].set_title('Williams %R (Last 2 Years)')
        axes[0,0].set_ylim(-100, 0)
        axes[0,0].grid(True, alpha=0.3)
        
        # Stochastic Oscillator
        axes[0,1].plot(plot_data.index, plot_data['Stoch_K'], label='%K', color='blue', alpha=0.7)
        axes[0,1].plot(plot_data.index, plot_data['Stoch_D'], label='%D', color='red', alpha=0.7)
        axes[0,1].axhline(y=80, color='r', linestyle='--', alpha=0.5)
        axes[0,1].axhline(y=20, color='g', linestyle='--', alpha=0.5)
        axes[0,1].set_title('Stochastic Oscillator (Last 2 Years)')
        axes[0,1].set_ylim(0, 100)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # ATR
        axes[1,0].plot(plot_data.index, plot_data['ATR'], color='purple', alpha=0.7)
        axes[1,0].set_title('Average True Range (ATR) - Last 2 Years')
        axes[1,0].grid(True, alpha=0.3)
        
        # CCI
        axes[1,1].plot(plot_data.index, plot_data['CCI'], color='brown', alpha=0.7)
        axes[1,1].axhline(y=100, color='r', linestyle='--', alpha=0.5)
        axes[1,1].axhline(y=-100, color='g', linestyle='--', alpha=0.5)
        axes[1,1].set_title('Commodity Channel Index (CCI) - Last 2 Years')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plots['advanced_indicators'] = fig_to_base64(fig)
        plt.close(fig)
        
        # 4. Options Analysis (if available)
        if has_options and (not call_options.empty or not put_options.empty):
            # Main Options Analysis - Combined Calls and Puts
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            
            # Volatility Smile - Combined Calls and Puts
            if not call_options.empty:
                axes[0,0].scatter(call_options['strike'], call_options['impliedVolatility'], 
                                  s=80, alpha=0.7, c='blue', label='Call Options', marker='o')
                if len(call_options) > 2:
                    z = np.polyfit(call_options['strike'], call_options['impliedVolatility'], 2)
                    p = np.poly1d(z)
                    axes[0,0].plot(call_options['strike'], p(call_options['strike']), "b--", alpha=0.8, linewidth=2)
            
            if not put_options.empty:
                axes[0,0].scatter(put_options['strike'], put_options['impliedVolatility'], 
                                  s=80, alpha=0.7, c='red', label='Put Options', marker='s')
                if len(put_options) > 2:
                    z = np.polyfit(put_options['strike'], put_options['impliedVolatility'], 2)
                    p = np.poly1d(z)
                    axes[0,0].plot(put_options['strike'], p(put_options['strike']), "r--", alpha=0.8, linewidth=2)
            
            axes[0,0].set_xlabel('Strike Price ($)')
            axes[0,0].set_ylabel('Implied Volatility')
            axes[0,0].set_title('Volatility Smile - Calls vs Puts')
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].legend()
            
            # Options Pricing - Combined
            if not call_options.empty:
                axes[0,1].scatter(call_options['strike'], call_options['bid'], s=60, alpha=0.8, 
                                  c='lightgreen', label='Call Bid', marker='o')
                axes[0,1].scatter(call_options['strike'], call_options['ask'], s=60, alpha=0.8, 
                                  c='darkgreen', label='Call Ask', marker='o')
                axes[0,1].scatter(call_options['strike'], call_options['lastPrice'], s=80, alpha=0.9, 
                                  c='blue', label='Call Last', marker='^')
            
            if not put_options.empty:
                axes[0,1].scatter(put_options['strike'], put_options['bid'], s=60, alpha=0.8, 
                                  c='lightcoral', label='Put Bid', marker='s')
                axes[0,1].scatter(put_options['strike'], put_options['ask'], s=60, alpha=0.8, 
                                  c='darkred', label='Put Ask', marker='s')
                axes[0,1].scatter(put_options['strike'], put_options['lastPrice'], s=80, alpha=0.9, 
                                  c='red', label='Put Last', marker='v')
            
            axes[0,1].set_xlabel('Strike Price ($)')
            axes[0,1].set_ylabel('Option Price ($)')
            axes[0,1].set_title('Options Chain Pricing - Calls vs Puts')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].legend()
            
            # Volume Analysis - Combined
            if not call_options.empty:
                axes[1,0].bar(call_options['strike'], call_options['volume'], alpha=0.7, 
                               color='skyblue', edgecolor='navy', label='Call Volume')
            if not put_options.empty:
                axes[1,0].bar(put_options['strike'], put_options['volume'], alpha=0.7, 
                               color='lightcoral', edgecolor='darkred', label='Put Volume')
            
            axes[1,0].set_xlabel('Strike Price ($)')
            axes[1,0].set_ylabel('Volume')
            axes[1,0].set_title('Trading Volume - Calls vs Puts')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].legend()
            
            # Open Interest - Combined
            if not call_options.empty:
                axes[1,1].bar(call_options['strike'], call_options['openInterest'], alpha=0.7, 
                               color='lightblue', edgecolor='blue', label='Call OI')
            if not put_options.empty:
                axes[1,1].bar(put_options['strike'], put_options['openInterest'], alpha=0.7, 
                               color='lightpink', edgecolor='red', label='Put OI')
            
            axes[1,1].set_xlabel('Strike Price ($)')
            axes[1,1].set_ylabel('Open Interest')
            axes[1,1].set_title('Open Interest - Calls vs Puts')
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].legend()
            
            plt.tight_layout()
            plots['options_analysis'] = fig_to_base64(fig)
            plt.close(fig)
            
            # Call Options Analysis - Dedicated Section
            if not call_options.empty:
                fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                
                # Call Options Volatility Analysis
                axes[0,0].scatter(call_options['strike'], call_options['impliedVolatility'], 
                                  s=80, alpha=0.7, c='blue', label='Call IV')
                if len(call_options) > 2:
                    z = np.polyfit(call_options['strike'], call_options['impliedVolatility'], 2)
                    p = np.poly1d(z)
                    axes[0,0].plot(call_options['strike'], p(call_options['strike']), "b--", alpha=0.8, linewidth=2)
                axes[0,0].set_xlabel('Strike Price ($)')
                axes[0,0].set_ylabel('Implied Volatility')
                axes[0,0].set_title('Call Options Volatility Skew')
                axes[0,0].grid(True, alpha=0.3)
                axes[0,0].legend()
                
                # Call Options Pricing Analysis
                axes[0,1].scatter(call_options['strike'], call_options['bid'], s=60, alpha=0.8, 
                                  c='lightgreen', label='Bid', marker='o')
                axes[0,1].scatter(call_options['strike'], call_options['ask'], s=60, alpha=0.8, 
                                  c='darkgreen', label='Ask', marker='o')
                axes[0,1].scatter(call_options['strike'], call_options['lastPrice'], s=80, alpha=0.9, 
                                  c='blue', label='Last', marker='^')
                axes[0,1].set_xlabel('Strike Price ($)')
                axes[0,1].set_ylabel('Option Price ($)')
                axes[0,1].set_title('Call Options Pricing')
                axes[0,1].grid(True, alpha=0.3)
                axes[0,1].legend()
                
                # Call Options Volume Analysis
                axes[1,0].bar(call_options['strike'], call_options['volume'], alpha=0.7, 
                               color='skyblue', edgecolor='navy')
                axes[1,0].set_xlabel('Strike Price ($)')
                axes[1,0].set_ylabel('Volume')
                axes[1,0].set_title('Call Options Trading Volume')
                axes[1,0].grid(True, alpha=0.3)
                
                # Call Options Open Interest Analysis
                axes[1,1].bar(call_options['strike'], call_options['openInterest'], alpha=0.7, 
                               color='lightblue', edgecolor='blue')
                axes[1,1].set_xlabel('Strike Price ($)')
                axes[1,1].set_ylabel('Open Interest')
                axes[1,1].set_title('Call Options Open Interest')
                axes[1,1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plots['call_options_analysis'] = fig_to_base64(fig)
                plt.close(fig)
            
            # Put Options Analysis - Dedicated Section
            if not put_options.empty:
                fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                
                # Put Options Volatility Analysis
                axes[0,0].scatter(put_options['strike'], put_options['impliedVolatility'], 
                                  s=80, alpha=0.7, c='red', label='Put IV')
                if len(put_options) > 2:
                    z = np.polyfit(put_options['strike'], put_options['impliedVolatility'], 2)
                    p = np.poly1d(z)
                    axes[0,0].plot(put_options['strike'], p(put_options['strike']), "r--", alpha=0.8, linewidth=2)
                axes[0,0].set_xlabel('Strike Price ($)')
                axes[0,0].set_ylabel('Implied Volatility')
                axes[0,0].set_title('Put Options Volatility Skew')
                axes[0,0].grid(True, alpha=0.3)
                axes[0,0].legend()
                
                # Put Options Pricing Analysis
                axes[0,1].scatter(put_options['strike'], put_options['bid'], s=60, alpha=0.8, 
                                  c='lightcoral', label='Bid', marker='s')
                axes[0,1].scatter(put_options['strike'], put_options['ask'], s=60, alpha=0.8, 
                                  c='darkred', label='Ask', marker='s')
                axes[0,1].scatter(put_options['strike'], put_options['lastPrice'], s=80, alpha=0.9, 
                                  c='red', label='Last', marker='v')
                axes[0,1].set_xlabel('Strike Price ($)')
                axes[0,1].set_ylabel('Option Price ($)')
                axes[0,1].set_title('Put Options Pricing')
                axes[0,1].grid(True, alpha=0.3)
                axes[0,1].legend()
                
                # Put Options Volume Analysis
                axes[1,0].bar(put_options['strike'], put_options['volume'], alpha=0.7, 
                               color='lightcoral', edgecolor='darkred')
                axes[1,0].set_xlabel('Strike Price ($)')
                axes[1,0].set_ylabel('Volume')
                axes[1,0].set_title('Put Options Trading Volume')
                axes[1,0].grid(True, alpha=0.3)
                
                # Put Options Open Interest Analysis
                axes[1,1].bar(put_options['strike'], put_options['openInterest'], alpha=0.7, 
                               color='lightpink', edgecolor='red')
                axes[1,1].set_xlabel('Strike Price ($)')
                axes[1,1].set_ylabel('Open Interest')
                axes[1,1].set_title('Put Options Open Interest')
                axes[1,1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plots['put_options_analysis'] = fig_to_base64(fig)
                plt.close(fig)
            
            # Advanced Comparative Analysis - Both Calls and Puts
            if not call_options.empty and not put_options.empty:
                fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                
                # Put-Call Volume Ratio Analysis
                put_call_ratio = put_options['volume'].sum() / call_options['volume'].sum() if call_options['volume'].sum() > 0 else 0
                
                # Volume Comparison by Strike
                axes[0,0].bar(call_options['strike'], call_options['volume'], alpha=0.7, 
                               color='skyblue', edgecolor='navy', label='Call Volume')
                axes[0,0].bar(put_options['strike'], put_options['volume'], alpha=0.7, 
                               color='lightcoral', edgecolor='darkred', label='Put Volume')
                axes[0,0].set_xlabel('Strike Price ($)')
                axes[0,0].set_ylabel('Volume')
                axes[0,0].set_title(f'Volume Comparison (P/C Ratio: {put_call_ratio:.2f})')
                axes[0,0].grid(True, alpha=0.3)
                axes[0,0].legend()
                
                # Put-Call Open Interest Ratio Analysis
                put_call_oi_ratio = put_options['openInterest'].sum() / call_options['openInterest'].sum() if call_options['openInterest'].sum() > 0 else 0
                
                # Open Interest Comparison by Strike
                axes[0,1].bar(call_options['strike'], call_options['openInterest'], alpha=0.7, 
                               color='lightblue', edgecolor='blue', label='Call OI')
                axes[0,1].bar(put_options['strike'], put_options['openInterest'], alpha=0.7, 
                               color='lightpink', edgecolor='red', label='Put OI')
                axes[0,1].set_xlabel('Strike Price ($)')
                axes[0,1].set_ylabel('Open Interest')
                axes[0,1].set_title(f'Open Interest Comparison (P/C OI Ratio: {put_call_oi_ratio:.2f})')
                axes[0,1].grid(True, alpha=0.3)
                axes[0,1].legend()
                
                # Volatility Comparison
                axes[1,0].scatter(call_options['strike'], call_options['impliedVolatility'], 
                                  s=80, alpha=0.7, c='blue', label='Call IV', marker='o')
                axes[1,0].scatter(put_options['strike'], put_options['impliedVolatility'], 
                                  s=80, alpha=0.7, c='red', label='Put IV', marker='s')
                if len(call_options) > 2 and len(put_options) > 2:
                    # Fit trend lines
                    z_call = np.polyfit(call_options['strike'], call_options['impliedVolatility'], 2)
                    p_call = np.poly1d(z_call)
                    z_put = np.polyfit(put_options['strike'], put_options['impliedVolatility'], 2)
                    p_put = np.poly1d(z_put)
                    
                    axes[1,0].plot(call_options['strike'], p_call(call_options['strike']), "b--", alpha=0.8, linewidth=2)
                    axes[1,0].plot(put_options['strike'], p_put(put_options['strike']), "r--", alpha=0.8, linewidth=2)
                
                axes[1,0].set_xlabel('Strike Price ($)')
                axes[1,0].set_ylabel('Implied Volatility')
                axes[1,0].set_title('Volatility Comparison - Calls vs Puts')
                axes[1,0].grid(True, alpha=0.3)
                axes[1,0].legend()
                
                # Moneyness Analysis
                if 'inTheMoney' in call_options.columns and 'inTheMoney' in put_options.columns:
                    itm_calls = call_options[call_options['inTheMoney'] == True]
                    otm_calls = call_options[call_options['inTheMoney'] == False]
                    itm_puts = put_options[put_options['inTheMoney'] == True]
                    otm_puts = put_options[put_options['inTheMoney'] == False]
                    
                    if not itm_calls.empty:
                        axes[1,1].scatter(itm_calls['strike'], itm_calls['lastPrice'], 
                                          s=100, alpha=0.8, c='green', label='ITM Calls', marker='o')
                    if not otm_calls.empty:
                        axes[1,1].scatter(otm_calls['strike'], otm_calls['lastPrice'], 
                                          s=80, alpha=0.6, c='blue', label='OTM Calls', marker='o')
                    if not itm_puts.empty:
                        axes[1,1].scatter(itm_puts['strike'], itm_puts['lastPrice'], 
                                          s=100, alpha=0.8, c='darkgreen', label='ITM Puts', marker='s')
                    if not otm_puts.empty:
                        axes[1,1].scatter(otm_puts['strike'], otm_puts['lastPrice'], 
                                          s=80, alpha=0.6, c='red', label='OTM Puts', marker='s')
                    
                    axes[1,1].set_xlabel('Strike Price ($)')
                    axes[1,1].set_ylabel('Option Price ($)')
                    axes[1,1].set_title('Moneyness Analysis - ITM vs OTM')
                    axes[1,1].grid(True, alpha=0.3)
                    axes[1,1].legend()
                else:
                    # Fallback: Price comparison
                    axes[1,1].scatter(call_options['strike'], call_options['lastPrice'], 
                                      s=80, alpha=0.7, c='blue', label='Call Last', marker='o')
                    axes[1,1].scatter(put_options['strike'], put_options['lastPrice'], 
                                      s=80, alpha=0.7, c='red', label='Put Last', marker='s')
                    axes[1,1].set_xlabel('Strike Price ($)')
                    axes[1,1].set_ylabel('Option Price ($)')
                    axes[1,1].set_title('Price Comparison - Calls vs Puts')
                    axes[1,1].grid(True, alpha=0.3)
                    axes[1,1].legend()
                
                plt.tight_layout()
                plots['advanced_options_analysis'] = fig_to_base64(fig)
                plt.close(fig)
        
        # 5. Fundamental Ratios (if available)
        if has_fundamentals:
            ratios = {}
            
            # Solvency ratios
            if 'Current Assets' in balance_sheet.index and 'Current Liabilities' in balance_sheet.index:
                ratios['Current Ratio'] = balance_sheet.loc['Current Assets'] / balance_sheet.loc['Current Liabilities']
            
            if all(x in balance_sheet.index for x in ['Current Assets', 'Inventory', 'Current Liabilities']):
                ratios['Quick Ratio'] = (balance_sheet.loc['Current Assets'] - balance_sheet.loc['Inventory']) / balance_sheet.loc['Current Liabilities']
            
            # Profitability margins
            if all(x in income_statement.index for x in ['Gross Profit', 'Total Revenue']):
                ratios['Gross Margin'] = income_statement.loc['Gross Profit'] / income_statement.loc['Total Revenue']
            
            if all(x in income_statement.index for x in ['Operating Income', 'Total Revenue']):
                ratios['Operating Margin'] = income_statement.loc['Operating Income'] / income_statement.loc['Total Revenue']
            
            if all(x in income_statement.index for x in ['Net Income', 'Total Revenue']):
                ratios['Net Margin'] = income_statement.loc['Net Income'] / income_statement.loc['Total Revenue']
            
            # Leverage ratios
            if all(x in balance_sheet.index for x in ['Total Debt', 'Stockholders Equity']):
                ratios['Debt-to-Equity'] = balance_sheet.loc['Total Debt'] / balance_sheet.loc['Stockholders Equity']
            
            # ROA and ROE
            if all(x in balance_sheet.index for x in ['Total Assets']) and 'Net Income' in income_statement.index:
                ratios['ROA'] = income_statement.loc['Net Income'] / balance_sheet.loc['Total Assets']
            
            if all(x in balance_sheet.index for x in ['Stockholders Equity']) and 'Net Income' in income_statement.index:
                ratios['ROE'] = income_statement.loc['Net Income'] / balance_sheet.loc['Stockholders Equity']
            
            # Convert to DataFrame and clean up
            ratios_df = pd.DataFrame.from_dict(ratios, orient='index')
            ratios_df = ratios_df.replace([np.inf, -np.inf], np.nan)
            ratios_df = ratios_df.round(4)
            
            # Create fundamental ratios plot
            if not ratios_df.empty:
                fig, ax = plt.subplots(figsize=(12, 8))
                ratios_df.plot(kind='bar', ax=ax, color='skyblue', alpha=0.7)
                ax.set_title(f'{ticker.upper()} - Key Financial Ratios')
                ax.set_xlabel('Ratio')
                ax.set_ylabel('Value')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plots['fundamental_ratios'] = fig_to_base64(fig)
                plt.close(fig)
            
            # Create revenue and margins plot
            try:
                # Check if we have the required data for revenue and margins analysis
                required_metrics = ['Total Revenue', 'Cost Of Revenue', 'Operating Income', 'Pretax Income', 'Tax Provision', 'Net Income']
                if all(metric in income_statement.index for metric in required_metrics):
                    fig, ax1 = plt.subplots(figsize=(14, 8))
                    
                    # Get the datetime index from the income statement
                    dates = income_statement.columns
                    
                    # Calculate margins
                    cost_margin = (income_statement.loc['Cost Of Revenue'] / income_statement.loc['Total Revenue']) * 100
                    operating_margin = (income_statement.loc['Operating Income'] / income_statement.loc['Total Revenue']) * 100
                    
                    # Handle potential division by zero for NOPAT margin
                    if income_statement.loc['Pretax Income'].sum() != 0:
                        tax_rate = income_statement.loc['Tax Provision'] / income_statement.loc['Pretax Income']
                        nopat_margin = ((income_statement.loc['Operating Income'] * (1 - tax_rate)) / income_statement.loc['Total Revenue']) * 100
                    else:
                        nopat_margin = pd.Series([0] * len(dates), index=dates)
                    
                    net_margin = (income_statement.loc['Net Income'] / income_statement.loc['Total Revenue']) * 100
                    
                    # Plot revenue as bars on primary y-axis
                    bars = ax1.plot(dates, income_statement.loc['Total Revenue'], alpha=0.7, color='skyblue', label='Total Revenue')
                    ax1.set_ylabel('Revenue ($)', color='blue', fontsize=12)
                    ax1.tick_params(axis='y', labelcolor='blue')
                    
                    # Create secondary y-axis for margins
                    ax2 = ax1.twinx()
                    
                    # Plot margins as lines on secondary y-axis
                    ax2.plot(dates, cost_margin, 'o-', color='red', linewidth=2, markersize=8, label='Cost of Revenue Margin')
                    ax2.plot(dates, operating_margin, 's-', color='green', linewidth=2, markersize=8, label='Operating Margin')
                    ax2.plot(dates, nopat_margin, '^-', color='purple', linewidth=2, markersize=8, label='NOPAT Margin')
                    ax2.plot(dates, net_margin, 'd-', color='orange', linewidth=2, markersize=8, label='Net Income Margin')
                    
                    ax2.set_ylabel('Margin (%)', color='red', fontsize=12)
                    ax2.tick_params(axis='y', labelcolor='red')
                    ax2.grid(True, alpha=0.3)
                    
                    # Customize the plot
                    ax1.set_title(f'{ticker.upper()} - Revenue and Margin Analysis', fontsize=16, fontweight='bold')
                    ax1.set_xlabel('Year', fontsize=12)
                    
                    # Rotate x-axis labels for better readability
                    ax1.tick_params(axis='x', rotation=45)
                    
                    # Add legends
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0, 1))
                    
                    plt.tight_layout()
                    plots['revenue_margins'] = fig_to_base64(fig)
                    plt.close(fig)
                    
                    print(f"✅ Revenue and margins plot generated for {ticker.upper()}")
                else:
                    print(f"⚠️  Missing required metrics for revenue and margins plot for {ticker.upper()}")
            except Exception as e:
                print(f"⚠️  Error generating revenue and margins plot for {ticker.upper()}: {e}")
            
            # Create leverage ratios plot
            try:
                # Check if we have the required data for leverage ratios analysis
                required_balance_metrics = ['Total Debt', 'Stockholders Equity', 'Total Assets', 'Long Term Debt']
                required_income_metrics = ['Operating Income']
                
                if all(metric in balance_sheet.index for metric in required_balance_metrics) and all(metric in income_statement.index for metric in required_income_metrics):
                    fig, ax = plt.subplots(figsize=(14, 8))
                    
                    # Get the datetime index from the balance sheet
                    dates = balance_sheet.columns
                    
                    # Calculate leverage ratios
                    debt_to_equity = balance_sheet.loc['Total Debt'] / balance_sheet.loc['Stockholders Equity']
                    debt_to_assets = balance_sheet.loc['Total Debt'] / balance_sheet.loc['Total Assets']
                    long_term_debt_to_equity = balance_sheet.loc['Long Term Debt'] / balance_sheet.loc['Stockholders Equity']
                    debt_to_capital = balance_sheet.loc['Total Debt'] / (balance_sheet.loc['Total Debt'] + balance_sheet.loc['Stockholders Equity'])
                    
                    # Calculate interest coverage if available
                    interest_coverage = None
                    if 'Interest Paid Cfo' in income_statement.index:
                        # Handle potential division by zero
                        interest_paid = income_statement.loc['Interest Paid Cfo']
                        interest_coverage = income_statement.loc['Operating Income'] / abs(interest_paid) if interest_paid.sum() != 0 else None
                    
                    # Plot all leverage ratios
                    ax.plot(dates, debt_to_equity, 'o-', color='red', linewidth=2, markersize=8, label='Debt-to-Equity')
                    ax.plot(dates, debt_to_assets, 's-', color='blue', linewidth=2, markersize=8, label='Debt-to-Assets')
                    ax.plot(dates, long_term_debt_to_equity, '^-', color='green', linewidth=2, markersize=8, label='Long-term Debt-to-Equity')
                    ax.plot(dates, debt_to_capital, 'd-', color='purple', linewidth=2, markersize=8, label='Debt-to-Capital')
                    
                    # Add interest coverage if available
                    if interest_coverage is not None:
                        ax.plot(dates, interest_coverage, 'v-', color='orange', linewidth=2, markersize=8, label='Interest Coverage')
                    
                    # Customize the plot
                    ax.set_title(f'{ticker.upper()} - Leverage Ratios Analysis', fontsize=16, fontweight='bold')
                    ax.set_xlabel('Year', fontsize=12)
                    ax.set_ylabel('Ratio Value', fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper left')
                    
                    # Rotate x-axis labels for better readability
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Add horizontal reference lines for common thresholds
                    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Debt-to-Equity = 1.0')
                    ax.axhline(y=0.5, color='blue', linestyle='--', alpha=0.5, label='Debt-to-Assets = 0.5')
                    
                    plt.tight_layout()
                    plots['leverage_ratios'] = fig_to_base64(fig)
                    plt.close(fig)
                    
                    print(f"✅ Leverage ratios plot generated for {ticker.upper()}")
                else:
                    print(f"⚠️  Missing required metrics for leverage ratios plot for {ticker.upper()}")
            except Exception as e:
                print(f"⚠️  Error generating leverage ratios plot for {ticker.upper()}: {e}")
        
        # Create CAPEX analysis plot
        try:
            # Check if we have the required data for CAPEX analysis
            if 'Capital Expenditure' in cash_flow.index and not cash_flow.empty:
                fig, ax1 = plt.subplots(figsize=(14, 8))
                
                # Get the datetime index from the cash flow statement
                dates = cash_flow.columns
                
                # Get CAPEX values (negative values, so we'll make them positive for visualization)
                capex_values = abs(cash_flow.loc['Capital Expenditure'])
                
                # Calculate CAPEX growth rate and handle NaN
                capex_growth = capex_values.pct_change(fill_method=None) * 100
                
                # Create a clean annual index for bars (extract just the year)
                annual_dates = [pd.Timestamp(date).year for date in dates]
                annual_dates_str = [str(year) for year in annual_dates]
                
                # Plot CAPEX values as bars using clean annual index
                bars = ax1.bar(annual_dates_str, capex_values, alpha=0.7, color='darkblue', label='Capital Expenditure', width=0.6)
                ax1.set_ylabel('CAPEX ($)', color='darkblue', fontsize=12)
                ax1.tick_params(axis='y', labelcolor='darkblue')
                
                # Create secondary y-axis for growth rate
                ax2 = ax1.twinx()
                
                # Convert dates to years for the growth rate line to match the bar x-axis
                # Skip first NaN value
                valid_years = annual_dates_str[1:]  # Skip first year since growth rate will be NaN
                valid_growth = capex_growth[1:]  # Skip first NaN value
                
                ax2.plot(valid_years, valid_growth, 'o-', color='red', linewidth=2, markersize=8, label='CAPEX Growth Rate (%)')
                ax2.set_ylabel('Growth Rate (%)', color='red', fontsize=12)
                ax2.tick_params(axis='y', labelcolor='red')
                ax2.grid(True, alpha=0.3)
                
                # Add horizontal reference line at 0% growth
                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No Growth')
                
                # Customize the plot
                ax1.set_title(f'{ticker.upper()} - Capital Expenditure Analysis', fontsize=16, fontweight='bold')
                ax1.set_xlabel('Year', fontsize=12)
                
                # Rotate x-axis labels for better readability
                ax1.tick_params(axis='x', rotation=45)
                
                # Add legends
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0, 1))
                
                plt.tight_layout()
                plots['capex_analysis'] = fig_to_base64(fig)
                plt.close(fig)
                
                print(f"✅ CAPEX analysis plot generated for {ticker.upper()}")
            else:
                print(f"⚠️  Missing required metrics for CAPEX analysis plot for {ticker.upper()}")
        except Exception as e:
            print(f"⚠️  Error generating CAPEX analysis plot for {ticker.upper()}: {e}")
        
        # Create liquidity ratios plot
        try:
            # Check if we have the required data for liquidity ratios analysis
            required_metrics = ['Current Assets', 'Current Liabilities', 'Inventory', 'Cash And Cash Equivalents', 'Working Capital', 'Total Assets']
            
            if all(metric in balance_sheet.index for metric in required_metrics) and not balance_sheet.empty:
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Get the datetime index from the balance sheet
                dates = balance_sheet.columns
                
                # Calculate liquidity ratios
                current_ratio = balance_sheet.loc['Current Assets'] / balance_sheet.loc['Current Liabilities']
                quick_ratio = (balance_sheet.loc['Current Assets'] - balance_sheet.loc['Inventory']) / balance_sheet.loc['Current Liabilities']
                cash_ratio = balance_sheet.loc['Cash And Cash Equivalents'] / balance_sheet.loc['Current Liabilities']
                working_capital_ratio = balance_sheet.loc['Working Capital'] / balance_sheet.loc['Total Assets']
                
                # Create a clean annual index for x-axis
                annual_dates = [pd.Timestamp(date).year for date in dates]
                annual_dates_str = [str(year) for year in annual_dates]
                
                # Plot all liquidity ratios
                ax.plot(annual_dates_str, current_ratio, 'o-', color='blue', linewidth=2, markersize=8, label='Current Ratio')
                ax.plot(annual_dates_str, quick_ratio, 's-', color='green', linewidth=2, markersize=8, label='Quick Ratio')
                ax.plot(annual_dates_str, cash_ratio, '^-', color='red', linewidth=2, markersize=8, label='Cash Ratio')
                ax.plot(annual_dates_str, working_capital_ratio, 'd-', color='purple', linewidth=2, markersize=8, label='Working Capital Ratio')
                
                # Customize the plot
                ax.set_title(f'{ticker.upper()} - Liquidity Ratios Analysis', fontsize=16, fontweight='bold')
                ax.set_xlabel('Year', fontsize=12)
                ax.set_ylabel('Ratio Value', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper left')
                
                # Rotate x-axis labels for better readability
                ax.tick_params(axis='x', rotation=45)
                
                # Add horizontal reference lines for common thresholds
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Current Ratio = 1.0')
                ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Quick Ratio = 0.5')
                ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Cash Ratio = 0.2')
                
                plt.tight_layout()
                plots['liquidity_ratios'] = fig_to_base64(fig)
                plt.close(fig)
                
                print(f"✅ Liquidity ratios plot generated for {ticker.upper()}")
            else:
                print(f"⚠️  Missing required metrics for liquidity ratios plot for {ticker.upper()}")
        except Exception as e:
            print(f"⚠️  Error generating liquidity ratios plot for {ticker.upper()}: {e}")
        
        # Summary statistics
        summary_stats = {
            'current_price': stock_data['Close'].iloc[-1],
            'price_change_1d': stock_data['daily_return'].iloc[-1],
            'price_change_1w': (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-6] - 1) if len(stock_data) > 6 else None,
            'price_change_1m': (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-22] - 1) if len(stock_data) > 22 else None,
            'volatility_20d': stock_data['20_volatility'].iloc[-1] if '20_volatility' in stock_data.columns else None,
            'rsi': stock_data['RSI'].iloc[-1] if 'RSI' in stock_data.columns else None,
            'beta_60d': stock_data['beta_60'].iloc[-1] if 'beta_60' in stock_data.columns else None,
            'has_options': has_options,
            'has_fundamentals': has_fundamentals
        }
        
        # Add fundamental summary statistics if available
        if has_fundamentals and not income_statement.empty:
            try:
                # IMPORTANT: yfinance returns data with most recent dates as FIRST column (index 0)
                # So we use .iloc[:, 0] for latest data, not .iloc[:, -1]
                latest_data = income_statement.iloc[:, 0]
                
                if 'Total Revenue' in latest_data.index:
                    summary_stats['latest_revenue'] = latest_data['Total Revenue']
                
                if all(metric in latest_data.index for metric in ['Operating Income', 'Total Revenue']):
                    summary_stats['latest_operating_margin'] = (latest_data['Operating Income'] / latest_data['Total Revenue']) * 100
                
                if all(metric in latest_data.index for metric in ['Net Income', 'Total Revenue']):
                    summary_stats['latest_net_margin'] = (latest_data['Net Income'] / latest_data['Total Revenue']) * 100
                
                # Calculate revenue growth if we have multiple periods
                # Note: Column 0 = most recent, Column 1 = previous period
                if income_statement.shape[1] >= 2:
                    current_revenue = income_statement.iloc[:, 0]['Total Revenue'] if 'Total Revenue' in income_statement.iloc[:, 0].index else None
                    previous_revenue = income_statement.iloc[:, 1]['Total Revenue'] if 'Total Revenue' in income_statement.iloc[:, 1].index else None
                    
                    if current_revenue and previous_revenue and previous_revenue != 0:
                        summary_stats['revenue_growth_yoy'] = ((current_revenue - previous_revenue) / previous_revenue) * 100
                
                # Add leverage ratio summary statistics if available
                if not balance_sheet.empty:
                    latest_balance = balance_sheet.iloc[:, 0]
                    
                                        
                    if all(metric in latest_balance.index for metric in ['Total Debt', 'Stockholders Equity']):
                        summary_stats['latest_debt_to_equity'] = latest_balance['Total Debt'] / latest_balance['Stockholders Equity']
                    
                    if all(metric in latest_balance.index for metric in ['Total Debt', 'Total Assets']):
                        summary_stats['latest_debt_to_assets'] = latest_balance['Total Debt'] / latest_balance['Total Assets']
                    
                    if all(metric in latest_balance.index for metric in ['Long Term Debt', 'Stockholders Equity']):
                        summary_stats['latest_long_term_debt_to_equity'] = latest_balance['Long Term Debt'] / latest_balance['Stockholders Equity']
                    
                # Calculate interest coverage if available
                if 'Interest Paid Cfo' in latest_data.index and latest_data['Interest Paid Cfo'] != 0:
                    summary_stats['latest_interest_coverage'] = latest_data['Operating Income'] / abs(latest_data['Interest Paid Cfo'])
                
                # Add CAPEX summary statistics if available
                if not cash_flow.empty:
                    latest_cash_flow = cash_flow.iloc[:, 0]
                                        
                    if 'Capital Expenditure' in latest_cash_flow.index:
                        summary_stats['latest_capex'] = abs(latest_cash_flow['Capital Expenditure'])
                    
                    # Calculate CAPEX growth if we have multiple periods
                    if cash_flow.shape[1] >= 2:
                        current_capex = abs(cash_flow.iloc[:, 0]['Capital Expenditure']) if 'Capital Expenditure' in cash_flow.iloc[:, 0].index else None
                        previous_capex = abs(cash_flow.iloc[:, 1]['Capital Expenditure']) if 'Capital Expenditure' in cash_flow.iloc[:, 1].index else None
                        
                        if current_capex and previous_capex and previous_capex != 0:
                            summary_stats['capex_growth_yoy'] = ((current_capex - previous_capex) / previous_capex) * 100
                    
                    # Add liquidity ratio summary statistics if available
                    if not balance_sheet.empty:
                        latest_balance = balance_sheet.iloc[:, 0]
                        
                        if all(metric in latest_balance.index for metric in ['Current Assets', 'Current Liabilities']):
                            summary_stats['latest_current_ratio'] = latest_balance['Current Assets'] / latest_balance['Current Liabilities']
                        
                        if all(metric in latest_balance.index for metric in ['Current Assets', 'Inventory', 'Current Liabilities']):
                            summary_stats['latest_quick_ratio'] = (latest_balance['Current Assets'] - latest_balance['Inventory']) / latest_balance['Current Liabilities']
                        
                        if all(metric in latest_balance.index for metric in ['Cash And Cash Equivalents', 'Current Liabilities']):
                            summary_stats['latest_cash_ratio'] = latest_balance['Cash And Cash Equivalents'] / latest_balance['Current Liabilities']
                        
                        if all(metric in latest_balance.index for metric in ['Working Capital', 'Total Assets']):
                            summary_stats['latest_working_capital_ratio'] = latest_balance['Working Capital'] / latest_balance['Total Assets']
                    
            except Exception as e:
                print(f"⚠️  Error calculating fundamental summary stats for {ticker.upper()}: {e}")
        
        # Add options-specific statistics if available
        if has_options:
            if not call_options.empty:
                summary_stats['call_options_count'] = len(call_options)
                summary_stats['call_avg_iv'] = call_options['impliedVolatility'].mean() if 'impliedVolatility' in call_options.columns else None
                summary_stats['call_total_volume'] = call_options['volume'].sum() if 'volume' in call_options.columns else None
                summary_stats['call_total_oi'] = call_options['openInterest'].sum() if 'openInterest' in call_options.columns else None
            
            if not put_options.empty:
                summary_stats['put_options_count'] = len(put_options)
                summary_stats['put_avg_iv'] = put_options['impliedVolatility'].mean() if 'impliedVolatility' in put_options.columns else None
                summary_stats['put_total_volume'] = put_options['volume'].sum() if 'volume' in put_options.columns else None
                summary_stats['put_total_oi'] = put_options['openInterest'].sum() if 'openInterest' in put_options.columns else None
            
            # Calculate put-call ratios
            if not call_options.empty and not put_options.empty:
                if 'volume' in call_options.columns and 'volume' in put_options.columns:
                    summary_stats['put_call_volume_ratio'] = put_options['volume'].sum() / call_options['volume'].sum() if call_options['volume'].sum() > 0 else 0
                if 'openInterest' in call_options.columns and 'openInterest' in put_options.columns:
                    summary_stats['put_call_oi_ratio'] = put_options['openInterest'].sum() / call_options['openInterest'].sum() if call_options['openInterest'].sum() > 0 else 0
        
        print(f"✅ Quick report generated for {ticker.upper()}")
        
        return render_template('quick_report.html',
                             ticker=ticker.upper(),
                             plots=plots,
                             summary_stats=summary_stats,
                             advanced_metrics=stock_data[['Williams_R', 'Stoch_K', 'Stoch_D', 'ATR', 'CCI']].iloc[-1].to_dict() if all(col in stock_data.columns for col in ['Williams_R', 'Stoch_K', 'Stoch_D', 'ATR', 'CCI']) else {},
                             risk_analysis=risk_metrics,
                             sentiment_analysis=sentiment_data,
                             portfolio_recommendations=portfolio_recs,
                             date=datetime.now().strftime("%Y-%m-%d @ %H:%M"))
                             
    except Exception as e:
        print(f"❌ Error generating quick report for {ticker}: {e}")
        return f"Error generating report for {ticker.upper()}: {str(e)}", 500

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def save_stock_data_to_db(stock_data, ticker, db_manager):
    """Save stock data to database using database manager"""
    try:
        if not db_manager or not db_manager.health_check():
            print("⚠️ Database manager not available or unhealthy")
            return False
        
        # Prepare data for database insertion
        # Reset index to make date a column
        data_for_db = stock_data.reset_index()
        data_for_db['ticker'] = ticker.upper()
        data_for_db['created_at'] = datetime.now()
        
        # Rename columns to match database schema (if needed)
        column_mapping = {
            'Date': 'date',
            'Open': 'open_price',
            'High': 'high_price', 
            'Low': 'low_price',
            'Close': 'close_price',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        # Rename columns that exist in the data
        for old_name, new_name in column_mapping.items():
            if old_name in data_for_db.columns:
                data_for_db = data_for_db.rename(columns={old_name: new_name})
        
        # Select only the columns we want to save
        columns_to_save = ['ticker', 'date', 'open_price', 'high_price', 'low_price', 
                          'close_price', 'volume', 'adj_close', 'created_at']
        
        # Filter to only include columns that exist
        existing_columns = [col for col in columns_to_save if col in data_for_db.columns]
        data_for_db = data_for_db[existing_columns]
        
        # Convert to list of dictionaries for bulk insert
        records = data_for_db.to_dict('records')
        
        # Use database manager to save data
        # Note: This assumes you have a table structure for stock data
        # You may need to adjust the table name and structure based on your schema
        
        # Option 1: Use pandas to_sql with database manager connection
        try:
            with db_manager.get_db_connection() as conn:
                data_for_db.to_sql(
                    name='stock_data',  # Adjust table name as needed
                    con=conn,
                    if_exists='append',
                    index=False,
                    method='multi',
                    chunksize=1000
                )
            print(f"✅ Stock data saved to database table 'stock_data' for {ticker.upper()}")
            return True
        except Exception as e:
            print(f"⚠️ Failed to save to 'stock_data' table: {str(e)}")
            
            # Option 2: Try alternative table name
            try:
                with db_manager.get_db_connection() as conn:
                    data_for_db.to_sql(
                        name=f'stock_data_{ticker.lower()}',  # Ticker-specific table
                        con=conn,
                        if_exists='replace',  # Replace for ticker-specific tables
                        index=False,
                        method='multi',
                        chunksize=1000
                    )
                print(f"✅ Stock data saved to database table 'stock_data_{ticker.lower()}' for {ticker.upper()}")
                return True
            except Exception as e2:
                print(f"⚠️ Failed to save to ticker-specific table: {str(e2)}")
                return False
                
    except Exception as e:
        print(f"❌ Error saving stock data to database: {str(e)}")
        return False

def get_stock_data_from_db(ticker, db_manager):
    """Retrieve historical stock data from the database for a given ticker."""
    try:
        if not db_manager or not db_manager.health_check():
            print("⚠️ Database manager not available or unhealthy for data retrieval.")
            return None

        # Define the table name based on the ticker
        table_name = f'stock_data_{ticker.lower()}'

        # Construct the SQL query to get the latest data
        query = f"""
            SELECT date, open_price, high_price, low_price, close_price, volume, adj_close
            FROM {table_name}
            WHERE date >= CURRENT_DATE - INTERVAL '1 year' -- Get data for the last year
            ORDER BY date DESC
            LIMIT 504 -- Limit to 2 years of data (252 trading days)
        """

        with db_manager.get_db_connection() as conn:
            stock_data = pd.read_sql_query(query, conn)
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            stock_data.set_index('date', inplace=True)
            return stock_data
    except Exception as e:
        print(f"❌ Error retrieving data from database for {ticker}: {e}")
        return None

def save_technical_indicators_to_db(stock_data, ticker, db_manager):
    """Save technical indicators and analysis results to the database."""
    try:
        if not db_manager or not db_manager.health_check():
            print("⚠️ Database manager not available or unhealthy for saving technical indicators.")
            return False
        
        # Prepare technical indicators data for database insertion
        # Get the latest data point
        latest_data = stock_data.iloc[-1].copy()
        
        # Create a DataFrame with the latest technical indicators
        indicators_data = pd.DataFrame({
            'ticker': [ticker.upper()],
            'date': [latest_data.name],  # Index is the date
            'close_price': [latest_data.get('Close', None)],
            'rsi': [latest_data.get('RSI', None)],
            'macd': [latest_data.get('MACD', None)],
            'macd_signal': [latest_data.get('MACD_signal', None)],
            'bollinger_upper': [latest_data.get('BB_upper', None)],
            'bollinger_lower': [latest_data.get('BB_lower', None)],
            'bollinger_middle': [latest_data.get('BB_middle', None)],
            'williams_r': [latest_data.get('Williams_R', None)],
            'stoch_k': [latest_data.get('Stoch_K', None)],
            'stoch_d': [latest_data.get('Stoch_D', None)],
            'atr': [latest_data.get('ATR', None)],
            'cci': [latest_data.get('CCI', None)],
            'volume': [latest_data.get('Volume', None)],
            'created_at': [datetime.now()]
        })
        
        # Use database manager to save technical indicators
        try:
            with db_manager.get_db_connection() as conn:
                indicators_data.to_sql(
                    name='technical_indicators',  # Adjust table name as needed
                    con=conn,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
            print(f"✅ Technical indicators saved to database table 'technical_indicators' for {ticker.upper()}")
            return True
        except Exception as e:
            print(f"⚠️ Failed to save to 'technical_indicators' table: {str(e)}")
            
            # Try alternative table name
            try:
                with db_manager.get_db_connection() as conn:
                    indicators_data.to_sql(
                        name=f'technical_indicators_{ticker.lower()}',  # Ticker-specific table
                        con=conn,
                        if_exists='replace',  # Replace for ticker-specific tables
                        index=False,
                        method='multi'
                    )
                print(f"✅ Technical indicators saved to database table 'technical_indicators_{ticker.lower()}' for {ticker.upper()}")
                return True
            except Exception as e2:
                print(f"⚠️ Failed to save to ticker-specific technical indicators table: {str(e2)}")
                return False
                
    except Exception as e:
        print(f"❌ Error saving technical indicators to database: {str(e)}")
        return False

def save_risk_analysis_to_db(risk_metrics, ticker, db_manager):
    """Save risk analysis results to the database."""
    try:
        if not db_manager or not db_manager.health_check():
            print("⚠️ Database manager not available or unhealthy for saving risk analysis.")
            return False
        
        # Prepare risk analysis data for database insertion
        # Get the latest data point
        latest_data = pd.DataFrame({
            'ticker': [ticker.upper()],
            'date': [datetime.now()], # Current date
            'var_95_1d': [risk_metrics.get('VaR_95_1d', None)],
            'var_95_5d': [risk_metrics.get('VaR_95_5d', None)],
            'var_95_20d': [risk_metrics.get('VaR_95_20d', None)],
            'cvar_95_1d': [risk_metrics.get('CVaR_95_1d', None)],
            'max_drawdown': [risk_metrics.get('Max_Drawdown', None)],
            'sharpe_ratio': [risk_metrics.get('Sharpe_Ratio', None)],
            'sortino_ratio': [risk_metrics.get('Sortino_Ratio', None)],
            'created_at': [datetime.now()]
        })
        
        # Use database manager to save risk analysis
        try:
            with db_manager.get_db_connection() as conn:
                latest_data.to_sql(
                    name='risk_analysis',  # Adjust table name as needed
                    con=conn,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
            print(f"✅ Risk analysis saved to database table 'risk_analysis' for {ticker.upper()}")
            return True
        except Exception as e:
            print(f"⚠️ Failed to save to 'risk_analysis' table: {str(e)}")
            
            # Try alternative table name
            try:
                with db_manager.get_db_connection() as conn:
                    latest_data.to_sql(
                        name=f'risk_analysis_{ticker.lower()}',  # Ticker-specific table
                        con=conn,
                        if_exists='replace',  # Replace for ticker-specific tables
                        index=False,
                        method='multi'
                    )
                print(f"✅ Risk analysis saved to database table 'risk_analysis_{ticker.lower()}' for {ticker.upper()}")
                return True
            except Exception as e2:
                print(f"⚠️ Failed to save to ticker-specific risk analysis table: {str(e2)}")
                return False
                
    except Exception as e:
        print(f"❌ Error saving risk analysis to database: {str(e)}")
        return False

def save_sentiment_portfolio_to_db(sentiment_data, portfolio_recs, ticker, db_manager):
    """Save sentiment analysis and portfolio recommendations to the database."""
    try:
        if not db_manager or not db_manager.health_check():
            print("⚠️ Database manager not available or unhealthy for saving sentiment and portfolio analysis.")
            return False
        
        # Prepare sentiment and portfolio data for database insertion
        analysis_data = pd.DataFrame({
            'ticker': [ticker.upper()],
            'date': [datetime.now()],
            'news_sentiment': [sentiment_data.get('news_sentiment', None)],
            'social_sentiment': [sentiment_data.get('social_sentiment', None)],
            'analyst_rating': [sentiment_data.get('analyst_rating', None)],
            'price_target': [sentiment_data.get('price_target', None)],
            'confidence_score': [sentiment_data.get('confidence_score', None)],
            'position_size': [portfolio_recs.get('position_size', None)],
            'entry_strategy': [portfolio_recs.get('entry_strategy', None)],
            'stop_loss': [portfolio_recs.get('stop_loss', None)],
            'take_profit': [portfolio_recs.get('take_profit', None)],
            'holding_period': [portfolio_recs.get('holding_period', None)],
            'risk_level': [portfolio_recs.get('risk_level', None)],
            'sector_allocation': [portfolio_recs.get('sector_allocation', None)],
            'created_at': [datetime.now()]
        })
        
        # Use database manager to save sentiment and portfolio analysis
        try:
            with db_manager.get_db_connection() as conn:
                analysis_data.to_sql(
                    name='sentiment_portfolio_analysis',  # Adjust table name as needed
                    con=conn,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
            print(f"✅ Sentiment and portfolio analysis saved to database table 'sentiment_portfolio_analysis' for {ticker.upper()}")
            return True
        except Exception as e:
            print(f"⚠️ Failed to save to 'sentiment_portfolio_analysis' table: {str(e)}")
            
            # Try alternative table name
            try:
                with db_manager.get_db_connection() as conn:
                    analysis_data.to_sql(
                        name=f'sentiment_portfolio_{ticker.lower()}',  # Ticker-specific table
                        con=conn,
                        if_exists='replace',  # Replace for ticker-specific tables
                        index=False,
                        method='multi'
                    )
                print(f"✅ Sentiment and portfolio analysis saved to database table 'sentiment_portfolio_{ticker.lower()}' for {ticker.upper()}")
                return True
            except Exception as e2:
                print(f"⚠️ Failed to save to ticker-specific sentiment and portfolio table: {str(e2)}")
                return False
                
    except Exception as e:
        print(f"❌ Error saving sentiment and portfolio analysis to database: {str(e)}")
        return False

def save_options_data_to_db(call_options, put_options, ticker, db_manager):
    """Save options data to the database."""
    try:
        if not db_manager or not db_manager.health_check():
            print("⚠️ Database manager not available or unhealthy for saving options data.")
            return False
        
        # Prepare call options data for database insertion
        if not call_options.empty:
            call_data = call_options.copy()
            call_data['ticker'] = ticker.upper()
            call_data['option_type'] = 'call'
            call_data['created_at'] = datetime.now()
            
            # Rename columns to match database schema
            column_mapping = {
                'strike': 'strike_price',
                'lastPrice': 'last_price',
                'bid': 'bid_price',
                'ask': 'ask_price',
                'volume': 'volume',
                'openInterest': 'open_interest',
                'impliedVolatility': 'implied_volatility'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in call_data.columns:
                    call_data = call_data.rename(columns={old_name: new_name})
            
            # Select only the columns we want to save
            columns_to_save = ['ticker', 'option_type', 'strike_price', 'last_price', 'bid_price', 
                              'ask_price', 'volume', 'open_interest', 'implied_volatility', 'created_at']
            existing_columns = [col for col in columns_to_save if col in call_data.columns]
            call_data = call_data[existing_columns]
            
            # Save call options to database
            try:
                with db_manager.get_db_connection() as conn:
                    call_data.to_sql(
                        name='options_data',  # Adjust table name as needed
                        con=conn,
                        if_exists='append',
                        index=False,
                        method='multi',
                        chunksize=1000
                    )
                print(f"✅ Call options data saved to database for {ticker.upper()}")
            except Exception as e:
                print(f"⚠️ Failed to save call options to 'options_data' table: {str(e)}")
                
                # Try alternative table name
                try:
                    with db_manager.get_db_connection() as conn:
                        call_data.to_sql(
                            name=f'options_data_{ticker.lower()}',  # Ticker-specific table
                            con=conn,
                            if_exists='append',
                            index=False,
                            method='multi',
                            chunksize=1000
                        )
                    print(f"✅ Call options data saved to ticker-specific table for {ticker.upper()}")
                except Exception as e2:
                    print(f"⚠️ Failed to save call options to ticker-specific table: {str(e2)}")
        
        # Prepare put options data for database insertion
        if not put_options.empty:
            put_data = put_options.copy()
            put_data['ticker'] = ticker.upper()
            put_data['option_type'] = 'put'
            put_data['created_at'] = datetime.now()
            
            # Rename columns to match database schema
            for old_name, new_name in column_mapping.items():
                if old_name in put_data.columns:
                    put_data = put_data.rename(columns={old_name: new_name})
            
            # Select only the columns we want to save
            existing_columns = [col for col in columns_to_save if col in put_data.columns]
            put_data = put_data[existing_columns]
            
            # Save put options to database
            try:
                with db_manager.get_db_connection() as conn:
                    put_data.to_sql(
                        name='options_data',  # Adjust table name as needed
                        con=conn,
                        if_exists='append',
                        index=False,
                        method='multi',
                        chunksize=1000
                    )
                print(f"✅ Put options data saved to database for {ticker.upper()}")
            except Exception as e:
                print(f"⚠️ Failed to save put options to 'options_data' table: {str(e)}")
                
                # Try alternative table name
                try:
                    with db_manager.get_db_connection() as conn:
                        put_data.to_sql(
                            name=f'options_data_{ticker.lower()}',  # Ticker-specific table
                            con=conn,
                            if_exists='append',
                            index=False,
                            method='multi',
                            chunksize=1000
                        )
                    print(f"✅ Put options data saved to ticker-specific table for {ticker.upper()}")
                except Exception as e2:
                    print(f"⚠️ Failed to save put options to ticker-specific table: {str(e2)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving options data to database: {str(e)}")
        return False

def save_fundamental_data_to_db(balance_sheet, cash_flow, income_statement, ticker, db_manager):
    """Save fundamental data to the database."""
    try:
        if not db_manager or not db_manager.health_check():
            print("⚠️ Database manager not available or unhealthy for saving fundamental data.")
            return False
        
        # Save balance sheet data
        if not balance_sheet.empty:
            balance_data = balance_sheet.reset_index()
            balance_data['ticker'] = ticker.upper()
            balance_data['statement_type'] = 'balance_sheet'
            balance_data['created_at'] = datetime.now()
            
            # Rename the index column to 'metric'
            balance_data = balance_data.rename(columns={'index': 'metric'})
            
            # Melt the data to long format for easier database storage
            balance_melted = balance_data.melt(
                id_vars=['ticker', 'statement_type', 'created_at', 'metric'],
                var_name='date',
                value_name='value'
            )
            
            # Save balance sheet to database
            try:
                with db_manager.get_db_connection() as conn:
                    balance_melted.to_sql(
                        name='fundamental_data',  # Adjust table name as needed
                        con=conn,
                        if_exists='append',
                        index=False,
                        method='multi',
                        chunksize=1000
                    )
                print(f"✅ Balance sheet data saved to database for {ticker.upper()}")
            except Exception as e:
                print(f"⚠️ Failed to save balance sheet to 'fundamental_data' table: {str(e)}")
                
                # Try alternative table name
                try:
                    with db_manager.get_db_connection() as conn:
                        balance_melted.to_sql(
                            name=f'fundamental_data_{ticker.lower()}',  # Ticker-specific table
                            con=conn,
                            if_exists='append',
                            index=False,
                            method='multi',
                            chunksize=1000
                        )
                    print(f"✅ Balance sheet data saved to ticker-specific table for {ticker.upper()}")
                except Exception as e2:
                    print(f"⚠️ Failed to save balance sheet to ticker-specific table: {str(e2)}")
        
        # Save cash flow data
        if not cash_flow.empty:
            cash_flow_data = cash_flow.reset_index()
            cash_flow_data['ticker'] = ticker.upper()
            cash_flow_data['statement_type'] = 'cash_flow'
            cash_flow_data['created_at'] = datetime.now()
            
            # Rename the index column to 'metric'
            cash_flow_data = cash_flow_data.rename(columns={'index': 'metric'})
            
            # Melt the data to long format
            cash_flow_melted = cash_flow_data.melt(
                id_vars=['ticker', 'statement_type', 'created_at', 'metric'],
                var_name='date',
                value_name='value'
            )
            
            # Save cash flow to database
            try:
                with db_manager.get_db_connection() as conn:
                    cash_flow_melted.to_sql(
                        name='fundamental_data',  # Adjust table name as needed
                        con=conn,
                        if_exists='append',
                        index=False,
                        method='multi',
                        chunksize=1000
                    )
                print(f"✅ Cash flow data saved to database for {ticker.upper()}")
            except Exception as e:
                print(f"⚠️ Failed to save cash flow to 'fundamental_data' table: {str(e)}")
                
                # Try alternative table name
                try:
                    with db_manager.get_db_connection() as conn:
                        cash_flow_melted.to_sql(
                            name=f'fundamental_data_{ticker.lower()}',  # Ticker-specific table
                            con=conn,
                            if_exists='append',
                            index=False,
                            method='multi',
                            chunksize=1000
                        )
                    print(f"✅ Cash flow data saved to ticker-specific table for {ticker.upper()}")
                except Exception as e2:
                    print(f"⚠️ Failed to save cash flow to ticker-specific table: {str(e2)}")
        
        # Save income statement data
        if not income_statement.empty:
            income_data = income_statement.reset_index()
            income_data['ticker'] = ticker.upper()
            income_data['statement_type'] = 'income_statement'
            income_data['created_at'] = datetime.now()
            
            # Rename the index column to 'metric'
            income_data = income_data.rename(columns={'index': 'metric'})
            
            # Melt the data to long format
            income_melted = income_data.melt(
                id_vars=['ticker', 'statement_type', 'created_at', 'metric'],
                var_name='date',
                value_name='value'
            )
            
            # Save income statement to database
            try:
                with db_manager.get_db_connection() as conn:
                    income_melted.to_sql(
                        name='fundamental_data',  # Adjust table name as needed
                        con=conn,
                        if_exists='append',
                        index=False,
                        method='multi',
                        chunksize=1000
                    )
                print(f"✅ Income statement data saved to database for {ticker.upper()}")
            except Exception as e:
                print(f"⚠️ Failed to save income statement to 'fundamental_data' table: {str(e)}")
                
                # Try alternative table name
                try:
                    with db_manager.get_db_connection() as conn:
                        income_melted.to_sql(
                            name=f'fundamental_data_{ticker.lower()}',  # Ticker-specific table
                            con=conn,
                            if_exists='append',
                            index=False,
                            method='multi',
                            chunksize=1000
                        )
                    print(f"✅ Income statement data saved to ticker-specific table for {ticker.upper()}")
                except Exception as e2:
                    print(f"⚠️ Failed to save income statement to ticker-specific table: {str(e2)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving fundamental data to database: {str(e)}")
        return False

# ============================================
# Additional Analysis Functions
# - Add your new processes here
# ============================================

def calculate_advanced_metrics(stock_data):
    """Calculate advanced technical indicators and metrics"""
    try:
        # Williams %R
        high_14 = stock_data['High'].rolling(window=14).max()
        low_14 = stock_data['Low'].rolling(window=14).min()
        stock_data['Williams_R'] = -100 * (high_14 - stock_data['Close']) / (high_14 - low_14)
        
        # Stochastic Oscillator
        stock_data['Stoch_K'] = 100 * (stock_data['Close'] - low_14) / (high_14 - low_14)
        stock_data['Stoch_D'] = stock_data['Stoch_K'].rolling(window=3).mean()
        
        # Average True Range (ATR)
        tr1 = stock_data['High'] - stock_data['Low']
        tr2 = abs(stock_data['High'] - stock_data['Close'].shift(1))
        tr3 = abs(stock_data['Low'] - stock_data['Close'].shift(1))
        stock_data['ATR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(window=14).mean()
        
        # Commodity Channel Index (CCI)
        typical_price = (stock_data['High'] + stock_data['Low'] + stock_data['Close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        stock_data['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
        
        return stock_data
    except Exception as e:
        print(f"Error calculating advanced metrics: {e}")
        return stock_data

def generate_risk_analysis(stock_data):
    """Generate comprehensive risk analysis"""
    try:
        risk_metrics = {}
        
        # Value at Risk (VaR) calculations
        returns = stock_data['daily_return'].dropna()
        
        # Historical VaR (95% confidence)
        risk_metrics['VaR_95_1d'] = np.percentile(returns, 5)
        risk_metrics['VaR_95_5d'] = risk_metrics['VaR_95_1d'] * np.sqrt(5)
        risk_metrics['VaR_95_20d'] = risk_metrics['VaR_95_1d'] * np.sqrt(20)
        
        # Expected Shortfall (CVaR)
        var_threshold = risk_metrics['VaR_95_1d']
        tail_returns = returns[returns <= var_threshold]
        risk_metrics['CVaR_95_1d'] = tail_returns.mean() if len(tail_returns) > 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        risk_metrics['Max_Drawdown'] = drawdown.min()
        
        # Sharpe Ratio (assuming 0% risk-free rate)
        risk_metrics['Sharpe_Ratio'] = returns.mean() / returns.std() * np.sqrt(252)
        
        # Sortino Ratio
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        risk_metrics['Sortino_Ratio'] = returns.mean() / downside_deviation if downside_deviation > 0 else 0
        
        return risk_metrics
    except Exception as e:
        print(f"Error generating risk analysis: {e}")
        return {}

def analyze_market_sentiment(ticker):
    """Analyze market sentiment for a given ticker"""
    try:
        # This is a placeholder for sentiment analysis
        # You could integrate with news APIs, social media sentiment, etc.
        sentiment_data = {
            'news_sentiment': 'neutral',  # Could be positive, negative, neutral
            'social_sentiment': 'positive',
            'analyst_rating': 'buy',
            'price_target': None,
            'confidence_score': 0.75
        }
        
        # Example: Get analyst recommendations from yfinance
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            recommendations = stock.recommendations
            if not recommendations.empty:
                latest_rec = recommendations.iloc[-1]
                sentiment_data['analyst_rating'] = latest_rec['To Grade'].lower()
                sentiment_data['confidence_score'] = 0.8
        except:
            pass
            
        return sentiment_data
    except Exception as e:
        print(f"Error analyzing market sentiment: {e}")
        return {}

def generate_portfolio_recommendations(ticker, risk_profile='moderate'):
    """Generate portfolio recommendations based on analysis"""
    try:
        recommendations = {
            'position_size': 'medium',
            'entry_strategy': 'dollar_cost_averaging',
            'stop_loss': None,
            'take_profit': None,
            'holding_period': 'medium_term',
            'risk_level': 'moderate',
            'sector_allocation': 'diversified'
        }
        
        # This is where you'd implement your portfolio logic
        # based on technical analysis, fundamental analysis, etc.
        
        if risk_profile == 'conservative':
            recommendations['position_size'] = 'small'
            recommendations['holding_period'] = 'long_term'
            recommendations['risk_level'] = 'low'
        elif risk_profile == 'aggressive':
            recommendations['position_size'] = 'large'
            recommendations['holding_period'] = 'short_term'
            recommendations['risk_level'] = 'high'
            
        return recommendations
    except Exception as e:
        print(f"Error generating portfolio recommendations: {e}")
        return {}

def generate_enhanced_report(ticker, include_advanced=True, include_sentiment=True, include_portfolio=True, db_manager=None):
    """Generate an enhanced report with additional analysis"""
    try:
        # Get the basic quick report first
        basic_report = generate_quick_report(ticker, db_manager)
        
        if isinstance(basic_report, str) and basic_report.startswith("Error"):
            return basic_report
            
        # Import yfinance for additional data
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            stock_data = yf.download(ticker, period='1y', interval='1d', auto_adjust=True)
        except ImportError:
            return "Error: yfinance package not installed", 500
            
        enhanced_data = {}
        
        if include_advanced:
            # Calculate advanced technical indicators
            stock_data = calculate_advanced_metrics(stock_data)
            enhanced_data['advanced_metrics'] = stock_data[['Williams_R', 'Stoch_K', 'Stoch_D', 'ATR', 'CCI']].iloc[-1].to_dict()
            
            # Generate risk analysis
            risk_metrics = generate_risk_analysis(stock_data)
            enhanced_data['risk_analysis'] = risk_metrics
            
        if include_sentiment:
            # Analyze market sentiment
            sentiment = analyze_market_sentiment(ticker)
            enhanced_data['sentiment_analysis'] = sentiment
            
        if include_portfolio:
            # Generate portfolio recommendations
            portfolio_recs = generate_portfolio_recommendations(ticker)
            enhanced_data['portfolio_recommendations'] = portfolio_recs
            
        # You could extend this to create enhanced plots, additional analysis, etc.
        
        return enhanced_data
        
    except Exception as e:
        print(f"Error generating enhanced report: {e}")
        return f"Error generating enhanced report: {str(e)}", 500
