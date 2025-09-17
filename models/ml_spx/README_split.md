# Data Preparation Pipeline - Split Architecture

This document describes the split of the original `data_prep.py` into two focused modules for better maintainability and separation of concerns.

## File Structure

### 1. `api_data_collection.py`
**Purpose**: API data calling, name sanitization, resampling, and lagging

**Key Functions**:
- `daily()` - Download market data from Yahoo Finance
- `data()` - Load FRED economic data
- `sanitize_symbol()` - Convert symbols to column-friendly format
- `resample_fred_data()` - Intelligent resampling based on frequency comparison
- `resample_data()` - Standard resampling for market data
- `compare_frequencies()` - Compare native vs target frequencies

**Responsibilities**:
- Load configuration from `variables.json`
- Connect to PostgreSQL database
- Download market data (equities, futures, FX)
- Download FRED economic indicators
- Apply intelligent resampling based on data frequency
- Apply configured lags to all data
- Create target variable (GSPC next-period log return)
- Calculate yield spreads
- Comprehensive NaN analysis and reporting

**Output**: Clean, lagged dataset ready for feature calculation

### 2. `feature_calculation.py`
**Purpose**: Technical indicators and feature engineering

**Key Functions**:
- `calculate_market_features()` - Calculate technical indicators for market data
- `calculate_economic_features()` - Calculate log % change for economic indicators
- `comprehensive_nan_analysis()` - Detailed NaN reporting
- `save_to_database()` - Save data in chunked format with manifest
- `compute_z_scores()` - Calculate standardized features

**Responsibilities**:
- Calculate technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Calculate rolling statistics (returns, volatility, skewness, kurtosis)
- Calculate economic indicator log % changes
- Remove original OHLCV columns after feature extraction
- Comprehensive NaN analysis at each stage
- Save final data and z-scores to database
- Handle problematic data gracefully

**Output**: Feature-rich dataset with technical indicators and economic features

### 3. `run_pipeline.py`
**Purpose**: Integration script to run both modules in sequence

**Key Functions**:
- `run_api_data_collection()` - Execute API data collection
- `run_feature_calculation()` - Execute feature calculation
- `main()` - Orchestrate the complete pipeline

**Responsibilities**:
- Coordinate execution of both modules
- Handle errors and provide clear feedback
- Display comprehensive progress reporting
- Ensure data flows correctly between modules

## Usage

### Option 1: Run Complete Pipeline
```bash
python run_pipeline.py
```

### Option 2: Run Individual Modules
```bash
# Step 1: API data collection
python api_data_collection.py

# Step 2: Feature calculation (requires Step 1 output)
python feature_calculation.py
```

## Key Improvements

### 1. **Separation of Concerns**
- API data collection focuses purely on data acquisition and preprocessing
- Feature calculation focuses purely on technical analysis and feature engineering
- Clear boundaries between data loading and feature creation

### 2. **Comprehensive NaN Reporting**
Both files include detailed NaN analysis at multiple stages:
- Before and after data loading
- Before and after feature calculation
- Final cleaned data analysis
- Z-scores data analysis

### 3. **Error Handling**
- Graceful handling of missing data
- Clear error messages and suggestions
- Robust processing of problematic columns

### 4. **Database Integration**
- Chunked storage for large datasets
- Manifest files for easy data reconstruction
- Separate storage for original data and z-scores

### 5. **Configuration Management**
- Centralized configuration loading
- Validation of configuration structure
- Flexible resampling and lagging parameters

## Data Flow

```
variables.json → api_data_collection.py → Database
                                    ↓
feature_calculation.py ← Database ← Database
                                    ↓
                              Final Dataset + Z-scores
```

## NaN Analysis Features

Both modules provide comprehensive NaN analysis including:
- Total NaN counts and percentages
- Column-wise NaN analysis
- Row-wise NaN analysis
- Target variable specific analysis
- Feature completeness analysis
- Infinite value detection
- Memory usage reporting
- Data type analysis

## Configuration

The pipeline uses `variables.json` for configuration:
- Symbol definitions with lag values
- Resampling frequency settings
- Technical indicator parameters
- Date range specifications

## Database Schema

### Tables Created:
- `ml_spx_target` - Target variable only
- `ml_spx_data_p001`, `ml_spx_data_p002`, ... - Chunked feature data
- `ml_spx_data_manifest` - Manifest for feature data
- `ml_spx_zscores_p001`, `ml_spx_zscores_p002`, ... - Chunked z-scores
- `ml_spx_zscores_manifest` - Manifest for z-scores data

## Benefits of Split Architecture

1. **Maintainability**: Easier to modify individual components
2. **Testability**: Can test API collection and feature calculation separately
3. **Reusability**: Feature calculation can work with different data sources
4. **Debugging**: Easier to identify issues in specific pipeline stages
5. **Performance**: Can optimize each stage independently
6. **Documentation**: Clearer understanding of each component's purpose

