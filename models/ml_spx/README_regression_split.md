# Regression Analysis Pipeline - Split Architecture

This document describes the split of the original `data_analysis.py` into focused modules for better maintainability and separation of concerns.

## File Structure

### 1. `data_refining.py`
**Purpose**: Feature selection, pruning, and data refinement

**Key Functions**:
- `load_data_from_database()` - Load z-scores data with manifest support
- `load_target_variable()` - Load target variable from separate table
- `comprehensive_nan_analysis()` - Detailed NaN reporting at each stage
- `prepare_data_for_pruning()` - Data validation and preparation
- `feature_pruning_ols_based()` - OLS-based feature significance ranking
- `correlation_pruning()` - Remove highly correlated features
- `vif_analysis()` - Variance Inflation Factor analysis
- `greedy_forward_selection()` - Greedy forward feature selection
- `save_refined_data()` - Save refined data to database

**Responsibilities**:
- Load z-scores data from database
- Apply OLS-based feature significance ranking
- Perform correlation-based pruning (remove highly correlated features)
- Run VIF analysis to identify multicollinearity
- Apply greedy forward selection with guardrails
- Comprehensive NaN analysis at each stage
- Save refined dataset to database

**Output**: Refined dataset with selected features and comprehensive diagnostics

### 2. `data_analysis_clean.py`
**Purpose**: Regression analysis and classification experiments

**Key Functions**:
- `load_refined_data()` - Load refined data from database
- `load_fallback_data()` - Fallback to original z-scores if refined data unavailable
- `comprehensive_nan_analysis()` - Detailed NaN reporting
- `run_ols_regression()` - OLS regression with diagnostics
- `run_classification_experiments()` - Multiple classifier experiments

**Responsibilities**:
- Load refined data from database
- Run OLS regression analysis with comprehensive diagnostics
- Perform rolling R-squared analysis
- Run stability diagnostics (subperiod analysis)
- Execute classification experiments (Logistic Regression, SVC, Random Forest, Gradient Boosting)
- Generate comprehensive analysis reports

**Output**: Regression results, classification metrics, and diagnostic reports

### 3. `run_regression_pipeline.py`
**Purpose**: Integration script to orchestrate the complete regression analysis pipeline

**Key Functions**:
- `run_data_refining()` - Execute data refining process
- `run_regression_analysis()` - Execute regression analysis
- `check_database_connection()` - Verify database connectivity
- `check_configuration()` - Validate configuration file
- `main()` - Orchestrate complete pipeline

**Responsibilities**:
- Coordinate execution of both modules
- Perform pre-flight checks (database, configuration)
- Handle errors and provide clear feedback
- Display comprehensive progress reporting
- Support individual step execution

## Usage

### Option 1: Run Complete Pipeline
```bash
python run_regression_pipeline.py
```

### Option 2: Run Individual Steps
```bash
# Step 1: Data refining only
python run_regression_pipeline.py --refining-only

# Step 2: Regression analysis only (requires Step 1)
python run_regression_pipeline.py --analysis-only
```

### Option 3: Run Individual Modules
```bash
# Data refining
python data_refining.py

# Regression analysis
python data_analysis_clean.py
```

## Key Improvements

### 1. **Separation of Concerns**
- Data refining focuses purely on feature selection and pruning
- Regression analysis focuses purely on statistical modeling and evaluation
- Clear boundaries between data preparation and analysis

### 2. **Comprehensive NaN Reporting**
Both files include detailed NaN analysis at multiple stages:
- Before and after data loading
- Before and after feature selection
- Before and after correlation pruning
- Before and after greedy selection
- Final refined data analysis
- Before and after regression analysis

### 3. **Robust Error Handling**
- Graceful handling of missing data
- Fallback mechanisms for data loading
- Clear error messages and suggestions
- Pre-flight checks for dependencies

### 4. **Database Integration**
- Chunked storage for large datasets
- Manifest files for easy data reconstruction
- Separate storage for refined data
- Support for both refined and fallback data

### 5. **Configuration Management**
- Centralized configuration loading
- Validation of configuration structure
- Flexible feature selection parameters

## Data Flow

```
Z-Scores Data → data_refining.py → Refined Data → data_analysis_clean.py → Analysis Results
                     ↓                                        ↓
              NaN Analysis                            NaN Analysis
                     ↓                                        ↓
              Database Storage                        Final Reports
```

## Feature Selection Process

### 1. **OLS-Based Ranking**
- Run initial OLS regression on all features
- Rank features by p-value significance
- Select most significant features

### 2. **Correlation Pruning**
- Identify highly correlated feature pairs (|r| >= 0.90)
- Keep more significant feature from each pair
- Remove redundant features

### 3. **VIF Analysis**
- Compute Variance Inflation Factors
- Identify multicollinearity issues
- Report condition numbers

### 4. **Greedy Forward Selection**
- Start with most significant features
- Add features one by one based on adjusted R² improvement
- Apply correlation and condition number guardrails
- Stop when no significant improvement

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

## Database Schema

### Tables Created by data_refining.py:
- `ml_spx_refined_features` - Selected features
- `ml_spx_refined_target` - Target variable
- `ml_spx_refined_feature_list` - Feature metadata

### Tables Used by data_analysis_clean.py:
- `ml_spx_refined_features` - Primary data source
- `ml_spx_refined_target` - Target variable
- `ml_spx_zscores_manifest` - Fallback data source
- `ml_spx_target` - Fallback target source

## Configuration Parameters

The pipeline uses `variables.json` for configuration:
- `top_features`: Number of top features to select
- `drop_bottom_features`: Number of bottom features to drop
- `resampling`: Data resampling frequency
- Symbol definitions with lag values
- Technical indicator parameters

## Benefits of Split Architecture

1. **Maintainability**: Easier to modify individual components
2. **Testability**: Can test feature selection and analysis separately
3. **Reusability**: Analysis can work with different refined datasets
4. **Debugging**: Easier to identify issues in specific pipeline stages
5. **Performance**: Can optimize each stage independently
6. **Documentation**: Clearer understanding of each component's purpose
7. **Flexibility**: Can run individual steps as needed

## Error Handling

- **Database Connection**: Automatic fallback to alternative data sources
- **Missing Data**: Comprehensive NaN analysis and handling
- **Configuration**: Validation and clear error messages
- **Feature Selection**: Graceful handling of problematic features
- **Analysis**: Robust error handling in statistical computations

## Performance Considerations

- **Chunked Data Loading**: Efficient handling of large datasets
- **Memory Management**: Defragmentation and optimization
- **Parallel Processing**: Support for multi-threaded operations where appropriate
- **Caching**: Database storage for intermediate results
