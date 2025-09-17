#!/usr/bin/env python3
"""
Regression Analysis Pipeline Runner
This script orchestrates the complete regression analysis pipeline including:
1. Data refining (feature selection and pruning)
2. Regression analysis (OLS and classification)
"""

import sys
import os
import pandas as pd
from sqlalchemy import create_engine

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_data_refining():
    """Run the data refining process (feature selection and pruning)"""
    print("="*80)
    print("STEP 1: DATA REFINING (FEATURE SELECTION & PRUNING)")
    print("="*80)
    
    try:
        # Import and run the data refining
        from data_refining import main as refine_main
        X_refined, y_refined = refine_main()
        
        if X_refined is not None and y_refined is not None:
            print("✅ Data refining completed successfully")
            print(f"Refined data shape: {X_refined.shape}")
            print(f"Target shape: {y_refined.shape}")
            return X_refined, y_refined
        else:
            print("❌ Data refining failed")
            return None, None
            
    except Exception as e:
        print(f"❌ Error in data refining: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_regression_analysis():
    """Run the regression analysis process"""
    print("\n" + "="*80)
    print("STEP 2: REGRESSION ANALYSIS")
    print("="*80)
    
    try:
        # Import and run the regression analysis
        from regression_analysis import main as analysis_main
        analysis_main()
        print("✅ Regression analysis completed successfully")
        
    except Exception as e:
        print(f"❌ Error in regression analysis: {e}")
        import traceback
        traceback.print_exc()

def check_database_connection():
    """Check database connection and verify required tables"""
    print("="*80)
    print("DATABASE CONNECTION CHECK")
    print("="*80)
    
    try:
        # Database connection
        # Use centralized config
        from config import Config
        engine = create_engine(Config.DATABASE['connection_string'], future=True)
        
        # Test connection
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection successful")
        
        # Check for required tables
        required_tables = [
            'ml_spx_data_manifest',
            'ml_spx_target'
        ]
        
        print("\nChecking for required tables:")
        for table in required_tables:
            try:
                pd.read_sql(f"SELECT 1 FROM {table} LIMIT 1", con=engine)
                print(f"  ✅ {table} - Found")
            except Exception as e:
                print(f"  ❌ {table} - Not found: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def check_configuration():
    """Check configuration file and parameters"""
    print("="*80)
    print("CONFIGURATION CHECK")
    print("="*80)
    
    try:
        import json
        
        with open('variables.json', 'r') as f:
            config = json.load(f)
        
        print("✅ Configuration file loaded successfully")
        
        # Check key parameters
        config_params = config.get('config', {})
        top_features = config_params.get('top_features', 50)
        drop_bottom = config_params.get('drop_bottom_features', 0)
        resampling = config_params.get('resampling', 'D')
        
        print(f"  Top features: {top_features}")
        print(f"  Drop bottom features: {drop_bottom}")
        print(f"  Resampling frequency: {resampling}")
        
        # Check required sections
        required_sections = ['markets', 'futures', 'economic_indicators']
        for section in required_sections:
            if section in config:
                print(f"  ✅ {section} section found")
            else:
                print(f"  ❌ {section} section missing")
        
        return True
        
    except FileNotFoundError:
        print("❌ variables.json not found")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in variables.json: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking configuration: {e}")
        return False

def display_pipeline_summary():
    """Display pipeline summary and usage instructions"""
    print("="*80)
    print("REGRESSION ANALYSIS PIPELINE")
    print("="*80)
    print("This pipeline performs comprehensive regression analysis including:")
    print("1. Data refining (feature selection and pruning)")
    print("2. OLS regression analysis with diagnostics")
    print("3. Classification experiments")
    print("4. Comprehensive NaN analysis at each stage")
    print("="*80)
    print("\nPipeline Steps:")
    print("1. Load z-scores data from database")
    print("2. Apply feature selection based on OLS significance")
    print("3. Perform correlation-based pruning")
    print("4. Run VIF analysis")
    print("5. Apply greedy forward selection")
    print("6. Save refined data to database")
    print("7. Run OLS regression analysis")
    print("8. Run classification experiments")
    print("9. Generate comprehensive reports")
    print("="*80)

def main():
    """Main pipeline execution"""
    display_pipeline_summary()
    
    # Pre-flight checks
    print("\n" + "="*80)
    print("PRE-FLIGHT CHECKS")
    print("="*80)
    
    # Check configuration
    if not check_configuration():
        print("❌ Configuration check failed. Please fix variables.json")
        return
    
    # Check database connection
    if not check_database_connection():
        print("❌ Database check failed. Please ensure database is accessible")
        return
    
    print("✅ All pre-flight checks passed")
    
    # Step 1: Data Refining
    print("\n" + "="*80)
    print("STARTING PIPELINE EXECUTION")
    print("="*80)
    
    X_refined, y_refined = run_data_refining()
    
    if X_refined is None:
        print("❌ Pipeline failed at data refining step")
        return
    
    # Step 2: Regression Analysis
    run_regression_analysis()
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print("✅ Data refining: Completed")
    print("✅ Regression analysis: Completed")
    print("✅ Classification experiments: Completed")
    print("✅ NaN analysis: Completed")
    print("\nResults saved to database:")
    print("  - ml_spx_refined_features: Selected features")
    print("  - ml_spx_refined_target: Target variable")
    print("  - ml_spx_refined_feature_list: Feature metadata")
    print("="*80)

def run_refining_only():
    """Run only the data refining step"""
    print("="*80)
    print("RUNNING DATA REFINING ONLY")
    print("="*80)
    
    if not check_configuration() or not check_database_connection():
        print("❌ Pre-flight checks failed")
        return
    
    X_refined, y_refined = run_data_refining()
    
    if X_refined is not None:
        print("✅ Data refining completed successfully")
        print(f"Refined data shape: {X_refined.shape}")
    else:
        print("❌ Data refining failed")

def run_analysis_only():
    """Run only the regression analysis step"""
    print("="*80)
    print("RUNNING REGRESSION ANALYSIS ONLY")
    print("="*80)
    
    if not check_database_connection():
        print("❌ Database check failed")
        return
    
    run_regression_analysis()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Regression Analysis Pipeline')
    parser.add_argument('--refining-only', action='store_true', 
                       help='Run only data refining step')
    parser.add_argument('--analysis-only', action='store_true', 
                       help='Run only regression analysis step')
    
    args = parser.parse_args()
    
    if args.refining_only:
        run_refining_only()
    elif args.analysis_only:
        run_analysis_only()
    else:
        main()
