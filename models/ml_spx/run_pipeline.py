#!/usr/bin/env python3
"""
Integration script to run the complete data preparation pipeline
This script demonstrates how to use both api_data_collection.py and feature_calculation.py
"""

import sys
import os
import pandas as pd
from sqlalchemy import create_engine

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_api_data_collection():
    """Run the API data collection and lagging process"""
    print("="*80)
    print("STEP 1: API DATA COLLECTION AND LAGGING")
    print("="*80)
    
    try:
        # Import and run the API data collection
        from api_data_collection import (
            variables_config, combined_data, resampling_freq, 
            engine, symbol_lag_mapping, markets_symbols, 
            economic_indicators_symbols
        )
        
        print("✅ API data collection completed successfully")
        print(f"Data shape: {combined_data.shape}")
        print(f"Resampling frequency: {resampling_freq}")
        print(f"Total symbols processed: {len(symbol_lag_mapping)}")
        
        return combined_data, variables_config, engine
        
    except Exception as e:
        print(f"❌ Error in API data collection: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def run_feature_calculation(combined_data, variables_config, engine):
    """Run the feature calculation process"""
    print("\n" + "="*80)
    print("STEP 2: FEATURE CALCULATION")
    print("="*80)
    
    try:
        # Import feature calculation functions
        from feature_calculation import (
            calculate_market_features, calculate_economic_features,
            comprehensive_nan_analysis, save_to_database, compute_z_scores
        )
        
        # NaN analysis before feature calculation
        comprehensive_nan_analysis(combined_data, "BEFORE FEATURE CALCULATION")
        
        # Calculate market features
        print("\nCalculating market features...")
        market_features = calculate_market_features(combined_data, variables_config)
        
        # Calculate economic features
        print("\nCalculating economic features...")
        economic_features = calculate_economic_features(combined_data, variables_config)
        
        # Combine all features
        all_features = {**market_features, **economic_features}
        
        # Create features dataframe
        print("Creating features dataframe...")
        new_features_data = pd.DataFrame(all_features, index=combined_data.index)
        
        # Add new features to combined_data
        print("Adding calculated features to dataset...")
        combined_data = pd.concat([combined_data, new_features_data], axis=1)
        
        # Remove original OHLCV columns
        print("Removing original lagged OHLCV columns...")
        ohlcv_columns = [col for col in combined_data.columns if any(ohlcv in col for ohlcv in ['Open', 'High', 'Low', 'Close', 'Volume'])]
        combined_data = combined_data.drop(columns=ohlcv_columns)
        
        # Remove original economic indicator columns that were transformed
        econ_cols_to_drop = [col for col in combined_data.columns if col in economic_features.keys()]
        if econ_cols_to_drop:
            combined_data = combined_data.drop(columns=econ_cols_to_drop)
            print(f"✓ Removed {len(econ_cols_to_drop)} original economic indicator columns")
        
        # NaN analysis after feature calculation
        comprehensive_nan_analysis(combined_data, "AFTER FEATURE CALCULATION")
        
        # Final data cleanup
        print("\nFinal data cleanup...")
        combined_data = combined_data.dropna(how='all')
        combined_data = combined_data.ffill()
        
        # Fill any remaining NaNs with 0
        remaining_nans = combined_data.isna().sum().sum()
        if remaining_nans > 0:
            print(f"Filling {remaining_nans} remaining NaNs with 0...")
            combined_data = combined_data.fillna(0)
        
        # Defragment DataFrame
        combined_data = combined_data.copy()
        
        # Final NaN analysis
        comprehensive_nan_analysis(combined_data, "FINAL CLEANED DATA")
        
        # Save to database
        print("\nSaving to database...")
        save_to_database(combined_data, engine, 'ml_spx_data_p', 'ml_spx_data_manifest')
        
        # Compute and save z-scores
        print("\nComputing z-scores...")
        zscores_df, problematic_cols = compute_z_scores(combined_data)
        
        # Z-scores NaN analysis
        comprehensive_nan_analysis(zscores_df, "Z-SCORES DATA")
        
        # Save z-scores to database
        save_to_database(zscores_df, engine, 'ml_spx_zscores_p', 'ml_spx_zscores_manifest')
        
        print("✅ Feature calculation completed successfully")
        print(f"Final data shape: {combined_data.shape}")
        print(f"Z-scores shape: {zscores_df.shape}")
        print(f"Problematic columns: {len(problematic_cols)}")
        
        return combined_data, zscores_df
        
    except Exception as e:
        print(f"❌ Error in feature calculation: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main pipeline execution"""
    print("="*80)
    print("COMPLETE DATA PREPARATION PIPELINE")
    print("="*80)
    print("This pipeline runs both API data collection and feature calculation")
    print("="*80)
    
    # Step 1: API Data Collection
    combined_data, variables_config, engine = run_api_data_collection()
    
    if combined_data is None:
        print("❌ Pipeline failed at API data collection step")
        return
    
    # Step 2: Feature Calculation
    final_data, zscores_data = run_feature_calculation(combined_data, variables_config, engine)
    
    if final_data is None:
        print("❌ Pipeline failed at feature calculation step")
        return
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"✅ API data collection: Completed")
    print(f"✅ Feature calculation: Completed")
    print(f"✅ Database storage: Completed")
    print(f"✅ Z-scores computation: Completed")
    print(f"✅ NaN analysis: Completed")
    print("\nFinal Results:")
    print(f"  - Original data shape: {combined_data.shape}")
    print(f"  - Final data shape: {final_data.shape}")
    print(f"  - Z-scores shape: {zscores_data.shape}")
    print(f"  - Target variable: GSPC_log_return_next_period")
    print("="*80)

if __name__ == "__main__":
    main()

