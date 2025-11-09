#!/usr/bin/env python3
"""
Test 2: Data Loading and Verification
Loads survey data and validates its structure.
"""

import sys
import pandas as pd

def test_data_loading():
    """Test loading and validating survey data."""
    print("\n" + "="*60)
    print("TEST 2: DATA LOADING AND VERIFICATION")
    print("="*60 + "\n")
    
    try:
        # Load survey data
        print("📊 Loading survey data...")
        df = pd.read_csv('data/investor_survey_data_with_timestamps.csv')
        
        print(f"\n✅ Data loaded successfully!")
        print(f"   • Respondents: {len(df)}")
        print(f"   • Questions: {len(df.columns)}")
        print(f"   • Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Validate expected structure
        expected_cols = 38
        if len(df.columns) == expected_cols:
            print(f"   ✅ Column count matches expected ({expected_cols})")
        else:
            print(f"   ⚠️  Expected {expected_cols} columns, found {len(df.columns)}")
        
        # Check for missing data
        missing = df.isnull().sum().sum()
        print(f"   • Missing values: {missing}")
        
        # Display sample data
        print(f"\n📋 Sample Data (First 3 respondents):\n")
        print(df.head(3).to_string(max_colwidth=30, max_cols=5))
        
        print("\n" + "="*60)
        print("✅ RESULT: Data loading successful!")
        print("="*60 + "\n")
        return True
        
    except FileNotFoundError:
        print("\n❌ ERROR: Survey data file not found!")
        print("   Expected: data/investor_survey_data_with_timestamps.csv")
        print("   Please ensure you're in the project root directory.")
        print("\n" + "="*60 + "\n")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n" + "="*60 + "\n")
        return False

if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)
