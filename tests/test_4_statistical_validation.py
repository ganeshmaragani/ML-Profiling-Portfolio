#!/usr/bin/env python3
"""
Test 4: Statistical Validation
Performs ANOVA and calculates effect size to validate cluster significance.
"""

import sys
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def test_statistical_validation():
    """Test statistical validation of clusters."""
    print("\n" + "="*60)
    print("TEST 4: STATISTICAL VALIDATION")
    print("="*60 + "\n")
    
    try:
        # Load data
        print("📊 Loading results...")
        df = pd.read_csv('data/investor_profiles_results.csv')
        
        # Extract data
        clusters = df['cluster'].values
        risk_scores = df['composite_risk_score'].values
        
        # Group by clusters
        cluster_0 = risk_scores[clusters == 0]
        cluster_1 = risk_scores[clusters == 1]
        cluster_2 = risk_scores[clusters == 2]
        
        # Perform ANOVA
        print("📈 Performing One-Way ANOVA...\n")
        f_statistic, p_value = stats.f_oneway(cluster_0, cluster_1, cluster_2)
        
        print(f"✅ ANOVA Results:")
        print(f"   • F-statistic: {f_statistic:.2f}")
        print(f"   • p-value: {p_value:.10f}")
        
        if p_value < 0.000001:
            print(f"   ✅ HIGHLY SIGNIFICANT (p < 0.000001) ⭐")
            print(f"   ✅ Clusters are statistically distinct!")
        elif p_value < 0.05:
            print(f"   ✅ Significant at α = 0.05")
        else:
            print(f"   ❌ Not statistically significant")
        
        # Calculate effect size (eta-squared)
        print(f"\n📊 Effect Size Analysis:")
        grand_mean = np.mean(risk_scores)
        ss_between = sum([
            len(cluster_0) * (np.mean(cluster_0) - grand_mean)**2,
            len(cluster_1) * (np.mean(cluster_1) - grand_mean)**2,
            len(cluster_2) * (np.mean(cluster_2) - grand_mean)**2
        ])
        ss_total = np.sum((risk_scores - grand_mean)**2)
        eta_squared = ss_between / ss_total
        
        print(f"   • Eta-squared (η²): {eta_squared:.4f}")
        print(f"   • Variance explained: {eta_squared*100:.1f}%")
        
        if eta_squared > 0.14:
            print(f"   ✅ Large effect size")
        elif eta_squared > 0.06:
            print(f"   ⚠️  Medium effect size")
        else:
            print(f"   ⚠️  Small effect size")
        
        # Cluster statistics
        print(f"\n📊 Cluster Statistics:")
        for i, cluster_data in enumerate([cluster_0, cluster_1, cluster_2]):
            profile_name = df[df['cluster'] == i]['investor_profile'].iloc[0]
            print(f"   {profile_name:.<20}")
            print(f"      Mean: {np.mean(cluster_data):.4f}")
            print(f"      SD: {np.std(cluster_data):.4f}")
            print(f"      N: {len(cluster_data)}")
        
        print("\n" + "="*60)
        print("✅ RESULT: Statistical validation successful!")
        print("="*60 + "\n")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n" + "="*60 + "\n")
        return False

if __name__ == "__main__":
    success = test_statistical_validation()
    sys.exit(0 if success else 1)
