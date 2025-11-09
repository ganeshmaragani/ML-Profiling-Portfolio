#!/usr/bin/env python3
"""
Test 3: Clustering Analysis
Performs K-Means clustering and validates results.
"""

import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def test_clustering():
    """Test K-Means clustering analysis."""
    print("\n" + "="*60)
    print("TEST 3: CLUSTERING ANALYSIS")
    print("="*60 + "\n")
    
    try:
        # Load processed results
        print("📊 Loading processed results...")
        df = pd.read_csv('data/investor_profiles_results.csv')
        
        print(f"✅ Data loaded: {len(df)} respondents\n")
        
        # Extract features for clustering
        X = df[['composite_risk_score']].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply K-Means
        print("🤖 Running K-Means clustering (k=3)...")
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, clusters)
        
        # Assign profile names
        df['cluster_new'] = clusters
        cluster_means = df.groupby('cluster_new')['composite_risk_score'].mean().sort_values()
        profile_mapping = {
            cluster_means.index[0]: 'Conservative',
            cluster_means.index[1]: 'Balanced',
            cluster_means.index[2]: 'Aggressive'
        }
        df['profile_new'] = df['cluster_new'].map(profile_mapping)
        
        print("\n✅ Clustering complete!\n")
        print("📊 Cluster Distribution:")
        for profile in ['Conservative', 'Balanced', 'Aggressive']:
            count = len(df[df['profile_new'] == profile])
            percentage = count / len(df) * 100
            mean_score = df[df['profile_new'] == profile]['composite_risk_score'].mean()
            print(f"   • {profile:.<20} {count:>3} investors ({percentage:>5.1f}%) - Mean: {mean_score:.3f}")
        
        print(f"\n📊 Quality Metrics:")
        print(f"   • Silhouette Score: {silhouette:.4f}")
        if silhouette > 0.5:
            print(f"   ✅ Good cluster separation")
        elif silhouette > 0.25:
            print(f"   ⚠️  Moderate cluster separation")
        else:
            print(f"   ❌ Poor cluster separation")
        
        print("\n" + "="*60)
        print("✅ RESULT: Clustering analysis successful!")
        print("="*60 + "\n")
        return True
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Required data file not found!")
        print(f"   {str(e)}")
        print("\n" + "="*60 + "\n")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n" + "="*60 + "\n")
        return False

if __name__ == "__main__":
    success = test_clustering()
    sys.exit(0 if success else 1)
