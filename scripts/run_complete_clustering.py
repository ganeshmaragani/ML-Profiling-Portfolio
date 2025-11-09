"""
Complete Clustering Analysis for 38 Investor Responses
Generates updated investor_profiles_results.csv and all visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("INVESTOR PROFILING - COMPLETE CLUSTERING ANALYSIS")
print("="*80)

# Load the dataset
file_path = 'investor_survey_data_with_timestamps.csv'
df = pd.read_csv(file_path)

print(f"\n✓ Loaded dataset: {df.shape[0]} responses, {df.shape[1]} columns")
print(f"Note: Working with {df.shape[0]} valid survey responses")

# Create a clean copy
df_clean = df.copy()

# ============================================================================
# STEP 1: CREATE COMPOSITE RISK SCORE
# ============================================================================
print("\n" + "="*80)
print("STEP 1: CREATING COMPOSITE RISK SCORE")
print("="*80)

# Define risk-related questions
risk_questions = {
    'Q14': '14. If your investment dropped 20% in value over a short period what would you do?',
    'Q15': '15. Which statement best describes your investment approach?',
    'Q16': '16. For an investment that has potential for higher returns but also higher risk what percentage of your portfolio would you allocate?',
    'Q17': '17. Rate your agreement with this statement: "I am willing to accept significant short-term losses for potentially higher long-term returns."',
    'Q18': '18. Which scenario would you prefer?',
    'Q19': '19. If you had ₹100000 to invest which option would you choose?'
}

# Mapping dictionaries for each question
q14_map = {
    'Sell everything and move to safer investments': 0,
    'Sell everything to avoid further losses': 0,
    'Sell a portion to reduce risk': 1,
    'Become concerned but take no action': 2,
    'Do nothing and wait for recovery': 3,
    'Buy more at the lower price': 4,
    'See it as an opportunity to buy more': 4,
    'Panic and consider selling investments': 0
}

q15_map = {
    'I prefer guaranteed returns even if they are low': 0,
    'I prefer a mix of stable and growth investments': 2,
    'I prefer growth investments even with higher volatility': 3,
    'I seek maximum growth and can accept significant volatility': 4
}

q16_map = {
    'Less than 1%': 0,
    '1% - 25%': 1,
    '26% - 50%': 2,
    '51% - 75%': 3,
    'More than 75%': 4
}

q17_map = {
    'Strongly disagree': 0,
    'Disagree': 1,
    'Neither agree nor disagree': 2,
    'Agree': 3,
    'Strongly agree': 4
}

q18_map = {
    'Investment A: 7% average annual return with minimal fluctuations': 1,
    'Investment B: 10% average annual return with moderate fluctuations': 2,
    'Investment C: 13% average annual return with significant fluctuations': 3
}

q19_map = {
    'A guaranteed return of ₹5000 (5%)': 1,
    'A guaranteed return of ₹5,000 (5%)': 1,
    '50% chance of earning ₹12000 and 50% chance of earning ₹2000': 2,
    '50% chance of earning ₹12,000 and 50% chance of earning ₹2,000': 2,
    '30% chance of earning ₹25000 and 70% chance of earning ₹0': 3,
    '30% chance of earning ₹25,000 and 70% chance of earning ₹0': 3
}

# Apply mappings
if risk_questions['Q14'] in df_clean.columns:
    df_clean['risk_score_q14'] = df_clean[risk_questions['Q14']].map(q14_map)
    print(f"✓ Q14 mapped: {df_clean['risk_score_q14'].notna().sum()} values")

if risk_questions['Q15'] in df_clean.columns:
    df_clean['risk_score_q15'] = df_clean[risk_questions['Q15']].map(q15_map)
    print(f"✓ Q15 mapped: {df_clean['risk_score_q15'].notna().sum()} values")

if risk_questions['Q16'] in df_clean.columns:
    df_clean['risk_score_q16'] = df_clean[risk_questions['Q16']].map(q16_map)
    print(f"✓ Q16 mapped: {df_clean['risk_score_q16'].notna().sum()} values")

if risk_questions['Q17'] in df_clean.columns:
    df_clean['risk_score_q17'] = df_clean[risk_questions['Q17']].map(q17_map)
    print(f"✓ Q17 mapped: {df_clean['risk_score_q17'].notna().sum()} values")

if risk_questions['Q18'] in df_clean.columns:
    df_clean['risk_score_q18'] = df_clean[risk_questions['Q18']].map(q18_map)
    print(f"✓ Q18 mapped: {df_clean['risk_score_q18'].notna().sum()} values")

if risk_questions['Q19'] in df_clean.columns:
    df_clean['risk_score_q19'] = df_clean[risk_questions['Q19']].map(q19_map)
    print(f"✓ Q19 mapped: {df_clean['risk_score_q19'].notna().sum()} values")

# Calculate composite risk score
risk_score_columns = [col for col in df_clean.columns if col.startswith('risk_score_q')]
print(f"\n✓ Using {len(risk_score_columns)} risk questions for composite score")

if risk_score_columns:
    # Calculate average of all risk scores
    df_clean['composite_risk_score'] = df_clean[risk_score_columns].mean(axis=1)
    
    # Normalize to 0-1 scale
    min_risk = df_clean['composite_risk_score'].min()
    max_risk = df_clean['composite_risk_score'].max()
    df_clean['composite_risk_score'] = ((df_clean['composite_risk_score'] - min_risk) / 
                                         (max_risk - min_risk))
    
    print(f"\n✓ Composite Risk Score Statistics:")
    print(f"  Mean: {df_clean['composite_risk_score'].mean():.4f}")
    print(f"  Std Dev: {df_clean['composite_risk_score'].std():.4f}")
    print(f"  Min: {df_clean['composite_risk_score'].min():.4f}")
    print(f"  Max: {df_clean['composite_risk_score'].max():.4f}")

# ============================================================================
# STEP 2: PREPARE FEATURES FOR CLUSTERING
# ============================================================================
print("\n" + "="*80)
print("STEP 2: PREPARING FEATURES FOR K-MEANS CLUSTERING")
print("="*80)

# Select features for clustering
features_for_clustering = ['composite_risk_score']

# Add investment horizon if available
horizon_col = '9. What is your investment time horizon for your primary financial goal?'
if horizon_col in df_clean.columns:
    # Map horizon to years
    horizon_map = {
        '1-3 years': 2,
        '3-5 years': 4,
        '5-10 years': 7.5,
        'More than 10 years': 15
    }
    df_clean['investment_horizon_years'] = df_clean[horizon_col].map(horizon_map)
    features_for_clustering.append('investment_horizon_years')
    print(f"✓ Added investment horizon feature")

print(f"\n✓ Features selected for clustering: {features_for_clustering}")

# Create feature matrix
X = df_clean[features_for_clustering].dropna()
print(f"✓ Feature matrix shape: {X.shape}")

# Keep track of indices
valid_indices = X.index

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✓ Features standardized (mean=0, std=1)")

# ============================================================================
# STEP 3: DETERMINE OPTIMAL CLUSTERS (ELBOW & SILHOUETTE)
# ============================================================================
print("\n" + "="*80)
print("STEP 3: DETERMINING OPTIMAL NUMBER OF CLUSTERS")
print("="*80)

inertia_scores = []
silhouette_scores = []
k_range = range(2, min(11, len(X_scaled)))

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_scores.append(kmeans.inertia_)
    
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X_scaled, labels)
    silhouette_scores.append(silhouette_avg)
    
    print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_avg:.4f}")

# Plot Elbow Method and Silhouette Scores
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(list(k_range), inertia_scores, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
ax1.set_ylabel('Inertia (WCSS)', fontsize=12)
ax1.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=3, color='r', linestyle='--', alpha=0.5, label='k=3 (Selected)')
ax1.legend()

ax2.plot(list(k_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Analysis for Optimal k', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=3, color='r', linestyle='--', alpha=0.5, label='k=3 (Selected)')
ax2.axhline(y=silhouette_scores[1], color='r', linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.savefig('cluster_optimization_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: cluster_optimization_analysis.png")
plt.close()

# ============================================================================
# STEP 4: APPLY K-MEANS CLUSTERING (k=3)
# ============================================================================
print("\n" + "="*80)
print("STEP 4: APPLYING K-MEANS CLUSTERING (k=3)")
print("="*80)

optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Calculate final silhouette score
final_silhouette = silhouette_score(X_scaled, cluster_labels)
print(f"\n✓ Final Silhouette Score: {final_silhouette:.4f}")
print(f"✓ Final Inertia: {kmeans_final.inertia_:.2f}")

# Add cluster labels to dataframe
df_clean.loc[valid_indices, 'cluster'] = cluster_labels

# Map clusters to investor profiles based on risk score
cluster_means = df_clean.loc[valid_indices].groupby('cluster')['composite_risk_score'].mean()
cluster_mapping = dict(zip(cluster_means.sort_values().index, 
                          ['Conservative', 'Balanced', 'Aggressive']))

df_clean.loc[valid_indices, 'investor_profile'] = df_clean.loc[valid_indices, 'cluster'].map(cluster_mapping)
df_clean.loc[valid_indices, 'risk_profile'] = df_clean.loc[valid_indices, 'investor_profile']

# Display cluster distribution
print(f"\n✓ Investor Profile Distribution:")
profile_counts = df_clean['investor_profile'].value_counts()
for profile, count in profile_counts.items():
    percentage = (count / len(valid_indices)) * 100
    print(f"  {profile}: {count} investors ({percentage:.1f}%)")

# ============================================================================
# STEP 5: ANALYZE CLUSTER CHARACTERISTICS
# ============================================================================
print("\n" + "="*80)
print("STEP 5: CLUSTER CHARACTERISTICS ANALYSIS")
print("="*80)

print("\nCluster Centers (Standardized):")
for i, profile in enumerate(['Conservative', 'Balanced', 'Aggressive']):
    cluster_num = [k for k, v in cluster_mapping.items() if v == profile][0]
    cluster_num = int(cluster_num)  # Ensure it's an integer
    print(f"\n{profile} (Cluster {cluster_num}):")
    for j, feature in enumerate(features_for_clustering):
        print(f"  {feature}: {kmeans_final.cluster_centers_[cluster_num, j]:.4f}")

print("\nCluster Statistics (Original Scale):")
for profile in ['Conservative', 'Balanced', 'Aggressive']:
    cluster_data = df_clean[df_clean['investor_profile'] == profile]
    print(f"\n{profile} Profile:")
    print(f"  Mean Risk Score: {cluster_data['composite_risk_score'].mean():.4f}")
    print(f"  Std Risk Score: {cluster_data['composite_risk_score'].std():.4f}")
    if 'investment_horizon_years' in features_for_clustering:
        print(f"  Mean Horizon: {cluster_data['investment_horizon_years'].mean():.2f} years")

# ============================================================================
# STEP 6: CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: CREATING VISUALIZATIONS")
print("="*80)

# Visualization 1: Cluster Distribution
plt.figure(figsize=(10, 6))
profile_order = ['Conservative', 'Balanced', 'Aggressive']
colors = ['#2E7D32', '#FFA726', '#D32F2F']
ax = sns.countplot(data=df_clean, x='investor_profile', order=profile_order, palette=colors)
plt.title(f'Distribution of Investor Profiles ({len(df_clean)} Responses)', fontsize=16, fontweight='bold')
plt.xlabel('Investor Profile', fontsize=12)
plt.ylabel('Number of Investors', fontsize=12)
for container in ax.containers:
    ax.bar_label(container, fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('investor_profile_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: investor_profile_distribution.png")
plt.close()

# Visualization 2: Risk Score Distribution by Profile
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_clean, x='investor_profile', y='composite_risk_score', 
            order=profile_order, palette=colors)
sns.swarmplot(data=df_clean, x='investor_profile', y='composite_risk_score', 
              order=profile_order, color='black', alpha=0.5, size=4)
plt.title('Risk Tolerance Score by Investor Profile', fontsize=16, fontweight='bold')
plt.xlabel('Investor Profile', fontsize=12)
plt.ylabel('Composite Risk Score (0-1)', fontsize=12)
plt.tight_layout()
plt.savefig('risk_score_by_profile.png', dpi=300, bbox_inches='tight')
print("✓ Saved: risk_score_by_profile.png")
plt.close()

# Visualization 3: 2D Cluster Scatter Plot
if len(features_for_clustering) >= 2:
    plt.figure(figsize=(12, 8))
    colors_map = {'Conservative': '#2E7D32', 'Balanced': '#FFA726', 'Aggressive': '#D32F2F'}
    
    for profile in profile_order:
        mask = df_clean['investor_profile'] == profile
        plt.scatter(df_clean.loc[mask, 'composite_risk_score'],
                   df_clean.loc[mask, 'investment_horizon_years'],
                   label=profile, alpha=0.7, s=100, 
                   c=colors_map[profile], edgecolors='black', linewidth=1)
    
    # Plot centroids
    for i, profile in enumerate(profile_order):
        cluster_num = [k for k, v in cluster_mapping.items() if v == profile][0]
        cluster_num = int(cluster_num)  # Ensure it's an integer
        centroid = scaler.inverse_transform(kmeans_final.cluster_centers_[cluster_num:cluster_num+1])[0]
        plt.scatter(centroid[0], centroid[1], marker='X', s=300, 
                   c=colors_map[profile], edgecolors='black', linewidth=2,
                   label=f'{profile} Centroid')
    
    plt.xlabel('Composite Risk Score', fontsize=12)
    plt.ylabel('Investment Horizon (Years)', fontsize=12)
    plt.title('K-Means Clustering of Investor Profiles', fontsize=16, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('kmeans_cluster_scatter.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: kmeans_cluster_scatter.png")
    plt.close()

# Visualization 4: PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
for i, profile in enumerate(profile_order):
    cluster_num = [k for k, v in cluster_mapping.items() if v == profile][0]
    mask = cluster_labels == cluster_num
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               label=profile, alpha=0.7, s=100,
               c=colors_map[profile], edgecolors='black', linewidth=1)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
plt.title('PCA Visualization of Investor Clusters', fontsize=16, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_cluster_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: pca_cluster_visualization.png")
plt.close()

# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 7: SAVING RESULTS")
print("="*80)

# Save updated results
df_clean.to_csv('investor_profiles_results.csv', index=False)
print(f"✓ Saved: investor_profiles_results.csv ({len(df_clean)} rows)")

# Save cluster summary
summary_data = []
for profile in profile_order:
    cluster_data = df_clean[df_clean['investor_profile'] == profile]
    summary_data.append({
        'Profile': profile,
        'Count': len(cluster_data),
        'Percentage': f"{(len(cluster_data)/len(valid_indices)*100):.1f}%",
        'Mean_Risk_Score': f"{cluster_data['composite_risk_score'].mean():.4f}",
        'Std_Risk_Score': f"{cluster_data['composite_risk_score'].std():.4f}",
        'Mean_Horizon': f"{cluster_data['investment_horizon_years'].mean():.2f}" if 'investment_horizon_years' in cluster_data else 'N/A'
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('cluster_summary_statistics.csv', index=False)
print(f"✓ Saved: cluster_summary_statistics.csv")

print("\n" + "="*80)
print("CLUSTERING ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nGenerated Files:")
print("  1. investor_profiles_results.csv")
print("  2. cluster_summary_statistics.csv")
print("  3. cluster_optimization_analysis.png")
print("  4. investor_profile_distribution.png")
print("  5. risk_score_by_profile.png")
print("  6. kmeans_cluster_scatter.png")
print("  7. pca_cluster_visualization.png")
print("\nFinal Metrics:")
print(f"  - Total Investors Analyzed: {len(valid_indices)}")
print(f"  - Silhouette Score: {final_silhouette:.4f}")
print(f"  - Number of Clusters: {optimal_k}")
print("="*80)
