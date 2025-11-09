#!/usr/bin/env python3
"""
Statistical Validation of Clustering Results
Performs ANOVA, Chi-square tests, and bootstrap validation
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("STATISTICAL VALIDATION OF CLUSTERING RESULTS")
print("="*80)

# Load results
df = pd.read_csv('investor_profiles_results.csv')
print(f"\n✓ Loaded {len(df)} investor profiles")

# ============================================================================
# 1. ANOVA F-TEST: Test if cluster means are significantly different
# ============================================================================
print("\n" + "="*80)
print("STEP 1: ANOVA F-TEST FOR CLUSTER SEPARATION")
print("="*80)

# Test composite risk score across clusters
conservative = df[df['investor_profile'] == 'Conservative']['composite_risk_score']
balanced = df[df['investor_profile'] == 'Balanced']['composite_risk_score']
aggressive = df[df['investor_profile'] == 'Aggressive']['composite_risk_score']

f_stat, p_value = stats.f_oneway(conservative, balanced, aggressive)

print(f"\nComposite Risk Score by Profile:")
print(f"  Conservative (n={len(conservative)}): Mean={conservative.mean():.4f}, Std={conservative.std():.4f}")
print(f"  Balanced (n={len(balanced)}): Mean={balanced.mean():.4f}, Std={balanced.std():.4f}")
print(f"  Aggressive (n={len(aggressive)}): Mean={aggressive.mean():.4f}, Std={aggressive.std():.4f}")

print(f"\nANOVA Results:")
print(f"  F-statistic: {f_stat:.4f}")
print(f"  p-value: {p_value:.6f}")

if p_value < 0.05:
    print(f"  ✓ SIGNIFICANT: Cluster means are statistically different (p < 0.05)")
else:
    print(f"  ✗ NOT SIGNIFICANT: Cluster means are not statistically different (p >= 0.05)")

# Effect size (eta-squared)
grand_mean = df['composite_risk_score'].mean()
ss_between = sum([len(df[df['investor_profile'] == profile]) * 
                  (df[df['investor_profile'] == profile]['composite_risk_score'].mean() - grand_mean)**2 
                  for profile in ['Conservative', 'Balanced', 'Aggressive']])
ss_total = sum((df['composite_risk_score'] - grand_mean)**2)
eta_squared = ss_between / ss_total

print(f"  Effect Size (η²): {eta_squared:.4f}")
if eta_squared > 0.14:
    print(f"  ✓ LARGE effect size (η² > 0.14)")
elif eta_squared > 0.06:
    print(f"  ✓ MEDIUM effect size (η² > 0.06)")
else:
    print(f"  ○ SMALL effect size (η² <= 0.06)")

# ============================================================================
# 2. POST-HOC PAIRWISE COMPARISONS (Tukey HSD)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: POST-HOC PAIRWISE COMPARISONS")
print("="*80)

from scipy.stats import ttest_ind

# Conservative vs Balanced
t_stat_cb, p_cb = ttest_ind(conservative, balanced)
print(f"\nConservative vs Balanced:")
print(f"  Mean difference: {balanced.mean() - conservative.mean():.4f}")
print(f"  t-statistic: {t_stat_cb:.4f}, p-value: {p_cb:.6f}")
print(f"  {'✓ SIGNIFICANT' if p_cb < 0.05 else '✗ NOT SIGNIFICANT'}")

# Conservative vs Aggressive
t_stat_ca, p_ca = ttest_ind(conservative, aggressive)
print(f"\nConservative vs Aggressive:")
print(f"  Mean difference: {aggressive.mean() - conservative.mean():.4f}")
print(f"  t-statistic: {t_stat_ca:.4f}, p-value: {p_ca:.6f}")
print(f"  {'✓ SIGNIFICANT' if p_ca < 0.05 else '✗ NOT SIGNIFICANT'}")

# Balanced vs Aggressive
t_stat_ba, p_ba = ttest_ind(balanced, aggressive)
print(f"\nBalanced vs Aggressive:")
print(f"  Mean difference: {aggressive.mean() - balanced.mean():.4f}")
print(f"  t-statistic: {t_stat_ba:.4f}, p-value: {p_ba:.6f}")
print(f"  {'✓ SIGNIFICANT' if p_ba < 0.05 else '✗ NOT SIGNIFICANT'}")

# ============================================================================
# 3. CHI-SQUARE TESTS: Test association with categorical variables
# ============================================================================
print("\n" + "="*80)
print("STEP 3: CHI-SQUARE TESTS FOR CATEGORICAL ASSOCIATIONS")
print("="*80)

# Test Age Group
age_col = [col for col in df.columns if 'age' in col.lower() and col != 'composite_risk_score'][0]
contingency_age = pd.crosstab(df[age_col], df['investor_profile'])
chi2_age, p_age, dof_age, expected_age = stats.chi2_contingency(contingency_age)

print(f"\nAge Group vs Investor Profile:")
print(contingency_age)
print(f"\n  Chi-square statistic: {chi2_age:.4f}")
print(f"  p-value: {p_age:.6f}")
print(f"  Degrees of freedom: {dof_age}")
print(f"  {'✓ SIGNIFICANT association' if p_age < 0.05 else '✗ NO significant association'}")

# Test Gender
gender_col = [col for col in df.columns if 'gender' in col.lower()][0]
contingency_gender = pd.crosstab(df[gender_col], df['investor_profile'])
chi2_gender, p_gender, dof_gender, expected_gender = stats.chi2_contingency(contingency_gender)

print(f"\nGender vs Investor Profile:")
print(contingency_gender)
print(f"\n  Chi-square statistic: {chi2_gender:.4f}")
print(f"  p-value: {p_gender:.6f}")
print(f"  Degrees of freedom: {dof_gender}")
print(f"  {'✓ SIGNIFICANT association' if p_gender < 0.05 else '✗ NO significant association'}")

# Test Education
education_col = [col for col in df.columns if 'education' in col.lower()][0]
contingency_education = pd.crosstab(df[education_col], df['investor_profile'])
chi2_education, p_education, dof_education, expected_education = stats.chi2_contingency(contingency_education)

print(f"\nEducation vs Investor Profile:")
print(contingency_education)
print(f"\n  Chi-square statistic: {chi2_education:.4f}")
print(f"  p-value: {p_education:.6f}")
print(f"  Degrees of freedom: {dof_education}")
print(f"  {'✓ SIGNIFICANT association' if p_education < 0.05 else '✗ NO significant association'}")

# ============================================================================
# 4. BOOTSTRAP VALIDATION: Test clustering stability
# ============================================================================
print("\n" + "="*80)
print("STEP 4: BOOTSTRAP VALIDATION (1000 iterations)")
print("="*80)

# Prepare features
features = df[['composite_risk_score', 'investment_horizon_years']].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

n_bootstrap = 1000
silhouette_scores = []

print(f"\nRunning bootstrap validation...")
for i in range(n_bootstrap):
    # Resample with replacement
    X_resampled = resample(features_scaled, random_state=i)
    
    # Fit K-means
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_resampled)
    
    # Calculate silhouette score
    score = silhouette_score(X_resampled, labels)
    silhouette_scores.append(score)
    
    if (i + 1) % 200 == 0:
        print(f"  Completed {i + 1}/{n_bootstrap} iterations...")

silhouette_scores = np.array(silhouette_scores)

print(f"\nBootstrap Silhouette Score Statistics:")
print(f"  Mean: {silhouette_scores.mean():.4f}")
print(f"  Std Dev: {silhouette_scores.std():.4f}")
print(f"  95% CI: [{np.percentile(silhouette_scores, 2.5):.4f}, {np.percentile(silhouette_scores, 97.5):.4f}]")
print(f"  Min: {silhouette_scores.min():.4f}")
print(f"  Max: {silhouette_scores.max():.4f}")

# ============================================================================
# 5. CLUSTER STABILITY: Jaccard Index
# ============================================================================
print("\n" + "="*80)
print("STEP 5: CLUSTER STABILITY ANALYSIS")
print("="*80)

from sklearn.metrics import adjusted_rand_score

# Run K-means 100 times with different random states
rand_scores = []
for i in range(100):
    kmeans1 = KMeans(n_clusters=3, n_init=10, random_state=i)
    kmeans2 = KMeans(n_clusters=3, n_init=10, random_state=i+100)
    
    labels1 = kmeans1.fit_predict(features_scaled)
    labels2 = kmeans2.fit_predict(features_scaled)
    
    rand_score = adjusted_rand_score(labels1, labels2)
    rand_scores.append(rand_score)

rand_scores = np.array(rand_scores)

print(f"\nAdjusted Rand Index (Cluster Stability):")
print(f"  Mean: {rand_scores.mean():.4f}")
print(f"  Std Dev: {rand_scores.std():.4f}")
print(f"  Min: {rand_scores.min():.4f}")
print(f"  Max: {rand_scores.max():.4f}")

if rand_scores.mean() > 0.9:
    print(f"  ✓ EXCELLENT stability (mean ARI > 0.9)")
elif rand_scores.mean() > 0.7:
    print(f"  ✓ GOOD stability (mean ARI > 0.7)")
elif rand_scores.mean() > 0.5:
    print(f"  ○ MODERATE stability (mean ARI > 0.5)")
else:
    print(f"  ✗ POOR stability (mean ARI <= 0.5)")

# ============================================================================
# 6. SAVE VALIDATION RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: SAVING VALIDATION RESULTS")
print("="*80)

validation_results = {
    'Test': [
        'ANOVA F-test (Risk Score)',
        'Eta-squared (Effect Size)',
        'Conservative vs Balanced t-test',
        'Conservative vs Aggressive t-test',
        'Balanced vs Aggressive t-test',
        'Chi-square (Age)',
        'Chi-square (Gender)',
        'Chi-square (Education)',
        'Bootstrap Silhouette (Mean)',
        'Bootstrap Silhouette (95% CI Lower)',
        'Bootstrap Silhouette (95% CI Upper)',
        'Cluster Stability (Adj Rand Index)'
    ],
    'Statistic': [
        f'{f_stat:.4f}',
        f'{eta_squared:.4f}',
        f'{t_stat_cb:.4f}',
        f'{t_stat_ca:.4f}',
        f'{t_stat_ba:.4f}',
        f'{chi2_age:.4f}',
        f'{chi2_gender:.4f}',
        f'{chi2_education:.4f}',
        f'{silhouette_scores.mean():.4f}',
        f'{np.percentile(silhouette_scores, 2.5):.4f}',
        f'{np.percentile(silhouette_scores, 97.5):.4f}',
        f'{rand_scores.mean():.4f}'
    ],
    'p-value': [
        f'{p_value:.6f}',
        'N/A',
        f'{p_cb:.6f}',
        f'{p_ca:.6f}',
        f'{p_ba:.6f}',
        f'{p_age:.6f}',
        f'{p_gender:.6f}',
        f'{p_education:.6f}',
        'N/A',
        'N/A',
        'N/A',
        'N/A'
    ],
    'Result': [
        'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT',
        'LARGE' if eta_squared > 0.14 else 'MEDIUM' if eta_squared > 0.06 else 'SMALL',
        'SIGNIFICANT' if p_cb < 0.05 else 'NOT SIGNIFICANT',
        'SIGNIFICANT' if p_ca < 0.05 else 'NOT SIGNIFICANT',
        'SIGNIFICANT' if p_ba < 0.05 else 'NOT SIGNIFICANT',
        'SIGNIFICANT' if p_age < 0.05 else 'NOT SIGNIFICANT',
        'SIGNIFICANT' if p_gender < 0.05 else 'NOT SIGNIFICANT',
        'SIGNIFICANT' if p_education < 0.05 else 'NOT SIGNIFICANT',
        f'Std={silhouette_scores.std():.4f}',
        'Bootstrap CI',
        'Bootstrap CI',
        'EXCELLENT' if rand_scores.mean() > 0.9 else 'GOOD' if rand_scores.mean() > 0.7 else 'MODERATE'
    ]
}

validation_df = pd.DataFrame(validation_results)
validation_df.to_csv('statistical_validation_results.csv', index=False)
print(f"\n✓ Saved: statistical_validation_results.csv")

# Save detailed cross-tabulations
with open('crosstab_details.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("DETAILED CROSS-TABULATIONS\n")
    f.write("="*80 + "\n\n")
    
    f.write("Age Group vs Investor Profile:\n")
    f.write(str(contingency_age) + "\n\n")
    
    f.write("Gender vs Investor Profile:\n")
    f.write(str(contingency_gender) + "\n\n")
    
    f.write("Education vs Investor Profile:\n")
    f.write(str(contingency_education) + "\n\n")

print(f"✓ Saved: crosstab_details.txt")

print("\n" + "="*80)
print("STATISTICAL VALIDATION COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nGenerated Files:")
print("  1. statistical_validation_results.csv")
print("  2. crosstab_details.txt")
print("\nKey Findings:")
print(f"  - ANOVA p-value: {p_value:.6f} ({'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'})")
print(f"  - Effect size η²: {eta_squared:.4f} ({'LARGE' if eta_squared > 0.14 else 'MEDIUM' if eta_squared > 0.06 else 'SMALL'})")
print(f"  - Bootstrap silhouette: {silhouette_scores.mean():.4f} ± {silhouette_scores.std():.4f}")
print(f"  - Cluster stability: {rand_scores.mean():.4f} ({'EXCELLENT' if rand_scores.mean() > 0.9 else 'GOOD' if rand_scores.mean() > 0.7 else 'MODERATE'})")
print("="*80 + "\n")
