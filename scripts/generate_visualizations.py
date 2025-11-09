import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Create directory for images
os.makedirs('/Users/ganeshmaragani/Documents/994/00_MBA_BITS/4th sem/MI-Project/docs/images', exist_ok=True)

# Load investor profiles data
try:
    df = pd.read_csv('/Users/ganeshmaragani/Documents/994/00_MBA_BITS/4th sem/MI-Project/investor_profiles_results.csv')
    print("Loaded investor profiles data")
except:
    print("Could not load investor profiles data, creating sample data")
    # Create sample data
    df = pd.DataFrame({
        'composite_risk_score': np.random.random(30),
        'investment_horizon_years': np.random.choice([1, 3, 5, 10, 15], size=30),
        'investor_profile': np.random.choice(['Conservative', 'Balanced', 'Aggressive'], size=30)
    })

# Recreate portfolio allocation visualizations
portfolio_allocations = {
    'Conservative': {
        'Large Cap Equity': 15,
        'Mid & Small Cap Equity': 5,
        'Government Bonds': 40,
        'Corporate Bonds': 25,
        'Gold': 5,
        'Cash': 10
    },
    'Balanced': {
        'Large Cap Equity': 30,
        'Mid & Small Cap Equity': 15,
        'Government Bonds': 20,
        'Corporate Bonds': 20,
        'Gold': 10,
        'Cash': 5
    },
    'Aggressive': {
        'Large Cap Equity': 40,
        'Mid & Small Cap Equity': 30,
        'Government Bonds': 5,
        'Corporate Bonds': 10,
        'Gold': 10,
        'Cash': 5
    }
}

# Create a DataFrame for the allocations
allocation_df = pd.DataFrame(portfolio_allocations)

# Plot and save the portfolio allocations
plt.figure(figsize=(14, 10))

# Bar chart
plt.subplot(2, 1, 1)
allocation_df.plot(kind='bar', ax=plt.gca())
plt.title('Recommended Portfolio Allocations by Investor Profile', fontsize=14)
plt.xlabel('Asset Class', fontsize=12)
plt.ylabel('Allocation (%)', fontsize=12)
plt.legend(title='Investor Profile')
plt.grid(axis='y', alpha=0.3)

# For each bar, add a label showing the percentage
for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt='%d%%')

# Pie charts for each profile
plt.subplot(2, 3, 4)
allocation_df['Conservative'].plot(kind='pie', autopct='%1.0f%%', title='Conservative Portfolio')
plt.ylabel('')

plt.subplot(2, 3, 5)
allocation_df['Balanced'].plot(kind='pie', autopct='%1.0f%%', title='Balanced Portfolio')
plt.ylabel('')

plt.subplot(2, 3, 6)
allocation_df['Aggressive'].plot(kind='pie', autopct='%1.0f%%', title='Aggressive Portfolio')
plt.ylabel('')

plt.tight_layout()
plt.savefig('/Users/ganeshmaragani/Documents/994/00_MBA_BITS/4th sem/MI-Project/docs/images/portfolio_allocations.png', dpi=300)
print("Portfolio allocations saved")

# Plot risk profile distribution
plt.figure(figsize=(10, 6))
profile_counts = df['investor_profile'].value_counts()
profile_counts.plot.pie(autopct='%1.1f%%', startangle=90, explode=[0.05, 0.05, 0.05])
plt.title('Distribution of Investor Profiles', fontsize=14)
plt.ylabel('')
plt.savefig('/Users/ganeshmaragani/Documents/994/00_MBA_BITS/4th sem/MI-Project/docs/images/investor_profile_distribution.png', dpi=300)
print("Investor profile distribution saved")

# If we have the required columns, create PCA visualization
if 'composite_risk_score' in df.columns and 'investment_horizon_years' in df.columns:
    plt.figure(figsize=(12, 8))
    
    # Plot each investor profile with different color
    for profile, color in zip(['Conservative', 'Balanced', 'Aggressive'], ['blue', 'green', 'red']):
        subset = df[df['investor_profile'] == profile]
        plt.scatter(
            subset['composite_risk_score'], 
            subset['investment_horizon_years'], 
            c=color, label=profile, alpha=0.7
        )
    
    plt.title('Investor Clusters Visualization', fontsize=14)
    plt.xlabel('Risk Score', fontsize=12)
    plt.ylabel('Investment Horizon (Years)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/Users/ganeshmaragani/Documents/994/00_MBA_BITS/4th sem/MI-Project/docs/images/investor_clusters.png', dpi=300)
    print("Investor clusters visualization saved")

print("All visualizations have been saved to the docs/images directory")