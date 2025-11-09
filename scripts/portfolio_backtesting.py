"""
Portfolio Backtesting Script
============================
Calculates returns for Conservative, Balanced, and Aggressive portfolios
using historical market data and compares against benchmarks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PORTFOLIO BACKTESTING ANALYSIS")
print("="*70)

# ============================================================================
# 1. LOAD MARKET DATA
# ============================================================================
print("\n[1/6] Loading market data...")

try:
    nifty50 = pd.read_csv('market_data_nifty50.csv', index_col=0)
    nifty50.index = pd.to_datetime(nifty50.index)
    
    nifty_midcap = pd.read_csv('market_data_nifty_midcap.csv', index_col=0)
    nifty_midcap.index = pd.to_datetime(nifty_midcap.index)
    
    gold = pd.read_csv('market_data_gold.csv', index_col=0)
    gold.index = pd.to_datetime(gold.index)
    
    govt_bonds = pd.read_csv('market_data_govt_bonds.csv', index_col=0)
    govt_bonds.index = pd.to_datetime(govt_bonds.index)
    
    corp_bonds = pd.read_csv('market_data_corp_bonds.csv', index_col=0)
    corp_bonds.index = pd.to_datetime(corp_bonds.index)
    
    print("✅ All market data loaded successfully")
    print(f"   Date range: {nifty50.index[0].date()} to {nifty50.index[-1].date()}")
    print(f"   Trading days: {len(nifty50)}")
    
except Exception as e:
    print(f"❌ Error loading market data: {e}")
    exit(1)

# ============================================================================
# 2. DEFINE PORTFOLIO ALLOCATIONS
# ============================================================================
print("\n[2/6] Defining portfolio allocations...")

portfolios = {
    'Conservative': {
        'Large Cap (Nifty50)': 0.15,
        'Mid Cap': 0.05,
        'Govt Bonds': 0.40,
        'Corp Bonds': 0.25,
        'Gold': 0.05,
        'Cash': 0.10
    },
    'Balanced': {
        'Large Cap (Nifty50)': 0.30,
        'Mid Cap': 0.15,
        'Govt Bonds': 0.20,
        'Corp Bonds': 0.20,
        'Gold': 0.10,
        'Cash': 0.05
    },
    'Aggressive': {
        'Large Cap (Nifty50)': 0.40,
        'Mid Cap': 0.30,
        'Govt Bonds': 0.05,
        'Corp Bonds': 0.10,
        'Gold': 0.10,
        'Cash': 0.05
    }
}

for profile, allocation in portfolios.items():
    print(f"\n{profile} Portfolio:")
    for asset, weight in allocation.items():
        print(f"  {asset}: {weight*100:.0f}%")

# ============================================================================
# 3. CALCULATE PORTFOLIO RETURNS
# ============================================================================
print("\n[3/6] Calculating portfolio returns...")

# Combine all returns into one DataFrame
returns_df = pd.DataFrame({
    'Nifty50': nifty50['Returns'],
    'Midcap': nifty_midcap['Returns'],
    'Govt_Bonds': govt_bonds['Returns'],
    'Corp_Bonds': corp_bonds['Returns'],
    'Gold': gold['Returns']
})

# Fill missing values with 0 (for days when markets were closed)
returns_df = returns_df.fillna(0)

# Align all data to common index
common_index = nifty50.index
returns_df = returns_df.reindex(common_index, fill_value=0)

# Cash returns (assume 3-4% annual ~ 0.0001 daily)
cash_return = 0.0001

# Calculate portfolio returns for each profile
portfolio_returns = {}

for profile_name, allocation in portfolios.items():
    daily_returns = (
        returns_df['Nifty50'] * allocation['Large Cap (Nifty50)'] +
        returns_df['Midcap'] * allocation['Mid Cap'] +
        returns_df['Govt_Bonds'] * allocation['Govt Bonds'] +
        returns_df['Corp_Bonds'] * allocation['Corp Bonds'] +
        returns_df['Gold'] * allocation['Gold'] +
        cash_return * allocation['Cash']
    )
    
    portfolio_returns[profile_name] = daily_returns

# Create DataFrame with all portfolio returns
portfolio_returns_df = pd.DataFrame(portfolio_returns, index=common_index)

# Add benchmark (Nifty50)
portfolio_returns_df['Nifty50_Benchmark'] = returns_df['Nifty50']

print("✅ Portfolio returns calculated")

# ============================================================================
# 4. CALCULATE CUMULATIVE RETURNS
# ============================================================================
print("\n[4/6] Calculating cumulative returns...")

# Calculate cumulative returns (compound growth)
cumulative_returns = (1 + portfolio_returns_df).cumprod()

# Display final values (₹100,000 initial investment)
initial_investment = 100000
final_values = cumulative_returns.iloc[-1] * initial_investment

print("\n📈 FINAL PORTFOLIO VALUES (₹1,00,000 invested in Jan 2015):")
print("="*70)
for profile in final_values.index:
    value = final_values[profile]
    total_return = ((value / initial_investment) - 1) * 100
    print(f"{profile:25s}: ₹{value:12,.0f}  (Total Return: {total_return:6.2f}%)")

# ============================================================================
# 5. CALCULATE ANNUAL RETURNS
# ============================================================================
print("\n[5/6] Calculating annualized metrics...")

# Calculate annualized returns and volatility
annual_metrics = {}

for column in portfolio_returns_df.columns:
    returns = portfolio_returns_df[column]
    
    # Annualized return (CAGR)
    total_return = cumulative_returns[column].iloc[-1] - 1
    years = (common_index[-1] - common_index[0]).days / 365.25
    annual_return = ((1 + total_return) ** (1/years)) - 1
    
    # Annualized volatility (std dev)
    annual_volatility = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio (assuming 6% risk-free rate)
    risk_free_rate = 0.06
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
    
    annual_metrics[column] = {
        'Annual Return (%)': annual_return * 100,
        'Annual Volatility (%)': annual_volatility * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown (%)': max_drawdown * 100,
        'Final Value (₹)': final_values[column]
    }

# Create summary table
metrics_df = pd.DataFrame(annual_metrics).T
metrics_df = metrics_df.round(2)

print("\n📊 PORTFOLIO PERFORMANCE METRICS (2015-2025):")
print("="*70)
print(metrics_df.to_string())

# Save results
metrics_df.to_csv('portfolio_performance_metrics.csv')
portfolio_returns_df.to_csv('portfolio_daily_returns.csv')
cumulative_returns.to_csv('portfolio_cumulative_returns.csv')

print("\n✅ Performance metrics saved to CSV files")

# ============================================================================
# 6. CREATE VISUALIZATIONS
# ============================================================================
print("\n[6/6] Creating visualizations...")

# Plot 1: Cumulative Returns Over Time
plt.figure(figsize=(14, 8))
for column in cumulative_returns.columns:
    plt.plot(cumulative_returns.index, cumulative_returns[column] * initial_investment, 
             label=column, linewidth=2)

plt.title('Portfolio Growth: ₹1,00,000 Initial Investment (2015-2025)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Portfolio Value (₹)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualization_portfolio_growth.png', dpi=300)
print("✅ Saved: visualization_portfolio_growth.png")
plt.close()

# Plot 2: Risk-Return Scatter Plot
plt.figure(figsize=(12, 8))
for profile in metrics_df.index:
    x = metrics_df.loc[profile, 'Annual Volatility (%)']
    y = metrics_df.loc[profile, 'Annual Return (%)']
    
    # Different colors for different portfolios
    if 'Conservative' in profile:
        color = 'green'
        marker = 's'
    elif 'Balanced' in profile:
        color = 'orange'
        marker = 'o'
    elif 'Aggressive' in profile:
        color = 'red'
        marker = '^'
    else:
        color = 'blue'
        marker = 'D'
    
    plt.scatter(x, y, s=200, c=color, marker=marker, label=profile, alpha=0.7, edgecolors='black')
    plt.annotate(profile, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.xlabel('Annualized Volatility (%)', fontsize=12, fontweight='bold')
plt.ylabel('Annualized Return (%)', fontsize=12, fontweight='bold')
plt.title('Risk-Return Profile of Portfolios (2015-2025)', fontsize=16, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualization_risk_return.png', dpi=300)
print("✅ Saved: visualization_risk_return.png")
plt.close()

# Plot 3: Portfolio Allocations Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (profile_name, allocation) in enumerate(portfolios.items()):
    # Remove 'Large Cap (Nifty50)' prefix for cleaner labels
    labels = [k.replace(' (Nifty50)', '') for k in allocation.keys()]
    sizes = list(allocation.values())
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9']
    
    axes[idx].pie(sizes, labels=labels, autopct='%1.0f%%', startangle=90, colors=colors)
    axes[idx].set_title(f'{profile_name} Portfolio', fontsize=14, fontweight='bold')

plt.suptitle('Portfolio Allocations by Investor Profile', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualization_portfolio_allocations.png', dpi=300)
print("✅ Saved: visualization_portfolio_allocations.png")
plt.close()

# Plot 4: Drawdown Analysis
plt.figure(figsize=(14, 8))
for column in portfolio_returns_df.columns:
    returns = portfolio_returns_df[column]
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = ((cumulative - running_max) / running_max) * 100
    
    plt.plot(drawdown.index, drawdown, label=column, linewidth=1.5, alpha=0.8)

plt.title('Portfolio Drawdowns Over Time (2015-2025)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Drawdown (%)', fontsize=12)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualization_drawdowns.png', dpi=300)
print("✅ Saved: visualization_drawdowns.png")
plt.close()

print("\n" + "="*70)
print("BACKTESTING COMPLETE!")
print("="*70)
print("\n📁 Generated Files:")
print("  1. portfolio_performance_metrics.csv")
print("  2. portfolio_daily_returns.csv")
print("  3. portfolio_cumulative_returns.csv")
print("  4. visualization_portfolio_growth.png")
print("  5. visualization_risk_return.png")
print("  6. visualization_portfolio_allocations.png")
print("  7. visualization_drawdowns.png")
print("\n✅ Ready for final report!")
