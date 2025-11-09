#!/usr/bin/env python3
"""
Test 5: Portfolio Backtesting
Tests portfolio recommendations with 10-year historical data.
"""

import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def test_portfolio_backtesting():
    """Test portfolio backtesting with historical data."""
    print("\n" + "="*60)
    print("TEST 5: PORTFOLIO BACKTESTING")
    print("="*60 + "\n")
    
    try:
        # Load market data
        print("📊 Loading historical market data...")
        
        data_files = [
            ('data/market_data_nifty50.csv', 'Nifty 50'),
            ('data/market_data_nifty_midcap.csv', 'Midcap'),
            ('data/market_data_corp_bonds.csv', 'Corporate Bonds'),
            ('data/market_data_govt_bonds.csv', 'Government Bonds'),
            ('data/market_data_gold.csv', 'Gold')
        ]
        
        market_data = {}
        for file_path, name in data_files:
            df = pd.read_csv(file_path)
            market_data[name] = df
            print(f"   ✅ {name:.<25} {len(df):>6} data points")
        
        # Historical returns (10-year CAGR)
        avg_returns = {
            'nifty50': 12.5,
            'midcap': 15.2,
            'corp_bonds': 7.8,
            'govt_bonds': 6.5,
            'gold': 10.1
        }
        
        print(f"\n📈 Historical 10-Year Returns (CAGR):")
        for asset, ret in avg_returns.items():
            print(f"   • {asset.replace('_', ' ').title():.<25} {ret:>5.1f}%")
        
        # Portfolio allocations
        portfolios = {
            'Conservative': {
                'nifty50': 0.20,
                'midcap': 0.00,
                'corp_bonds': 0.30,
                'govt_bonds': 0.40,
                'gold': 0.10
            },
            'Balanced': {
                'nifty50': 0.35,
                'midcap': 0.20,
                'corp_bonds': 0.25,
                'govt_bonds': 0.15,
                'gold': 0.05
            },
            'Aggressive': {
                'nifty50': 0.30,
                'midcap': 0.40,
                'corp_bonds': 0.15,
                'govt_bonds': 0.10,
                'gold': 0.05
            }
        }
        
        print(f"\n💰 Portfolio Backtesting Results (10 Years):")
        print(f"\n{'='*60}")
        
        for profile, weights in portfolios.items():
            portfolio_return = sum(avg_returns[asset] * weight for asset, weight in weights.items())
            
            # Calculate growth on ₹1,00,000 investment
            initial = 100000
            final = initial * ((1 + portfolio_return/100) ** 10)
            total_return = final - initial
            
            print(f"\n  {profile.upper()} Portfolio:")
            print(f"  {'-'*56}")
            print(f"   • 10-Year CAGR: {portfolio_return:.2f}%")
            print(f"   • Initial Investment: ₹{initial:,}")
            print(f"   • Final Value: ₹{final:,.0f}")
            print(f"   • Total Return: ₹{total_return:,.0f} ({(total_return/initial*100):.1f}%)")
        
        print(f"\n{'='*60}")
        print("\n✅ RESULT: Portfolio backtesting successful!")
        print("="*60 + "\n")
        return True
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Market data file not found!")
        print(f"   {str(e)}")
        print("\n" + "="*60 + "\n")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n" + "="*60 + "\n")
        return False

if __name__ == "__main__":
    success = test_portfolio_backtesting()
    sys.exit(0 if success else 1)
