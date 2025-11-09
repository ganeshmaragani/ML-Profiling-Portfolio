#!/usr/bin/env python3
"""
Test 6: Visualizations Verification
Verifies all generated visualization files exist and are valid.
"""

import sys
import os

def test_visualizations():
    """Test that all visualization files exist."""
    print("\n" + "="*60)
    print("TEST 6: VISUALIZATIONS VERIFICATION")
    print("="*60 + "\n")
    
    try:
        images_dir = "images"
        
        expected_files = [
            "cluster_distribution.png",
            "risk_score_distribution.png",
            "cluster_comparison.png",
            "feature_importance.png",
            "age_vs_risk.png",
            "income_vs_risk.png",
            "portfolio_allocation_conservative.png",
            "portfolio_allocation_balanced.png",
            "portfolio_allocation_aggressive.png",
            "portfolio_backtest.png"
        ]
        
        print(f"📁 Checking images directory: {images_dir}/\n")
        
        if not os.path.exists(images_dir):
            print(f"❌ ERROR: Images directory '{images_dir}/' not found!")
            print(f"   Run generate_visualizations.py to create images.")
            print("\n" + "="*60 + "\n")
            return False
        
        found_files = []
        missing_files = []
        
        for filename in expected_files:
            filepath = os.path.join(images_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                found_files.append((filename, size))
                print(f"   ✅ {filename:.<45} {size/1024:>6.1f} KB")
            else:
                missing_files.append(filename)
                print(f"   ❌ {filename:.<45} MISSING")
        
        print(f"\n{'='*60}")
        print(f"\n📊 Verification Summary:")
        print(f"   • Total expected: {len(expected_files)} files")
        print(f"   • Found: {len(found_files)} files")
        print(f"   • Missing: {len(missing_files)} files")
        
        if missing_files:
            print(f"\n❌ Missing files:")
            for f in missing_files:
                print(f"      • {f}")
            print(f"\n   Run: python3 generate_visualizations.py")
            print("\n" + "="*60 + "\n")
            return False
        else:
            total_size = sum(size for _, size in found_files)
            print(f"   • Total size: {total_size/1024:.1f} KB")
            print(f"\n✅ RESULT: All visualizations verified!")
            print("="*60 + "\n")
            return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n" + "="*60 + "\n")
        return False

if __name__ == "__main__":
    success = test_visualizations()
    sys.exit(0 if success else 1)
