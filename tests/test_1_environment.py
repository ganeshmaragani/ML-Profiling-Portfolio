#!/usr/bin/env python3
"""
Test 1: Environment Verification
Verifies Python version and all required libraries are installed.
"""

import sys

def test_environment():
    """Test Python environment and library installations."""
    print("\n" + "="*60)
    print("TEST 1: ENVIRONMENT VERIFICATION")
    print("="*60 + "\n")
    
    # Test Python version
    print(f"✅ Python Version: {sys.version.split()[0]}")
    
    # Test required libraries
    libraries = [
        ('pandas', 'Data manipulation'),
        ('numpy', 'Numerical computing'),
        ('sklearn', 'Machine learning'),
        ('matplotlib', 'Plotting'),
        ('seaborn', 'Statistical visualization'),
        ('scipy', 'Scientific computing')
    ]
    
    print("\n📦 Library Versions:\n")
    all_ok = True
    
    for lib_name, description in libraries:
        try:
            if lib_name == 'sklearn':
                import sklearn
                lib = sklearn
            else:
                lib = __import__(lib_name)
            
            version = getattr(lib, '__version__', 'Unknown')
            print(f"   ✅ {lib_name:.<20} {version:>10}   ({description})")
        except ImportError as e:
            print(f"   ❌ {lib_name:.<20} NOT FOUND   ({description})")
            all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("✅ RESULT: All libraries installed successfully!")
        print("="*60 + "\n")
        return True
    else:
        print("❌ RESULT: Some libraries are missing. Please run:")
        print("   pip install -r requirements.txt")
        print("="*60 + "\n")
        return False

if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1)
