#!/bin/bash
# run_all_tests.sh - Run all test scripts sequentially

echo ""
echo "======================================================================"
echo "  INVESTOR PROFILING SYSTEM - COMPLETE TEST SUITE"
echo "======================================================================"
echo ""
echo "Running all tests sequentially..."
echo ""

# Track results
PASSED=0
FAILED=0
TOTAL=6

# Test 1: Environment
echo "→ Running Test 1/6: Environment Verification..."
python3 tests/test_1_environment.py
if [ $? -eq 0 ]; then
    ((PASSED++))
else
    ((FAILED++))
fi
echo ""

# Test 2: Data Loading
echo "→ Running Test 2/6: Data Loading Validation..."
python3 tests/test_2_data_loading.py
if [ $? -eq 0 ]; then
    ((PASSED++))
else
    ((FAILED++))
fi
echo ""

# Test 3: Clustering
echo "→ Running Test 3/6: Clustering Analysis..."
python3 tests/test_3_clustering.py
if [ $? -eq 0 ]; then
    ((PASSED++))
else
    ((FAILED++))
fi
echo ""

# Test 4: Statistical Validation
echo "→ Running Test 4/6: Statistical Validation..."
python3 tests/test_4_statistical_validation.py
if [ $? -eq 0 ]; then
    ((PASSED++))
else
    ((FAILED++))
fi
echo ""

# Test 5: Portfolio Backtesting
echo "→ Running Test 5/6: Portfolio Backtesting..."
python3 tests/test_5_portfolio_backtesting.py
if [ $? -eq 0 ]; then
    ((PASSED++))
else
    ((FAILED++))
fi
echo ""

# Test 6: Visualizations
echo "→ Running Test 6/6: Visualizations Verification..."
python3 tests/test_6_visualizations.py
if [ $? -eq 0 ]; then
    ((PASSED++))
else
    ((FAILED++))
fi
echo ""

# Final Summary
echo "======================================================================"
echo "  TEST SUITE RESULTS"
echo "======================================================================"
echo ""
echo "   Total Tests: $TOTAL"
echo "   Passed: $PASSED"
echo "   Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ ALL TESTS PASSED! System is working correctly."
    echo ""
    echo "======================================================================"
    echo ""
    exit 0
else
    echo "❌ SOME TESTS FAILED. Please review the output above."
    echo ""
    echo "======================================================================"
    echo ""
    exit 1
fi
