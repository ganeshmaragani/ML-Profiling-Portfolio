# Testing Guide

Complete guide for testing the Investor Profiling and Portfolio Recommendation System.

## 📋 Overview

This folder contains **6 executable test scripts** that verify the complete system functionality:

1. **test_1_environment.py** - Verify Python version and required libraries
2. **test_2_data_loading.py** - Validate survey data structure
3. **test_3_clustering.py** - Test K-Means clustering analysis
4. **test_4_statistical_validation.py** - Verify ANOVA statistical validation
5. **test_5_portfolio_backtesting.py** - Test portfolio backtesting with historical data
6. **test_6_visualizations.py** - Verify all visualization files exist

These tests allow professors and reviewers to **independently verify** the system works correctly.

---

## 🚀 Quick Start

### Run All Tests at Once
```bash
# Make sure you're in the project root directory
cd ML-Profiling-Portfolio

# Run the automated test suite
./tests/run_all_tests.sh
```

### Run Individual Tests
```bash
# Test 1: Check environment
python3 tests/test_1_environment.py

# Test 2: Validate data loading
python3 tests/test_2_data_loading.py

# Test 3: Run clustering analysis
python3 tests/test_3_clustering.py

# Test 4: Statistical validation
python3 tests/test_4_statistical_validation.py

# Test 5: Portfolio backtesting
python3 tests/test_5_portfolio_backtesting.py

# Test 6: Verify visualizations
python3 tests/test_6_visualizations.py
```

---

## 📦 Prerequisites

### 1. Python Version
- **Required:** Python 3.11 or higher
- **Check version:** `python3 --version`

### 2. Required Libraries
Install all dependencies using:
```bash
pip install -r requirements.txt
```

Required libraries:
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scipy >= 1.10.0

---

## 📊 Expected Test Results

### Test 1: Environment ✅
```
Library Versions:
  ✓ pandas ............. 2.3.3
  ✓ numpy .............. 2.3.4
  ✓ scikit-learn ....... 1.7.2
  ✓ matplotlib ......... 3.10.7
  ✓ seaborn ............ 0.13.2
  ✓ scipy .............. 1.16.2

RESULT: All libraries found!
```

### Test 2: Data Loading ✅
```
Loading survey data: data/investor_survey_data_with_timestamps.csv

Dataset Information:
  • Total respondents: 37
  • Total features: 38
  • Missing values: 0

Sample Data (first 3 rows):
  [Shows Timestamp, Age, Gender, Income, etc.]

RESULT: Data loaded successfully!
```

### Test 3: Clustering ✅
```
Cluster Distribution:
  • Conservative: 11 investors (29.7%)
  • Balanced: 14 investors (37.8%)
  • Aggressive: 12 investors (32.4%)

Silhouette Score: 0.51 (GOOD - Clusters well separated)

RESULT: Clustering successful!
```

### Test 4: Statistical Validation ✅
```
ANOVA Results:
  • F-statistic: 68.03
  • p-value: < 0.000001
  • η² (effect size): 0.8001 (LARGE effect)

RESULT: Clusters are statistically significant!
```

### Test 5: Portfolio Backtesting ✅
```
CONSERVATIVE Portfolio:
  • 10-Year CAGR: 8.45%
  • Initial Investment: ₹1,00,000
  • Final Value: ₹2,24,820
  • Total Return: ₹1,24,820 (124.8%)

BALANCED Portfolio:
  • 10-Year CAGR: 11.27%
  • Final Value: ₹2,90,940

AGGRESSIVE Portfolio:
  • 10-Year CAGR: 12.16%
  • Final Value: ₹3,16,460

RESULT: Portfolio backtesting successful!
```

### Test 6: Visualizations ✅
```
Checking images directory: images/

  ✅ cluster_distribution.png ............ 45.2 KB
  ✅ risk_score_distribution.png ......... 42.8 KB
  ✅ cluster_comparison.png .............. 58.3 KB
  ... (10 total files)

RESULT: All visualizations verified!
```

---

## 🔄 Adding New Survey Data

### Step 1: Understand the CSV Format

The survey data must have **38 columns** in this exact order:

```csv
Timestamp,Age,Gender,Annual_Income,Investment_Experience,Primary_Investment_Goal,
Risk_Tolerance,Time_Horizon,Liquidity_Needs,Investment_Knowledge,
Market_Volatility_Reaction,Loss_Tolerance,Return_Expectations,Debt_Comfort,
Portfolio_Diversification,ESG_Preference,Professional_Advice,Investment_Review_Frequency,
Risk_Capacity,Emergency_Fund,Financial_Obligations,Health_Insurance,
Life_Insurance,Career_Stability,Income_Stability,Retirement_Planning,
Education_Planning,Home_Purchase,Major_Purchase,Tax_Planning,
Business_Investment,Alternative_Investments,Cryptocurrency,Real_Estate,
Fixed_Deposits,Mutual_Funds,Stocks,Bonds
```

### Step 2: Sample Data for Each Profile

**Conservative Investor Example:**
```csv
2024-01-25 10:30:00,28,Female,500000,Less than 1 year,Capital Preservation,
Very Conservative,Less than 1 year,High liquidity needed,Beginner,
Sell everything,0-5%,5-7%,Very uncomfortable,Low diversification,
Not important,Prefer professional management,Quarterly,Low,Yes,
Moderate obligations,Yes,Yes,Very stable,Very stable,
Not started,No plans,Not planning,Not planning,No interest,
Not interested,Not interested,Not interested,No,Yes,No,Yes
```

**Balanced Investor Example:**
```csv
2024-01-25 11:15:00,35,Male,1200000,3-5 years,Balanced Growth,
Moderate,3-5 years,Moderate liquidity,Intermediate,
Hold and wait,10-20%,10-12%,Somewhat comfortable,Moderate diversification,
Somewhat important,Mix of both,Semi-annually,Moderate,Yes,
Moderate obligations,Yes,Yes,Stable,Stable,
Planning,Planning soon,Planning in 2-3 years,Planning,Considering,
Somewhat interested,Not interested,Considering,Yes,Yes,Yes,Yes
```

**Aggressive Investor Example:**
```csv
2024-01-25 14:45:00,42,Male,2500000,More than 10 years,Aggressive Growth,
Very Aggressive,More than 10 years,Low liquidity needed,Advanced,
Buy more,More than 30%,15%+,Very comfortable,Highly diversified,
Very important,Self-managed,Annually,High,Yes,
Low obligations,Yes,Yes,Very stable,Very stable,
Advanced planning,Fully funded,Already purchased,Planned,Actively investing,
Very interested,Actively trading,Actively investing,Yes,Yes,Yes,Yes
```

### Step 3: Add New Data

#### Option A: Manual CSV Edit
1. Open `data/investor_survey_data_with_timestamps.csv`
2. Add new row at the end with proper format
3. Ensure **all 38 columns** are filled
4. Save the file

#### Option B: Google Forms (Recommended)
1. Create a Google Form with all 38 questions
2. Export responses to CSV
3. Replace or append to `investor_survey_data_with_timestamps.csv`

### Step 4: Rerun the System

After adding new data:

```bash
# Step 1: Reload data and re-cluster
python3 tests/test_2_data_loading.py
python3 tests/test_3_clustering.py

# Step 2: Regenerate processed results
jupyter notebook investor_profiling_analysis.ipynb
# Or run the notebook cells manually

# Step 3: Re-test everything
./tests/run_all_tests.sh
```

**Important Notes:**
- New data will change cluster distributions
- Silhouette score may vary slightly
- Statistical validation (F-statistic, p-value) will update
- Portfolio recommendations remain the same (based on profiles)

---

## 🛠️ Troubleshooting

### Error: "No module named 'pandas'"
**Solution:**
```bash
pip install -r requirements.txt
```

### Error: "FileNotFoundError: data/investor_survey_data_with_timestamps.csv"
**Solution:**
- Ensure you're in the project root directory
- Check the file exists: `ls -la data/`
- Verify correct spelling

### Error: "Expected 38 columns, found X"
**Solution:**
- Check your CSV has all 38 required columns
- Verify no missing commas or extra columns
- Use the sample data format above

### Error: "Python version X.X is not 3.11+"
**Solution:**
```bash
# Check installed Python versions
python3 --version

# Install Python 3.11+ if needed (macOS)
brew install python@3.11

# Or download from python.org
```

### Test 6 Fails: Missing Visualizations
**Solution:**
```bash
# Generate visualizations
python3 generate_visualizations.py

# Verify images created
ls -la images/
```

---

## 📈 Interpreting Results

### Silhouette Score (Test 3)
- **0.71-1.0:** Excellent clustering
- **0.51-0.70:** Good clustering ✅ (Expected: ~0.51)
- **0.26-0.50:** Weak clustering
- **< 0.25:** Poor clustering

### ANOVA F-Statistic (Test 4)
- **F > 50:** Very strong separation between clusters ✅ (Expected: F=68.03)
- **F > 10:** Strong separation
- **F < 10:** Weak separation

### Effect Size η² (Test 4)
- **0.14+:** Large effect ✅ (Expected: η²=0.8001)
- **0.06-0.13:** Medium effect
- **< 0.06:** Small effect

### p-value (Test 4)
- **p < 0.05:** Statistically significant ✅ (Expected: p < 0.000001)
- **p > 0.05:** Not significant

---

## 🎓 For Professors/Reviewers

To independently verify this research:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ganeshmaragani/ML-Profiling-Portfolio.git
   cd ML-Profiling-Portfolio
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run complete test suite:**
   ```bash
   ./tests/run_all_tests.sh
   ```

4. **Expected outcome:**
   - All 6 tests should pass ✅
   - Results match the expected values above
   - Total runtime: ~2-3 minutes

5. **View detailed analysis:**
   - Open `investor_profiling_analysis.ipynb` in Jupyter
   - Open `investor_clustering_viva.ipynb` for VIVA demo

---

## 📞 Support

If you encounter any issues:

1. Check the Troubleshooting section above
2. Verify prerequisites (Python 3.11+, all libraries installed)
3. Review the main README.md in project root
4. Check the Jupyter notebooks for detailed explanations

---

## ✅ Success Checklist

Before submitting/demonstrating:

- [ ] All 6 tests pass when running `./tests/run_all_tests.sh`
- [ ] Python version is 3.11 or higher
- [ ] All required libraries installed (no import errors)
- [ ] Survey data loads correctly (37 respondents, 38 columns)
- [ ] Clustering produces 3 distinct profiles
- [ ] ANOVA shows statistical significance (p < 0.05)
- [ ] Portfolio backtesting shows reasonable returns
- [ ] All 10 visualization files exist in `images/` folder
- [ ] Jupyter notebooks run without errors

---

**Last Updated:** January 2025  
**Version:** 1.0  
**Author:** Ganesh Maragani  
**GitHub:** https://github.com/ganeshmaragani/ML-Profiling-Portfolio
