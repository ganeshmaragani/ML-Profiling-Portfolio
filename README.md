# 🎯 ML-Based Investor Profiling & Portfolio Recommendation System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-red.svg)](LICENSE)

> **Machine Learning application for personalized investment portfolio recommendations using behavioral finance and K-Means clustering**

---

## 📊 Project Overview

This dissertation project develops an ML-based investor profiling system that segments retail investors into three distinct risk profiles using K-Means clustering and behavioral finance principles. The system provides personalized portfolio recommendations validated through 10-year historical backtesting.

**Institution:** BITS Pilani - Wilp  
**Program:** MBA (FinTech)  
**Author:** Ganesh Maragani (2023mb53560)  
**Supervisor:** Dr. Charu Surana  
**Date:** November 2025

---

## 🎯 Key Results

- **37 Investors Surveyed** with 38 behavioral & demographic features
- **3 Investor Profiles** identified: Conservative (40.5%), Balanced (35.1%), Aggressive (24.3%)
- **F=68.03, p<0.000001** - Highly significant statistical validation
- **η²=0.815** - Explains 81.5% of variance (large effect size)
- **Silhouette Score: 0.6380** - Exceeds "good" threshold (>0.5)
- **10-Year Backtesting:** 8.5%-12.1% CAGR, all portfolios beat 6% inflation
- **Case Study:** Ramesh gained ₹3.7 lakh extra wealth with ML-personalized portfolio

---

## 📂 Repository Structure

```
ML-Profiling-Portfolio/
│
├── README.md                          # Project overview
├── requirements.txt                   # Python dependencies
├── LICENSE                            # Academic license
│
├── data/                              # Datasets (9 files)
│   ├── investor_survey_data_with_timestamps.csv
│   ├── investor_profiles_results.csv
│   └── market_data_*.csv (6 files)
│
├── notebooks/                         # Jupyter analysis (2 files)
│   ├── investor_profiling_analysis.ipynb
│   └── investor_clustering_viva.ipynb
│
├── scripts/                           # Python scripts (7 files)
│   ├── run_complete_clustering.py
│   ├── statistical_validation.py
│   └── portfolio_backtesting.py
│
├── images/                            # Visualizations (10 charts)
│   ├── 1_profile_distribution.png
│   └── ...
│
├── docs/                              # Documentation (4 files)
│   ├── DISSERTATION_REPORT.md
│   ├── METHODOLOGY.md
│   └── Survey_Questionnaire.md
│
└── presentation/                      # PowerPoint (1 file)
    └── FINAL_PRESENTATION_WITH_RAMESH_STORY.pptx
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip package manager
- Jupyter Notebook

### Installation

```bash
# Clone repository
git clone https://github.com/ganeshmaragani/ML-Profiling-Portfolio.git
cd ML-Profiling-Portfolio

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook
```

### Run Analysis

```bash
# Option 1: Run Jupyter notebooks
jupyter notebook notebooks/investor_profiling_analysis.ipynb

# Option 2: Run Python scripts
python scripts/run_complete_clustering.py
```

---

## 📊 Methodology

1. **Data Collection:** 37 respondents, 30-question behavioral finance survey
2. **Feature Engineering:** Composite behavioral risk score (6 dimensions)
3. **Clustering:** K-Means (k=3) with optimal validation
4. **Validation:** 5 statistical tests (ANOVA, η², Silhouette, Bootstrap, Davies-Bouldin)
5. **Portfolio Design:** Tailored allocations (Equity/Debt/Gold)
6. **Backtesting:** 10-year historical simulation (2015-2025)

---

## 💡 Key Contributions

1. First Indian market-specific ML profiling study
2. Composite behavioral risk score methodology
3. Multi-method validation framework (5 tests)
4. 10-year historical backtesting with real data
5. Commercial viability analysis (₹870 Cr TAM)
6. Explainable AI with clear interpretations

---

## 📈 Portfolio Recommendations

| Profile | Risk Score | Allocation | Expected CAGR |
|---------|------------|------------|---------------|
| **Conservative** | 0.0-0.35 | 10% Equity / 60% Debt / 30% Gold | 8-9% |
| **Balanced** | 0.35-0.65 | 50% Equity / 40% Debt / 10% Gold | 10-11% |
| **Aggressive** | 0.65-1.0 | 80% Equity / 5% Debt / 15% Gold | 12-14% |

---

## 📞 Contact

**Author:** Ganesh Maragani  
**Email:** 2023mb53560@wilp.bits-pilani.ac.in  
**Institution:** BITS Pilani - Wilp  
**LinkedIn:** [linkedin.com/in/ganeshmaragani](https://linkedin.com/in/ganeshmaragani)

---

## 📄 License

Academic License - For educational and research purposes.

---

## 🙏 Acknowledgments
T R Srinath - Dissertation supervisor
Pinki Saha choudhury - Additional Examinar
- 37 Survey respondents
*⭐ Star this repository if you find it useful!**
