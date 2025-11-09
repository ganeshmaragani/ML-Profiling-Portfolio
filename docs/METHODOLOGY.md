# METHODOLOGY

## Research Design Overview

This research employs a **quantitative, exploratory, cross-sectional design** utilizing unsupervised machine learning techniques to discover naturally occurring investor segments within the Indian retail investment population. The methodology integrates primary data collection through structured surveys with advanced statistical analysis and historical portfolio validation.

---

## Research Type and Approach

### **Research Classification**

**Primary Research Type:** Exploratory Research  
**Secondary Research Type:** Descriptive and Analytical Research

**Methodological Framework:**
- **Quantitative Approach:** Structured data collection and statistical analysis
- **Positivist Paradigm:** Objective measurement of risk attitudes and investment behaviors
- **Inductive Reasoning:** Data-driven discovery of investor profiles (bottom-up approach)
- **Cross-Sectional Design:** Single time-point observation across diverse investor sample

**Research Philosophy:**
This study adopts a **pragmatic research philosophy**, combining behavioral finance theory with data science techniques to solve a practical problem in wealth management. The approach prioritizes actionable insights over purely theoretical contributions.

---

## Population and Sampling Framework

### **Target Population**

**Definition:** Indian retail investors with active demat accounts who make independent investment decisions in equity and debt markets.

**Population Characteristics:**
- Geographic: Primarily urban India
- Age: 18-65 years
- Investment experience: Minimum 1 year of active investing
- Account type: Individual demat account holders (not corporate/institutional)
- Language: English proficiency for survey comprehension

**Estimated Population Size:** ~87 million active demat accounts in India (as of 2024)

### **Sampling Method**

**Technique:** Non-probability convenience and snowball sampling

**Justification:**
- Absence of comprehensive investor database for random sampling
- Behavioral finance studies typically use convenience samples
- Focus on pattern discovery (exploratory) rather than population estimation
- Resource constraints for large-scale random sampling

**Sampling Process:**
1. Initial contact through professional networks (LinkedIn, MBA alumni groups)
2. Distribution via Google Forms with informed consent
3. Snowball expansion through participant referrals
4. Screening questions to ensure eligibility

### **Sample Size Determination**

**Target Sample:** 35-50 investors  
**Achieved Sample:** 37 investors

**Sample Size Rationale:**
- **Minimum for statistical tests:** n ≥ 30 for normality assumptions (Central Limit Theorem)
- **Clustering guidelines:** Sample size ≥ 2^k (where k=number of clusters); for k=3, minimum n=8 per cluster achieved (11, 14, 12)
- **Comparable studies:** Behavioral finance surveys typically use n=30-200 for exploratory research
- **Feasibility:** Balanced depth (36 questions) vs. participation rate

**Sample Demographics (Achieved):**
- Age range: 26-65 years (diverse life stages)
- Gender: Male and female representation
- Income levels: <₹25,000 to >₹2,00,000 monthly
- Occupations: Salaried employees, business owners, professionals
- Investment experience: 1-20+ years

---

## Hypothesized Model

### **Conceptual Framework**

```
[Investor Characteristics]
         ↓
    ┌────────────────────────────────┐
    │  Demographic Variables         │
    │  - Age, Gender, Income         │
    │  - Education, Occupation       │
    └────────────────────────────────┘
                 ↓
    ┌────────────────────────────────┐
    │  Financial Circumstances       │
    │  - Savings rate                │
    │  - Debt levels                 │
    │  - Emergency fund              │
    └────────────────────────────────┘
                 ↓
    ┌────────────────────────────────┐
    │  Behavioral Characteristics    │
    │  - Risk tolerance              │ → [COMPOSITE RISK SCORE]
    │  - Loss aversion               │
    │  - Investment horizon          │
    │  - Return expectations         │
    └────────────────────────────────┘
                 ↓
    ┌────────────────────────────────┐
    │  Financial Literacy            │
    │  - Compound interest           │
    │  - Return calculation          │
    └────────────────────────────────┘
                 ↓
         [K-MEANS CLUSTERING]
                 ↓
    ┌─────────────────────────────────────────┐
    │      INVESTOR PROFILES                  │
    │                                         │
    │  Conservative │ Balanced │ Aggressive  │
    └─────────────────────────────────────────┘
                 ↓
    ┌─────────────────────────────────────────┐
    │    PORTFOLIO RECOMMENDATIONS            │
    │                                         │
    │  25% Equity   │ 55% Equity │ 80% Equity│
    │  60% Debt     │ 30% Debt   │ 10% Debt  │
    └─────────────────────────────────────────┘
                 ↓
         [HISTORICAL VALIDATION]
              (2015-2025)
```

### **Theoretical Foundation**

**1. Behavioral Finance Theory (Kahneman & Tversky, 1979)**
- Prospect theory: Loss aversion and risk attitudes
- Framing effects in investment decisions
- Systematic biases in financial decision-making

**2. Modern Portfolio Theory (Markowitz, 1952)**
- Risk-return optimization
- Diversification benefits
- Efficient frontier concept

**3. Life-Cycle Investment Theory (Modigliani & Brumberg, 1954)**
- Age-related risk capacity
- Time horizon influence on asset allocation
- Income stability effects

**4. Financial Literacy Framework (Lusardi & Mitchell, 2011)**
- Knowledge-behavior linkage
- Compound interest understanding
- Risk-return comprehension

### **Research Hypotheses**

**H1:** Indian retail investors can be segmented into distinct clusters based on risk attitudes, behavioral characteristics, and financial circumstances, with statistically significant differences between groups.

**H2:** Composite risk scores derived from behavioral questions will demonstrate strong discriminatory power in separating investor profiles (ANOVA F-statistic significant at p<0.05).

**H3:** Portfolio allocations aligned with identified investor profiles will exhibit differentiated risk-return characteristics consistent with Modern Portfolio Theory predictions.

**H4:** Conservative profiles will demonstrate lower returns and volatility, while aggressive profiles will show higher returns and volatility, with balanced profiles occupying middle ground.

---

## Research Tools and Instruments

### **1. Survey Instrument**

**Tool:** Structured questionnaire (36 questions)  
**Platform:** Google Forms  
**Administration:** Self-administered, online

**Survey Structure:**

**Section A: Demographic Profile (6 questions)**
- Age, gender, education, location, occupation, income
- Purpose: Control variables and profile characterization

**Section B: Financial Circumstances (6 questions)**
- Savings rate, debt-to-income ratio, emergency fund
- Income stability, financial goals, investment horizon
- Purpose: Financial capacity assessment

**Section C: Risk Attitudes & Investment Behavior (12 questions)**
- Risk tolerance scenarios (market drop, return preferences)
- Portfolio allocation preferences
- Investment decision-making approach
- Market volatility response
- Purpose: **Composite risk score calculation (6 questions selected)**

**Section D: Behavioral Biases (6 questions)**
- Loss aversion measurement
- Overconfidence assessment
- Herding behavior identification
- Regret aversion evaluation
- Purpose: Behavioral tendency profiling

**Section E: Financial Literacy (6 questions)**
- Compound interest calculation
- Return percentage computation
- Risk diversification understanding
- Financial planning knowledge
- Purpose: Knowledge level assessment

**Questionnaire Validation:**
- Face validity: Expert review by finance faculty
- Content validity: Literature-based question design
- Pilot testing: 5 investors for clarity and timing
- Internal consistency: Cronbach's alpha calculated post-collection

### **2. Data Processing Tools**

**Programming Language:** Python 3.11  

**Libraries and Frameworks:**
```python
# Data manipulation
- pandas 2.0.3       # DataFrame operations
- numpy 1.24.3       # Numerical computations

# Machine learning
- scikit-learn 1.3.0 # K-Means clustering, metrics
- scipy 1.11.1       # Statistical tests

# Visualization
- matplotlib 3.7.2   # Plotting
- seaborn 0.12.2     # Statistical graphics

# Financial analysis
- yfinance 0.2.28    # Market data download
```

**Development Environment:**
- Jupyter Notebook for exploratory analysis
- VS Code for production scripts
- Git/GitHub for version control

### **3. Statistical Analysis Tools**

**Clustering Algorithm:**
- **K-Means (scikit-learn implementation)**
  - Distance metric: Euclidean
  - Initialization: k-means++ (smart centroid initialization)
  - Iterations: Maximum 300
  - Random state: 42 (reproducibility)

**Validation Tests:**
- One-way ANOVA (scipy.stats.f_oneway)
- Chi-square test (scipy.stats.chi2_contingency)
- Bootstrap resampling (custom implementation)
- Silhouette analysis (sklearn.metrics.silhouette_score)

**Performance Metrics:**
- CAGR calculation (custom function)
- Sharpe ratio computation (returns / volatility)
- Maximum drawdown analysis

---

## Detailed Methodology Steps

### **Phase 1: Survey Design and Data Collection (Weeks 1-4)**

**Step 1.1: Questionnaire Development**
1. Literature review of investor profiling instruments
2. Question drafting based on validated scales
3. Expert review by finance faculty (faculty mentor)
4. Pilot testing with 5 investors
5. Refinement based on feedback

**Step 1.2: Survey Administration**
1. Google Forms setup with logic flows
2. Informed consent statement
3. Distribution via professional networks
4. Snowball sampling through participants
5. Data collection period: 2 weeks (October 2025)

**Step 1.3: Data Quality Control**
1. Duplicate response detection (email, timestamp)
2. Completeness verification (no missing values)
3. Logical consistency checks (percentages sum to 100)
4. Outlier identification (Grubbs' test)

**Deliverable:** Clean dataset with 37 complete responses

---

### **Phase 2: Feature Engineering (Week 5)**

**Step 2.1: Composite Risk Score Calculation**

**Formula:**
```
Risk Score = Σ(Normalized Question Score) / 6

Where:
Q14: Market drop response (0-1 scale)
Q15: Investment approach (0-1 scale)
Q16: Risk allocation percentage (0-1 scale)
Q17: Loss tolerance agreement (0-1 scale)
Q18: Return scenario preference (0-1 scale)
Q19: Investment choice (0-1 scale)
```

**Normalization Method:**
- Min-Max scaling to [0, 1] range
- Higher values indicate higher risk tolerance

**Step 2.2: Variable Transformation**
1. Categorical encoding (one-hot for gender, education)
2. Ordinal encoding (age groups, income levels)
3. Investment horizon conversion to years (numeric)
4. Standardization (z-score normalization for clustering)

**Step 2.3: Feature Selection**
- Primary feature: Composite risk score
- Secondary features: Investment horizon, age group
- Rationale: High correlation with portfolio preferences, theoretical grounding

**Deliverable:** Feature matrix ready for clustering

---

### **Phase 3: K-Means Clustering Implementation (Week 6)**

**Step 3.1: Optimal Cluster Determination**

**Method:** Elbow method + Silhouette analysis

```python
# Pseudo-code
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(features)
    silhouette = silhouette_score(features, labels)
    # Plot silhouette scores
```

**Evaluation Criteria:**
- Silhouette coefficient (target: ≥ 0.5)
- Within-cluster sum of squares (WCSS)
- Interpretability of profiles

**Step 3.2: Final Model Training**

```python
# Optimal k=3 selected
kmeans_final = KMeans(n_clusters=3, n_init=10, random_state=42)
cluster_labels = kmeans_final.fit_predict(features_scaled)
```

**Step 3.3: Profile Characterization**
- Calculate mean risk scores per cluster
- Compute demographic distributions
- Identify distinguishing characteristics
- Assign profile labels (Conservative, Balanced, Aggressive)

**Deliverable:** Three distinct investor profiles with statistical profiles

---

### **Phase 4: Statistical Validation (Week 7)**

**Step 4.1: Inter-Cluster Difference Testing**

**One-Way ANOVA:**
```python
from scipy.stats import f_oneway
F_stat, p_value = f_oneway(
    conservative_risk_scores,
    balanced_risk_scores,
    aggressive_risk_scores
)
```

**Null Hypothesis:** No significant difference in mean risk scores across clusters  
**Alternative Hypothesis:** At least one cluster differs significantly  
**Significance Level:** α = 0.05

**Step 4.2: Effect Size Calculation**

**Eta-Squared (η²):**
```
η² = SS_between / SS_total

Interpretation:
η² < 0.01: Small effect
η² = 0.06: Medium effect
η² ≥ 0.14: Large effect
```

**Step 4.3: Bootstrap Confidence Intervals**

```python
# 1,000 bootstrap iterations
for i in range(1000):
    sample = resample(data, replace=True)
    cluster_means[i] = calculate_means(sample)

# 95% CI
ci_lower = percentile(cluster_means, 2.5)
ci_upper = percentile(cluster_means, 97.5)
```

**Step 4.4: Cluster Stability Analysis**

```python
# 100 random initializations
ari_scores = []
for seed in range(100):
    kmeans = KMeans(n_clusters=3, random_state=seed)
    labels = kmeans.fit_predict(features)
    ari = adjusted_rand_score(baseline_labels, labels)
    ari_scores.append(ari)

stability = mean(ari_scores)  # Target: ≥ 0.8
```

**Deliverable:** Statistical validation report confirming clustering robustness

---

### **Phase 5: Portfolio Design (Week 8)**

**Step 5.1: Asset Allocation Framework**

**Conservative Profile:**
- Equity: 25% (capital appreciation potential)
- Debt: 60% (stability and income)
- Gold: 10% (hedge against volatility)
- Cash: 5% (liquidity buffer)

**Balanced Profile:**
- Equity: 55% (balanced growth)
- Debt: 30% (moderate stability)
- Gold: 10% (diversification)
- Cash: 5% (liquidity)

**Aggressive Profile:**
- Equity: 80% (maximum growth)
- Debt: 10% (minimal stability)
- Gold: 5% (limited hedge)
- Cash: 5% (liquidity)

**Design Principles:**
- Modern Portfolio Theory alignment
- Age-appropriate risk allocation
- Regulatory compliance (diversification requirements)
- Practical implementability (liquid assets only)

**Deliverable:** Three portfolio templates with asset allocations

---

### **Phase 6: Historical Backtesting (Weeks 9-10)**

**Step 6.1: Data Collection**

**Market Data Sources:**
```python
import yfinance as yf

# Equity: Nifty 50 Index
nifty = yf.download('^NSEI', start='2015-01-01', end='2025-10-31')

# Debt: 10-Year G-Sec (proxy via bond ETF)
gsec = yf.download('GOLDBEES.NS', start='2015-01-01', end='2025-10-31')

# Gold: Gold ETF
gold = yf.download('GOLDBEES.NS', start='2015-01-01', end='2025-10-31')
```

**Data Cleaning:**
- Handle missing values (forward fill)
- Align dates across assets
- Calculate monthly returns

**Step 6.2: Portfolio Performance Simulation**

```python
# Portfolio returns calculation
portfolio_returns = (
    weight_equity * equity_returns +
    weight_debt * debt_returns +
    weight_gold * gold_returns +
    weight_cash * cash_rate
)

# Cumulative growth
portfolio_value = (1 + portfolio_returns).cumprod() * 100
```

**Step 6.3: Performance Metrics**

**CAGR (Compound Annual Growth Rate):**
```
CAGR = (Final Value / Initial Value)^(1/Years) - 1
```

**Annualized Volatility:**
```
σ_annual = σ_monthly * √12
```

**Sharpe Ratio:**
```
Sharpe = (Portfolio Return - Risk-Free Rate) / σ
```

**Deliverable:** 10-year performance results for three portfolios

---

### **Phase 7: Results Compilation and Documentation (Weeks 11-12)**

**Step 7.1: Visualization Creation**
- Cluster scatter plots (PCA for 2D visualization)
- Profile distribution charts
- Performance comparison graphs
- Statistical validation summaries

**Step 7.2: Dissertation Writing**
- Abstract, introduction, literature review
- Methodology documentation
- Results presentation with tables/charts
- Discussion and implications
- Conclusions and future research

**Step 7.3: Quality Assurance**
- Plagiarism check (target: <25%)
- Statistical verification
- Code review and documentation
- Peer review by supervisor

**Deliverable:** Complete dissertation report (50-200 pages)

---

## Methodological Rigor and Limitations

### **Strengths**
✅ Validated survey instrument based on established scales  
✅ Rigorous statistical testing (ANOVA, bootstrap, stability analysis)  
✅ Triangulation through multiple validation methods  
✅ Transparent methodology enabling reproducibility  
✅ Real-world validation through historical backtesting  
✅ Alignment with academic research standards  

### **Limitations and Mitigations**

| Limitation | Mitigation Strategy |
|------------|---------------------|
| Small sample size (n=37) | Bootstrap resampling for robustness; Conservative statistical tests |
| Convenience sampling | Demographic diversity verification; Transparency about generalizability limits |
| Self-reported data | Anonymous responses; Consistency checks; Validated question scales |
| Cross-sectional design | 10-year backtesting for temporal validation |
| Simplistic asset classes | Focus on core retail portfolios (80%+ coverage) |
| K-Means limitations (assumes spherical clusters) | Silhouette validation; Comparison with alternative algorithms |

---

## Ethical Considerations

✅ **Informed Consent:** All participants provided explicit consent  
✅ **Anonymity:** No personally identifiable information collected beyond email (for duplicate check, then discarded)  
✅ **Voluntary Participation:** Right to withdraw at any time  
✅ **Data Security:** Encrypted storage, access-controlled  
✅ **Transparency:** Methodology fully disclosed for reproducibility  
✅ **No Harm:** Survey posed no financial, psychological, or social risks  

---

## Timeline Summary

| Phase | Duration | Key Activities |
|-------|----------|---------------|
| 1. Survey Design & Data Collection | Weeks 1-4 | Questionnaire development, pilot testing, data collection |
| 2. Feature Engineering | Week 5 | Risk score calculation, variable transformation |
| 3. K-Means Clustering | Week 6 | Optimal k selection, model training, profile characterization |
| 4. Statistical Validation | Week 7 | ANOVA, bootstrap, stability analysis |
| 5. Portfolio Design | Week 8 | Asset allocation framework development |
| 6. Historical Backtesting | Weeks 9-10 | Data collection, performance simulation, metrics calculation |
| 7. Documentation | Weeks 11-12 | Visualization, dissertation writing, quality assurance |
| **Total Project Duration** | **12 weeks** | **January - November 2025** |

---

**Student:** Ganesh Maragani (2023mb53546)  
**Program:** MBA - Data Science and Analytics  
**Institution:** BITS Pilani - WILP  
**Date:** November 7, 2025
