<p align="center">
  <h1 align="center">🏦 Loan Approval Risk Analysis</h1>
  <p align="center">
    <strong>An Intelligent Banking Analytics Dashboard for Borrower Risk Assessment & Loan Prediction</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#tech-stack">Tech Stack</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#screenshots">Screenshots</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
    <img src="https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
    <img src="https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white" />
    <img src="https://img.shields.io/badge/Plotly-5.15+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
  </p>
</p>

---

## 📌 About The Project

A **Python-powered banking analytics dashboard** that allows users to upload any loan-related dataset (CSV/Excel) and automatically generates borrower risk analysis, loan approval insights, interactive charts, ML predictions, and downloadable PDF reports.

The system uses **intelligent column detection** to work with ANY loan dataset — no hardcoded column names. Upload a Kaggle dataset, a bank's internal CSV, or a custom spreadsheet, and the dashboard automatically maps columns like income, credit score, loan amount, and approval status.

### 🎯 Problem Statement

Banks process thousands of loan applications daily. Manual evaluation is slow, biased, and error-prone. Wrong approvals lead to **loan defaults** (financial loss), and wrong rejections cause **lost revenue**. This project automates the entire loan analysis pipeline — from raw data to actionable insights — using data science and machine learning.

---

## ✨ Features

### 📊 10-Page Interactive Dashboard
| Page | Description |
|------|-------------|
| **Dashboard Overview** | KPI cards, approval donut chart, risk gauge, executive summary |
| **Upload & Preview** | CSV/Excel upload with auto column detection and manual override |
| **Data Cleaning** | One-click cleaning — missing values, duplicates, type correction |
| **Exploratory Analysis** | 3-tab view — numeric histograms, categorical bars, box plots |
| **Correlation Analysis** | Heatmap + scatter plots with OLS trendlines |
| **Risk Segmentation** | Classify borrowers into Low/Medium/High risk with scoring algorithm |
| **Loan Insights** | Approval by education, gender, property area, credit history |
| **ML Predictions** | Train models, evaluate metrics, predict individual applicants |
| **Recommendations** | Auto-generated lending advice based on data patterns |
| **Download Reports** | Export cleaned CSV, text report, and professional PDF |

### 🤖 Machine Learning
- **Random Forest Classifier** — 100-tree ensemble for high accuracy
- **Logistic Regression** — Interpretable baseline model
- **Single Applicant Prediction** — Real-time prediction with confidence percentage
- **Green/Red Result Cards** — Visual ✅ APPROVED / ❌ DENIED indicators
- **Feature Importance** — Understand which factors drive the model's decisions
- **Confusion Matrix** — See exactly where the model makes mistakes

### 🔍 Intelligent Column Detection
- Auto-maps columns using **keyword matching** across 14 standard roles
- Works with Kaggle datasets, banking CSVs, and custom spreadsheets
- No manual configuration required — just upload and go

### 📈 15+ Chart Types
Histogram, KDE, Donut, Grouped Bar, Heatmap, Box Plot, Violin, Scatter, Gauge, Feature Importance, Confusion Matrix — all styled with a consistent dark banking theme.

### 📥 Report Generation
- **PDF Report** — Professional cover page, sections, formatted tables
- **Text Report** — Comprehensive plain-text analysis summary
- **Auto Recommendations** — Data-driven lending advice

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.11+ | Core programming |
| **Dashboard** | Streamlit | Interactive web UI |
| **Data Processing** | Pandas, NumPy | Data manipulation & cleaning |
| **Static Charts** | Matplotlib, Seaborn | Histograms, heatmaps, confusion matrix |
| **Interactive Charts** | Plotly | Donut, bar, box, violin, gauge, scatter |
| **Machine Learning** | Scikit-learn | Random Forest, Logistic Regression |
| **PDF Generation** | FPDF2 | Professional PDF reports |
| **Excel Support** | OpenPyXL, xlrd | .xlsx and .xls file reading |
| **Styling** | Custom CSS | Dark theme, glassmorphism, Google Fonts |

---

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/loan-approval-risk-analysis.git
   cd loan-approval-risk-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```
   Or alternatively:
   ```bash
   python -m streamlit run app.py
   ```

4. **Open in browser**
   ```
   http://localhost:8501
   ```

---

## 📖 Usage

### Quick Start
1. Launch the dashboard with `streamlit run app.py`
2. Navigate to **Upload & Preview** page
3. Upload any loan dataset (CSV or Excel)
4. The system auto-detects columns — verify the mapping
5. Go to **Data Cleaning** → Click "Clean Dataset"
6. Explore **EDA**, **Correlation**, **Risk Segmentation** pages
7. Train an ML model on **ML Predictions** page
8. Predict individual loan applications
9. Download reports from **Download Reports** page

### Sample Datasets Included

| File | Rows | Approval Rate | Description |
|------|------|---------------|-------------|
| `sample_dataset.csv` | 1,500 | ~65% | General testing dataset |
| `sample_best_case.csv` | 1,500 | ~94% | Healthy portfolio demo |
| `sample_worst_case.csv` | 1,500 | ~24% | High-risk portfolio demo |

---

## 🏗️ Architecture

```
loan-approval-risk-analysis/
│
├── app.py                  # Main Streamlit dashboard (10 pages, CSS theme)
├── utils.py                # Column detection, risk scoring, formatting
├── data_cleaning.py        # Missing values, duplicates, type correction
├── visualization.py        # 15+ chart functions (Matplotlib + Plotly)
├── ml_model.py             # ML training, evaluation, prediction
├── report_generator.py     # Text/PDF reports, recommendations engine
├── generate_sample.py      # Synthetic 1500-row dataset generator
├── generate_showcase.py    # Best/worst case dataset generator
├── requirements.txt        # Python dependencies
├── sample_dataset.csv      # Default sample data
├── sample_best_case.csv    # Best-case showcase data
└── sample_worst_case.csv   # Worst-case showcase data
```

### Module Responsibilities

```
┌──────────────────────────────────────────────────────────┐
│                    app.py (Dashboard UI)                 │
│         10 Pages • CSS Theme • Session State             │
├──────────┬─────────────┬──────────────┬──────────────────┤
│ utils.py │data_cleaning│visualization │   ml_model.py    │
│          │    .py      │    .py       │                  │
│ • Column │ • Missing   │ • 15+ Charts│ • Train/Evaluate  │
│   Detect │   Values    │ • Banking   │ • Random Forest   │
│ • Risk   │ • Duplicates│   Theme     │ • Logistic Reg    │
│   Score  │ • Type Fix  │ • Plotly +  │ • Single Predict  │
│ • Format │ • Encoding  │   Matplotlib│ • Feature Import  │
├──────────┴─────────────┴──────────────┴──────────────────┤
│                 report_generator.py                      │
│       Text Report • PDF Report • Recommendations         │
└──────────────────────────────────────────────────────────┘
```

### Data Flow
```
Upload CSV → Auto-Detect Columns → Clean Data → EDA Charts
                                              → Correlation
                                              → Risk Segmentation
                                              → ML Training → Predict
                                              → Generate Reports → Download
```

---

## 🤖 ML Models

### Random Forest Classifier
- **100 decision trees** with max depth 10
- Balanced class weights for handling imbalanced datasets
- Extracts feature importance for interpretability
- Typical accuracy: **85-92%**

### Logistic Regression
- Interpretable linear classifier with sigmoid activation
- Provides exact feature coefficients (weights)
- Ideal for regulatory compliance and explainability
- Typical accuracy: **78-85%**

### Evaluation Metrics
- **Accuracy** — Overall prediction correctness
- **Precision** — Of predicted approvals, how many were correct?
- **Recall** — Of actual approvals, how many were detected?
- **F1 Score** — Harmonic mean of precision and recall
- **Confusion Matrix** — Visual breakdown of TP, TN, FP, FN

### Risk Scoring Algorithm
Each borrower is scored on 4 factors:
| Factor | Low Risk (0) | Medium (1) | High Risk (3) |
|--------|-------------|------------|---------------|
| Credit Score | ≥700 | 580-700 | <580 |
| Loan/Income Ratio | <2.5x | 2.5-5x | >5x |
| Debt-to-Income | <20% | 20-40% | >40% |
| Employment | Employed | — | Unemployed |

---

## 📊 Visualizations

| Chart | Library | Purpose |
|-------|---------|---------|
| Income Histogram + KDE | Matplotlib | Income distribution with mean/median |
| Loan Amount Histogram | Matplotlib | Loan size distribution |
| Credit Score Zones | Matplotlib | Risk zones: Red (<580), Yellow (580-700), Green (>700) |
| Approval Donut | Plotly | Approved vs Denied ratio |
| Gender Approval Bar | Plotly | Gender bias analysis |
| Correlation Heatmap | Seaborn | Feature correlation matrix |
| Risk Donut | Plotly | Low/Medium/High risk breakdown |
| Risk Bar Chart | Plotly | Risk counts with percentages |
| Box Plots | Plotly | Outlier detection, grouped distributions |
| Feature Importance | Matplotlib | ML model's most important features |
| Confusion Matrix | Seaborn | Model prediction accuracy breakdown |
| Category Approval | Plotly | Approval by education, property, etc. |
| Debt Violin Plot | Plotly | Debt comparison: Approved vs Denied |
| Scatter + Trendline | Plotly | Two-variable relationships with OLS |
| Risk Gauge | Plotly | Portfolio health speedometer |

---

## 🎨 UI Design

- **Dark banking theme** with `#0f172a` background
- **Glassmorphism effects** — frosted glass cards with backdrop blur
- **Google Fonts** — Inter (body) + Plus Jakarta Sans (headings)
- **Neon glow metric cards** with hover animations
- **Green/Red prediction cards** for Approved/Denied
- **Responsive layout** using Streamlit columns



## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) — For the amazing dashboard framework
- [Scikit-learn](https://scikit-learn.org/) — For machine learning tools
- [Plotly](https://plotly.com/) — For interactive visualizations
- [Kaggle](https://www.kaggle.com/) — For loan dataset inspiration

---

<p align="center">
  Made with ❤️ using Python
</p>
