# 🎓 Graduate_Admission_Prediction

I'm **Caglar Uysal**, a Computer Science and Engineering student at Sabancı University, currently taking **DSA210 – Introduction to Data Science**.  

This repository contains my **term project** for the course.

**For reproducing my analysis you can run run_complete_analysis.py file after install requirements**

---

## 📋 Feedback Addressed

This project addresses the following feedback points:

1. **✅ Dataset Merging Strategy:**
   - Detailed explanation in `md_files/DATASET_MERGING_STRATEGY.md`
   - Uses **fuzzy string matching with rapidfuzz** to map university names
   - Direct matching with QS rankings using canonical university names

2. **✅ Research Question 3 Clarification:**
   - We analyze the **relationship between university rating and applicant characteristics** (GPA/TOEFL scores)
   - Note: Official university requirements are not in the dataset, but we can analyze patterns in applicant data

3. **✅ Socioeconomic Pattern Analysis:**
   - Integrated into hypothesis testing script (H6)
   - Analyzes admission patterns using QS rankings and cost of living data
   - Tests correlations using statistical methods

4. **✅ Hypotheses with Null/Alternative and Testing Methods:**
   - Each hypothesis now includes:
     - Null hypothesis (H0)
     - Alternative hypothesis (H1)
     - Specific testing method
   - All documented in hypothesis testing script

---

## 🧭 Table of Contents
1. Project Overview  
2. Objectives  
3. Motivation  
4. Data Collection & Sources  
5. Research Questions  
6. Tools & Libraries Used  
7. Hypotheses and Expected Results  
8. Key Findings  
9. Outputs Generated  
10. Future Work  

---

## 📘 Project Overview
Graduate admissions are among the most competitive and uncertain processes students face.  
This project analyzes a **public graduate admission dataset** to identify which factors most strongly influence a student's chance of admission to a graduate program.

The main goal is to **analyze and identify** which factors most strongly influence a student's chance of admission to a graduate program using statistical analysis and hypothesis testing.  
The project also investigates how institutional factors—such as **university ranking** or **cost of living**—affect admission outcomes.

---

## 🎯 Objectives
- Analyze which academic and qualitative factors have the strongest impact on graduate admission.  
- Quantify and visualize the influence of each feature through statistical analysis.  
- Apply end-to-end data science methodologies — from data collection to hypothesis testing.  
- Contribute insights that can help future applicants make more informed decisions.

---

## 💡 Motivation
Students often invest heavily in test preparation without understanding how much weight these metrics actually carry in admission decisions.  
By applying data science methods to historical data, this project seeks to answer a **practical question**:  
> “Which factors matter most for getting accepted to graduate school?”

The project not only helps students **prioritize their efforts**, but also promotes **transparency and fairness** in admissions by visualizing relationships between academic and non-academic metrics.

---

## 📊 Data Collection & Sources

### 🧩 Primary Data Source
**Dataset:** [GradCafe Admission Data (GitHub)](https://github.com/deedy/gradcafe_data)  
This dataset contains real admission data scraped from the GradCafe forum, including:
- **uni_name** - University name (cleaned, 2,708 distinct universities)
- **major** - Intended major (e.g., Computer Science, Economics)
- **degree** - Degree type (PhD, MS, MA, etc.)
- **decision** - Admission decision (Accepted, Rejected, Wait listed, Interview)
- **gre_verbal** - GRE Verbal score
- **gre_quant** - GRE Quantitative score
- **gre_writing** - GRE Writing score (0.0-6.0)
- **ugrad_gpa** - Undergraduate GPA (typically 4.0 scale, sometimes 10.0)
- **status** - Applicant status (American, International, etc.)
- **season** - Admission season (e.g., F15 for Fall 2015)
- **decision_date** - Date decision was made
- And more fields (see [GradCafe Data Schema](https://github.com/deedy/gradcafe_data))

**Note:** The dataset contains 271,807 total results (all majors) and 27,822 computer science results. This is real-world data from actual applicants posting their results on GradCafe.

---

### 🌍 Enrichment Data Sources
To add originality and external context, the dataset will be combined with:
- **QS World University Rankings 2025** (to represent institutional prestige)
  - Link: https://www.kaggle.com/datasets/melissamonfared/qs-world-university-rankings-2025
- **Cost of Living Index by Country** (to account for financial context by country)
  - Link: https://www.kaggle.com/datasets/debdutta/cost-of-living-index-by-country

**Note on Dataset Merging:** GradCafe data includes university names (uni_name), allowing **direct matching** with QS rankings! This is much more accurate than rating-based mapping. We match university names directly using `university_mappings_from_qs.csv` and merge QS ranking information. See `DATASET_MERGING_STRATEGY.md` for detailed explanation.

**Data Processing Pipeline:**
1. Load raw data from `data/raw/all.csv` (GradCafe admission data)
2. Apply university name mappings from `university_mappings_from_qs.csv`
3. Merge with QS Rankings data (from `data/raw/`)
4. Merge with Cost of Living data (from `data/raw/`)
5. Save processed dataset to `data/processed/admissions_merged.csv`
6. Final combined dataset saved as `final_combined_dataset.csv`

These additions will allow testing whether **economic or institutional factors** influence admission trends.

---

## 🧠 Research Questions
1. Which feature has the strongest correlation with admission probability?  
2. Does research experience significantly increase the likelihood of acceptance?  
3. How does university rating relate to applicant GPA or GRE scores?
   - **Note:** We analyze the relationship between university rating and applicant characteristics (not official requirements, as requirement data is not available in the dataset)
4. Are there socioeconomic patterns revealed by QS rankings and cost of living data?
   - **Addressed in:** Hypothesis testing script (H6)

---

## ⚙️ Tools & Libraries Used
### 1. Programming Language  
- **Python** (v3.10+)

### 2. Data Analysis & Statistical Testing  
- **pandas**, **numpy** – Data cleaning and manipulation  
- **scipy** – Statistical tests (t-test, Mann-Whitney U, Chi-square, correlation tests)
- **statsmodels** – Statistical inference

### 3. Visualization  
- **matplotlib**, **seaborn** – Correlation and feature importance plots  
- **plotly** (optional) – Interactive admission probability graphs  

### 4. Data Processing  
- **rapidfuzz** – Fuzzy string matching for university name mapping

---

## 🚀 Usage

### Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Setup Kaggle API (optional): Place `kaggle.json` in `~/.kaggle/` directory
3. Run complete analysis: `python run_complete_analysis.py`

### Recommended: Run with Full Logging
```bash
python run_complete_analysis.py
```
This will:
- Run all analysis steps (Data Collection, EDA, Hypothesis Testing)
- Log all outputs to `logs/complete_analysis_TIMESTAMP.log`
- Generate summary report in `logs/analysis_summary_TIMESTAMP.txt`
- Show progress in console while saving everything to files

### Alternative: Run Individual Steps
```bash
python data_collection_new.py   # Data collection and merging
python eda.py                   # Exploratory Data Analysis
python hypothesis_testing.py    # Hypothesis testing
```

### Project Structure
```
c/
├── data/
│   ├── raw/              # Raw downloaded datasets
│   │   ├── all.csv                           # GradCafe admission data
│   │   ├── QS World University Rankings 2025 (Top global universities).csv
│   │   └── Cost_of_living_index.csv
│   └── processed/        # Cleaned and merged datasets
│       ├── admissions_merged.csv
│       ├── qs_rankings_clean.csv
│       └── cost_of_living_clean.csv
├── outputs/              # Analysis results, plots, reports
│   ├── eda_report.txt
│   ├── hypothesis_testing_report.txt
│   ├── correlation_heatmap.png
│   ├── distributions.png
│   ├── boxplots.png
│   ├── feature_relationships.png
│   ├── categorical_analysis.png
│   └── hypothesis_*.png
├── logs/                 # Execution logs and summaries
│   ├── complete_analysis_TIMESTAMP.log
│   └── analysis_summary_TIMESTAMP.txt
├── md_files/             # Documentation
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── DATASET_MERGING_STRATEGY.md
│   ├── FEEDBACK_RESPONSE.md
│   └── PROJECT_SUMMARY.md
├── data_collection_new.py    # Data collection and merging script
├── eda.py                    # Exploratory Data Analysis
├── hypothesis_testing.py     # Hypothesis testing (with H0/H1)
├── run_complete_analysis.py  # Complete pipeline with logging (RECOMMENDED)
├── map_universities_from_qs.py # University name mapping utility
├── final_combined_dataset.csv # Final merged dataset
├── merged_data_combined.csv  # Intermediate merged dataset
├── university_mappings_from_qs.csv # University name mappings
├── university_mappings_*.json     # Mapping JSON files
└── requirements.txt            # Dependencies
```  

---

## 📈 Hypotheses and Expected Results

Each hypothesis includes **null hypothesis (H0)**, **alternative hypothesis (H1)**, and **testing method**:

| ID | Hypothesis | Null (H0) | Alternative (H1) | Testing Method | Expected Outcome |
|----|------------|-----------|------------------|----------------|------------------|
| H1 | GRE scores correlate with Chance of Admit | ρ = 0 | ρ ≠ 0 | Pearson correlation test (α=0.05) | Moderate to strong positive correlation |
| H2 | GPA is the most predictive academic factor | GPA corr ≤ max(other factors) | GPA corr > max(other factors) | Correlation comparison | High correlation (>0.7) and highest among factors |
| H3 | University research intensity (QS Research Intensity) affects acceptance probability | μ_high_res = μ_low_res | μ_high_res ≠ μ_low_res | Mann-Whitney U test | Significant difference between research intensity levels |
| H4 | SOP and LOR scores moderately affect predictions | ρ = 0 OR \|ρ\| not in [0.4,0.6] | 0.4 ≤ \|ρ\| ≤ 0.6 | Pearson correlation + range validation | Medium correlation (0.4–0.6) |
| H5 | University prestige (QS Overall Score) correlates with applicant GRE scores | ρ_s = 0 | ρ_s > 0 | Spearman rank correlation | Positive correlation |
| H6 | QS ranking and cost-of-living data reveal socioeconomic patterns | No additional insights | Meaningful patterns exist | Correlation analysis | Partial correlation; richer insights expected |

---

## 🔍 Key Findings
Based on the analysis:
- QS Academic Reputation Score shows the strongest correlation with admission outcomes.  
- University research intensity significantly affects acceptance patterns.  
- Higher prestige universities attract applicants with higher GRE scores.  
- Institutional prestige and country-specific economics reveal patterns in admission data.  

These insights aim to **inform both applicants and universities** about the factors driving admission outcomes.

---

## 📊 Outputs Generated

After running the analysis, the following outputs are generated:

### Reports (in `outputs/` directory)
- `eda_report.txt` - Comprehensive EDA summary
- `hypothesis_testing_report.txt` - Detailed hypothesis test results (with H0/H1 and methods)

### Visualizations (in `outputs/` directory)
- `correlation_heatmap.png` - Correlation matrix
- `distributions.png` - Distribution plots
- `boxplots.png` - Outlier detection
- `feature_relationships.png` - Scatter plots with target
- `categorical_analysis.png` - Categorical variable analysis
- `hypothesis_1.png` through `hypothesis_6.png` - Hypothesis-specific plots

### Execution Logs (in `logs/` directory)
- `complete_analysis_TIMESTAMP.log` - Full execution log with all outputs
- `analysis_summary_TIMESTAMP.txt` - Quick summary report

---

## 🚀 Future Work
To extend this project:
- Collect more recent or domain-specific datasets (e.g., AI or CS graduate programs).  
- Integrate additional data sources for SOP/LOR analysis.  
- Explore additional fairness metrics and bias detection methods.

---

## 📚 Documentation

Additional documentation files:
- `md_files/DATASET_MERGING_STRATEGY.md` - Detailed explanation of dataset merging approach
- `md_files/QUICKSTART.md` - Quick start guide

---

## 🧾 Example Data Table (Sample)

| uni_name | GRE Score | GRE Verbal | GRE Quant | GPA | qs_Overall_Score | qs_RANK_2025 | Chance of Admit |
|:---------:|:-----------:|:------------:|:-----------:|:----:|:----------------:|:-------------:|:----------------:|
| University of Wisconsin-Madison | 654 | 292 | 322 | 3.72 | 65.3 | 101-150 | 1.0 |
| Massachusetts Institute of Technology | 800 | 400 | 400 | 3.95 | 100.0 | 1 | 1.0 |
| University of California, Berkeley | 700 | 310 | 330 | 3.80 | 89.4 | 10 | 0.0 |

---

This project demonstrates how data science can be applied to a **real-world, student-relevant problem** — understanding graduate admissions quantitatively.  
By combining statistical analysis and hypothesis testing, it provides insights to guide future applicants.

---
