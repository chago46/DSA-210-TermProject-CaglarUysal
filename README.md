# 🎓 Graduate_Admission_Prediction

I'm **Caglar Uysal**, a Computer Science and Engineering student at Sabancı University, currently taking **DSA210 – Introduction to Data Science**.  

This repository contains my **term project** for the course.

---

## 🧭 Table of Contents
1. Project Overview  
2. Objectives  
3. Motivation  
4. Data Collection & Sources  
 • Primary Data Source  
 • Enrichment Data Sources  
5. Research Questions  
6. Tools & Libraries Used  
7. Hypotheses and Expected Results  
8. Machine Learning Models & Prediction Plan  
9. Key Insights (Expected)  
10. Future Work  
11. Example Data Table  

---

## 📘 Project Overview
Graduate admissions are among the most competitive and uncertain processes students face.  
This project analyzes a **public graduate admission dataset** to identify which factors most strongly influence a student's chance of admission to a graduate program.

The main goal is to build a **predictive and interpretable model** that estimates the probability of admission based on quantitative and qualitative variables (GRE, TOEFL, GPA, Research Experience, etc.).  
The project also investigates how institutional factors—such as **university ranking** or **cost of living**—affect admission outcomes.

---

## 🎯 Objectives
- Analyze which academic and qualitative factors have the strongest impact on graduate admission.  
- Build regression and classification models to predict admission probabilities.  
- Quantify and visualize the influence of each feature (e.g., GPA vs. SOP score).  
- Apply end-to-end data science methodologies — from data cleaning to model interpretation.  
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
**Dataset:** [Graduate Admission Dataset (Kaggle)](https://www.kaggle.com/mohansacharya/graduate-admissions)  
This dataset includes 500 anonymized records of graduate applicants, each with the following features:
- GRE Score  
- TOEFL Score  
- University Rating  
- Statement of Purpose (SOP) Strength  
- Letter of Recommendation (LOR) Strength  
- Undergraduate GPA  
- Research Experience (0 or 1)  
- Chance of Admit (0–1)

---

### 🌍 Enrichment Data Sources
To add originality and external context, the dataset will be combined with:
- **QS World University Rankings** (to represent institutional prestige)  
- **Numbeo Cost of Living API** (to account for financial context by country)  

These additions will allow testing whether **economic or institutional factors** influence admission trends.

---

## 🧠 Research Questions
1. Which feature has the strongest correlation with admission probability?  
2. Does research experience significantly increase the likelihood of acceptance?  
3. How does university rating (or QS rank) relate to GPA or TOEFL requirements?  
4. Can we predict a student’s admission chance with reasonable accuracy using regression or tree-based models?  
5. Are there fairness or bias implications across university rating tiers?

---

## ⚙️ Tools & Libraries Used
### 1. Programming Language  
- **Python** (v3.10+)

### 2. Data Analysis & Machine Learning  
- **pandas**, **numpy** – Data cleaning and manipulation  
- **scikit-learn** – ML modeling (Linear Regression, Random Forest, XGBoost)  
- **SHAP**, **statsmodels** – Interpretability and statistical inference  

### 3. Visualization  
- **matplotlib**, **seaborn** – Correlation and feature importance plots  
- **plotly** (optional) – Interactive admission probability graphs  

### 4. Data Access  
- **Kaggle API** – Download admission dataset  
- **requests** – Fetch QS ranking or cost-of-living data  

---

## 📈 Hypotheses and Expected Results

| ID | Hypothesis | Expected Outcome |
|----|-------------|------------------|
| H1 | GRE and TOEFL scores strongly correlate with Chance of Admit | Moderate to strong positive correlation |
| H2 | GPA is the most predictive academic factor | High correlation (>0.7) |
| H3 | Research experience significantly increases acceptance probability | Yes, especially for high GPA candidates |
| H4 | SOP and LOR scores moderately affect predictions | Medium correlation (0.4–0.6) |
| H5 | University rating (prestige) correlates with required GRE/TOEFL | Positive correlation |
| H6 | Adding QS ranking or cost-of-living data will reveal socioeconomic patterns | Partial correlation; richer insights expected |

---

## 🤖 Machine Learning Models & Prediction Plan

### Dataset Overview
After enrichment, the dataset will contain:
- ~500 observations × 8–10 features  
- Mix of numeric and ordinal variables  

### Models to Be Tested
- **Linear Regression** (baseline)  
- **Random Forest Regressor** (non-linear relationships)  
- **XGBoost Regressor** (optimized gradient boosting)  

### Evaluation Metrics
- **RMSE (Root Mean Squared Error)**  
- **R² Score**  
- **Cross-Validation Mean Error**  

### Feature Interpretation
- SHAP feature importance analysis  
- Correlation heatmaps and scatter plots  

---

## 🔍 Key Insights (Expected)
- GPA and GRE will likely be the most influential features.  
- Research experience will have a measurable impact on acceptance probability.  
- SOP/LOR strength may differentiate candidates within similar score ranges.  
- Institutional prestige and country-specific economics may reveal subtle biases in admission patterns.  

These insights aim to **inform both applicants and universities** about the factors driving admission outcomes.

---

## 🚀 Future Work
To extend this project beyond DSA210:
- Collect more recent or domain-specific datasets (e.g., AI or CS graduate programs).  
- Integrate natural language analysis of SOPs or essays.  
- Explore fairness metrics and bias detection in the model.  
- Build a web dashboard that allows users to input scores and see predicted admission probabilities interactively.

---

## 🧾 Example Data Table (Sample)

| GRE Score | TOEFL Score | University Rating | SOP | LOR | GPA | Research | Chance of Admit |
|:-----------:|:------------:|:----------------:|:---:|:---:|:----:|:----------:|:----------------:|
| 337 | 118 | 4 | 4.5 | 4.5 | 9.65 | 1 | 0.92 |
| 324 | 107 | 4 | 4.0 | 4.5 | 8.87 | 1 | 0.76 |
| 316 | 104 | 3 | 3.0 | 3.5 | 8.00 | 1 | 0.72 |
| 322 | 110 | 3 | 3.5 | 2.5 | 8.67 | 1 | 0.80 |
| 314 | 103 | 2 | 2.0 | 3.0 | 8.21 | 0 | 0.65 |

---

This project demonstrates how data science can be applied to a **real-world, student-relevant problem** — understanding graduate admissions quantitatively.  
By combining statistical analysis and machine learning, it provides both predictive power and interpretability to guide future applicants.

---
