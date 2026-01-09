# Graduate Admission Prediction with Real-World Applicant Data

**DSA210 – Introduction to Data Science · Term Project**

A full data‑science pipeline on 30K+ GradCafe applications, enriched with QS World University Rankings 2025 and global cost‑of‑living indicators to understand what truly drives graduate admission outcomes.

**Student:** Caglar Uysal  
**Course:** DSA‑210 – Introduction to Data Science  
**Institution:** Sabancı University

---

## Motivation

**Why this project?**

Graduate admissions are competitive, expensive, and often opaque. Applicants invest heavily in exams like the GRE without clearly knowing how much these scores actually matter compared to GPA, university prestige, or economic context.

This project uses real-world admission data to answer a practical question: **"Which factors matter most for getting accepted to graduate school?"**

The goal is to provide evidence‑based guidance for future applicants and to bring more transparency to how academic, institutional, and socioeconomic factors interact in graduate admissions.

---

## Data Source & Collection

**Datasets used**

### Primary dataset – GradCafe Admission Data

- Public dataset of **real admission results** scraped from GradCafe (GitHub).
- Contains 271K+ total results (27K+ in Computer Science) with fields such as university, major, degree, decision (accepted/rejected), GRE Verbal/Quant/Writing, GPA, season, and applicant status.

### Enrichment datasets

- **QS World University Rankings 2025** (Kaggle) – overall score, academic reputation, research intensity, and other ranking metrics.
- **Global Cost of Living Index** (Kaggle) – country-level cost of living, rent, groceries, and purchasing power indices.

### Collection & merging workflow

- Raw GradCafe data loaded and cleaned; university names standardized and **matched directly** to QS universities using mapping files.
- QS ranking features and cost‑of‑living indicators merged into a single **final combined dataset** with 30,135 observations and 72 features.

---

## Data Analysis Pipeline

**Methods & stages**

### 1. Data cleaning & integration

- Handled missing values, ensured no duplicate rows, standardized key fields.
- Merged GradCafe data with QS rankings and cost‑of‑living indices using university name mappings and country information.

### 2. Exploratory Data Analysis (EDA)

- Described distributions of key variables (GRE, GPA, QS scores, cost‑of‑living).
- Explored correlations between academic metrics, institutional prestige, and admission outcomes.
- Generated visualizations: distributions, boxplots, correlation heatmap, categorical analysis, and feature‑relationship plots.

### 3. Hypothesis testing

- Formulated six hypotheses with explicit **H₀/H₁** and appropriate tests (Pearson/Spearman correlation, Mann–Whitney U, etc.).
- Tested, for example, whether GRE scores strongly predict admission, whether GPA is the most predictive academic factor, and how QS rankings and cost‑of‑living relate to outcomes.

### 4. Machine learning modelling

- Trained 7 models (Decision Tree, Random Forest, Logistic/Linear Regression, Neural Network, Voting and Stacking ensembles) to predict admission decisions.
- Selected a **Stacking Classifier** as the best model based on cross‑validation, test accuracy, ROC‑AUC, and F1‑score.
  - *Note: Stacking combines multiple models: first, several base models make predictions; then a "meta-learner" learns how to best combine those predictions for the final decision. Models used a selected subset of features including QS rankings, GRE scores, GPA, and categorical variables, not all 72 features.*
- Performed **K‑means clustering** (k = 5) to identify applicant segments with different acceptance rates.

---

## Key Findings

**What the data says**

| Metric | Value |
|--------|-------|
| **Dataset** | 30,135 observations · 72 total features |
| **Best model** | Stacking Classifier (Acc ≈ 63.5% · ROC‑AUC ≈ 0.69) |
| **Clusters** | 5 distinct applicant groups |

### Key Insights

- **GRE scores alone are weak predictors** of admission (correlation ≈ −0.013); they are necessary but not sufficient.

- **GPA is important but not the most predictive factor**; in this dataset, GRE subject score and several institutional metrics show stronger associations with outcomes.

- **University prestige matters:** QS Academic Reputation and overall scores show significant correlations with admission patterns and with required GRE levels—higher prestige universities tend to require higher GRE scores and appear more selective.

- **Research intensity affects admission patterns:** universities with very high research intensity exhibit clearly different acceptance behavior compared to lower‑intensity institutions.

- **Cost of living indices show little direct effect** on admission probability, suggesting that economic context is less influential than academic and institutional factors in this dataset.

- **Clustering reveals heterogeneous applicant profiles** with acceptance rates ranging roughly from low‑40% to above 60%, emphasizing that combinations of features (major, season, scores, prestige) define distinct opportunity levels.

---

## Limitations & Future Work

**What could be improved**

### Limitations

- Data is self‑reported by GradCafe users and may contain **selection and reporting bias**.
- Critical qualitative features such as **SOP quality, LOR strength, and research projects** are not available and could not be tested.
- QS rankings and cost‑of‑living indices are used as proxies and may not perfectly reflect individual program policies or personal financial constraints.

### Future directions

- Collect richer data on SOP/LOR, research experience, and funding to better model holistic review.
- Extend the analysis to **more recent or domain‑specific cohorts** (e.g., AI/ML programs) and include fairness/bias diagnostics.
- Explore more advanced models (e.g., deep learning, sequence models for application histories) and interpretability tools such as SHAP or LIME.
- Turn the trained model and insights into an **interactive decision‑support tool** where students can input their profile and explore realistic target schools.

---

**Tags:** End‑to‑end data science pipeline · Real‑world student impact · Evidence‑based application strategy

---

*This markdown file summarizes the full project; for full code, logs, and figures see the accompanying repository.*

