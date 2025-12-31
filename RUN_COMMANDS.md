# Command-Line Commands to Run the Project

## Prerequisites

### 1. Install Dependencies
```cmd
pip install -r requirements.txt
```

---

## Quick Start: Run Complete Pipeline

### Option 1: Run Everything at Once (Recommended)
```cmd
python scripts/00_run_complete_analysis.py
```

This will run:
- Data Collection
- Exploratory Data Analysis
- Hypothesis Testing
- Machine Learning Analysis

**Output:** All results saved to `outputs/` and `ml_outputs/` directories

---

## Run Individual Scripts

### 1. Data Collection and Merging
```cmd
python scripts/01_data_collection.py
```
**Output:** Creates `data/processed/merged_data_combined.csv`

### 2. Exploratory Data Analysis (EDA)
```cmd
python scripts/02_exploratory_data_analysis.py
```
**Output:** 
- `outputs/eda_report.txt`
- `outputs/correlation_heatmap.png`
- `outputs/distributions.png`
- `outputs/boxplots.png`
- `outputs/feature_relationships.png`
- `outputs/categorical_analysis.png`

### 3. Hypothesis Testing
```cmd
python scripts/03_hypothesis_testing.py
```
**Output:**
- `outputs/hypothesis_testing_report.txt`
- `outputs/hypothesis_1.png` through `hypothesis_6.png`

### 4. Machine Learning Analysis
```cmd
python scripts/04_machine_learning.py
```
**Output:**
- `ml_outputs/ml_analysis_report.txt`
- `ml_outputs/decision_tree_analysis.png`
- `ml_outputs/model_comparison.png`
- `ml_outputs/clustering_analysis.png`
- `ml_outputs/cluster_comparison.png`
- `ml_outputs/best_model.pkl`
- `ml_outputs/scaler.pkl`
- `ml_outputs/model_metadata.json`

---

## Utility Scripts

### Create University Mappings
```cmd
python scripts/utils_map_universities.py
```
**Output:** `data/mappings/university_mappings_from_qs.csv`

### Create Mapping JSON Files
```cmd
python scripts/utils_create_mappings.py
```
**Output:** JSON mapping files in `data/mappings/`

### Save Final Dataset
```cmd
python scripts/utils_save_dataset.py
```
**Output:** `data/processed/final_combined_dataset.csv`

---

## Complete Workflow Example

### Step-by-Step Execution:
```cmd
REM Step 1: Install dependencies
pip install -r requirements.txt

REM Step 2: Run data collection
python scripts/01_data_collection.py

REM Step 3: Run EDA
python scripts/02_exploratory_data_analysis.py

REM Step 4: Run hypothesis testing
python scripts/03_hypothesis_testing.py

REM Step 5: Run ML analysis
python scripts/04_machine_learning.py
```

---

## PowerShell Commands (Alternative)

If you prefer PowerShell:

```powershell
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python scripts/00_run_complete_analysis.py

# Or run individually:
python scripts/01_data_collection.py
python scripts/02_exploratory_data_analysis.py
python scripts/03_hypothesis_testing.py
python scripts/04_machine_learning.py
```

---

## Troubleshooting

### If you get "Module not found" errors:
```cmd
pip install pandas numpy matplotlib seaborn scikit-learn scipy rapidfuzz
```

### If you get "File not found" errors:
Make sure you're running commands from the project root directory:
```cmd
cd C:\Users\batu0\Desktop\DSA-210-TermProject-CaglarUysal
```

### Check if data file exists:
```cmd
dir data\processed\final_combined_dataset.csv
```

---

## Expected Runtime

- **Data Collection:** ~1-2 minutes
- **EDA:** ~2-3 minutes
- **Hypothesis Testing:** ~1-2 minutes
- **ML Analysis:** ~5-10 minutes (depending on your CPU)
- **Complete Pipeline:** ~10-15 minutes total

---

## Quick Reference

| Task | Command |
|------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Run everything | `python scripts/00_run_complete_analysis.py` |
| Data collection only | `python scripts/01_data_collection.py` |
| EDA only | `python scripts/02_exploratory_data_analysis.py` |
| Hypothesis testing only | `python scripts/03_hypothesis_testing.py` |
| ML analysis only | `python scripts/04_machine_learning.py` |


