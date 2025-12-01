# 🚀 Quick Start Guide

This guide will help you get started with the Graduate Admission Prediction project quickly.

## Prerequisites Check

Before starting, ensure you have:
- ✅ Python 3.10+ installed
- ✅ pip package manager
- ✅ Internet connection (for downloading datasets)

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset

Ensure the following datasets are in `data/raw/` directory:
- `all.csv` - GradCafe admission data
- `QS World University Rankings 2025 (Top global universities).csv` - QS Rankings
- `Cost_of_living_index.csv` - Cost of Living data

### 3. Run the Analysis

**Complete Pipeline with Full Logging (RECOMMENDED):**
```bash
python run_complete_analysis.py
```

This will:
- Run all analysis steps
- Log everything to `logs/complete_analysis_TIMESTAMP.log`
- Generate summary report in `logs/analysis_summary_TIMESTAMP.txt`
- Show progress in console and save to file simultaneously

**Or run individual steps:**
```bash
# Step 1: Collect and merge data
python data_collection_new.py

# Step 2: Exploratory Data Analysis
python eda.py

# Step 3: Hypothesis Testing
python hypothesis_testing.py
```

## Expected Output

After running the analysis, you should see:

```
outputs/
├── correlation_heatmap.png
├── distributions.png
├── boxplots.png
├── feature_relationships.png
├── categorical_analysis.png
├── hypothesis_1.png
├── hypothesis_2.png
├── hypothesis_3.png
├── hypothesis_5.png
├── hypothesis_6.png
├── eda_report.txt
└── hypothesis_testing_report.txt

logs/
├── complete_analysis_YYYYMMDD_HHMMSS.log  # Full execution log
└── analysis_summary_YYYYMMDD_HHMMSS.txt   # Summary report
```

## Troubleshooting

### Issue: "No data file found"
**Solution:**
- Ensure datasets are in `data/raw/` directory
- Check that CSV files are not corrupted
- Required files: `all.csv`, QS Rankings CSV, and Cost of Living CSV
- Run `data_collection_new.py` first

### Issue: "Module not found"
**Solution:**
- Run `pip install -r requirements.txt`
- Ensure you're in the project root directory
- Check that virtual environment is activated (if using one)
- Make sure `rapidfuzz` is installed: `pip install rapidfuzz`

## Next Steps

1. Review `outputs/eda_report.txt` for data insights
2. Check `outputs/hypothesis_testing_report.txt` for hypothesis results
3. Examine visualization files in `outputs/` directory
4. Refer to `README.md` for detailed documentation

## Need Help?

- Check the main `README.md` for detailed documentation
- Review code comments in individual scripts
- Ensure all dependencies are installed correctly

---

**Happy Analyzing! 🎓**

