"""
Exploratory Data Analysis (EDA) Script

This script performs comprehensive exploratory data analysis on the graduate
admission dataset, including:
- Descriptive statistics
- Data quality checks
- Correlation analysis
- Distribution visualizations
- Outlier detection
- Feature relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Directories
DATA_DIR = Path('data')
PROCESSED_DIR = DATA_DIR / 'processed'
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load the processed dataset"""
    # Try final combined dataset first
    final_path = Path('final_combined_dataset.csv')
    merged_path = Path('merged_data_combined.csv')
    processed_path = PROCESSED_DIR / 'admissions_merged.csv'
    
    if final_path.exists():
        print(f"Loading data from {final_path}")
        df = pd.read_csv(final_path, low_memory=False)
    elif merged_path.exists():
        print(f"Loading data from {merged_path}")
        df = pd.read_csv(merged_path, low_memory=False)
    elif processed_path.exists():
        print(f"Loading data from {processed_path}")
        df = pd.read_csv(processed_path, low_memory=False)
    else:
        # Try raw data as fallback
        raw_path = DATA_DIR / 'raw'
        csv_files = list(raw_path.glob('*.csv'))
        if csv_files:
            print(f"Loading data from {csv_files[0]}")
            df = pd.read_csv(csv_files[0], low_memory=False)
        else:
            raise FileNotFoundError("No data file found. Please run data_collection_new.py first.")
    
    return df

def check_data_quality(df):
    """Perform data quality checks"""
    print("=" * 60)
    print("DATA QUALITY CHECKS")
    print("=" * 60)
    
    print(f"\n1. Dataset Shape: {df.shape}")
    print(f"   - Rows: {df.shape[0]}")
    print(f"   - Columns: {df.shape[1]}")
    
    print(f"\n2. Column Names and Types:")
    for col in df.columns:
        dtype = df[col].dtype
        print(f"   - {col}: {dtype}")
    
    print(f"\n3. Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    if len(missing_df) > 0:
        print(missing_df.to_string())
    else:
        print("   [OK] No missing values found")
    
    print(f"\n4. Duplicate Rows: {df.duplicated().sum()}")
    
    print(f"\n5. Data Types Summary:")
    print(f"   - Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"   - Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
    
    return missing_df

def descriptive_statistics(df):
    """Generate descriptive statistics"""
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("\nNumeric Variables Summary:")
    print(df[numeric_cols].describe().round(2))
    
    # Check for target variable
    if 'Chance of Admit' in df.columns or 'Chance_of_Admit' in df.columns:
        target_col = 'Chance of Admit' if 'Chance of Admit' in df.columns else 'Chance_of_Admit'
        print(f"\nTarget Variable ({target_col}) Distribution:")
        print(df[target_col].describe().round(3))
    
    return df[numeric_cols].describe()

def correlation_analysis(df):
    """Perform correlation analysis"""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))
    
    # Find target variable
    target_col = None
    for col in ['Chance of Admit', 'Chance_of_Admit', 'chance_of_admit']:
        if col in df.columns:
            target_col = col
            break
    
    if target_col:
        print(f"\nCorrelations with {target_col}:")
        correlations = corr_matrix[target_col].sort_values(ascending=False)
        correlations = correlations[correlations.index != target_col]
        print(correlations.round(3))
    
    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"\n[OK] Correlation heatmap saved to {OUTPUT_DIR / 'correlation_heatmap.png'}")
    plt.close()
    
    return corr_matrix

def distribution_analysis(df):
    """Analyze distributions of variables"""
    print("\n" + "=" * 60)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Create distribution plots
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        if idx < len(axes):
            ax = axes[idx]
            df[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
            ax.set_title(f'Distribution of {col}', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distributions.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Distribution plots saved to {OUTPUT_DIR / 'distributions.png'}")
    plt.close()
    
    # Box plots for outlier detection
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        if idx < len(axes):
            ax = axes[idx]
            df.boxplot(column=col, ax=ax, vert=True)
            ax.set_title(f'Box Plot of {col}', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'boxplots.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Box plots saved to {OUTPUT_DIR / 'boxplots.png'}")
    plt.close()

def feature_relationships(df):
    """Analyze relationships between features and target"""
    print("\n" + "=" * 60)
    print("FEATURE RELATIONSHIPS")
    print("=" * 60)
    
    # Find target variable
    target_col = None
    for col in ['Chance of Admit', 'Chance_of_Admit', 'chance_of_admit']:
        if col in df.columns:
            target_col = col
            break
    
    if not target_col:
        print("[WARN] Target variable not found. Skipping feature relationship analysis.")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    # Scatter plots with target
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(feature_cols):
        if idx < len(axes):
            ax = axes[idx]
            ax.scatter(df[feature], df[target_col], alpha=0.5, s=30)
            ax.set_xlabel(feature)
            ax.set_ylabel(target_col)
            ax.set_title(f'{feature} vs {target_col}', fontweight='bold')
            
            # Add trend line (only if we have enough data points)
            feature_data = df[feature].dropna()
            target_data = df[target_col][df[feature].notna()]
            if len(feature_data) > 1 and len(target_data) > 1:
                try:
                    z = np.polyfit(feature_data, target_data, 1)
                    p = np.poly1d(z)
                    sorted_feature = df[feature].sort_values().dropna()
                    if len(sorted_feature) > 0:
                        ax.plot(sorted_feature, p(sorted_feature), 
                               "r--", alpha=0.8, linewidth=2)
                except (TypeError, np.linalg.LinAlgError):
                    # Skip trend line if polyfit fails
                    pass
            ax.grid(True, alpha=0.3)
    
    for idx in range(len(feature_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_relationships.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Feature relationship plots saved to {OUTPUT_DIR / 'feature_relationships.png'}")
    plt.close()

def categorical_analysis(df):
    """Analyze categorical variables"""
    print("\n" + "=" * 60)
    print("CATEGORICAL ANALYSIS")
    print("=" * 60)
    
    # Check for research experience
    research_cols = [col for col in df.columns if 'research' in col.lower() or 'Research' in col]
    
    if research_cols:
        research_col = research_cols[0]
        value_counts = df[research_col].value_counts()
        
        if len(value_counts) > 0:
            print(f"\n{research_col} Distribution:")
            print(value_counts)
            print(f"\n{research_col} Percentages:")
            print(df[research_col].value_counts(normalize=True).round(3) * 100)
            
            # Visualize
            if 'Chance of Admit' in df.columns or 'Chance_of_Admit' in df.columns:
                target_col = 'Chance of Admit' if 'Chance of Admit' in df.columns else 'Chance_of_Admit'
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Count plot
                value_counts.plot(kind='bar', ax=axes[0], color=['skyblue', 'salmon'])
                axes[0].set_title(f'Distribution of {research_col}', fontweight='bold')
                axes[0].set_xlabel(research_col)
                axes[0].set_ylabel('Count')
                axes[0].tick_params(axis='x', rotation=0)
                
                # Box plot by research experience
                try:
                    valid_data = df[[research_col, target_col]].dropna()
                    if len(valid_data) > 0 and len(valid_data[research_col].unique()) > 0:
                        valid_data.boxplot(column=target_col, by=research_col, ax=axes[1])
                        axes[1].set_title(f'{target_col} by {research_col}', fontweight='bold')
                        axes[1].set_xlabel(research_col)
                        axes[1].set_ylabel(target_col)
                    else:
                        axes[1].text(0.5, 0.5, 'Insufficient data for boxplot', 
                                    ha='center', va='center', transform=axes[1].transAxes)
                        axes[1].set_title(f'{target_col} by {research_col}', fontweight='bold')
                except (ValueError, KeyError):
                    axes[1].text(0.5, 0.5, 'Could not create boxplot', 
                                ha='center', va='center', transform=axes[1].transAxes)
                    axes[1].set_title(f'{target_col} by {research_col}', fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / 'categorical_analysis.png', dpi=300, bbox_inches='tight')
                print(f"[OK] Categorical analysis plots saved to {OUTPUT_DIR / 'categorical_analysis.png'}")
                plt.close()
        else:
            print(f"[WARN] {research_col} has no valid values to analyze")
    else:
        print("[WARN] No research column found in dataset")

def generate_eda_report(df):
    """Generate comprehensive EDA report with enhanced analysis"""
    print("\n" + "=" * 60)
    print("GENERATING EDA REPORT")
    print("=" * 60)
    
    report = []
    report.append("=" * 80)
    report.append("EXPLORATORY DATA ANALYSIS REPORT - COMPREHENSIVE")
    report.append("=" * 80)
    report.append(f"\nDataset Shape: {df.shape}")
    report.append(f"Date Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Observations: {len(df):,}")
    report.append(f"Total Features: {len(df.columns)}")
    
    # Data quality
    missing_df = check_data_quality(df)
    report.append("\n\n" + "=" * 80)
    report.append("DATA QUALITY SUMMARY")
    report.append("=" * 80)
    report.append(f"Missing Values: {df.isnull().sum().sum():,}")
    report.append(f"Duplicate Rows: {df.duplicated().sum()}")
    report.append(f"Data Completeness: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%")
    
    # Column breakdown
    qs_cols = [c for c in df.columns if c.startswith('qs_')]
    col_cols = [c for c in df.columns if c.startswith('col_')]
    original_cols = [c for c in df.columns if not c.startswith('qs_') and not c.startswith('col_')]
    
    report.append("\n\nCOLUMN BREAKDOWN:")
    report.append(f"  Original columns (GradCafe data): {len(original_cols)}")
    report.append(f"  QS Rankings columns: {len(qs_cols)}")
    report.append(f"  Cost of Living columns: {len(col_cols)}")
    report.append(f"  Total columns: {len(df.columns)}")
    
    # Key columns analysis
    report.append("\n\nKEY COLUMNS ANALYSIS:")
    key_columns = {
        'Target Variable': ['Chance of Admit', 'is_accepted'],
        'Academic Metrics': ['GRE Score', 'GPA', 'GRE Verbal', 'GRE Quant'],
        'University Metrics': ['qs_Overall_Score', 'qs_RANK_2025', 'qs_Academic_Reputation_Score'],
        'Economic Metrics': ['col_Cost of Living Index_mean', 'col_Rent Index_mean']
    }
    
    for category, cols in key_columns.items():
        report.append(f"\n  {category}:")
        for col in cols:
            if col in df.columns:
                non_null = df[col].notna().sum()
                pct = (non_null / len(df)) * 100
                if non_null > 0:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        mean_val = df[col].mean()
                        report.append(f"    - {col}: {non_null:,} values ({pct:.1f}%), Mean: {mean_val:.2f}")
                    else:
                        unique_count = df[col].nunique()
                        report.append(f"    - {col}: {non_null:,} values ({pct:.1f}%), Unique: {unique_count}")
                else:
                    report.append(f"    - {col}: No data")
    
    # Descriptive stats
    desc_stats = descriptive_statistics(df)
    report.append("\n\n" + "=" * 80)
    report.append("DESCRIPTIVE STATISTICS")
    report.append("=" * 80)
    report.append("\nKey Numeric Variables Summary:")
    key_numeric = ['GRE Score', 'GPA', 'qs_Overall_Score', 'qs_Academic_Reputation_Score', 
                   'col_Cost of Living Index_mean', 'Chance of Admit']
    for col in key_numeric:
        if col in df.columns and df[col].notna().sum() > 0:
            stats = df[col].describe()
            report.append(f"\n  {col}:")
            report.append(f"    Count: {stats['count']:,.0f}")
            report.append(f"    Mean: {stats['mean']:.2f}")
            report.append(f"    Std: {stats['std']:.2f}")
            report.append(f"    Min: {stats['min']:.2f}")
            report.append(f"    Max: {stats['max']:.2f}")
            report.append(f"    Median: {stats['50%']:.2f}")
    
    # Correlation
    corr_matrix = correlation_analysis(df)
    target_col = None
    for col in ['Chance of Admit', 'Chance_of_Admit', 'chance_of_admit']:
        if col in df.columns:
            target_col = col
            break
    
    if target_col:
        correlations = corr_matrix[target_col].sort_values(ascending=False)
        correlations = correlations[correlations.index != target_col]
        
        report.append("\n\n" + "=" * 80)
        report.append("TOP CORRELATIONS WITH TARGET (Chance of Admit)")
        report.append("=" * 80)
        
        # Positive correlations
        positive_corr = correlations[correlations > 0].head(10)
        if len(positive_corr) > 0:
            report.append("\n  Top 10 Positive Correlations:")
            for feature, corr in positive_corr.items():
                report.append(f"    {feature}: {corr:.3f}")
        
        # Negative correlations
        negative_corr = correlations[correlations < 0].head(10)
        if len(negative_corr) > 0:
            report.append("\n  Top 10 Negative Correlations:")
            for feature, corr in negative_corr.items():
                report.append(f"    {feature}: {corr:.3f}")
        
        # QS Rankings correlations
        qs_corrs = [(k, v) for k, v in correlations.items() if k.startswith('qs_')]
        if qs_corrs:
            qs_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
            report.append("\n  QS Rankings Correlations (Top 5):")
            for feature, corr in qs_corrs[:5]:
                report.append(f"    {feature}: {corr:.3f}")
        
        # Cost of Living correlations
        col_corrs = [(k, v) for k, v in correlations.items() if k.startswith('col_')]
        if col_corrs:
            col_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
            report.append("\n  Cost of Living Correlations (Top 5):")
            for feature, corr in col_corrs[:5]:
                report.append(f"    {feature}: {corr:.3f}")
    
    # University analysis
    if 'uni_name' in df.columns:
        report.append("\n\n" + "=" * 80)
        report.append("UNIVERSITY ANALYSIS")
        report.append("=" * 80)
        report.append(f"  Unique Universities: {df['uni_name'].nunique():,}")
        report.append(f"\n  Top 10 Universities by Application Count:")
        top_unis = df['uni_name'].value_counts().head(10)
        for idx, (uni, count) in enumerate(top_unis.items(), 1):
            pct = (count / len(df)) * 100
            report.append(f"    {idx}. {uni}: {count:,} applications ({pct:.1f}%)")
    
    # QS Rankings coverage
    if 'qs_Overall_Score' in df.columns:
        qs_coverage = df['qs_Overall_Score'].notna().sum() / len(df) * 100
        report.append(f"\n  QS Rankings Coverage: {qs_coverage:.1f}%")
    
    # Cost of Living coverage
    if 'col_Cost of Living Index_mean' in df.columns:
        col_coverage = df['col_Cost of Living Index_mean'].notna().sum() / len(df) * 100
        report.append(f"  Cost of Living Coverage: {col_coverage:.1f}%")
    
    # Save report
    report_text = "\n".join([str(r) for r in report])
    with open(OUTPUT_DIR / 'eda_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"[OK] EDA report saved to {OUTPUT_DIR / 'eda_report.txt'}")

def main():
    """Main EDA function"""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Load data
    try:
        df = load_data()
        print(f"\n[OK] Data loaded successfully: {df.shape}")
    except Exception as e:
        print(f"\n[ERROR] Error loading data: {e}")
        return
    
    # Perform EDA
    check_data_quality(df)
    descriptive_statistics(df)
    correlation_analysis(df)
    distribution_analysis(df)
    feature_relationships(df)
    categorical_analysis(df)
    generate_eda_report(df)
    
    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

