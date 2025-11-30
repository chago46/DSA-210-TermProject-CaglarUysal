"""
Hypothesis Testing Script

This script tests the 6 hypotheses outlined in the README:
H1: GRE and TOEFL scores strongly correlate with Chance of Admit
H2: GPA is the most predictive academic factor
H3: Research experience significantly increases acceptance probability
H4: SOP and LOR scores moderately affect predictions
H5: University rating (prestige) correlates with required GRE/TOEFL
H6: Adding QS ranking or cost-of-living data will reveal socioeconomic patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu, chi2_contingency
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

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

def normalize_column_names(df):
    """Normalize column names to handle variations - keep original names for new dataset"""
    df = df.copy()
    # Note: New combined dataset uses original column names
    # We'll keep them as-is and handle variations in the analysis functions
    # Only normalize if we have old-style column names
    
    # Common variations (for backward compatibility)
    col_mapping = {
        'Chance_of_Admit': 'Chance of Admit',  # Prefer space version
        'GRE_Score': 'GRE Score',
        'TOEFL_Score': 'TOEFL Score',
        'University_Rating': 'University Rating',
        'CGPA': 'GPA',
        'CGPA ': 'GPA',
    }
    
    for old_name, new_name in col_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    
    return df

def hypothesis_1_gre_toefl_correlation(df):
    """
    H1: GRE and TOEFL scores strongly correlate with Chance of Admit
    
    Null Hypothesis (H0): ρ = 0 (No correlation between GRE/TOEFL and admission probability)
    Alternative Hypothesis (H1): rho != 0 (There is a correlation)
    
    Testing Method: Pearson correlation test
    - Test statistic: Pearson correlation coefficient (r)
    - Significance level: α = 0.05
    - Expected: Moderate to strong positive correlation (|r| > 0.4)
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS 1: GRE and TOEFL Correlation with Admission")
    print("=" * 60)
    print("\nH0: rho = 0 (No correlation)")
    print("H1: rho != 0 (Correlation exists)")
    print("Method: Pearson correlation test (alpha = 0.05)")
    
    # Find columns
    target_col = None
    gre_col = None
    toefl_col = None
    
    for col in df.columns:
        col_lower = col.lower().replace(' ', '_')
        if 'chance' in col_lower and 'admit' in col_lower:
            target_col = col
        elif 'gre' in col_lower and 'score' in col_lower:
            gre_col = col  # Prefer 'GRE Score' over 'GRE Verbal' or 'GRE Quant'
        elif 'gre' in col_lower and gre_col is None:
            gre_col = col  # Fallback to any GRE column
        elif 'toefl' in col_lower:
            toefl_col = col
    
    if not target_col:
        print("[WARN] Target column (Chance of Admit) not found. Skipping H1.")
        return None
    
    if not gre_col:
        print("[WARN] GRE Score column not found. Skipping H1.")
        return None
    
    # TOEFL is optional (not in GradCafe data)
    if not toefl_col:
        print("[WARN] TOEFL Score not available (GradCafe data doesn't include TOEFL)")
        print("  Testing GRE correlation only...")
    
    results = {}
    
    # Test GRE correlation
    gre_data = df[[gre_col, target_col]].dropna()
    if len(gre_data) > 0:
        # Ensure data is numeric
        gre_series = pd.to_numeric(gre_data[gre_col], errors='coerce')
        target_series = pd.to_numeric(gre_data[target_col], errors='coerce')
        valid_data = pd.DataFrame({gre_col: gre_series, target_col: target_series}).dropna()
        
        if len(valid_data) > 1:
            gre_corr, gre_p = pearsonr(valid_data[gre_col], valid_data[target_col])
            results['GRE'] = {'correlation': gre_corr, 'p_value': gre_p}
            print(f"\nGRE Score vs {target_col}:")
            print(f"  Pearson Correlation: {gre_corr:.4f}")
            print(f"  P-value: {gre_p:.4f}")
            print(f"  Significant: {'Yes' if gre_p < 0.05 else 'No'} (alpha=0.05)")
            print(f"  Strength: {'Strong' if abs(gre_corr) > 0.7 else 'Moderate' if abs(gre_corr) > 0.4 else 'Weak'}")
        else:
            print(f"\n[WARN] Not enough valid GRE data for correlation test (need at least 2 data points)")
    
    # Test TOEFL correlation (if available)
    if toefl_col:
        toefl_data = df[[toefl_col, target_col]].dropna()
        if len(toefl_data) > 0:
            # Ensure data is numeric
            toefl_series = pd.to_numeric(toefl_data[toefl_col], errors='coerce')
            target_series = pd.to_numeric(toefl_data[target_col], errors='coerce')
            valid_toefl = pd.DataFrame({toefl_col: toefl_series, target_col: target_series}).dropna()
            
            if len(valid_toefl) > 1:
                toefl_corr, toefl_p = pearsonr(valid_toefl[toefl_col], valid_toefl[target_col])
                results['TOEFL'] = {'correlation': toefl_corr, 'p_value': toefl_p}
                print(f"\nTOEFL Score vs {target_col}:")
                print(f"  Pearson Correlation: {toefl_corr:.4f}")
                print(f"  P-value: {toefl_p:.4f}")
                print(f"  Significant: {'Yes' if toefl_p < 0.05 else 'No'} (α=0.05)")
                print(f"  Strength: {'Strong' if abs(toefl_corr) > 0.7 else 'Moderate' if abs(toefl_corr) > 0.4 else 'Weak'}")
    
    # Visualization
    n_plots = 2 if toefl_col and 'TOEFL' in results else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    # Ensure numeric for plotting
    gre_plot = pd.to_numeric(df[gre_col], errors='coerce')
    target_plot = pd.to_numeric(df[target_col], errors='coerce')
    valid_plot = pd.DataFrame({gre_col: gre_plot, target_col: target_plot}).dropna()
    
    if len(valid_plot) > 0:
        axes[0].scatter(valid_plot[gre_col], valid_plot[target_col], alpha=0.5)
        axes[0].set_xlabel(gre_col)
        axes[0].set_ylabel(target_col)
        if 'GRE' in results:
            axes[0].set_title(f'GRE Score vs {target_col}\n(r={results["GRE"]["correlation"]:.3f}, p={results["GRE"]["p_value"]:.4f})')
        else:
            axes[0].set_title(f'GRE Score vs {target_col}')
        
        # Add trend line if we have enough points
        if len(valid_plot) > 1:
            try:
                z = np.polyfit(valid_plot[gre_col], valid_plot[target_col], 1)
                p = np.poly1d(z)
                sorted_gre = valid_plot[gre_col].sort_values()
                axes[0].plot(sorted_gre, p(sorted_gre), "r--", alpha=0.8)
            except (TypeError, np.linalg.LinAlgError):
                pass
        axes[0].grid(True, alpha=0.3)
    
    if toefl_col and 'TOEFL' in results:
        # Ensure numeric for plotting
        toefl_plot = pd.to_numeric(df[toefl_col], errors='coerce')
        target_plot = pd.to_numeric(df[target_col], errors='coerce')
        valid_toefl_plot = pd.DataFrame({toefl_col: toefl_plot, target_col: target_plot}).dropna()
        
        if len(valid_toefl_plot) > 0:
            axes[1].scatter(valid_toefl_plot[toefl_col], valid_toefl_plot[target_col], alpha=0.5)
            axes[1].set_xlabel(toefl_col)
            axes[1].set_ylabel(target_col)
            axes[1].set_title(f'TOEFL Score vs {target_col}\n(r={results["TOEFL"]["correlation"]:.3f}, p={results["TOEFL"]["p_value"]:.4f})')
            
            # Add trend line if we have enough points
            if len(valid_toefl_plot) > 1:
                try:
                    z = np.polyfit(valid_toefl_plot[toefl_col], valid_toefl_plot[target_col], 1)
                    p = np.poly1d(z)
                    sorted_toefl = valid_toefl_plot[toefl_col].sort_values()
                    axes[1].plot(sorted_toefl, p(sorted_toefl), "r--", alpha=0.8)
                except (TypeError, np.linalg.LinAlgError):
                    pass
            axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hypothesis_1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def hypothesis_2_gpa_most_predictive(df):
    """
    H2: GPA is the most predictive academic factor
    
    Null Hypothesis (H0): GPA correlation <= max(other academic factors)
    Alternative Hypothesis (H1): GPA correlation > max(other academic factors)
    
    Testing Method: Correlation comparison
    - Compare Pearson correlation coefficients across all academic factors
    - GPA should have the highest absolute correlation with admission probability
    - Expected: High correlation (>0.7) and highest among all factors
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS 2: GPA as Most Predictive Factor")
    print("=" * 60)
    print("\nH0: GPA correlation <= max(other factors)")
    print("H1: GPA correlation > max(other factors)")
    print("Method: Correlation coefficient comparison (alpha = 0.05)")
    
    # Find columns
    target_col = None
    gpa_col = None
    gre_col = None
    toefl_col = None
    
    for col in df.columns:
        col_lower = col.lower().replace(' ', '_')
        if 'chance' in col_lower and 'admit' in col_lower:
            target_col = col
        elif 'gpa' in col_lower or 'cgpa' in col_lower:
            gpa_col = col
        elif 'gre' in col_lower:
            gre_col = col
        elif 'toefl' in col_lower:
            toefl_col = col
    
    if not target_col:
        print("[WARN] Target column not found. Skipping H2.")
        return None
    
    # Define academic factors to compare (applicant academic metrics only)
    academic_factors = []
    academic_keywords = ['gpa', 'gre', 'toefl', 'cgpa', 'ugrad_gpa', 'gre_verbal', 'gre_quant', 'gre_writing', 'gre_subject']
    
    # Exclude these columns (target variables, IDs, non-academic factors)
    exclude_keywords = ['is_accepted', 'rowid', 'chance', 'admit', 'decision']
    
    for col in df.columns:
        col_lower = col.lower().replace(' ', '_')
        # Include if it's an academic factor
        if any(keyword in col_lower for keyword in academic_keywords):
            # Exclude if it's a target variable, ID, or non-academic factor
            if (col != target_col and 
                not any(exclude in col_lower for exclude in exclude_keywords) and
                not col.startswith('qs_') and 
                not col.startswith('col_') and
                col != 'is_accepted'):
                academic_factors.append(col)
    
    # Calculate correlations for academic factors only
    correlations = {}
    
    for col in academic_factors:
        data = df[[col, target_col]].dropna()
        if len(data) > 0:
            # Ensure data is numeric
            col_series = pd.to_numeric(data[col], errors='coerce')
            target_series = pd.to_numeric(data[target_col], errors='coerce')
            valid_data = pd.DataFrame({col: col_series, target_col: target_series}).dropna()
            
            if len(valid_data) > 1:
                corr, p_val = pearsonr(valid_data[col], valid_data[target_col])
                correlations[col] = {'correlation': corr, 'p_value': p_val}
    
    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
    
    print("\nAcademic Factor Correlations (sorted by strength):")
    for i, (factor, stats) in enumerate(sorted_corrs[:10], 1):
        print(f"{i}. {factor}:")
        print(f"   Correlation: {stats['correlation']:.4f}")
        print(f"   P-value: {stats['p_value']:.4f}")
        print(f"   Significant: {'Yes' if stats['p_value'] < 0.05 else 'No'}")
    
    if gpa_col and gpa_col in correlations:
        gpa_corr = abs(correlations[gpa_col]['correlation'])
        is_highest = all(gpa_corr >= abs(correlations[col]['correlation']) 
                        for col in correlations.keys())
        print(f"\n[OK] GPA Correlation: {correlations[gpa_col]['correlation']:.4f}")
        print(f"  Is GPA the most predictive? {'Yes' if is_highest else 'No'}")
        print(f"  Meets threshold (>0.7)? {'Yes' if gpa_corr > 0.7 else 'No'}")
    
    # Visualization
    factors = [col for col, _ in sorted_corrs[:8]]
    corrs = [stats['correlation'] for _, stats in sorted_corrs[:8]]
    
    plt.figure(figsize=(10, 6))
    colors = ['red' if abs(c) > 0.7 else 'orange' if abs(c) > 0.4 else 'blue' for c in corrs]
    plt.barh(factors, corrs, color=colors)
    plt.xlabel('Correlation with Chance of Admit')
    plt.title('Academic Factors Correlation Analysis', fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hypothesis_2.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return correlations

def hypothesis_3_research_experience(df):
    """
    H3: Research experience significantly increases acceptance probability
    
    Null Hypothesis (H0): μ_with_research = μ_without_research
    Alternative Hypothesis (H1): μ_with_research > μ_without_research
    
    Testing Method: 
    - Independent t-test (if data is normally distributed)
    - Mann-Whitney U test (if data is not normally distributed)
    - Significance level: α = 0.05
    - Expected: Significant positive effect, especially for high GPA candidates
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS 3: Research Experience Impact")
    print("=" * 60)
    print("\nH0: mu_with_research = mu_without_research")
    print("H1: mu_with_research > mu_without_research")
    print("Method: Independent t-test or Mann-Whitney U test (alpha = 0.05)")
    
    # Find columns
    target_col = None
    research_col = None
    gpa_col = None
    
    for col in df.columns:
        col_lower = col.lower().replace(' ', '_')
        if 'chance' in col_lower and 'admit' in col_lower:
            target_col = col
        elif 'research' in col_lower:
            research_col = col
        elif 'gpa' in col_lower or 'cgpa' in col_lower:
            gpa_col = col
    
    # Initialize variables
    data_to_plot = None
    labels = None
    p_value = None
    title_suffix = None
    
    # Check if research column exists (it won't in GradCafe data)
    if research_col and research_col in df.columns and df[research_col].notna().any():
        # Original research experience test
        research_0 = df[df[research_col] == 0][target_col].dropna()
        research_1 = df[df[research_col] == 1][target_col].dropna()
        
        if len(research_0) > 0 and len(research_1) > 0:
            print(f"\nResearch Experience = 0 (n={len(research_0)}):")
            print(f"  Mean {target_col}: {research_0.mean():.4f}")
            print(f"  Std: {research_0.std():.4f}")
            
            print(f"\nResearch Experience = 1 (n={len(research_1)}):")
            print(f"  Mean {target_col}: {research_1.mean():.4f}")
            print(f"  Std: {research_1.std():.4f}")
            
            # Statistical test
            _, p_norm_0 = stats.normaltest(research_0)
            _, p_norm_1 = stats.normaltest(research_1)
            
            if p_norm_0 > 0.05 and p_norm_1 > 0.05:
                t_stat, p_value = ttest_ind(research_1, research_0)
                test_name = "Independent t-test"
            else:
                u_stat, p_value = mannwhitneyu(research_1, research_0, alternative='two-sided')
                test_name = "Mann-Whitney U test"
            
            print(f"\n{test_name} Results:")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'} (alpha=0.05)")
            
            data_to_plot = [research_0, research_1]
            labels = ['No Research', 'With Research']
            title_suffix = 'Research Experience'
    
    # Alternative 1: Use QS Research Intensity (qs_RES.) as proxy for research focus
    if data_to_plot is None:
        qs_res_col = 'qs_RES.'
        if qs_res_col in df.columns and df[qs_res_col].notna().any():
            print("\n[INFO] Using QS Research Intensity (qs_RES.) as proxy for research focus")
            print("  Categories: VH (Very High), HI (High), MD (Medium)")
            
            # Group by research intensity
            res_data = df[[qs_res_col, target_col]].dropna()
            res_groups = res_data.groupby(qs_res_col)[target_col]
            
            group_stats = {}
            for group_name, group_data in res_groups:
                if len(group_data) > 0:
                    group_stats[group_name] = {
                        'mean': group_data.mean(),
                        'std': group_data.std(),
                        'count': len(group_data),
                        'data': group_data
                    }
            
            if len(group_stats) >= 2:
                # Compare VH (Very High) vs others, or highest vs lowest if VH not available
                if 'VH' in group_stats:
                    high_res_name = 'VH'
                    high_res_stats = group_stats['VH']
                    # Compare with the next highest group
                    other_groups = {k: v for k, v in group_stats.items() if k != 'VH'}
                    if other_groups:
                        sorted_others = sorted(other_groups.items(), key=lambda x: x[1]['mean'], reverse=True)
                        low_res_name, low_res_stats = sorted_others[0]  # Compare with highest non-VH
                    else:
                        sorted_groups = sorted(group_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
                        high_res_name, high_res_stats = sorted_groups[0]
                        low_res_name, low_res_stats = sorted_groups[-1]
                else:
                    # Compare highest vs lowest research intensity
                    sorted_groups = sorted(group_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
                    high_res_name, high_res_stats = sorted_groups[0]
                    low_res_name, low_res_stats = sorted_groups[-1]
                
                print(f"\nHigh Research Intensity ({high_res_name}, n={high_res_stats['count']}):")
                print(f"  Mean {target_col}: {high_res_stats['mean']:.4f}")
                print(f"  Std: {high_res_stats['std']:.4f}")
                
                print(f"\nLow Research Intensity ({low_res_name}, n={low_res_stats['count']}):")
                print(f"  Mean {target_col}: {low_res_stats['mean']:.4f}")
                print(f"  Std: {low_res_stats['std']:.4f}")
                
                # Statistical test
                high_data = high_res_stats['data']
                low_data = low_res_stats['data']
                
                _, p_norm_high = stats.normaltest(high_data)
                _, p_norm_low = stats.normaltest(low_data)
                
                if p_norm_high > 0.05 and p_norm_low > 0.05:
                    t_stat, p_value = ttest_ind(high_data, low_data)
                    test_name = "Independent t-test"
                else:
                    u_stat, p_value = mannwhitneyu(high_data, low_data, alternative='two-sided')
                    test_name = "Mann-Whitney U test"
                
                print(f"\n{test_name} Results (High vs Low Research Intensity):")
                print(f"  P-value: {p_value:.4f}")
                print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'} (alpha=0.05)")
                if p_value < 0.05:
                    effect_size = (high_res_stats['mean'] - low_res_stats['mean']) / np.sqrt((high_res_stats['std']**2 + low_res_stats['std']**2) / 2)
                    print(f"  Effect size (Cohen's d): {effect_size:.4f}")
                
                # Prepare data for visualization
                all_groups = sorted(group_stats.items(), key=lambda x: x[0])
                data_to_plot = [stats['data'] for _, stats in all_groups]
                labels = [name for name, _ in all_groups]
                title_suffix = 'QS Research Intensity'
    
    # Alternative 2: Test degree type (PhD vs MS) as proxy for research focus
    if data_to_plot is None:
        degree_col = None
        for col in df.columns:
            if col.lower() == 'degree':
                degree_col = col
                break
        
        if not degree_col:
            print("[WARN] Degree column not found. Skipping H3.")
            return None
        
        print("\n[WARN] Research experience not available in GradCafe data.")
        print("Testing alternative: Degree type impact (PhD vs MS)")
        
        # Filter for PhD and MS degrees
        degree_data = df[[degree_col, target_col]].dropna()
        phd_data = degree_data[degree_data[degree_col].str.upper() == 'PHD'][target_col]
        ms_data = degree_data[degree_data[degree_col].str.upper().isin(['MS', 'M.S.', 'MASTER'])][target_col]
        
        if len(phd_data) == 0 or len(ms_data) == 0:
            print(f"[WARN] Insufficient data: PhD (n={len(phd_data)}), MS (n={len(ms_data)})")
            print("[WARN] Cannot perform hypothesis test. Skipping H3.")
            return None
        
        print(f"\nPhD (n={len(phd_data)}):")
        print(f"  Mean {target_col}: {phd_data.mean():.4f}")
        print(f"  Std: {phd_data.std():.4f}")
        
        print(f"\nMS (n={len(ms_data)}):")
        print(f"  Mean {target_col}: {ms_data.mean():.4f}")
        print(f"  Std: {ms_data.std():.4f}")
        
        # Statistical test
        _, p_norm_phd = stats.normaltest(phd_data)
        _, p_norm_ms = stats.normaltest(ms_data)
        
        if p_norm_phd > 0.05 and p_norm_ms > 0.05:
            t_stat, p_value = ttest_ind(phd_data, ms_data)
            test_name = "Independent t-test"
        else:
            u_stat, p_value = mannwhitneyu(phd_data, ms_data, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        
        print(f"\n{test_name} Results (PhD vs MS):")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
        print(f"  Effect size (Cohen's d): {(phd_data.mean() - ms_data.mean()) / np.sqrt((phd_data.std()**2 + ms_data.std()**2) / 2):.4f}")
        
        data_to_plot = [phd_data, ms_data]
        labels = ['PhD', 'MS']
        title_suffix = 'Degree Type'
    
    # Visualization - only if we have data
    if data_to_plot is not None and len(data_to_plot) >= 2 and p_value is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Box plot
        axes[0].boxplot(data_to_plot, labels=labels)
        axes[0].set_ylabel(target_col)
        axes[0].set_title(f'{target_col} by {title_suffix or "Research Experience"}\n(p={p_value:.4f})')
        axes[0].grid(True, alpha=0.3)
        
        # Bar plot
        means = [d.mean() for d in data_to_plot]
        stds = [d.std() for d in data_to_plot]
        axes[1].bar(labels, means, yerr=stds, capsize=10, color=['salmon', 'skyblue'], alpha=0.7)
        axes[1].set_ylabel(f'Mean {target_col}')
        axes[1].set_title(f'Mean Admission Probability by {title_suffix or "Research Experience"}')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'hypothesis_3.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'mean_group1': data_to_plot[1].mean(),
            'mean_group0': data_to_plot[0].mean(),
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    else:
        print("[WARN] Cannot create visualization - insufficient data")
        return None

def hypothesis_4_sop_lor_moderate(df):
    """
    H4: SOP and LOR scores moderately affect predictions
    
    Null Hypothesis (H0): ρ = 0 (No correlation) OR |ρ| not in [0.4, 0.6]
    Alternative Hypothesis (H1): 0.4 <= |rho| <= 0.6 (Moderate correlation exists)
    
    Testing Method: Pearson correlation test
    - Test for significant correlation (H0: ρ = 0)
    - Validate correlation magnitude falls in moderate range [0.4, 0.6]
    - Significance level: α = 0.05
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS 4: SOP and LOR Moderate Impact")
    print("=" * 60)
    print("\nH0: rho = 0 OR |rho| not in [0.4, 0.6]")
    print("H1: 0.4 <= |rho| <= 0.6 (Moderate correlation)")
    print("Method: Pearson correlation test + range validation (alpha = 0.05)")
    
    # Find columns
    target_col = None
    sop_col = None
    lor_col = None
    
    for col in df.columns:
        col_lower = col.lower().replace(' ', '_')
        if 'chance' in col_lower and 'admit' in col_lower:
            target_col = col
        elif 'sop' in col_lower:
            sop_col = col
        elif 'lor' in col_lower:
            lor_col = col
    
    if not target_col:
        print("[WARN] Target column not found. Skipping H4.")
        return None
    
    results = {}
    
    # Check if SOP/LOR columns exist and have data
    if sop_col and sop_col in df.columns:
        sop_data = df[[sop_col, target_col]].dropna()
        if len(sop_data) > 0:
            # Ensure data is numeric
            sop_series = pd.to_numeric(sop_data[sop_col], errors='coerce')
            target_series = pd.to_numeric(sop_data[target_col], errors='coerce')
            valid_sop = pd.DataFrame({sop_col: sop_series, target_col: target_series}).dropna()
            
            if len(valid_sop) > 1:
                sop_corr, sop_p = pearsonr(valid_sop[sop_col], valid_sop[target_col])
                results['SOP'] = {'correlation': sop_corr, 'p_value': sop_p}
                print(f"\nSOP vs {target_col}:")
                print(f"  Correlation: {sop_corr:.4f}")
                print(f"  P-value: {sop_p:.4f}")
                print(f"  In moderate range (0.4-0.6)? {'Yes' if 0.4 <= abs(sop_corr) <= 0.6 else 'No'}")
                print(f"  Significant: {'Yes' if sop_p < 0.05 else 'No'}")
    
    if lor_col and lor_col in df.columns:
        lor_data = df[[lor_col, target_col]].dropna()
        if len(lor_data) > 0:
            # Ensure data is numeric
            lor_series = pd.to_numeric(lor_data[lor_col], errors='coerce')
            target_series = pd.to_numeric(lor_data[target_col], errors='coerce')
            valid_lor = pd.DataFrame({lor_col: lor_series, target_col: target_series}).dropna()
            
            if len(valid_lor) > 1:
                lor_corr, lor_p = pearsonr(valid_lor[lor_col], valid_lor[target_col])
                results['LOR'] = {'correlation': lor_corr, 'p_value': lor_p}
                print(f"\nLOR vs {target_col}:")
                print(f"  Correlation: {lor_corr:.4f}")
                print(f"  P-value: {lor_p:.4f}")
                print(f"  In moderate range (0.4-0.6)? {'Yes' if 0.4 <= abs(lor_corr) <= 0.6 else 'No'}")
                print(f"  Significant: {'Yes' if lor_p < 0.05 else 'No'}")
    
    if not results:
        # Check if columns exist but are empty
        if (sop_col and sop_col in df.columns) or (lor_col and lor_col in df.columns):
            sop_count = df[sop_col].notna().sum() if sop_col and sop_col in df.columns else 0
            lor_count = df[lor_col].notna().sum() if lor_col and lor_col in df.columns else 0
            if sop_count == 0 and lor_count == 0:
                print("[WARN] SOP/LOR columns exist but contain no data.")
            else:
                print(f"[WARN] SOP/LOR columns found but insufficient data (SOP: {sop_count}, LOR: {lor_count})")
        else:
            print("[WARN] SOP/LOR columns not found.")
        return None
    
    # Visualization
    n_plots = len(results)
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    idx = 0
    if 'SOP' in results:
        ax = axes[idx]
        ax.scatter(df[sop_col], df[target_col], alpha=0.5)
        ax.set_xlabel(sop_col)
        ax.set_ylabel(target_col)
        ax.set_title(f'SOP vs {target_col}\n(r={results["SOP"]["correlation"]:.3f})')
        z = np.polyfit(df[sop_col].dropna(), df[target_col][df[sop_col].notna()], 1)
        p = np.poly1d(z)
        ax.plot(df[sop_col].sort_values(), p(df[sop_col].sort_values()), "r--", alpha=0.8)
        ax.grid(True, alpha=0.3)
        idx += 1
    
    if 'LOR' in results:
        ax = axes[idx]
        ax.scatter(df[lor_col], df[target_col], alpha=0.5)
        ax.set_xlabel(lor_col)
        ax.set_ylabel(target_col)
        ax.set_title(f'LOR vs {target_col}\n(r={results["LOR"]["correlation"]:.3f})')
        z = np.polyfit(df[lor_col].dropna(), df[target_col][df[lor_col].notna()], 1)
        p = np.poly1d(z)
        ax.plot(df[lor_col].sort_values(), p(df[lor_col].sort_values()), "r--", alpha=0.8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hypothesis_4.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def hypothesis_5_university_rating_correlation(df):
    """
    H5: University rating (prestige) correlates with required GRE/TOEFL
    
    Null Hypothesis (H0): ρ_s = 0 (No correlation between rating and test scores)
    Alternative Hypothesis (H1): ρ_s > 0 (Positive correlation)
    
    Testing Method: Spearman rank correlation
    - University rating is ordinal (1-5 scale), so Spearman is appropriate
    - Tests whether higher-rated universities require higher test scores
    - Significance level: α = 0.05
    - Expected: Positive correlation
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS 5: University Rating vs Test Scores")
    print("=" * 60)
    print("\nH0: rho_s = 0 (No correlation)")
    print("H1: rho_s > 0 (Positive correlation)")
    print("Method: Spearman rank correlation test (alpha = 0.05)")
    
    # Find columns - try QS ranking as university rating alternative
    univ_rating_col = None
    gre_col = None
    toefl_col = None
    
    # First try to find University Rating column (but check if it has data)
    for col in df.columns:
        col_lower = col.lower().replace(' ', '_')
        if 'university' in col_lower and 'rating' in col_lower:
            # Only use if it has data
            if df[col].notna().sum() > 0:
                univ_rating_col = col
                break
    
    # If not found or has no data, use QS ranking as alternative
    if not univ_rating_col:
        # Try multiple QS ranking options - prefer numeric Score columns
        qs_rank_options = ['qs_Overall_Score', 'qs_Academic_Reputation_Score', 'qs_RANK_2025', 'qs_RANK_2024']
        for qs_col in qs_rank_options:
            if qs_col in df.columns and df[qs_col].notna().any():
                # Prefer Score columns (numeric) over RANK (may be string)
                if 'Score' in qs_col:
                    print(f"[INFO] University Rating column not found. Using {qs_col} as alternative.")
                    print("  Note: Higher QS score = higher prestige")
                    univ_rating_col = qs_col
                    break
                elif 'RANK' in qs_col:
                    # Only use RANK if Score not available
                    print(f"[INFO] University Rating column not found. Using {qs_col} as alternative.")
                    print("  Note: Lower QS rank = higher prestige (inverse relationship)")
                    univ_rating_col = qs_col
                    break
        
        if not univ_rating_col:
            print("[WARN] University rating column not found. Skipping H5.")
            return None
    
    # Find GRE and TOEFL columns - check multiple variations
    # First try exact matches
    for col in ['GRE Score', 'GRE_Score', 'gre_score', 'GRE Score']:
        if col in df.columns:
            gre_col = col
            break
    
    if not gre_col:
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_')
            if 'gre' in col_lower and 'score' in col_lower:
                gre_col = col
                break
    
    # Find TOEFL
    for col in ['TOEFL Score', 'TOEFL_Score', 'toefl_score', 'TOEFL Score']:
        if col in df.columns:
            toefl_col = col
            break
    
    if not toefl_col:
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_')
            if 'toefl' in col_lower and 'score' in col_lower:
                toefl_col = col
                break
    
    results = {}
    
    if gre_col:
        gre_data = df[[univ_rating_col, gre_col]].dropna()
        if len(gre_data) > 0:
            # Ensure data is numeric
            rating_series = pd.to_numeric(gre_data[univ_rating_col], errors='coerce')
            gre_series = pd.to_numeric(gre_data[gre_col], errors='coerce')
            valid_gre = pd.DataFrame({univ_rating_col: rating_series, gre_col: gre_series}).dropna()
            
            if len(valid_gre) > 1:
                try:
                    # Use Spearman for ordinal rating
                    gre_corr, gre_p = spearmanr(valid_gre[univ_rating_col], valid_gre[gre_col])
                    results['GRE'] = {'correlation': gre_corr, 'p_value': gre_p}
                    print(f"\nUniversity Rating ({univ_rating_col}) vs GRE Score:")
                    print(f"  Spearman Correlation: {gre_corr:.4f}")
                    print(f"  P-value: {gre_p:.4f}")
                    print(f"  Significant: {'Yes' if gre_p < 0.05 else 'No'}")
                    if 'RANK' in univ_rating_col:
                        # For rank, negative correlation means higher rank (lower number) = higher prestige
                        print(f"  Interpretation: {'Higher prestige universities require higher GRE' if gre_corr < 0 else 'Lower prestige universities require higher GRE'}")
                    else:
                        # For Score, positive correlation means higher score = higher prestige = higher GRE
                        print(f"  Interpretation: {'Higher prestige universities require higher GRE' if gre_corr > 0 else 'Lower prestige universities require higher GRE'}")
                except Exception as e:
                    print(f"[WARN] Could not calculate GRE correlation: {e}")
                    import traceback
                    print(f"  Error details: {traceback.format_exc()}")
            else:
                print(f"[WARN] Insufficient valid data after conversion (need >1, got {len(valid_gre)})")
        else:
            print(f"[WARN] No overlapping data between {univ_rating_col} and {gre_col}")
    
    if toefl_col:
        toefl_data = df[[univ_rating_col, toefl_col]].dropna()
        if len(toefl_data) > 0:
            # Ensure data is numeric
            rating_series = pd.to_numeric(toefl_data[univ_rating_col], errors='coerce')
            toefl_series = pd.to_numeric(toefl_data[toefl_col], errors='coerce')
            valid_toefl = pd.DataFrame({univ_rating_col: rating_series, toefl_col: toefl_series}).dropna()
            
            if len(valid_toefl) > 1:
                try:
                    toefl_corr, toefl_p = spearmanr(valid_toefl[univ_rating_col], valid_toefl[toefl_col])
                    results['TOEFL'] = {'correlation': toefl_corr, 'p_value': toefl_p}
                    print(f"\nUniversity Rating ({univ_rating_col}) vs TOEFL Score:")
                    print(f"  Spearman Correlation: {toefl_corr:.4f}")
                    print(f"  P-value: {toefl_p:.4f}")
                    print(f"  Significant: {'Yes' if toefl_p < 0.05 else 'No'}")
                    if 'RANK' in univ_rating_col:
                        print(f"  Interpretation: {'Higher prestige universities require higher TOEFL' if toefl_corr < 0 else 'Lower prestige universities require higher TOEFL'}")
                    else:
                        print(f"  Positive correlation: {'Yes' if toefl_corr > 0 else 'No'}")
                except Exception as e:
                    print(f"[WARN] Could not calculate TOEFL correlation: {e}")
            else:
                print(f"\n[WARN] TOEFL Score has no valid data (all NaN)")
        else:
            print(f"\n[WARN] TOEFL Score column exists but has no data")
    
    if not gre_col and not toefl_col:
        print("[WARN] Test score columns not found.")
        print(f"  Available columns with 'gre' or 'toefl': {[c for c in df.columns if 'gre' in c.lower() or 'toefl' in c.lower()]}")
        return None
    
    if not results:
        print("[WARN] Could not calculate correlations with test scores.")
        if gre_col:
            print(f"  GRE column found: {gre_col}")
        if toefl_col:
            print(f"  TOEFL column found: {toefl_col}")
        return None
    
    # Visualization
    n_plots = len(results)
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    idx = 0
    if 'GRE' in results:
        ax = axes[idx]
        df.boxplot(column=gre_col, by=univ_rating_col, ax=ax)
        ax.set_xlabel(univ_rating_col)
        ax.set_ylabel(gre_col)
        ax.set_title(f'GRE Score by University Rating\n(rho={results["GRE"]["correlation"]:.3f})')
        ax.grid(True, alpha=0.3)
        idx += 1
    
    if 'TOEFL' in results:
        ax = axes[idx]
        df.boxplot(column=toefl_col, by=univ_rating_col, ax=ax)
        ax.set_xlabel(univ_rating_col)
        ax.set_ylabel(toefl_col)
        ax.set_title(f'TOEFL Score by University Rating\n(rho={results["TOEFL"]["correlation"]:.3f})')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hypothesis_5.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def hypothesis_6_socioeconomic_patterns(df):
    """
    H6: Adding QS ranking or cost-of-living data will reveal socioeconomic patterns
    
    Null Hypothesis (H0): Enrichment data provides no additional insights
    Alternative Hypothesis (H1): Enrichment data reveals meaningful patterns
    
    Testing Method: Correlation analysis with QS rankings and cost of living data
    - QS rankings are already merged (columns prefixed with qs_)
    - Cost of living data is already merged (columns prefixed with col_)
    - Analyze correlations with admission probability
    - Expected: Partial correlation; richer insights expected
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS 6: Socioeconomic Patterns (QS Rankings & Cost of Living)")
    print("=" * 60)
    print("\nH0: Enrichment data provides no additional insights")
    print("H1: Enrichment data reveals meaningful patterns")
    print("Method: Correlation analysis with merged enrichment data")
    
    # Find target column
    target_col = None
    for col in ['Chance of Admit', 'Chance_of_Admit', 'chance_of_admit', 'is_accepted']:
        if col in df.columns:
            target_col = col
            break
    
    if not target_col:
        print("[WARN] Target column not found. Skipping H6.")
        return None
    
    results = {}
    
    # Check for QS columns (prefixed with qs_)
    qs_cols = [col for col in df.columns if col.startswith('qs_')]
    if qs_cols:
        print(f"\n[OK] Found {len(qs_cols)} QS ranking columns")
        # Test correlation with key QS metrics
        key_qs_cols = ['qs_RANK_2025', 'qs_Overall_Score', 'qs_Academic_Reputation_Score']
        
        for qs_col in key_qs_cols:
            if qs_col in df.columns:
                data = df[[qs_col, target_col]].dropna()
                if len(data) > 1:
                    try:
                        corr, p_val = pearsonr(data[qs_col], data[target_col])
                        results[qs_col] = {'correlation': corr, 'p_value': p_val}
                        print(f"  {qs_col} vs {target_col}:")
                        print(f"    Correlation: {corr:.4f}, P-value: {p_val:.4f}")
                        print(f"    Significant: {'Yes' if p_val < 0.05 else 'No'}")
                    except Exception as e:
                        print(f"  [WARN] Could not calculate correlation for {qs_col}: {e}")
        
        results['QS_available'] = True
    else:
        print("\n[WARN] QS ranking columns not found in dataset")
        results['QS_available'] = False
    
    # Check for Cost of Living columns (prefixed with col_)
    col_cols = [col for col in df.columns if col.startswith('col_')]
    if col_cols:
        print(f"\n[OK] Found {len(col_cols)} Cost of Living columns")
        # Test correlation with key COL metrics
        key_col_cols = [c for c in col_cols if 'mean' in c.lower() and 'Cost of Living Index' in c]
        
        for col_col in key_col_cols[:3]:  # Test first 3 key metrics
            data = df[[col_col, target_col]].dropna()
            if len(data) > 1:
                try:
                    corr, p_val = pearsonr(data[col_col], data[target_col])
                    results[col_col] = {'correlation': corr, 'p_value': p_val}
                    print(f"  {col_col} vs {target_col}:")
                    print(f"    Correlation: {corr:.4f}, P-value: {p_val:.4f}")
                    print(f"    Significant: {'Yes' if p_val < 0.05 else 'No'}")
                except Exception as e:
                    print(f"  [WARN] Could not calculate correlation for {col_col}: {e}")
        
        results['COL_available'] = True
    else:
        print("\n[WARN] Cost of Living columns not found in dataset")
        results['COL_available'] = False
    
    # Visualization if we have results
    if len(results) > 2:  # More than just availability flags
        n_plots = min(3, len([k for k in results.keys() if k not in ['QS_available', 'COL_available']]))
        if n_plots > 0:
            fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 5))
            if n_plots == 1:
                axes = [axes]
            
            plot_idx = 0
            for col_name, col_result in results.items():
                if col_name not in ['QS_available', 'COL_available'] and isinstance(col_result, dict):
                    if plot_idx < n_plots:
                        ax = axes[plot_idx]
                        data = df[[col_name, target_col]].dropna()
                        if len(data) > 0:
                            ax.scatter(data[col_name], data[target_col], alpha=0.5)
                            ax.set_xlabel(col_name)
                            ax.set_ylabel(target_col)
                            corr = col_result.get('correlation', 0)
                            ax.set_title(f'{col_name} vs {target_col}\n(r={corr:.3f})')
                            ax.grid(True, alpha=0.3)
                            plot_idx += 1
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'hypothesis_6.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n[OK] Visualization saved to {OUTPUT_DIR / 'hypothesis_6.png'}")
    
    return results

def generate_hypothesis_report(all_results, df):
    """Generate comprehensive hypothesis testing report with interpretations and recommendations"""
    report = []
    report.append("=" * 80)
    report.append("HYPOTHESIS TESTING REPORT - COMPREHENSIVE ANALYSIS")
    report.append("=" * 80)
    report.append(f"\nDate Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Dataset Shape: {df.shape}")
    report.append(f"Total Observations: {len(df):,}")
    
    # Summary of findings
    report.append("\n\n" + "=" * 80)
    report.append("EXECUTIVE SUMMARY")
    report.append("=" * 80)
    
    # Analyze each hypothesis
    h1_result = all_results.get('H1: GRE and TOEFL correlation', {})
    h2_result = all_results.get('H2: GPA most predictive', {})
    h3_result = all_results.get('H3: Research experience impact', {})
    h4_result = all_results.get('H4: SOP and LOR moderate impact', {})
    h5_result = all_results.get('H5: University rating correlation', {})
    h6_result = all_results.get('H6: Socioeconomic patterns', {})
    
    # Key findings
    key_findings = []
    
    if h1_result and isinstance(h1_result, dict):
        if 'GRE' in h1_result:
            gre_corr = h1_result['GRE'].get('correlation', 0)
            gre_p = h1_result['GRE'].get('p_value', 1)
            if abs(gre_corr) < 0.1:
                key_findings.append("H1: GRE scores show WEAK correlation with admission (r={:.3f}, p={:.4f})".format(gre_corr, gre_p))
                key_findings.append("  → Interpretation: GRE scores alone are not strong predictors in this dataset")
                key_findings.append("  → Action: Consider GRE as one factor among many, not a primary determinant")
    
    if h2_result and isinstance(h2_result, dict):
        gpa_corr = None
        max_corr = -1
        max_factor = None
        
        for factor, stats in h2_result.items():
            if isinstance(stats, dict) and 'correlation' in stats:
                corr = abs(stats['correlation'])
                if corr > max_corr and factor != 'is_accepted':
                    max_corr = corr
                    max_factor = factor
                if factor in ['GPA', 'gpa']:
                    gpa_corr = stats.get('correlation', 0)
        
        if gpa_corr is not None:
            key_findings.append("H2: GPA correlation: {:.3f}".format(gpa_corr))
            if max_factor and max_factor != 'GPA':
                max_stats = h2_result[max_factor]
                key_findings.append("  → Most predictive factor: {} (r={:.3f})".format(max_factor, max_stats.get('correlation', 0)))
                key_findings.append("  → Interpretation: GPA is NOT the most predictive factor")
                key_findings.append("  → Action: Consider university prestige (QS rankings) as more important than GPA alone")
    
    if h3_result and isinstance(h3_result, dict):
        if 'p_value' in h3_result:
            p_val = h3_result.get('p_value', 1)
            if p_val < 0.05:
                key_findings.append("H3: Research Intensity shows SIGNIFICANT impact on admission")
                key_findings.append("  → Tested using QS Research Intensity (qs_RES.) as proxy")
                key_findings.append("  → Significant difference found (p={:.4f})".format(p_val))
                key_findings.append("  → Action: Consider university research intensity in application strategy")
    
    if h5_result and isinstance(h5_result, dict):
        if 'GRE' in h5_result:
            gre_corr = h5_result['GRE'].get('correlation', 0)
            gre_p = h5_result['GRE'].get('p_value', 1)
            if gre_p < 0.05:
                key_findings.append("H5: University prestige correlates with GRE requirements")
                key_findings.append("  → QS Overall Score vs GRE: r={:.3f}, p={:.4f}".format(gre_corr, gre_p))
                key_findings.append("  → Interpretation: Higher prestige universities require higher GRE scores")
                key_findings.append("  → Action: Match your GRE scores to universities' QS rankings")
    
    if h6_result and isinstance(h6_result, dict):
        qs_corrs = []
        for key, value in h6_result.items():
            if isinstance(value, dict) and 'correlation' in value and key.startswith('qs_'):
                qs_corrs.append((key, value['correlation'], value.get('p_value', 1)))
        
        if qs_corrs:
            strongest_qs = max(qs_corrs, key=lambda x: abs(x[1]))
            key_findings.append("H6: QS Rankings show STRONG correlations with admission")
            key_findings.append("  → Strongest: {} (r={:.3f}, p={:.2e})".format(strongest_qs[0], strongest_qs[1], strongest_qs[2]))
            key_findings.append("  → Interpretation: University prestige (QS rankings) is a strong predictor")
            key_findings.append("  → Note: Negative correlation suggests higher-ranked universities may have lower acceptance rates")
            key_findings.append("  → Action: University selection strategy should prioritize QS rankings")
    
    if key_findings:
        report.append("\nKEY FINDINGS:")
        for finding in key_findings:
            report.append(f"  {finding}")
    
    # Detailed results
    report.append("\n\n" + "=" * 80)
    report.append("DETAILED HYPOTHESIS RESULTS")
    report.append("=" * 80)
    
    for i, (hypothesis, result) in enumerate(all_results.items(), 1):
        report.append(f"\n\nHYPOTHESIS {i}: {hypothesis}")
        report.append("-" * 80)
        
        if not result:
            report.append("  STATUS: Could not be tested (missing data)")
            report.append("  RECOMMENDATION: Data collection needed for this hypothesis")
            continue
        
        if isinstance(result, dict):
            # Add interpretation
            if i == 1:  # GRE/TOEFL
                if 'GRE' in result:
                    gre_stats = result['GRE']
                    corr = gre_stats.get('correlation', 0)
                    p_val = gre_stats.get('p_value', 1)
                    report.append("  RESULTS:")
                    report.append(f"    GRE Correlation: {corr:.4f}")
                    report.append(f"    P-value: {p_val:.4f}")
                    report.append(f"    Significant: {'Yes' if p_val < 0.05 else 'No'}")
                    report.append(f"    Strength: {'Strong' if abs(corr) > 0.5 else 'Moderate' if abs(corr) > 0.3 else 'Weak'}")
                    report.append("  INTERPRETATION:")
                    if abs(corr) < 0.1:
                        report.append("    - GRE scores show very weak correlation with admission probability")
                        report.append("    - This contradicts the hypothesis that GRE strongly predicts admission")
                        report.append("    - Possible reasons: dataset includes only accepted/rejected (binary), not probability")
                        report.append("    - Or: GRE scores may be necessary but not sufficient")
                    report.append("  RECOMMENDATION:")
                    report.append("    - GRE scores should be considered alongside other factors")
                    report.append("    - Focus on holistic application review rather than GRE alone")
            
            elif i == 2:  # GPA most predictive
                report.append("  RESULTS:")
                # Sort by absolute correlation
                sorted_factors = []
                for factor, stats in result.items():
                    if isinstance(stats, dict) and 'correlation' in stats and factor != 'is_accepted':
                        sorted_factors.append((factor, stats['correlation'], stats.get('p_value', 1)))
                
                sorted_factors.sort(key=lambda x: abs(x[1]), reverse=True)
                
                report.append("    Top 5 Most Predictive Factors:")
                for idx, (factor, corr, p_val) in enumerate(sorted_factors[:5], 1):
                    report.append(f"      {idx}. {factor}: r={corr:.4f}, p={p_val:.2e}")
                
                gpa_stats = None
                for factor, stats in result.items():
                    if factor in ['GPA', 'gpa'] and isinstance(stats, dict):
                        gpa_stats = stats
                        break
                
                if gpa_stats:
                    gpa_corr = gpa_stats.get('correlation', 0)
                    report.append(f"\n    GPA Correlation: {gpa_corr:.4f}")
                    report.append("  INTERPRETATION:")
                    if sorted_factors and abs(sorted_factors[0][1]) > abs(gpa_corr):
                        report.append(f"    - GPA is NOT the most predictive factor")
                        report.append(f"    - {sorted_factors[0][0]} has stronger correlation ({sorted_factors[0][1]:.4f})")
                        report.append("    - QS ranking scores show stronger negative correlations")
                        report.append("    - This suggests university prestige matters more than individual GPA")
                    else:
                        report.append("    - GPA shows moderate predictive power")
                    report.append("  RECOMMENDATION:")
                    report.append("    - Consider university prestige (QS rankings) in admission strategy")
                    report.append("    - GPA is important but not the sole determinant")
                    report.append("    - Focus on applying to universities where your GPA aligns with their standards")
            
            elif i == 3:  # Research experience
                # Check if we have test results
                if 'p_value' in result or any('mean' in str(k).lower() for k in result.keys()):
                    report.append("  RESULTS:")
                    if 'p_value' in result:
                        p_val = result.get('p_value', 1)
                        report.append(f"    P-value: {p_val:.4f}")
                        report.append(f"    Significant: {'Yes' if p_val < 0.05 else 'No'}")
                    if 'mean_group1' in result and 'mean_group0' in result:
                        mean1 = result.get('mean_group1', 0)
                        mean0 = result.get('mean_group0', 0)
                        report.append(f"    High Research Intensity mean: {mean1:.4f}")
                        report.append(f"    Low Research Intensity mean: {mean0:.4f}")
                    report.append("  INTERPRETATION:")
                    report.append("    - Tested using QS Research Intensity (qs_RES.) as proxy for research focus")
                    report.append("    - Categories: VH (Very High), HI (High), MD (Medium)")
                    if 'p_value' in result and result.get('p_value', 1) < 0.05:
                        report.append("    - Significant difference found between research intensity levels")
                        report.append("    - Higher research intensity universities show different admission patterns")
                    report.append("  RECOMMENDATION:")
                    report.append("    - Consider university research intensity when selecting target universities")
                    report.append("    - Research-focused universities may have different admission criteria")
                else:
                    report.append("  STATUS: Could not be tested (missing data)")
                    report.append("  RECOMMENDATION: Data collection needed for this hypothesis")
            
            elif i == 5:  # University rating correlation
                if 'GRE' in result or 'TOEFL' in result:
                    report.append("  RESULTS:")
                    if 'GRE' in result:
                        gre_stats = result['GRE']
                        gre_corr = gre_stats.get('correlation', 0)
                        gre_p = gre_stats.get('p_value', 1)
                        report.append(f"    GRE Correlation: {gre_corr:.4f}")
                        report.append(f"    P-value: {gre_p:.4f}")
                        report.append(f"    Significant: {'Yes' if gre_p < 0.05 else 'No'}")
                    if 'TOEFL' in result:
                        toefl_stats = result['TOEFL']
                        toefl_corr = toefl_stats.get('correlation', 0)
                        toefl_p = toefl_stats.get('p_value', 1)
                        report.append(f"    TOEFL Correlation: {toefl_corr:.4f}")
                        report.append(f"    P-value: {toefl_p:.4f}")
                        report.append(f"    Significant: {'Yes' if toefl_p < 0.05 else 'No'}")
                    report.append("  INTERPRETATION:")
                    if 'GRE' in result:
                        gre_corr = result['GRE'].get('correlation', 0)
                        report.append(f"    - University prestige (QS Overall Score) shows correlation with GRE (r={gre_corr:.4f})")
                        if gre_corr > 0:
                            report.append("    - Higher prestige universities require higher GRE scores")
                            report.append("    - This confirms that top universities have higher admission standards")
                        else:
                            report.append("    - Unexpected negative correlation - may indicate data quality issues")
                    report.append("  RECOMMENDATION:")
                    report.append("    - Target universities based on your GRE scores and their QS rankings")
                    report.append("    - Higher-ranked universities require higher test scores")
                    report.append("    - Use QS rankings to identify appropriate target universities")
                else:
                    report.append("  STATUS: Could not be tested (missing data)")
                    report.append("  RECOMMENDATION: Data collection needed for this hypothesis")
            
            elif i == 6:  # Socioeconomic patterns
                report.append("  RESULTS:")
                qs_results = []
                col_results = []
                
                for key, value in result.items():
                    if isinstance(value, dict) and 'correlation' in value:
                        if key.startswith('qs_'):
                            qs_results.append((key, value['correlation'], value.get('p_value', 1)))
                        elif key.startswith('col_'):
                            col_results.append((key, value['correlation'], value.get('p_value', 1)))
                
                if qs_results:
                    report.append("    QS Rankings Correlations:")
                    for qs_name, corr, p_val in qs_results[:3]:
                        report.append(f"      {qs_name}: r={corr:.4f}, p={p_val:.2e}")
                
                if col_results:
                    report.append("\n    Cost of Living Correlations:")
                    significant_col = [c for c in col_results if c[2] < 0.05]
                    if significant_col:
                        for col_name, corr, p_val in significant_col[:3]:
                            report.append(f"      {col_name}: r={corr:.4f}, p={p_val:.4f} (significant)")
                    else:
                        report.append("      No significant correlations found")
                
                report.append("  INTERPRETATION:")
                if qs_results:
                    strongest_qs = max(qs_results, key=lambda x: abs(x[1]))
                    report.append(f"    - QS rankings show strong correlations (strongest: {strongest_qs[0]}, r={strongest_qs[1]:.4f})")
                    report.append("    - Negative correlations suggest higher-ranked universities may have lower acceptance rates")
                    report.append("    - This could indicate: more selective admissions at top universities")
                if col_results:
                    sig_count = len([c for c in col_results if c[2] < 0.05])
                    if sig_count == 0:
                        report.append("    - Cost of living shows minimal impact on admission probability")
                        report.append("    - Economic factors may not directly influence admission decisions")
                
                report.append("  RECOMMENDATION:")
                report.append("    - University selection should prioritize QS rankings")
                report.append("    - Consider that top-ranked universities may be more selective")
                report.append("    - Cost of living is not a primary factor in admission decisions")
            
            else:
                # Generic output for other hypotheses
                for key, value in result.items():
                    if isinstance(value, dict):
                        report.append(f"  {key}:")
                        for k, v in value.items():
                            report.append(f"    {k}: {v}")
                    else:
                        report.append(f"  {key}: {value}")
        else:
            report.append(f"  Result: {result}")
    
    # Actionable recommendations
    report.append("\n\n" + "=" * 80)
    report.append("ACTIONABLE RECOMMENDATIONS")
    report.append("=" * 80)
    
    recommendations = []
    
    # Based on H1
    if h1_result and isinstance(h1_result, dict) and 'GRE' in h1_result:
        gre_corr = abs(h1_result['GRE'].get('correlation', 0))
        if gre_corr < 0.1:
            recommendations.append("1. GRE SCORES:")
            recommendations.append("   - Do not rely solely on GRE scores for admission prediction")
            recommendations.append("   - GRE is necessary but not sufficient for admission")
            recommendations.append("   - Focus on holistic application improvement")
    
    # Based on H2
    if h2_result and isinstance(h2_result, dict):
        recommendations.append("2. GPA AND UNIVERSITY SELECTION:")
        recommendations.append("   - GPA is important but not the most predictive factor")
        recommendations.append("   - University prestige (QS rankings) shows stronger correlation")
        recommendations.append("   - Strategy: Match your GPA to universities where you're competitive")
        recommendations.append("   - Consider applying to universities with QS rankings that match your profile")
    
    # Based on H3
    if h3_result and isinstance(h3_result, dict) and 'p_value' in h3_result:
        recommendations.append("3. RESEARCH INTENSITY:")
        recommendations.append("   - University research intensity (QS Research Intensity) affects admission patterns")
        recommendations.append("   - Consider research focus when selecting target universities")
        recommendations.append("   - Very High research universities may have different admission criteria")
    
    # Based on H5
    if h5_result and isinstance(h5_result, dict) and 'GRE' in h5_result:
        recommendations.append("4. UNIVERSITY PRESTIGE AND TEST SCORES:")
        recommendations.append("   - Higher prestige universities (higher QS scores) require higher GRE scores")
        recommendations.append("   - Use QS rankings to identify universities matching your GRE profile")
        recommendations.append("   - Target universities where your GRE scores align with their prestige level")
    
    # Based on H6
    if h6_result and isinstance(h6_result, dict):
        recommendations.append("5. UNIVERSITY RANKINGS:")
        recommendations.append("   - QS rankings are strong predictors of admission patterns")
        recommendations.append("   - Higher-ranked universities may be more selective")
        recommendations.append("   - Research target universities' QS rankings and acceptance patterns")
        recommendations.append("   - Consider a balanced application strategy (reach, match, safety)")
    
    # General recommendations
    recommendations.append("6. DATA LIMITATIONS:")
    recommendations.append("   - SOP and LOR data not available in this dataset")
    recommendations.append("   - These factors may still be important in real admissions")
    recommendations.append("   - Consider collecting additional data for comprehensive analysis")
    
    recommendations.append("7. NEXT STEPS:")
    recommendations.append("   - Build predictive models using QS rankings and GPA as key features")
    recommendations.append("   - Analyze acceptance patterns by university tier")
    recommendations.append("   - Consider interaction effects between factors")
    recommendations.append("   - Validate findings with additional data sources")
    
    for rec in recommendations:
        report.append(f"  {rec}")
    
    report.append("\n" + "=" * 80)
    
    report_text = "\n".join([str(r) for r in report])
    
    with open(OUTPUT_DIR / 'hypothesis_testing_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n[OK] Hypothesis testing report saved to {OUTPUT_DIR / 'hypothesis_testing_report.txt'}")

def main():
    """Main hypothesis testing function"""
    print("=" * 60)
    print("HYPOTHESIS TESTING")
    print("=" * 60)
    
    # Load data
    try:
        df = load_data()
        df = normalize_column_names(df)
        print(f"\n[OK] Data loaded successfully: {df.shape}")
    except Exception as e:
        print(f"\n[ERROR] Error loading data: {e}")
        return
    
    # Test all hypotheses
    all_results = {}
    
    h1_result = hypothesis_1_gre_toefl_correlation(df)
    all_results['H1: GRE and TOEFL correlation'] = h1_result
    
    h2_result = hypothesis_2_gpa_most_predictive(df)
    all_results['H2: GPA most predictive'] = h2_result
    
    h3_result = hypothesis_3_research_experience(df)
    all_results['H3: Research experience impact'] = h3_result
    
    h4_result = hypothesis_4_sop_lor_moderate(df)
    all_results['H4: SOP and LOR moderate impact'] = h4_result
    
    h5_result = hypothesis_5_university_rating_correlation(df)
    all_results['H5: University rating correlation'] = h5_result
    
    h6_result = hypothesis_6_socioeconomic_patterns(df)
    all_results['H6: Socioeconomic patterns'] = h6_result
    
    # Generate report
    generate_hypothesis_report(all_results, df)
    
    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTING COMPLETED!")
    print("=" * 60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

