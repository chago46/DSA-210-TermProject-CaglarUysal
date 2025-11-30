"""
New Data Collection Script with Updated Combining Strategy

Strategy:
1. Read all.csv
2. Map university names using QS mappings
3. Remove unwanted columns (status, dates, timestamps, decision_method, comments, degree)
4. Merge with QS Rankings using university names
5. Merge with Cost of Living using QS country data
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Create data directories
DATA_DIR = Path('data')
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'

for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def load_all_csv():
    """Load all.csv file"""
    print("\n[INFO] Loading all.csv...")
    
    # GradCafe CSV column names (file doesn't have headers)
    column_names = [
        'rowid', 'uni_name', 'major', 'degree', 'season', 'decision', 
        'decision_method', 'decision_date', 'decision_timestamp', 'ugrad_gpa',
        'gre_verbal', 'gre_quant', 'gre_writing', 'is_new_gre', 'gre_subject',
        'status', 'post_data', 'post_timestamp', 'comments'
    ]
    
    all_csv = RAW_DIR / 'all.csv'
    if not all_csv.exists():
        print(f"  [ERROR] Could not find {all_csv}")
        return None
    
    try:
        df = pd.read_csv(all_csv, names=column_names, low_memory=False)
        print(f"  [OK] Loaded {len(df):,} rows from all.csv")
        return df
    except Exception as e:
        print(f"  [ERROR] Error reading all.csv: {e}")
        return None

def apply_university_mappings(df):
    """Apply university name mappings from QS Rankings"""
    print("\n[INFO] Applying university name mappings...")
    
    mapping_file = Path('university_mappings_from_qs.csv')
    if not mapping_file.exists():
        print(f"  [WARN] Mapping file not found: {mapping_file}")
        print(f"  [WARN] Continuing without mapping")
        return df, None
    
    try:
        mappings_df = pd.read_csv(mapping_file)
        # Create mapping dictionaries (case-insensitive)
        mappings_exact = dict(zip(mappings_df['original_name'], mappings_df['qs_canonical_name']))
        mappings_lower = dict(zip(mappings_df['original_name'].str.lower().str.strip(), mappings_df['qs_canonical_name']))
        
        # Also include canonical names themselves
        canonical_set = set(mappings_df['qs_canonical_name'].str.strip())
        for canonical in canonical_set:
            canonical_lower = str(canonical).lower().strip()
            if canonical_lower not in mappings_lower:
                mappings_lower[canonical_lower] = canonical
        
        def map_university(uni_name):
            if pd.isna(uni_name):
                return None
            uni_str = str(uni_name).strip()
            # Try exact match first
            if uni_str in mappings_exact:
                return mappings_exact[uni_str]
            # Try case-insensitive match
            uni_lower = uni_str.lower().strip()
            if uni_lower in mappings_lower:
                return mappings_lower[uni_lower]
            return None
        
        original_count = len(df)
        df['uni_name_mapped'] = df['uni_name'].apply(map_university)
        
        # Filter to only keep rows with mappings
        has_mapping = df['uni_name_mapped'].notna()
        mapped_count = has_mapping.sum()
        unmapped_count = (~has_mapping).sum()
        
        if unmapped_count > 0:
            print(f"  [OK] Mapped {mapped_count:,} universities ({mapped_count/original_count*100:.1f}%)")
            print(f"  [OK] Removing {unmapped_count:,} rows without mappings ({unmapped_count/original_count*100:.1f}%)")
            df = df[has_mapping].copy()
            # Replace original uni_name with mapped canonical name
            df['uni_name'] = df['uni_name_mapped']
            df = df.drop(columns=['uni_name_mapped'])
        else:
            print(f"  [OK] All {mapped_count:,} universities have mappings")
            df = df.drop(columns=['uni_name_mapped'])
        
        return df, mappings_df
    except Exception as e:
        print(f"  [ERROR] Error applying mappings: {e}")
        return df, None

def remove_unwanted_columns(df):
    """Remove unwanted columns"""
    print("\n[INFO] Removing unwanted columns...")
    
    columns_to_remove = ['status', 'decision_date', 'decision_timestamp', 
                         'post_data', 'post_timestamp', 'decision_method', 
                         'comments', 'degree']
    
    removed = []
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
            removed.append(col)
    
    if removed:
        print(f"  [OK] Removed {len(removed)} columns: {', '.join(removed)}")
    else:
        print(f"  [OK] No unwanted columns found to remove")
    
    return df

def merge_with_qs_rankings(df, qs_df):
    """Merge with QS Rankings using university names - include ALL QS columns"""
    print("\n[INFO] Merging with QS Rankings...")
    
    if qs_df is None:
        print("  [WARN] QS Rankings not available")
        return df
    
    # Find columns in QS data
    qs_uni_col = None
    for col in qs_df.columns:
        if 'institution' in col.lower() or ('university' in col.lower() and 'name' in col.lower()):
            qs_uni_col = col
            break
    
    if not qs_uni_col:
        print("  [ERROR] Could not find university name column in QS data")
        return df
    
    # Prepare for merge - include ALL QS columns
    qs_df_clean = qs_df.copy()
    qs_df_clean['uni_name_normalized'] = qs_df_clean[qs_uni_col].astype(str).str.strip()
    df['uni_name_normalized'] = df['uni_name'].astype(str).str.strip()
    
    # Merge with ALL QS columns (except the normalized key)
    qs_cols_to_merge = [col for col in qs_df_clean.columns if col != 'uni_name_normalized']
    
    # Rename QS columns to avoid conflicts (prefix with qs_)
    qs_df_for_merge = qs_df_clean[['uni_name_normalized'] + qs_cols_to_merge].copy()
    rename_dict = {col: f'qs_{col}' if col != qs_uni_col else 'qs_institution_name' 
                   for col in qs_cols_to_merge}
    qs_df_for_merge = qs_df_for_merge.rename(columns=rename_dict)
    
    # Merge
    merged_df = df.merge(
        qs_df_for_merge,
        on='uni_name_normalized',
        how='left'
    )
    
    # Drop temporary normalized column
    merged_df = merged_df.drop(columns=['uni_name_normalized'])
    
    # Check match rate using rank column
    rank_cols = [col for col in merged_df.columns if 'rank' in col.lower() and '2025' in col.lower()]
    if rank_cols:
        match_rate = (merged_df[rank_cols[0]].notna().sum() / len(merged_df)) * 100
        print(f"  [OK] Merged with QS Rankings: {match_rate:.1f}% matched")
        print(f"  [OK] Added {len(qs_cols_to_merge)} QS columns")
    
    return merged_df

def merge_with_cost_of_living(df, col_df):
    """Merge with Cost of Living using country data from QS - include ALL COL columns"""
    print("\n[INFO] Merging with Cost of Living...")
    
    if col_df is None:
        print("  [WARN] Cost of Living data not available")
        return df
    
    # Find country column (could be from QS merge - might be qs_Location)
    country_col = None
    # Check for various possible names
    possible_names = ['country', 'qs_Location', 'Location', 'qs_location', 'location']
    for name in possible_names:
        if name in df.columns:
            country_col = name
            break
    
    if not country_col:
        print("  [WARN] No country/location column found - cannot merge with Cost of Living")
        print(f"  Available columns: {[c for c in df.columns if 'location' in c.lower() or 'country' in c.lower()]}")
        return df
    
    # Find city column in COL data
    city_col = None
    for col in col_df.columns:
        if 'city' in col.lower():
            city_col = col
            break
    
    if not city_col:
        print("  [WARN] Could not find city column in Cost of Living data")
        return df
    
    # Extract country from city names (e.g., "Zurich, Switzerland" -> "Switzerland")
    col_df_clean = col_df.copy()
    
    def extract_country(city_str):
        if pd.isna(city_str):
            return None
        city_str = str(city_str)
        # Check if city name contains country (format: "City, Country")
        if ',' in city_str:
            parts = city_str.split(',')
            if len(parts) > 1:
                return parts[-1].strip()
        return None
    
    col_df_clean['country_from_city'] = col_df_clean[city_col].apply(extract_country)
    
    # Aggregate COL data by country (take mean/median for numeric columns)
    # Prepare COL data for merge - include ALL COL columns
    col_cols_to_merge = [col for col in col_df_clean.columns if col != 'country_from_city' and col != city_col]
    
    # Aggregate by country
    agg_dict = {}
    for col in col_cols_to_merge:
        if pd.api.types.is_numeric_dtype(col_df_clean[col]):
            agg_dict[col] = ['mean', 'median', 'std']
        else:
            agg_dict[col] = 'first'  # Take first value for non-numeric
    
    country_col_stats = col_df_clean.groupby('country_from_city').agg(agg_dict).reset_index()
    
    # Flatten column names
    country_col_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                  for col in country_col_stats.columns.values]
    country_col_stats = country_col_stats.rename(columns={'country_from_city': country_col})
    
    # Rename columns to prefix with col_
    rename_dict = {col: f'col_{col}' for col in country_col_stats.columns if col != country_col}
    country_col_stats = country_col_stats.rename(columns=rename_dict)
    
    # Merge with dataset using country column
    df = df.merge(country_col_stats, on=country_col, how='left')
    
    matched_count = df[list(rename_dict.values())[0]].notna().sum() if rename_dict else 0
    if matched_count > 0:
        print(f"  [OK] Merged Cost of Living by country: {matched_count:,} rows matched ({matched_count/len(df)*100:.1f}%)")
        print(f"  [OK] Added {len(col_cols_to_merge)} Cost of Living columns")
    
    return df

def main():
    """Main data collection function with new strategy"""
    print("=" * 80)
    print("DATA COLLECTION - NEW COMBINING STRATEGY")
    print("=" * 80)
    
    # Step 1: Load all.csv
    df = load_all_csv()
    if df is None:
        print("\n[ERROR] Could not load all.csv")
        return None
    
    # Step 2: Apply university name mappings
    df, mappings_df = apply_university_mappings(df)
    if df is None or len(df) == 0:
        print("\n[ERROR] No data after applying mappings")
        return None
    
    # Step 3: Remove unwanted columns
    df = remove_unwanted_columns(df)
    
    # Step 4: Load QS Rankings
    print("\n[INFO] Loading QS Rankings...")
    qs_df = None
    qs_files = list(RAW_DIR.glob('*QS*.csv')) + list(RAW_DIR.glob('*ranking*.csv'))
    if qs_files:
        try:
            qs_file = qs_files[0]
            # Try different encodings
            try:
                qs_df = pd.read_csv(qs_file, low_memory=False, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    qs_df = pd.read_csv(qs_file, low_memory=False, encoding='latin-1')
                except:
                    qs_df = pd.read_csv(qs_file, low_memory=False, encoding='cp1252')
            print(f"  [OK] Loaded QS Rankings from {qs_file.name}")
            print(f"       Shape: {qs_df.shape}")
        except Exception as e:
            print(f"  [WARN] Could not load QS Rankings: {e}")
    
    # Step 5: Load Cost of Living
    print("\n[INFO] Loading Cost of Living data...")
    col_df = None
    col_files = list(RAW_DIR.glob('*cost*.csv')) + list(RAW_DIR.glob('*living*.csv'))
    if col_files:
        try:
            col_file = col_files[0]
            col_df = pd.read_csv(col_file, low_memory=False)
            print(f"  [OK] Loaded Cost of Living from {col_file.name}")
            print(f"       Shape: {col_df.shape}")
        except Exception as e:
            print(f"  [WARN] Could not load Cost of Living: {e}")
    
    # Step 6: Merge with QS Rankings
    df = merge_with_qs_rankings(df, qs_df)
    
    # Step 7: Merge with Cost of Living
    df = merge_with_cost_of_living(df, col_df)
    
    # Step 8: Save merged dataset
    output_file = PROCESSED_DIR / 'admissions_merged.csv'
    df.to_csv(output_file, index=False)
    print(f"\n[OK] Merged dataset saved to {output_file}")
    print(f"     Final shape: {df.shape}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Final dataset: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Unique universities: {df['uni_name'].nunique():,}")
    if 'qs_rank' in df.columns:
        qs_coverage = df['qs_rank'].notna().sum() / len(df) * 100
        print(f"QS Rankings coverage: {qs_coverage:.1f}%")
    if 'cost_of_living_index_mean' in df.columns:
        col_coverage = df['cost_of_living_index_mean'].notna().sum() / len(df) * 100
        print(f"Cost of Living coverage: {col_coverage:.1f}%")
    
    return df

if __name__ == "__main__":
    merged_data = main()

