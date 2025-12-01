"""
Data Adapter for GradCafe Dataset

This module adapts GradCafe data structure to work with existing analysis scripts.
GradCafe data has different column names and structure than the original Kaggle dataset.
"""

import pandas as pd
import numpy as np

def adapt_gradcafe_data(df):
    """
    Adapt GradCafe data to match expected column names and structure
    
    GradCafe columns (from all_uisc_clean.csv):
    - rowid, uni_name, major, degree, season, decision, decision_method, 
      decision_date, decision_timestamp, ugrad_gpa, gre_verbal, gre_quant, 
      gre_writing, is_new_gre, gre_subject, status, post_data, post_timestamp, comments
    
    Expected columns (from original Kaggle dataset):
    - University Rating, GRE Score, TOEFL Score, SOP, LOR, GPA, Research, Chance of Admit
    """
    df = df.copy()
    
    # Check if columns need to be mapped (CSV might not have headers)
    # Only remap if columns are numeric (0, 1, 2...) or Unnamed, not if they already have proper names
    needs_remapping = (
        (df.columns[0] == 0 or isinstance(df.columns[0], (int, float))) or 
        ('Unnamed' in str(df.columns[0])) or
        (all(isinstance(c, (int, float)) for c in df.columns[:5]))
    )
    
    if needs_remapping and 'uni_name' not in df.columns:
        print("[INFO] Detected CSV without proper headers - mapping columns...")
        # Map based on GradCafe schema
        expected_cols = [
            'rowid', 'uni_name', 'major', 'degree', 'season', 'decision', 
            'decision_method', 'decision_date', 'decision_timestamp', 'ugrad_gpa',
            'gre_verbal', 'gre_quant', 'gre_writing', 'is_new_gre', 'gre_subject',
            'status', 'post_data', 'post_timestamp', 'comments'
        ]
        
        # If we have the right number of columns, rename them
        if len(df.columns) >= len(expected_cols):
            df.columns = expected_cols[:len(df.columns)] + [f'extra_{i}' for i in range(len(df.columns) - len(expected_cols))]
            print(f"  [OK] Mapped {len(expected_cols)} columns")
    
    # Check if this is GradCafe data
    is_gradcafe = 'uni_name' in df.columns or 'decision' in df.columns or any('uni' in str(c).lower() for c in df.columns)
    
    if not is_gradcafe:
        # Already in expected format (Kaggle dataset)
        return df
    
    print("[INFO] Adapting GradCafe data structure...")
    
    # Remove unwanted columns
    columns_to_remove = ['status', 'decision_date', 'decision_timestamp', 
                         'post_data', 'post_timestamp', 'decision_method', 'comments']
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"  [OK] Removed column: {col}")
    
    # 1. Convert decision to binary target (Accepted/Wait listed = 1, Others = 0)
    if 'decision' in df.columns:
        # Map: Accepted and Wait listed = 1 (accepted), all others = 0 (rejected)
        df['is_accepted'] = df['decision'].apply(
            lambda x: 1 if x in ['Accepted', 'Wait listed'] else 0
        )
        
        # Create Chance of Admit column (same as is_accepted for binary classification)
        df['Chance of Admit'] = df['is_accepted'].astype(float)
        
        # Remove original decision column
        df = df.drop(columns=['decision'])
        
        accepted_count = df['is_accepted'].sum()
        rejected_count = len(df) - accepted_count
        print(f"  [OK] Converted decision: {accepted_count} accepted, {rejected_count} rejected")
    
    # 2. Combine GRE scores (create total or average)
    # According to GradCafe README: is_new_gre=1 means 130-170 scale, is_new_gre=0 means 200-800 scale
    if 'gre_verbal' in df.columns and 'gre_quant' in df.columns:
        # Convert is_new_gre to numeric if it's not already
        if 'is_new_gre' in df.columns:
            df['is_new_gre'] = pd.to_numeric(df['is_new_gre'], errors='coerce')
        else:
            # If is_new_gre doesn't exist, infer from score ranges
            df['is_new_gre'] = np.nan
        
        # Normalize GRE scores based on is_new_gre field (per row)
        gre_v = pd.to_numeric(df['gre_verbal'], errors='coerce')
        gre_q = pd.to_numeric(df['gre_quant'], errors='coerce')
        
        # Initialize normalized scores
        gre_v_normalized = gre_v.copy()
        gre_q_normalized = gre_q.copy()
        
        # For rows where is_new_gre is 1 (or inferred), convert 130-170 to 200-800 scale
        # Formula: (score - 130) / 40 * 600 + 200
        if df['is_new_gre'].notna().any():
            new_gre_mask = df['is_new_gre'] == 1
            gre_v_normalized[new_gre_mask] = (gre_v[new_gre_mask] - 130) / 40 * 600 + 200
            gre_q_normalized[new_gre_mask] = (gre_q[new_gre_mask] - 130) / 40 * 600 + 200
        else:
            # If is_new_gre is missing, infer from score values
            # Scores <= 170 are likely new GRE, > 170 are old GRE
            new_gre_inferred = (gre_v <= 170) & (gre_q <= 170) & gre_v.notna() & gre_q.notna()
            gre_v_normalized[new_gre_inferred] = (gre_v[new_gre_inferred] - 130) / 40 * 600 + 200
            gre_q_normalized[new_gre_inferred] = (gre_q[new_gre_inferred] - 130) / 40 * 600 + 200
        
        # Create combined GRE score (average of verbal and quant)
        df['GRE Score'] = (gre_v_normalized + gre_q_normalized) / 2
        
        # Ensure GRE Score is numeric
        df['GRE Score'] = pd.to_numeric(df['GRE Score'], errors='coerce')
        
        # Keep individual scores too (ensure numeric)
        df['GRE Verbal'] = gre_v
        df['GRE Quant'] = gre_q
        
        valid_gre = df['GRE Score'].notna().sum()
        print(f"  [OK] Created combined GRE Score from verbal and quant ({valid_gre} valid scores)")
        
        # Drop rows where GRE Score is missing
        before_drop = len(df)
        df = df[df['GRE Score'].notna()].copy()
        dropped_count = before_drop - len(df)
        if dropped_count > 0:
            print(f"  [OK] Dropped {dropped_count:,} rows with missing GRE scores ({dropped_count/before_drop*100:.1f}%)")
    
    # 3. Map GPA and convert 10.0 scale to 4.0 scale
    if 'ugrad_gpa' in df.columns:
        # Ensure GPA is numeric
        df['GPA'] = pd.to_numeric(df['ugrad_gpa'], errors='coerce')
        
        # Convert 10.0 scale to 4.0 scale (GPAs > 4.0 are likely on 10.0 scale)
        # Formula: 4.0_scale = 10.0_scale / 2.5
        gpa_10_scale = df['GPA'] > 4.0
        if gpa_10_scale.any():
            converted_count = gpa_10_scale.sum()
            df.loc[gpa_10_scale, 'GPA'] = df.loc[gpa_10_scale, 'GPA'] / 2.5
            print(f"  [OK] Converted {converted_count} GPAs from 10.0 scale to 4.0 scale")
        
        valid_gpa = df['GPA'].notna().sum()
        print(f"  [OK] Mapped ugrad_gpa to GPA ({valid_gpa} valid values)")
    
    # 4. Create University Rating from QS rank (if available)
    if 'qs_rank' in df.columns:
        # Convert QS rank to rating (1-5 scale)
        qs_rank_numeric = pd.to_numeric(df['qs_rank'], errors='coerce')
        df['University Rating'] = pd.cut(
            qs_rank_numeric,
            bins=[0, 100, 200, 300, 500, float('inf')],
            labels=[5, 4, 3, 2, 1],
            include_lowest=True
        )
        # Convert to numeric (handles NaN properly)
        df['University Rating'] = pd.to_numeric(df['University Rating'], errors='coerce')
        valid_ratings = df['University Rating'].notna().sum()
        print(f"  [OK] Created University Rating from QS rank ({valid_ratings} valid ratings)")
    elif 'uni_name' in df.columns:
        # If we have university names but no QS rank, create placeholder
        # This will be filled when QS data is merged
        df['University Rating'] = np.nan
        print(f"  [WARN] University Rating not available (will be created after QS merge)")
    
    # 5. Handle missing columns (set to NaN or default values)
    # These columns don't exist in GradCafe data
    if 'TOEFL Score' not in df.columns:
        df['TOEFL Score'] = np.nan
        print(f"  [WARN] TOEFL Score not available in GradCafe data")
    
    if 'SOP' not in df.columns:
        df['SOP'] = np.nan
        print(f"  [WARN] SOP not available in GradCafe data")
    
    if 'LOR' not in df.columns:
        df['LOR'] = np.nan
        print(f"  [WARN] LOR not available in GradCafe data")
    
    if 'Research' not in df.columns:
        df['Research'] = np.nan
        print(f"  [WARN] Research Experience not available in GradCafe data")
    
    # 6. Remove degree column
    if 'degree' in df.columns:
        df = df.drop(columns=['degree'])
        print(f"  [OK] Removed column: degree")
    
    # 7. Keep original GradCafe columns for reference
    # They're already there, just add some aliases
    if 'uni_name' in df.columns:
        df['University Name'] = df['uni_name']
    
    if 'major' in df.columns:
        df['Major'] = df['major']
    
    print(f"  [OK] Data adaptation complete. Shape: {df.shape}")
    
    return df

def get_data_info(df):
    """Get information about the dataset structure"""
    info = {
        'is_gradcafe': 'uni_name' in df.columns or 'decision' in df.columns,
        'has_target': 'Chance of Admit' in df.columns or 'decision' in df.columns,
        'has_gre': 'GRE Score' in df.columns or ('gre_verbal' in df.columns and 'gre_quant' in df.columns),
        'has_gpa': 'GPA' in df.columns or 'ugrad_gpa' in df.columns,
        'has_toefl': 'TOEFL Score' in df.columns and df['TOEFL Score'].notna().any(),
        'has_sop_lor': 'SOP' in df.columns and df['SOP'].notna().any(),
        'has_research': 'Research' in df.columns and df['Research'].notna().any(),
        'has_university_rating': 'University Rating' in df.columns and df['University Rating'].notna().any(),
    }
    return info

