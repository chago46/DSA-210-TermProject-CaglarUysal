"""
Map university names from dataset to QS Rankings canonical names
Only maps high similarity matches (like "Wisconsin Madison" vs "Wisconsin-Madison")
"""

import pandas as pd
from rapidfuzz import fuzz, process
import re

def normalize_for_comparison(s):
    """Normalize string for comparison - removes 'university' and common words"""
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r'^the\s+', '', s)  # Remove leading "the"
    # Remove common university-related words
    s = re.sub(r'\b(university|univ|college|school|institute|institution)\b', '', s)
    s = re.sub(r'\s+', ' ', s).strip()  # Normalize whitespace
    s = re.sub(r'[^\w\s-]', '', s)  # Keep hyphens, remove other punctuation
    s = re.sub(r'\s+', ' ', s).strip()  # Normalize whitespace again after removing words
    return s

def find_best_qs_match(uni_name, qs_universities, threshold=85):
    """Find best matching QS university for a given university name"""
    if pd.isna(uni_name) or not uni_name:
        return None, 0
    
    norm_uni = normalize_for_comparison(uni_name)
    
    best_match = None
    best_score = 0
    
    for qs_uni in qs_universities:
        if pd.isna(qs_uni) or not qs_uni:
            continue
        
        norm_qs = normalize_for_comparison(qs_uni)
        
        # Try multiple similarity metrics
        ratio = fuzz.ratio(norm_uni, norm_qs)
        partial_ratio = fuzz.partial_ratio(norm_uni, norm_qs)
        token_sort_ratio = fuzz.token_sort_ratio(norm_uni, norm_qs)
        token_set_ratio = fuzz.token_set_ratio(norm_uni, norm_qs)
        
        max_score = max(ratio, partial_ratio, token_sort_ratio, token_set_ratio)
        
        # Also check if one contains the other (for abbreviations)
        if norm_uni in norm_qs or norm_qs in norm_uni:
            if len(norm_uni) > 5 and len(norm_qs) > 5:
                max_score = max(max_score, 90)
        
        if max_score > best_score:
            best_score = max_score
            best_match = qs_uni
    
    if best_score >= threshold:
        return best_match, best_score
    else:
        return None, best_score

def create_qs_based_mappings():
    """Create university mappings using QS Rankings as canonical source"""
    print("=" * 80)
    print("UNIVERSITY NAME MAPPING FROM QS RANKINGS")
    print("=" * 80)
    
    # Load QS Rankings
    print("\nLoading QS Rankings...")
    try:
        qs_df = pd.read_csv('data/raw/QS World University Rankings 2025 (Top global universities).csv', 
                           low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            qs_df = pd.read_csv('data/raw/QS World University Rankings 2025 (Top global universities).csv', 
                               low_memory=False, encoding='latin-1')
        except:
            qs_df = pd.read_csv('data/raw/QS World University Rankings 2025 (Top global universities).csv', 
                               low_memory=False, encoding='cp1252')
    
    # Get QS university names
    if 'Institution_Name' in qs_df.columns:
        qs_universities = qs_df['Institution_Name'].dropna().unique()
    else:
        print("[ERROR] Could not find Institution_Name column in QS rankings")
        print(f"Available columns: {list(qs_df.columns)}")
        return
    
    print(f"  [OK] Loaded {len(qs_universities):,} universities from QS Rankings")
    
    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv('merged_data_combined.csv', low_memory=False)
    
    if 'uni_name' in df.columns:
        dataset_universities = df['uni_name'].dropna().unique()
    else:
        print("[ERROR] Could not find uni_name column")
        return
    
    print(f"  [OK] Found {len(dataset_universities):,} unique universities in dataset")
    
    # Find mappings with different thresholds
    print("\n" + "=" * 80)
    print("FINDING MAPPINGS")
    print("=" * 80)
    
    thresholds = [95, 90, 85, 80, 70]
    all_mappings = {}
    
    for threshold in thresholds:
        print(f"\nAnalyzing with {threshold}% similarity threshold...")
        mappings = {}
        scores = {}
        
        for uni in dataset_universities:
            if uni in all_mappings:
                continue  # Already mapped at higher threshold
            
            best_match, score = find_best_qs_match(uni, qs_universities, threshold=threshold)
            if best_match:
                mappings[uni] = best_match
                scores[uni] = score
        
        if len(mappings) > 0:
            print(f"  Found {len(mappings)} mappings at {threshold}% threshold")
            all_mappings.update(mappings)
        else:
            print(f"  No new mappings found at {threshold}% threshold")
    
    print(f"\nTotal mappings found: {len(all_mappings)}")
    
    # Show Wisconsin-Madison example
    print("\n" + "=" * 80)
    print("WISCONSIN-MADISON EXAMPLE")
    print("=" * 80)
    wisconsin_variants = [u for u in all_mappings.keys() if 'wisconsin' in str(u).lower() and 'madison' in str(u).lower()]
    if wisconsin_variants:
        print(f"\nFound {len(wisconsin_variants)} Wisconsin-Madison variants:")
        for variant in sorted(wisconsin_variants):
            canonical = all_mappings[variant]
            norm1 = normalize_for_comparison(variant)
            norm2 = normalize_for_comparison(canonical)
            sim = max(fuzz.ratio(norm1, norm2), fuzz.token_sort_ratio(norm1, norm2))
            try:
                print(f"  {variant}")
                print(f"    -> {canonical} (similarity: {sim:.1f}%)")
            except UnicodeEncodeError:
                print(f"  [Variant mapped]")
                print(f"    -> [QS Canonical] (similarity: {sim:.1f}%)")
    else:
        # Check if there are any Wisconsin-Madison in dataset
        wisconsin_in_dataset = [u for u in dataset_universities if 'wisconsin' in str(u).lower() and 'madison' in str(u).lower()]
        if wisconsin_in_dataset:
            print(f"\nFound {len(wisconsin_in_dataset)} Wisconsin-Madison variants in dataset:")
            for variant in sorted(wisconsin_in_dataset[:5]):
                best_match, score = find_best_qs_match(variant, qs_universities, threshold=0)
                try:
                    print(f"  {variant}")
                    if best_match:
                        print(f"    -> {best_match} (similarity: {score:.1f}%)")
                    else:
                        print(f"    -> No match found (best score: {score:.1f}%)")
                except UnicodeEncodeError:
                    print(f"  [Variant]")
                    if best_match:
                        print(f"    -> [Match found] (similarity: {score:.1f}%)")
                    else:
                        print(f"    -> No match found (best score: {score:.1f}%)")
    
    # Save mappings
    print("\n" + "=" * 80)
    print("SAVING MAPPINGS")
    print("=" * 80)
    
    # Create mapping dataframe
    mapping_list = []
    for original, canonical in sorted(all_mappings.items()):
        norm1 = normalize_for_comparison(original)
        norm2 = normalize_for_comparison(canonical)
        sim = max(fuzz.ratio(norm1, norm2), fuzz.token_sort_ratio(norm1, norm2))
        # Only include mappings with similarity >= 73%
        if sim >= 73:
            mapping_list.append({
                'original_name': original,
                'qs_canonical_name': canonical,
                'similarity_score': sim
            })
    
    mapping_df = pd.DataFrame(mapping_list)
    mapping_df = mapping_df.sort_values(['qs_canonical_name', 'similarity_score'], ascending=[True, False])
    
    # Filter to only include mappings with similarity >= 73%
    mapping_df_filtered = mapping_df[mapping_df['similarity_score'] >= 73].copy()
    
    # Save CSV
    csv_file = 'university_mappings_from_qs.csv'
    mapping_df_filtered.to_csv(csv_file, index=False)
    print(f"\n[OK] Saved {len(mapping_df_filtered)} mappings (similarity >= 73%) to: {csv_file}")
    print(f"     Filtered out {len(mapping_df) - len(mapping_df_filtered)} mappings below 73% threshold")
    
    # Save text file
    txt_file = 'university_mappings_from_qs.txt'
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("UNIVERSITY NAME MAPPINGS FROM QS RANKINGS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total mappings: {len(all_mappings)}\n")
        f.write(f"QS Universities used as canonical source: {len(qs_universities)}\n\n")
        f.write("=" * 80 + "\n")
        f.write("MAPPINGS (Original -> QS Canonical)\n")
        f.write("=" * 80 + "\n\n")
        
        # Group by QS canonical name (only include similarity >= 73%)
        by_canonical = {}
        for original, canonical in all_mappings.items():
            norm1 = normalize_for_comparison(original)
            norm2 = normalize_for_comparison(canonical)
            sim = max(fuzz.ratio(norm1, norm2), fuzz.token_sort_ratio(norm1, norm2))
            # Only include mappings with similarity >= 73%
            if sim >= 73:
                if canonical not in by_canonical:
                    by_canonical[canonical] = []
                by_canonical[canonical].append((original, sim))
        
        for canonical in sorted(by_canonical.keys()):
            variants = sorted(by_canonical[canonical], key=lambda x: x[1], reverse=True)
            if len(variants) > 0:
                f.write(f"\n{'='*80}\n")
                f.write(f"QS CANONICAL: {canonical}\n")
                f.write(f"{'='*80}\n")
                for original, sim in variants:
                    f.write(f"  -> {original} (similarity: {sim:.1f}%)\n")
                f.write("\n")
    
    print(f"[OK] Saved detailed mappings to: {txt_file}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  - Total mappings found: {len(all_mappings)}")
    print(f"  - Mappings with similarity >= 73%: {len(mapping_df_filtered)}")
    print(f"  - Mappings filtered out (<73%): {len(all_mappings) - len(mapping_df_filtered)}")
    print(f"  - Unique QS canonical names used: {len(set(mapping_df_filtered['qs_canonical_name']))}")
    print(f"  - Coverage: {len(mapping_df_filtered) / len(dataset_universities) * 100:.1f}% of dataset universities")
    
    # Show sample mappings
    print(f"\nSample mappings (first 10):")
    for i, (original, canonical) in enumerate(list(all_mappings.items())[:10], 1):
        norm1 = normalize_for_comparison(original)
        norm2 = normalize_for_comparison(canonical)
        sim = max(fuzz.ratio(norm1, norm2), fuzz.token_sort_ratio(norm1, norm2))
        try:
            print(f"  {i}. {original}")
            print(f"     -> {canonical} ({sim:.1f}%)")
        except UnicodeEncodeError:
            print(f"  {i}. [Original name]")
            print(f"     -> [QS Canonical] ({sim:.1f}%)")
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nFiles created:")
    print(f"  - {csv_file} (CSV format)")
    print(f"  - {txt_file} (Human-readable format)")

if __name__ == "__main__":
    create_qs_based_mappings()

