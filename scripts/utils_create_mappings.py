"""
Create JSON file for university name mappings
"""

import pandas as pd
import json
from pathlib import Path
from rapidfuzz import fuzz

def normalize_for_comparison(s):
    """Normalize string for comparison - removes 'university' and common words"""
    if pd.isna(s):
        return ""
    import re
    s = str(s).lower()
    s = re.sub(r'^the\s+', '', s)  # Remove leading "the"
    # Remove common university-related words
    s = re.sub(r'\b(university|univ|college|school|institute|institution)\b', '', s)
    s = re.sub(r'\s+', ' ', s).strip()  # Normalize whitespace
    s = re.sub(r'[^\w\s-]', '', s)  # Keep hyphens, remove other punctuation
    s = re.sub(r'\s+', ' ', s).strip()  # Normalize whitespace again after removing words
    return s

def create_mappings_json():
    """Create JSON file from mappings"""
    print("=" * 80)
    print("CREATING UNIVERSITY MAPPINGS JSON")
    print("=" * 80)
    
    # Load mappings CSV
    print("\nLoading mappings...")
    possible_mapping_paths = [
        Path('data/mappings/university_mappings_from_qs.csv'),
        Path('university_mappings_from_qs.csv')  # Fallback
    ]
    
    mapping_df = None
    for path in possible_mapping_paths:
        if path.exists():
            mapping_df = pd.read_csv(path)
            print(f"  [OK] Loaded {len(mapping_df)} mappings from {path}")
            break
    
    if mapping_df is None:
        print("  [ERROR] university_mappings_from_qs.csv not found")
        print("  Please run scripts/utils_map_universities.py first.")
        return
    
    # Create JSON structure
    print("\nCreating JSON structure...")
    
    # Format 1: Simple dictionary (original -> canonical)
    simple_mappings = {}
    for _, row in mapping_df.iterrows():
        original = str(row['original_name']).strip()
        canonical = str(row['qs_canonical_name']).strip()
        simple_mappings[original] = canonical
    
    # Format 2: Detailed with metadata
    detailed_mappings = []
    for _, row in mapping_df.iterrows():
        original = str(row['original_name']).strip()
        canonical = str(row['qs_canonical_name']).strip()
        similarity = float(row['similarity_score'])
        
        # Calculate similarity again for consistency
        norm1 = normalize_for_comparison(original)
        norm2 = normalize_for_comparison(canonical)
        sim = max(fuzz.ratio(norm1, norm2), fuzz.token_sort_ratio(norm1, norm2))
        
        detailed_mappings.append({
            "original_name": original,
            "canonical_name": canonical,
            "similarity_score": round(sim, 2),
            "confidence": "high" if sim >= 90 else "medium" if sim >= 85 else "low"
        })
    
    # Sort by canonical name, then by similarity
    detailed_mappings.sort(key=lambda x: (x['canonical_name'], -x['similarity_score']))
    
    # Create grouped format (by canonical name)
    grouped_mappings = {}
    for item in detailed_mappings:
        canonical = item['canonical_name']
        if canonical not in grouped_mappings:
            grouped_mappings[canonical] = []
        grouped_mappings[canonical].append({
            "original_name": item['original_name'],
            "similarity_score": item['similarity_score'],
            "confidence": item['confidence']
        })
    
    mappings_dir = Path('data/mappings')
    mappings_dir.mkdir(parents=True, exist_ok=True)
    
    # Save simple JSON
    simple_file = mappings_dir / 'university_mappings_simple.json'
    with open(simple_file, 'w', encoding='utf-8') as f:
        json.dump(simple_mappings, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved simple mappings to: {simple_file}")
    print(f"       Format: {{'original_name': 'canonical_name'}}")
    
    # Save detailed JSON
    detailed_file = mappings_dir / 'university_mappings_detailed.json'
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_mappings, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved detailed mappings to: {detailed_file}")
    print(f"       Format: Array of objects with original_name, canonical_name, similarity_score, confidence")
    
    # Save grouped JSON
    grouped_file = mappings_dir / 'university_mappings_grouped.json'
    with open(grouped_file, 'w', encoding='utf-8') as f:
        json.dump(grouped_mappings, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved grouped mappings to: {grouped_file}")
    print(f"       Format: {{'canonical_name': [variants...]}}")
    
    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total mappings: {len(simple_mappings)}")
    print(f"Unique canonical names: {len(grouped_mappings)}")
    
    high_confidence = sum(1 for item in detailed_mappings if item['similarity_score'] >= 90)
    medium_confidence = sum(1 for item in detailed_mappings if 85 <= item['similarity_score'] < 90)
    low_confidence = sum(1 for item in detailed_mappings if item['similarity_score'] < 85)
    
    print(f"\nConfidence levels:")
    print(f"  High (>=90%): {high_confidence}")
    print(f"  Medium (85-90%): {medium_confidence}")
    print(f"  Low (<85%): {low_confidence}")
    
    # Show Wisconsin-Madison example
    print("\n" + "=" * 80)
    print("WISCONSIN-MADISON EXAMPLE")
    print("=" * 80)
    wisconsin_mappings = [item for item in detailed_mappings 
                         if 'wisconsin' in item['original_name'].lower() 
                         and 'madison' in item['original_name'].lower()]
    if wisconsin_mappings:
        print(f"\nFound {len(wisconsin_mappings)} Wisconsin-Madison mappings:")
        for item in sorted(wisconsin_mappings, key=lambda x: -x['similarity_score']):
            print(f"  {item['original_name']}")
            print(f"    -> {item['canonical_name']} ({item['similarity_score']}%, {item['confidence']})")
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nFiles created:")
    print(f"  1. {simple_file} - Simple dictionary format (easiest to edit)")
    print(f"  2. {detailed_file} - Detailed format with scores")
    print(f"  3. {grouped_file} - Grouped by canonical name")
    print(f"\nYou can edit any of these JSON files to remove incorrect mappings.")

if __name__ == "__main__":
    create_mappings_json()

