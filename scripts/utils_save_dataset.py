"""Save final combined dataset with all columns"""

import pandas as pd
from pathlib import Path

# Try multiple possible locations
possible_inputs = [
    Path('data/processed/merged_data_combined.csv'),
    Path('merged_data_combined.csv')  # Fallback
]

df = None
for path in possible_inputs:
    if path.exists():
        df = pd.read_csv(path, low_memory=False)
        break

if df is None:
    raise FileNotFoundError("Could not find merged_data_combined.csv")

output_file = Path('data/processed/final_combined_dataset.csv')

df.to_csv(output_file, index=False)

print(f"Saved final dataset with ALL columns to: {output_file}")
print(f"  Rows: {len(df):,}")
print(f"  Columns: {len(df.columns)}")
print(f"  File size: {Path(output_file).stat().st_size / (1024*1024):.2f} MB")

print(f"\nColumn summary:")
qs_cols = [c for c in df.columns if c.startswith('qs_')]
col_cols = [c for c in df.columns if c.startswith('col_')]
print(f"  - QS Rankings columns: {len(qs_cols)}")
print(f"  - Cost of Living columns: {len(col_cols)}")
print(f"  - Total columns: {len(df.columns)}")

print(f"\nSample QS columns: {qs_cols[:5]}")
print(f"Sample COL columns: {col_cols[:5]}")


