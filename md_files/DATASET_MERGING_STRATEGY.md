# Dataset Merging Strategy

## Data Source

We use **GradCafe Admission Data** from [GitHub (deedy/gradcafe_data)](https://github.com/deedy/gradcafe_data), which includes:
- University names (uni_name) - cleaned, with 2,708 distinct universities
- Real admission decisions from GradCafe forum
- Comprehensive applicant information (GRE scores, GPA, etc.)

## Advantage: Direct University Name Matching

GradCafe data **includes university names**, allowing direct matching with QS rankings using fuzzy string matching to handle name variations.

## Implementation Overview

The merging process is implemented in `data_collection_new.py` and consists of three main steps:

1. **University Name Mapping** - Uses rapidfuzz for fuzzy matching
2. **QS Rankings Integration** - Direct merge on canonical names
3. **Cost of Living Integration** - Country-level mapping via QS Location data

### Step 1: University Name Mapping with RapidFuzz

**Library Used:** `rapidfuzz` (fast, MIT-licensed alternative to fuzzywuzzy)

**Process:**

1. **Pre-processing (Name Normalization):**
   - Convert to lowercase
   - Strip whitespace
   - Remove common words ("university", "univ", "college", "school", "institute")
   - Remove punctuation (except hyphens)
   - Normalize whitespace

2. **Similarity Calculation:**
   For each GradCafe university name, calculate similarity with all QS universities using multiple metrics:
   - `fuzz.ratio()` - Standard Levenshtein-based similarity
   - `fuzz.partial_ratio()` - Best substring match
   - `fuzz.token_sort_ratio()` - Token-based, order-independent
   - `fuzz.token_set_ratio()` - Set-based token matching
   - Take the maximum of all metrics

3. **Progressive Threshold Strategy:**
   - Try thresholds in order: 95%, 90%, 85%, 80%, 70%
   - Stop at first match found (prevents lower-quality matches)

4. **Quality Control:**
   - Final filter: Only include mappings with similarity ≥ 73%
   - This balances match rate with accuracy
   - Prevents false matches from being included

5. **Output:**
   - Creates `university_mappings_from_qs.csv` with columns:
     - `original_name`: GradCafe university name
     - `qs_canonical_name`: QS Rankings canonical name
     - `similarity_score`: Similarity percentage (73-100%)

**Example Mappings:**
- "Wisconsin Madison" → "University of Wisconsin-Madison" (similarity: ~85%)
- "MIT" → "Massachusetts Institute of Technology" (similarity: ~90%)
- "UC Berkeley" → "University of California, Berkeley" (similarity: ~88%)

**Implementation:** The mapping file is created by `map_universities_from_qs.py` and applied during data collection in `data_collection_new.py`.

**Match Rate:** ~49.4% of records successfully mapped (134,314 out of 271,807)

### Step 2: QS Rankings Integration

**Approach:** Direct merge using mapped canonical university names

1. **Mapping Application:**
   - Load `university_mappings_from_qs.csv` mapping file
   - Apply mappings to standardize university names
   - Exact match first, then case-insensitive match
   - Filter to keep only records with successful mappings

2. **Direct Merge:**
   - Merge QS Rankings data on normalized canonical university names
   - Include all QS ranking columns (28 columns: rank, scores, reputation metrics, etc.)
   - **Match rate:** 100% for mapped universities

### Step 3: Cost of Living Integration

**Approach:** Country-level mapping via QS Location data

1. **Country Extraction:**
   - Use QS Location column to get country information
   - Extract country from location strings (e.g., "Zurich, Switzerland" → "Switzerland")

2. **Cost of Living Merge:**
   - Aggregate Cost of Living data by country (mean, median, std)
   - Merge with dataset using country column
   - Include 21 cost of living columns (indices with aggregated statistics)
   - **Match rate:** 100% for mapped universities

## Current Implementation Results

The implementation in `data_collection_new.py` produces:

1. **University Name Mapping:**
   - Uses `rapidfuzz` library for fuzzy string matching
   - Creates mapping file (`university_mappings_from_qs.csv`) with similarity scores
   - Handles university name variations automatically
   - Match rate: ~49.4% of records successfully mapped

2. **QS Rankings Integration:**
   - Direct merge using mapped canonical university names
   - 100% match rate for mapped universities
   - Includes all QS ranking columns (28 columns)

3. **Cost of Living Integration:**
   - Country-level mapping via QS Location data
   - Aggregated statistics (mean, median, std) by country
   - 100% match rate for mapped universities
   - Includes 21 cost of living columns

4. **Data Processing:**
   - Removes unwanted columns (status, timestamps, comments, etc.)
   - Filters to only keep records with successful university mappings
   - **Final dataset:** 134,314 rows × 60 columns

## Research Question 3 Clarification

**Original Question:** "How does university rating (or QS rank) relate to GPA or GRE requirements?"

**Available Data (GradCafe):**
- ✅ University names (uni_name) - matched to QS rankings
- ✅ QS rankings (directly merged)
- ✅ Applicant GPA (ugrad_gpa) and GRE scores (gre_verbal, gre_quant)
- ✅ Admission decisions (Accepted/Rejected)

**What We Can Analyze:**
- ✅ Average GPA/GRE scores by QS ranking tier
- ✅ Correlation between QS rank and applicant test scores
- ✅ Whether higher-ranked universities attract higher-scoring applicants
- ✅ Acceptance rates by QS ranking tier
- ✅ Score distributions for accepted vs rejected by university rank

**What We Cannot Analyze (without additional data):**
- ❌ Official university admission requirements (not in dataset)
- ❌ Minimum GPA/GRE thresholds per university

**Solution:** We analyze the **relationship between QS ranking and applicant characteristics** using real admission data. This provides valuable insights into actual admission patterns.

## Advantages of GradCafe Data

1. **Real University Names:**
   - ✅ Direct matching with QS rankings using fuzzy matching
   - ✅ No need for rating-based proxies
   - ✅ More accurate analysis

2. **Real Admission Data:**
   - ✅ Actual decisions from real applicants
   - ✅ Large sample size (271K+ records)
   - ✅ Multiple years of data

3. **Rich Feature Set:**
   - ✅ GRE scores (verbal, quant, writing)
   - ✅ GPA information
   - ✅ Applicant status (domestic/international)
   - ✅ Decision dates and methods

## Limitations

1. **Match Rate:**
   - ~49.4% of records successfully mapped (134,314 out of 271,807)
   - Reason: Many universities in GradCafe dataset are not in QS Rankings (smaller/regional universities)

2. **Missing Requirement Data:**
   - We analyze applicant scores, not official requirements
   - Cannot determine minimum GPA/GRE thresholds per university

## Validation

To validate the merging strategy:

1. **Data Quality Checks:**
   - Verify QS rankings are properly merged
   - Check for missing values after merging

2. **Statistical Validation:**
   - Compare distributions before/after merging
   - Check for unexpected correlations

---

**Note:** This strategy uses fuzzy string matching with rapidfuzz to handle university name variations, enabling accurate merging of GradCafe admission data with QS Rankings and Cost of Living datasets.
