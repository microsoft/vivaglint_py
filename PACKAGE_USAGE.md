# vivaglint Package Usage Guide

## Installation and Setup

```bash
pip install vivaglint

# With optional dev dependencies (pytest)
pip install "vivaglint[dev]"

# With optional docs dependencies (Sphinx)
pip install "vivaglint[docs]"
```

---

## Core Functions

### 1. Import and Validate Data

```python
from vivaglint import read_glint_survey, extract_questions, join_attributes

# Load a Glint CSV export
survey = read_glint_survey("FY25_Q2_Export.csv", emp_id_col="Employee ID")

# R equivalent:
# survey <- read_glint_survey("FY25_Q2_Export.csv")

# Access the data and metadata
print(survey.data.head())               # R: survey$data
print(survey.metadata["questions"])     # R: survey$metadata$questions
print(survey.metadata["n_respondents"]) # R: survey$metadata$n_respondents

# List all questions in the survey
questions = extract_questions(survey)

# Enrich with employee attributes
survey_enriched = join_attributes(survey, "employee_attributes.csv")
# Attribute column names are stored in survey.metadata["attribute_cols"]
```

### 2. Reshape Data

```python
from vivaglint import pivot_long, split_survey_data

# Wide to long — all responses
long_all = pivot_long(survey, data_type="all")

# Comments only
long_comments = pivot_long(survey, data_type="comments")

# Both at once as a dict
# R: both <- pivot_long(survey, data_type = "both")
# R: both$comments
both = pivot_long(survey, data_type="both")
comments_df = both["comments"]

# Split quantitative and qualitative
# R: parts <- split_survey_data(survey)
# R: parts$quantitative
parts = split_survey_data(survey)
numeric_data  = parts["quantitative"]
comment_data  = parts["qualitative"]
```

### 3. Summary Statistics

```python
from vivaglint import summarize_survey, get_response_dist

# All questions
summary = summarize_survey(survey, scale_points=5)

# Specific questions only
summary_subset = summarize_survey(
    survey,
    scale_points=5,
    questions=["My work is meaningful", "I feel valued at work"],
)

# Response value distributions
dist = get_response_dist(survey)

# With plot
summary_plot = summarize_survey(survey, scale_points=5, plot=True)
```

### 4. Multi-Cycle Comparisons

```python
from vivaglint import compare_cycles

survey_q1 = read_glint_survey("Q1.csv", emp_id_col="Employee ID")
survey_q2 = read_glint_survey("Q2.csv", emp_id_col="Employee ID")
survey_q3 = read_glint_survey("Q3.csv", emp_id_col="Employee ID")

# R: compare_cycles(survey_q1, survey_q2, survey_q3, scale_points = 5,
#                   cycle_names = c("Q1", "Q2", "Q3"))
trends = compare_cycles(
    survey_q1, survey_q2, survey_q3,
    scale_points=5,
    cycle_names=["Q1 FY25", "Q2 FY25", "Q3 FY25"],
)
```

### 5. Manager-Level Analysis

```python
from vivaglint import aggregate_by_manager

# Direct reports only
managers = aggregate_by_manager(survey, scale_points=5)

# Full org tree (all indirect reports included)
managers_full = aggregate_by_manager(survey, scale_points=5, full_tree=True)

# Filter to managers below threshold
# R: filter(managers, pct_favorable < 60)
low_scoring = managers[managers["pct_favorable"] < 60]
```

### 6. Demographic Segmentation

```python
from vivaglint import analyze_by_attributes

# Pass attribute file path or pre-joined survey
by_dept = analyze_by_attributes(
    survey_enriched,
    scale_points=5,
    attribute_cols=["Department", "Job Level"],
    min_group_size=10,   # suppress groups smaller than this (privacy)
)

# Filter to subpopulation before analysing
import copy
na_survey = copy.copy(survey_enriched)
na_survey.data = survey_enriched.data[
    survey_enriched.data["Region"] == "North America"
].copy()

na_results = analyze_by_attributes(na_survey, scale_points=5,
                                   attribute_cols="Department")
```

### 7. Attrition Risk Analysis

```python
from vivaglint import analyze_attrition

attrition = analyze_attrition(
    survey,
    attrition_file="terminations.csv",
    emp_id_col="Employee ID",
    term_date_col="Termination Date",
    scale_points=5,
    time_periods=[90, 180, 365],  # days post-survey
    min_group_size=5,
)

# Highest-risk questions
top_risk = attrition.sort_values("attrition_ratio", ascending=False)
```

### 8. Correlations and Factor Analysis

```python
from vivaglint import get_correlations, extract_survey_factors

# Correlations — long format (default)
corr = get_correlations(survey)

# Matrix format
corr_matrix = get_correlations(survey, format="matrix")

# Spearman (more robust for ordinal data)
corr_spearman = get_correlations(survey, method="spearman")

# R-style use parameter (also accepted)
# R: get_correlations(survey, use = "pairwise.complete.obs")
corr_pairwise = get_correlations(survey, use="pairwise.complete.obs")

# Factor analysis
factors = extract_survey_factors(survey, n_factors=3, rotation="oblimin")

# R: factors$factor_summary   →  Python: factors["factor_summary"]
# R: factors$fa_object        →  Python: factors["fa_object"]
print(factors["factor_summary"])

strong = factors["factor_summary"][
    factors["factor_summary"]["loading_label"] == "Strong"
]
```

### 9. Comment Search

```python
from vivaglint import search_comments

# Fuzzy match (default)
results = search_comments(survey, "work life balance")

# Exact case-sensitive match
# R: search_comments(survey, "burnout", exact = TRUE)
results_exact = search_comments(survey, "burnout", exact=True)

# Widen fuzzy tolerance
results_broad = search_comments(survey, "colaboration", max_distance=0.3)
```

---

## Running Tests

```bash
pytest
pytest --cov=vivaglint   # with coverage
```

---

## Building Documentation

```bash
sphinx-build -b html man man/_build/html
# Open man/_build/html/index.html
```

---

## Error Handling

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: File not found: 'export.csv'` | Path does not exist | Check the file path |
| `ValueError: Missing required standard column(s): 'Employee ID'` | `emp_id_col` does not match | Pass the exact column name from your CSV |
| `ValueError: No question columns found` | CSV has only standard columns | Confirm you are using a complete Glint export |
| `ValueError: Incomplete question column sets` | A question is missing one of its 4 columns | Check the CSV for truncated columns |
| `ValueError: scale_points must be an integer between 2 and 11` | Invalid scale | Pass the number of points on your rating scale (e.g. `5`) |
| `ValueError: use must be one of: ...` | Invalid `use` value in `get_correlations` | Use `"pairwise"`, `"complete"`, `"pairwise.complete.obs"`, or `"complete.obs"` |

---

## Notes

- **Response values**: Numeric columns must contain integers 1–N where N = `scale_points`. Non-numeric values are treated as skips (NaN).
- **Date format**: Glint exports dates as `DD-MM-YYYY HH:MM`. Parsed automatically to `datetime64[ns]`.
- **Comment columns**: The `_COMMENT`, `_COMMENT_TOPICS`, and `_SENSITIVE_COMMENT_FLAG` columns are preserved in all reshape operations.
- **Manager comparisons**: `aggregate_by_manager` requires `min_group_size` compliance to protect small teams.
- **Org tree**: `full_tree=True` in `aggregate_by_manager` uses a recursive traversal with cycle detection to handle imperfect org data.
- **Group suppression**: `analyze_by_attributes` and `analyze_attrition` return an empty DataFrame (with correct schema) when no groups meet the `min_group_size` threshold.
