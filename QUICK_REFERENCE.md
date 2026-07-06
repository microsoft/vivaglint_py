# vivaglint Quick Reference

## Import Functions

| Function | Returns | Description |
|---|---|---|
| `read_glint_survey(file_path, emp_id_col, encoding="UTF-8")` | `GlintSurvey` | Load and validate a Glint CSV export |
| `extract_questions(data, emp_id_col=None)` | `DataFrame` | Parse column names into question metadata |
| `join_attributes(survey, attribute_source, emp_id_col=None)` | `GlintSurvey` | Enrich survey with employee attributes |

## API Functions (Microsoft Graph)

| Function | Returns | Description |
|---|---|---|
| `glint_setup(tenant_id, client_id, client_secret, experience_name, save_to_env_file=False)` | `bool` | Configure Graph API credentials (via env vars) |
| `read_glint_survey_api(survey_uuid=None, cycle_id=None, mode=None, ..., start_date=None, end_date=None, save_zip_to=None, parse=True, experience_name=None)` | `GlintSurvey` / `dict` / `str` | Export & import survey data via the Graph API (`cycle`/`survey`/`daterange` modes) |

## Reshape Functions

| Function | Returns | Description |
|---|---|---|
| `pivot_long(survey, data_type="all", include_empty=False, include_standard_cols=True)` | `DataFrame` or `dict` | Convert wide survey to long format |
| `split_survey_data(survey, emp_id_col=None)` | `dict` | Split into quantitative and qualitative DataFrames |

## Analysis Functions

| Function | Returns | Description |
|---|---|---|
| `summarize_survey(survey, scale_points, questions="all", emp_id_col=None, plot=False)` | `DataFrame` | Mean, SD, Glint Score, favorability per question |
| `get_response_dist(survey, questions="all", plot=False)` | `DataFrame` | Response value distribution counts and percentages |
| `compare_cycles(*surveys, scale_points, cycle_names=None, plot=False)` | `DataFrame` | Track metrics across survey waves |
| `get_correlations(survey, method="pearson", format="long", use="pairwise", plot=False)` | `DataFrame` | Question-to-question correlations |
| `extract_survey_factors(survey, n_factors=None, rotation="oblimin", min_loading=0.3, fm="minres", plot=False)` | `dict` | Factor analysis loadings and variance |
| `analyze_by_attributes(survey, attribute_file=None, scale_points=None, attribute_cols=None, emp_id_col=None, min_group_size=5, plot=False)` | `DataFrame` | Metrics segmented by demographic groups |
| `analyze_attrition(survey, attrition_file, emp_id_col=None, term_date_col=None, scale_points=None, time_periods=None, attribute_cols=None, min_group_size=5, plot=False)` | `DataFrame` | Survey responses linked to turnover |
| `search_comments(survey, query, exact=False, max_distance=0.2)` | `DataFrame` | Full-text search across all comment columns |

## Hierarchy Functions

| Function | Returns | Description |
|---|---|---|
| `aggregate_by_manager(survey, scale_points, emp_id_col=None, manager_id_col=None, full_tree=False, plot=False)` | `DataFrame` | Roll up scores by manager |

---

## Example Workflow

```python
from vivaglint import *

# 1. Load data
survey = read_glint_survey("export.csv", emp_id_col="Employee ID")

# 2. Explore questions
questions = extract_questions(survey)

# 3. Summarise
summary = summarize_survey(survey, scale_points=5)

# 4. Response distributions
dist = get_response_dist(survey)

# 5. Multi-cycle trends
trends = compare_cycles(survey_q1, survey_q2, scale_points=5,
                        cycle_names=["Q1", "Q2"])

# 6. Manager roll-up
managers = aggregate_by_manager(survey, scale_points=5)

# 7. Demographic segmentation
survey_enriched = join_attributes(survey, "attributes.csv")
by_dept = analyze_by_attributes(survey_enriched, scale_points=5,
                                attribute_cols="Department")

# 8. Attrition risk
attrition = analyze_attrition(survey, attrition_file="terms.csv",
                              emp_id_col="Employee ID",
                              term_date_col="Termination Date",
                              scale_points=5)

# 9. Correlations
corr = get_correlations(survey)                          # long format
corr_matrix = get_correlations(survey, format="matrix")  # matrix

# 10. Factor analysis
factors = extract_survey_factors(survey, n_factors=3)
print(factors["factor_summary"])

# 11. Comment search
comments = search_comments(survey, "flexibility")

# 12. Long format / NLP pipeline
long = pivot_long(survey, data_type="comments")
parts = split_survey_data(survey)
```

---

## Key Metrics

| Column | Description |
|---|---|
| `mean` | Average raw response value |
| `sd` | Standard deviation (ddof=1, matching R) |
| `glint_score` | `round(((mean - 1) / (scale_points - 1)) * 100)` |
| `n_responses` | Respondents who answered |
| `n_skips` | Respondents who skipped |
| `n_total` | Total respondents |
| `response_rate` | `n_responses / n_total * 100` |
| `pct_favorable` | % of responses in the favorable band |
| `pct_neutral` | % of responses in the neutral band |
| `pct_unfavorable` | % of responses in the unfavorable band |
| `count_X` / `pct_X` | Per-value counts/percentages (response dist) |

---

## R-to-Python Translation

| R | Python |
|---|---|
| `survey$data` | `survey.data` |
| `survey$metadata` | `survey.metadata` |
| `survey$metadata$questions` | `survey.metadata["questions"]` |
| `result$factor_summary` | `result["factor_summary"]` |
| `both$comments` | `both["comments"]` |
| `parts$quantitative` | `parts["quantitative"]` |
| `filter(df, col > x)` | `df[df["col"] > x]` |
| `arrange(df, col)` | `df.sort_values("col")` |
| `left_join(a, b, by="ID")` | `a.merge(b, on="ID", how="left")` |
| `use = "pairwise.complete.obs"` | `use="pairwise.complete.obs"` *(also accepted)* |
| `exact = TRUE` | `exact=True` |
| `full_tree = FALSE` | `full_tree=False` |

---

## Favorability Thresholds by Scale

| Scale Points | Favorable | Neutral | Unfavorable |
|---|---|---|---|
| 2 | [2] | [] | [1] |
| 3 | [3] | [2] | [1] |
| 4 | [4] | [2,3] | [1] |
| 5 | [4,5] | [3] | [1,2] |
| 6 | [4,5,6] | [] | [1,2,3] |
| 7 | [6,7] | [4,5] | [1,2,3] |
| 8 | [6,7,8] | [4,5] | [1,2,3] |
| 9 | [7,8,9] | [4,5,6] | [1,2,3] |
| 10 | [8,9,10] | [4,5,6,7] | [1,2,3] |
| 11 | [10,11] | [8,9] | [1,2,3,4,5,6,7] |

---

## Common Patterns

```python
# Filter to low-scoring questions
low = summary[summary["glint_score"] < 60]

# Get all comments for a specific question
q_comments = search_comments(survey, "My work is meaningful", exact=True)

# Find managers below 50% favorable
at_risk = managers[
    (managers["question"] == "My work is meaningful")
    & (managers["pct_favorable"] < 50)
].sort_values("pct_favorable")

# Strong factor loadings only
strong = factors["factor_summary"][
    factors["factor_summary"]["loading_label"] == "Strong"
]

# Attrition risk ratio — highest first
top_risk = attrition.sort_values("attrition_ratio", ascending=False)
```

---

## Error Messages

```
ValueError: Missing required standard column(s): 'Employee ID'
→ Check emp_id_col matches the column name in your CSV exactly.

ValueError: Incomplete question column sets found
→ Each question needs 4 columns: base, _COMMENT, _COMMENT_TOPICS, _SENSITIVE_COMMENT_FLAG.

ValueError: scale_points must be an integer between 2 and 11
→ Pass the number of points on your survey's rating scale (e.g. scale_points=5).

ValueError: use must be one of: 'pairwise', 'complete', 'pairwise.complete.obs', 'complete.obs'
→ Use one of the listed values (R-style names are accepted).
```
