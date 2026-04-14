# vivaglint News

## vivaglint 0.1.0

Initial Python port of the Microsoft vivaglint R package.

### New features

**Import & Validation**
- `read_glint_survey()` — load and validate Viva Glint CSV exports with automatic date parsing
- `extract_questions()` — parse column names to extract survey question metadata
- `join_attributes()` — enrich survey data with employee attribute files
- `validate_glint_structure()` — internal validation with descriptive error messages

**Reshaping**
- `pivot_long()` — convert wide survey to long format; supports `"all"`, `"comments"`, `"both"`
- `split_survey_data()` — separate numeric responses from qualitative comment columns

**Core Analytics**
- `summarize_survey()` — mean, SD, Glint Score (0–100), response counts, favorability percentages
- `get_response_dist()` — response value distribution counts and percentages
- `compare_cycles()` — track question metrics across multiple survey waves with lag-based change columns

**Advanced Analytics**
- `get_correlations()` — Pearson/Spearman/Kendall correlations in long or matrix format; `use` parameter accepts both short (`"pairwise"`, `"complete"`) and R-style names (`"pairwise.complete.obs"`, `"complete.obs"`)
- `extract_survey_factors()` — factor analysis via `factor_analyzer`; Kaiser criterion for automatic factor count
- `search_comments()` — exact and fuzzy full-text search across all comment columns
- `analyze_by_attributes()` — aggregate metrics by demographic attribute groups with group suppression
- `analyze_attrition()` — correlate survey responses with employee turnover across configurable time windows

**Hierarchy**
- `aggregate_by_manager()` — roll up scores by direct reports or full org tree
