# vivaglint News

## vivaglint 0.1.1

### New features

**API Import (Microsoft Graph)**
- `glint_setup()` — configure Viva Glint API credentials via environment variables (Python analogue of R's `.Renviron`), with optional `save_to_env_file`
- `read_glint_survey_api()` — export and import survey data directly from the Microsoft Graph beta API. Supports `cycle`, `survey`, and `daterange` modes; environment-variable fallbacks for every input; `save_zip_to` to persist the raw export zip; and `parse=False` to download without parsing. Single-CSV exports return a `GlintSurvey`; multi-CSV exports return a `dict` keyed by filename.

### Improvements
- `read_glint_survey()` now accepts configurable standard-column names (`first_name_col`, `last_name_col`, `email_col`, `status_col`, `completion_date_col`, `sent_date_col`); pass `None` to opt out of a column your export lacks.
- New shared `build_glint_survey()` builder wraps in-memory DataFrames (used by both CSV and API import).
- `validate_glint_structure()` now requires only the employee ID column and emits warnings (instead of hard errors) for missing optional standard columns, matching the current R package behaviour.
- Added `pyproject.toml` for `pip install`, with `factor`, `plot`, `dev`, `docs`, and `all` extras. `requests` is now a core dependency; `factor_analyzer`, `matplotlib`, and `seaborn` are optional extras.
- Added a `tests/` suite (utils, import, analysis smoke tests, and fully-mocked API tests) plus sample fixtures under `tests/fixtures/`.

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
