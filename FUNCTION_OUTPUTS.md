# vivaglint Function Outputs

Column-level reference for every function's return value.

---

## `read_glint_survey()`

Returns a `GlintSurvey` object.

```python
survey.data        # pd.DataFrame — all survey responses
survey.metadata    # dict
```

**`survey.metadata` keys:**

| Key | Type | Description |
|---|---|---|
| `emp_id_col` | `str` | Employee ID column name |
| `standard_columns` | `list[str]` | The 8 standard Glint column names |
| `questions` | `DataFrame` | One row per question (see `extract_questions`) |
| `n_respondents` | `int` | Total rows in `survey.data` |
| `n_questions` | `int` | Number of survey questions |
| `file_path` | `str` | Path originally supplied |
| `attribute_cols` | `list[str]` | Populated by `join_attributes()` |

---

## `extract_questions()`

Returns `pd.DataFrame` — one row per question.

| Column | Type | Description |
|---|---|---|
| `question` | `str` | Base question text (stem) |
| `response_col` | `str` | Column containing numeric responses |
| `comment_col` | `str` | Column containing open-text comments |
| `topics_col` | `str` | Column containing Glint-assigned topic tags |
| `flag_col` | `str` | Column containing sensitive comment flags |

---

## `summarize_survey()`

Returns `pd.DataFrame` — one row per question.

| Column | Type | Description |
|---|---|---|
| `question` | `str` | Question text |
| `mean` | `float` | Mean response on the raw scale |
| `sd` | `float` | Standard deviation (ddof=1) |
| `glint_score` | `int` | `round(((mean-1)/(scale_points-1))*100)` |
| `n_responses` | `int` | Respondents who answered |
| `n_skips` | `int` | Respondents who skipped |
| `n_total` | `int` | Total respondents |
| `response_rate` | `float` | `n_responses / n_total * 100` |
| `pct_favorable` | `float` | % in the favorable band |
| `pct_neutral` | `float` | % in the neutral band |
| `pct_unfavorable` | `float` | % in the unfavorable band |

---

## `get_response_dist()`

Returns `pd.DataFrame` — one row per question, plus per-value columns.

Includes all columns from `summarize_survey()`, plus for each response value X (1 to `scale_points`):

| Column | Type | Description |
|---|---|---|
| `count_X` | `int` | Number of respondents who chose value X |
| `pct_X` | `float` | Percentage who chose value X |

Missing values default to `0` (not NaN).

---

## `compare_cycles()`

Returns `pd.DataFrame` — one row per question × cycle.

Includes all columns from `summarize_survey()`, plus:

| Column | Type | Description |
|---|---|---|
| `cycle` | `str` | Cycle name (from `cycle_names`) or `"Cycle 1"`, etc. |
| `change_from_previous` | `float` | Change in `mean` vs. prior cycle (NaN for first cycle) |
| `pct_change_from_previous` | `float` | % change in `mean` vs. prior cycle |

---

## `get_correlations()`

**Long format** (`format="long"`, default) — one row per question pair:

| Column | Type | Description |
|---|---|---|
| `question1` | `str` | First question |
| `question2` | `str` | Second question |
| `correlation` | `float` | Correlation coefficient |
| `p_value` | `float` | p-value (`0.0` for self-correlations) |
| `n` | `int` | Number of complete pairs used |

**Matrix format** (`format="matrix"`) — `pd.DataFrame` with questions as both index and columns.

---

## `extract_survey_factors()`

Returns `dict` with two keys:

| Key | Type | Description |
|---|---|---|
| `"factor_summary"` | `DataFrame` | One row per question × factor (see below) |
| `"fa_object"` | `FactorAnalyzer` | Raw `factor_analyzer.FactorAnalyzer` object |

**`factor_summary` columns:**

| Column | Type | Description |
|---|---|---|
| `question` | `str` | Question text |
| `factor` | `str` | Factor name (`"MR1"`, `"MR2"`, …) |
| `loading` | `float` | Factor loading |
| `loading_label` | `str` | `"Strong"` (≥0.75), `"Medium"` (0.60–0.74), `"Weak"` (<0.60) |
| `communality` | `float` | Communality (variance explained by all factors) |
| `factor_variance_pct` | `float` | % variance explained by this factor |

Rows with `abs(loading) < min_loading` are excluded.

---

## `pivot_long()`

**`data_type="all"` or `data_type="comments"`** — returns `pd.DataFrame`:

Standard columns (when `include_standard_cols=True`) + :

| Column | Type | Description |
|---|---|---|
| `question` | `str` | Question text |
| `response` | `float` | Numeric response value (NaN if skipped) |
| `comment` | `str` | Open-text comment (NaN if empty) |
| `comment_topics` | `str` | Topic tags from Glint (NaN if empty) |
| `sensitive_flag` | `bool` | Sensitive comment flag |

**`data_type="both"`** — returns `dict[str, pd.DataFrame]` with keys `"all"` and `"comments"`.

---

## `split_survey_data()`

Returns `dict[str, pd.DataFrame]` with two keys:

| Key | Contents |
|---|---|
| `"quantitative"` | Standard columns + one numeric response column per question |
| `"qualitative"` | Employee ID + all `_COMMENT`, `_COMMENT_TOPICS`, `_SENSITIVE_COMMENT_FLAG` columns |

---

## `aggregate_by_manager()`

Returns `pd.DataFrame` — one row per manager × question.

| Column | Type | Description |
|---|---|---|
| `manager_id` | `str` | Manager employee ID |
| `manager_name` | `str` | Manager name (from `First Name` + `Last Name`) |
| `question` | `str` | Question text |
| `team_size` | `int` | Number of reports included |
| `mean` | `float` | Mean response |
| `sd` | `float` | Standard deviation |
| `glint_score` | `int` | Glint Score (0–100) |
| `n_responses` | `int` | Respondents who answered |
| `n_skips` | `int` | Respondents who skipped |
| `n_total` | `int` | Total team members |
| `pct_favorable` | `float` | % favorable |
| `pct_neutral` | `float` | % neutral |
| `pct_unfavorable` | `float` | % unfavorable |

---

## `analyze_by_attributes()`

Returns `pd.DataFrame` — one row per attribute group × question.

| Column | Type | Description |
|---|---|---|
| `attribute` | `str` | Attribute name (e.g. `"Department"`) |
| `group` | `str` | Attribute value (e.g. `"Engineering"`) |
| `question` | `str` | Question text |
| `group_size` | `int` | Number of respondents in group |
| `mean` | `float` | Mean response |
| `sd` | `float` | Standard deviation |
| `glint_score` | `int` | Glint Score (0–100) |
| `n_responses` | `int` | Respondents who answered |
| `n_skips` | `int` | Respondents who skipped |
| `n_total` | `int` | Group total |
| `pct_favorable` | `float` | % favorable |
| `pct_neutral` | `float` | % neutral |
| `pct_unfavorable` | `float` | % unfavorable |

Groups with fewer than `min_group_size` respondents are suppressed.

---

## `analyze_attrition()`

Returns `pd.DataFrame` — one row per question × time period (× attribute group if `attribute_cols` is set).

| Column | Type | Description |
|---|---|---|
| `question` | `str` | Question text |
| `time_period_days` | `int` | Time window (e.g. `90`, `180`, `365`) |
| `favorability_group` | `str` | `"favorable"`, `"neutral"`, or `"unfavorable"` |
| `n_group` | `int` | Respondents in this favorability group |
| `n_attrition` | `int` | Who left within the time window |
| `attrition_rate` | `float` | `n_attrition / n_group` |
| `attrition_ratio` | `float` | Attrition rate vs. overall rate (>1 = higher risk) |

---

## `search_comments()`

Returns `pd.DataFrame` — one row per matching comment.

| Column | Type | Description |
|---|---|---|
| `question` | `str` | Question the comment was attached to |
| `response` | `float` | Numeric score given by the respondent |
| `comment` | `str` | Comment text |
| `topics` | `str` | Glint topic tags |

Returns an empty DataFrame (same schema) when no matches are found.
