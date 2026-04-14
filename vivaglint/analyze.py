"""
vivaglint.analyze
-----------------
Core survey analysis functions.

All functions are direct Python ports of the exported functions in R/analyze.R.
They accept either a :class:`~vivaglint.import_.GlintSurvey` object or a plain
``pd.DataFrame`` and return ``pd.DataFrame`` (or a ``dict`` for
``extract_survey_factors``).
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from vivaglint.import_ import GlintSurvey, extract_questions
from vivaglint.utils import (
    get_favorability_map,
    get_standard_columns,
    mean_to_glint_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_survey(
    survey: Union[GlintSurvey, pd.DataFrame],
    emp_id_col: Optional[str] = None,
) -> tuple[pd.DataFrame, list[str], Optional[str]]:
    """Return (data, question_list, emp_id_col) from a GlintSurvey or DataFrame.

    Parameters
    ----------
    survey:
        A GlintSurvey or a plain DataFrame.
    emp_id_col:
        Overrides the value stored in survey.metadata when provided.

    Returns
    -------
    tuple of (pd.DataFrame, list[str], Optional[str])
    """
    if isinstance(survey, GlintSurvey):
        resolved_emp_id_col = emp_id_col or survey.metadata.get("emp_id_col")
        data = survey.data
        questions = list(survey.metadata["questions"]["question"])
    elif isinstance(survey, pd.DataFrame):
        resolved_emp_id_col = emp_id_col
        data = survey
        questions = list(extract_questions(data, emp_id_col)["question"])
    else:
        raise TypeError("survey must be a GlintSurvey object or a pandas DataFrame")
    return data, questions, resolved_emp_id_col


def _validate_questions(
    questions_param,
    all_questions: list[str],
) -> list[str]:
    """Resolve the *questions* parameter to a concrete list.

    Mirrors the R logic: if questions == "all" return everything; otherwise
    validate each requested question exists and raise if any are missing.
    """
    if isinstance(questions_param, str) and questions_param == "all":
        return all_questions

    requested = list(questions_param)
    missing = [q for q in requested if q not in all_questions]
    if missing:
        missing_str = ", ".join(f"'{q}'" for q in missing)
        available_str = "\n  ".join(f"- {q}" for q in all_questions)
        raise ValueError(
            f"Question(s) not found: {missing_str}\n\n"
            f"Available questions:\n  {available_str}"
        )
    return requested


def _parallel_analysis(
    response_data: pd.DataFrame,
    n_iter: int = 20,
    random_state: int = 0,
) -> int:
    """Approximate psych::fa.parallel() to choose the number of factors."""
    corr = response_data.corr()
    if corr.isna().any().any():
        raise ValueError(
            "Parallel analysis cannot compute correlations because there "
            "are not enough complete rows of question responses "
            "to estimate all pairwise correlations."
        )
    observed_eigs = np.sort(np.linalg.eigvalsh(corr.to_numpy()))[::-1]
    rng = np.random.default_rng(random_state)
    n_rows, n_cols = response_data.shape
    simulated_eigs = np.empty((n_iter, n_cols), dtype=float)

    for i in range(n_iter):
        simulated = rng.standard_normal(size=(n_rows, n_cols))
        sim_corr = np.corrcoef(simulated, rowvar=False)
        simulated_eigs[i, :] = np.sort(np.linalg.eigvalsh(sim_corr))[::-1]

    mean_simulated = simulated_eigs.mean(axis=0)
    n_factors = int((observed_eigs > mean_simulated).sum())
    return max(1, n_factors)


# ---------------------------------------------------------------------------
# summarize_survey
# ---------------------------------------------------------------------------

def summarize_survey(
    survey: Union[GlintSurvey, pd.DataFrame],
    scale_points: int,
    questions="all",
    emp_id_col: Optional[str] = None,
    plot: bool = False,
) -> pd.DataFrame:
    """Calculate comprehensive metrics for survey questions.

    Returns mean, standard deviation, Glint Score, response counts, skip
    counts, and favorability percentages for each question.

    Ported from ``R/analyze.R::summarize_survey``.

    The Glint Score formula (verified against R source)::

        round(((mean - 1) / (scale_points - 1)) * 100)

    Favorability percentages are rounded to 1 decimal place, matching R's
    ``round(..., 1)`` convention.

    Parameters
    ----------
    survey:
        A :class:`~vivaglint.import_.GlintSurvey` or plain DataFrame.
    scale_points:
        Number of scale points (2-11).
    questions:
        ``"all"`` to analyze every question (default), or a list of
        question text strings to analyze a subset.
    emp_id_col:
        Employee ID column name. Resolved from survey metadata when omitted.
    plot:
        If ``True``, display a horizontal bar chart of Glint Scores and
        return the DataFrame.

    Returns
    -------
    pd.DataFrame
        One row per question with columns: ``question``, ``mean``, ``sd``,
        ``glint_score``, ``n_responses``, ``n_skips``, ``n_total``,
        ``pct_favorable``, ``pct_neutral``, ``pct_unfavorable``.

    Raises
    ------
    ValueError
        If *scale_points* is not 2-11, or requested questions are not found.
    """
    if scale_points not in range(2, 12):
        raise ValueError("scale_points must be an integer between 2 and 11")

    favorability = get_favorability_map(scale_points)
    fav_set = set(favorability["favorable"])
    neu_set = set(favorability["neutral"])
    unf_set = set(favorability["unfavorable"])

    data, all_questions, _ = _resolve_survey(survey, emp_id_col)
    questions_to_analyze = _validate_questions(questions, all_questions)

    rows = []
    for question_text in questions_to_analyze:
        responses = data[question_text]
        n_total = len(responses)

        valid_responses = responses.dropna()
        n_responses = len(valid_responses)
        n_skips = n_total - n_responses

        if n_responses > 0:
            mean_response = float(valid_responses.mean())
            # R uses sd() which is sample std (ddof=1)
            sd_response = (
                float(valid_responses.std(ddof=1)) if n_responses > 1 else float("nan")
            )

            n_favorable = int(valid_responses.isin(fav_set).sum())
            n_neutral = int(valid_responses.isin(neu_set).sum())
            n_unfavorable = int(valid_responses.isin(unf_set).sum())

            # R: round((n / n_responses) * 100, 1)
            pct_favorable = round((n_favorable / n_responses) * 100, 1)
            pct_neutral = round((n_neutral / n_responses) * 100, 1)
            pct_unfavorable = round((n_unfavorable / n_responses) * 100, 1)
        else:
            mean_response = float("nan")
            sd_response = float("nan")
            pct_favorable = float("nan")
            pct_neutral = float("nan")
            pct_unfavorable = float("nan")

        rows.append(
            {
                "question": question_text,
                "mean": mean_response,
                "sd": sd_response,
                # mean_to_glint_score: round(((mean - 1) / (scale_points - 1)) * 100)
                "glint_score": mean_to_glint_score(mean_response, scale_points),
                "n_responses": n_responses,
                "n_skips": n_skips,
                "n_total": n_total,
                "pct_favorable": pct_favorable,
                "pct_neutral": pct_neutral,
                "pct_unfavorable": pct_unfavorable,
            }
        )

    results = pd.DataFrame(rows)

    if plot:
        from vivaglint.plots import _plot_survey_summary
        fig = _plot_survey_summary(results)
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# get_response_dist
# ---------------------------------------------------------------------------

def get_response_dist(
    survey: Union[GlintSurvey, pd.DataFrame],
    questions="all",
    plot: bool = False,
    emp_id_col: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate the distribution of response values for survey questions.

    Returns counts and percentages for each response value in wide format
    (columns ``count_1``, ``pct_1``, ``count_2``, ``pct_2``, ...).

    Ported from ``R/analyze.R::get_response_dist``.

    The R source stores per-question dicts then expands them into a wide
    DataFrame column-by-column. Missing values default to 0, matching R's
    ``%||% 0`` fallback.

    Parameters
    ----------
    survey:
        A :class:`~vivaglint.import_.GlintSurvey` or plain DataFrame.
    questions:
        ``"all"`` (default) or a list of question text strings.
    plot:
        If ``True``, display a stacked bar chart and return the DataFrame.

    Returns
    -------
    pd.DataFrame
        One row per question with ``question`` plus ``count_X`` and ``pct_X``
        columns for each unique response value observed across all questions.

    Raises
    ------
    ValueError
        If requested questions are not found.
    """
    data, all_questions, _ = _resolve_survey(survey, emp_id_col)
    questions_to_analyze = _validate_questions(questions, all_questions)

    # First pass: collect per-question value count/pct dicts
    question_rows = []
    for question_text in questions_to_analyze:
        responses = data[question_text]
        valid_responses = responses.dropna()

        value_counts: dict[int, int] = {}
        value_pcts: dict[int, float] = {}

        if len(valid_responses) > 0:
            counts = valid_responses.value_counts().sort_index()
            total = counts.sum()
            for val, cnt in counts.items():
                v = int(val)
                value_counts[v] = int(cnt)
                value_pcts[v] = float(cnt / total * 100)

        question_rows.append(
            {
                "question": question_text,
                "_counts": value_counts,
                "_pcts": value_pcts,
            }
        )

    # All unique response values across all questions (sorted) — mirrors R's sort()
    all_values = sorted(
        {v for row in question_rows for v in row["_counts"].keys()}
    )

    # Second pass: build wide DataFrame, filling 0 for absent values
    final_rows = []
    for row in question_rows:
        final_row: dict = {"question": row["question"]}
        for val in all_values:
            final_row[f"count_{val}"] = row["_counts"].get(val, 0)
            final_row[f"pct_{val}"] = row["_pcts"].get(val, 0.0)
        final_rows.append(final_row)

    results = pd.DataFrame(final_rows)

    if plot:
        from vivaglint.plots import _plot_response_dist
        fig = _plot_response_dist(results)
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# compare_cycles
# ---------------------------------------------------------------------------

def compare_cycles(
    *surveys,
    scale_points: int,
    cycle_names: Optional[list[str]] = None,
    plot: bool = False,
) -> pd.DataFrame:
    """Compare question-level metrics across multiple survey cycles.

    Calculates change scores and trends over time using the same metrics as
    ``summarize_survey()``.

    Ported from ``R/analyze.R::compare_cycles``.

    Change columns mirror R's ``dplyr::lag()`` behavior:
    - ``change_from_previous`` — raw mean difference from previous cycle
    - ``pct_change_from_previous`` — percentage change from previous cycle mean
    - ``glint_score_change_from_previous`` — Glint Score point change

    Parameters
    ----------
    *surveys:
        Two or more :class:`~vivaglint.import_.GlintSurvey` objects or plain
        DataFrames.
    scale_points:
        Number of scale points (2-11).
    cycle_names:
        Optional list of names for each cycle. Defaults to
        ``["Cycle 1", "Cycle 2", ...]``.
    plot:
        If ``True``, display a line chart of Glint Score over cycles.

    Returns
    -------
    pd.DataFrame
        One row per question-cycle combination. Columns: ``cycle``,
        ``question``, ``mean``, ``sd``, ``glint_score``, ``n_responses``,
        ``n_skips``, ``n_total``, ``pct_favorable``, ``pct_neutral``,
        ``pct_unfavorable``, ``change_from_previous``,
        ``pct_change_from_previous``, ``glint_score_change_from_previous``.

    Raises
    ------
    ValueError
        If fewer than two surveys are supplied or cycle_names length mismatch.
    """
    if len(surveys) < 2:
        raise ValueError("At least two surveys are required for comparison")

    if cycle_names is None:
        cycle_names = [f"Cycle {i + 1}" for i in range(len(surveys))]
    elif len(cycle_names) != len(surveys):
        raise ValueError("Number of cycle_names must match number of surveys")

    frames = []
    for survey, cycle_name in zip(surveys, cycle_names):
        result = summarize_survey(survey, scale_points=scale_points)
        result["cycle"] = cycle_name
        frames.append(result)

    analyses = pd.concat(frames, ignore_index=True)

    # Reorder: cycle first, then question, then everything else (mirrors R's select())
    col_order = ["cycle", "question"] + [
        c for c in analyses.columns if c not in ("cycle", "question")
    ]
    analyses = analyses[col_order]

    # Sort by question then cycle (in insertion order, not lexicographic order).
    # Use an integer rank derived from cycle_names so that "Cycle 10" sorts
    # after "Cycle 9" regardless of string comparison.
    # This mirrors R's group_by(question) %>% mutate(... dplyr::lag(mean) ...)
    _cycle_rank = {name: i for i, name in enumerate(cycle_names)}
    analyses["_cycle_rank"] = analyses["cycle"].map(_cycle_rank)
    analyses = (
        analyses
        .sort_values(["question", "_cycle_rank"])
        .drop(columns=["_cycle_rank"])
        .reset_index(drop=True)
    )
    analyses["change_from_previous"] = analyses.groupby("question")["mean"].diff()
    analyses["pct_change_from_previous"] = (
        analyses.groupby("question")["mean"].diff()
        / analyses.groupby("question")["mean"].shift(1)
        * 100
    )
    analyses["glint_score_change_from_previous"] = (
        analyses.groupby("question")["glint_score"].diff()
    )

    if plot:
        from vivaglint.plots import _plot_compare_cycles
        fig = _plot_compare_cycles(analyses, cycle_names)
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass

    return analyses


# ---------------------------------------------------------------------------
# get_correlations
# ---------------------------------------------------------------------------

def get_correlations(
    survey: Union[GlintSurvey, pd.DataFrame],
    method: str = "pearson",
    format: str = "long",
    use: str = "pairwise",
    plot: bool = False,
    emp_id_col: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate correlations between all survey question response columns.

    Ported from ``R/analyze.R::get_correlations``.

    In long format, p-values are computed for each pair individually using
    ``scipy.stats``, mirroring R's ``cor.test(..., exact=FALSE)``.
    Self-correlations have ``p_value = 0`` (matching R).

    Parameters
    ----------
    survey:
        A :class:`~vivaglint.import_.GlintSurvey` or plain DataFrame.
    method:
        Correlation method: ``"pearson"`` (default), ``"spearman"``, or
        ``"kendall"``.
    format:
        ``"long"`` (default) — one row per question pair; ``"matrix"`` —
        square DataFrame with questions as rows and columns.
    use:
        How to handle missing values. ``"pairwise"`` (default, equivalent to
        R's ``"pairwise.complete.obs"``) uses all complete pairs per pair.
        ``"complete"`` (or ``"complete.obs"``) uses only rows that are complete
        across all questions. R-style full names are also accepted
        (``"pairwise.complete.obs"``, ``"complete.obs"``) for easy migration.
    plot:
        If ``True`` and *format* is ``"long"``, display a correlation heatmap.
        Ignored with a warning when *format* is ``"matrix"``.

    Returns
    -------
    pd.DataFrame
        Long-format with columns ``question1``, ``question2``,
        ``correlation``, ``p_value``, ``n``; or a square correlation
        matrix DataFrame when *format* is ``"matrix"``.

    Raises
    ------
    ValueError
        On invalid *method* or *format* values.
    """
    valid_methods = ("pearson", "spearman", "kendall")
    if method not in valid_methods:
        raise ValueError(
            f"method must be one of: {', '.join(repr(m) for m in valid_methods)}"
        )

    valid_formats = ("long", "matrix")
    if format not in valid_formats:
        raise ValueError(
            f"format must be one of: {', '.join(repr(f) for f in valid_formats)}"
        )

    # Accept both short forms ("pairwise", "complete") and R's full names
    # ("pairwise.complete.obs", "complete.obs") for easy migration from R.
    _use_normalised = use.replace(".obs", "").replace(".complete", "")
    valid_use = ("pairwise", "complete", "pairwise.complete.obs", "complete.obs")
    if use not in valid_use:
        raise ValueError(
            f"use must be one of: {', '.join(repr(u) for u in valid_use)}"
        )
    _use_complete = _use_normalised == "complete"

    data, questions, _ = _resolve_survey(survey, emp_id_col)
    response_data = data[questions].copy()

    # For "complete"/"complete.obs": drop any row with ANY missing value across
    # all question columns first (mirrors R's use="complete.obs").
    if _use_complete:
        response_data = response_data.dropna()

    if format == "matrix":
        if plot:
            warnings.warn(
                "plot=True is not supported when format='matrix'. "
                "Returning matrix without plot.",
                UserWarning,
                stacklevel=2,
            )
        # min_periods=1 for pairwise; for complete, NAs are already dropped
        min_p = None if _use_complete else 1
        return response_data.corr(method=method, min_periods=min_p)

    # Long format — compute correlation and p-value per pair individually
    from scipy import stats as scipy_stats

    rows = []
    for q1 in questions:
        for q2 in questions:
            x = response_data[q1]
            y = response_data[q2]

            # For pairwise: use complete pairs per pair (mirrors R's
            # "pairwise.complete.obs"). For complete: data is already filtered.
            mask = x.notna() & y.notna()
            n_complete = int(mask.sum())
            x_clean = x[mask].values
            y_clean = y[mask].values

            if n_complete > 0:
                if method == "pearson":
                    corr_val = float(np.corrcoef(x_clean, y_clean)[0, 1])
                elif method == "spearman":
                    corr_val, _ = scipy_stats.spearmanr(x_clean, y_clean)
                    corr_val = float(corr_val)
                else:  # kendall
                    corr_val, _ = scipy_stats.kendalltau(x_clean, y_clean)
                    corr_val = float(corr_val)
            else:
                corr_val = float("nan")

            # P-value: mirrors R's cor.test(..., exact=FALSE)
            if q1 == q2:
                # Self-correlation always has p_value = 0 (mirrors R source comment)
                p_value = 0.0
            elif n_complete > 2:
                if method == "pearson":
                    _, p_value = scipy_stats.pearsonr(x_clean, y_clean)
                elif method == "spearman":
                    _, p_value = scipy_stats.spearmanr(x_clean, y_clean)
                else:  # kendall
                    _, p_value = scipy_stats.kendalltau(x_clean, y_clean)
                p_value = float(p_value)
            else:
                p_value = float("nan")

            rows.append(
                {
                    "question1": q1,
                    "question2": q2,
                    "correlation": corr_val,
                    "p_value": p_value,
                    "n": n_complete,
                }
            )

    long_data = pd.DataFrame(rows)[
        ["question1", "question2", "correlation", "p_value", "n"]
    ]

    if plot:
        from vivaglint.plots import _plot_correlations
        fig = _plot_correlations(long_data)
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass

    return long_data


# ---------------------------------------------------------------------------
# extract_survey_factors
# ---------------------------------------------------------------------------

def extract_survey_factors(
    survey: Union[GlintSurvey, pd.DataFrame],
    n_factors: Optional[int] = None,
    rotation: str = "oblimin",
    min_loading: float = 0.3,
    fm: str = "minres",
    plot: bool = False,
) -> dict:
    """Perform exploratory factor analysis on survey question responses.

    Uses a pure numpy/scipy EFA implementation (equivalent to R's
    ``psych::fa``). Supports the same rotation options and factoring methods
    as the R source.

    Ported from ``R/analyze.R::extract_survey_factors``.

    Loading labels mirror R's case_when thresholds exactly:
    - ``"Strong"`` when ``abs(loading) >= 0.75``
    - ``"Medium"`` when ``abs(loading) >= 0.60`` (and < 0.75)
    - ``"Weak"`` when ``abs(loading) < 0.60``

    .. note::
        ``rotation="oblimin"`` and ``rotation="quartimin"`` are currently
        known to produce degenerate results (Bug 23).  Use ``"varimax"``
        (default, orthogonal) or ``"promax"`` (oblique, stable) instead.

    Parameters
    ----------
    survey:
        A :class:`~vivaglint.import_.GlintSurvey` or plain DataFrame.
    n_factors:
        Number of factors to extract. If ``None`` (default), uses a small
        parallel analysis simulation to estimate the optimal number of factors.
    rotation:
        ``"oblimin"`` (default), ``"varimax"``, ``"promax"``,
        ``"quartimin"``, ``"quartimax"``, ``"equamax"``, or ``"none"``.
    min_loading:
        Minimum absolute loading to include in ``factor_summary``
        (default 0.3, matching R).
    fm:
        Factoring method: ``"minres"`` (default), ``"ml"``, ``"pa"``,
        ``"wls"``, ``"gls"``, or ``"uls"``.
    plot:
        If ``True``, display a factor loading dot-plot.

    Returns
    -------
    dict
        * ``"factor_summary"`` — :class:`pd.DataFrame` filtered to
          ``abs(loading) >= min_loading``. Columns: ``question``, ``factor``,
          ``loading``, ``loading_label``, ``communality``,
          ``factor_variance_pct``. Sorted by factor asc, then loading desc.
        * ``"fa_object"`` — the fitted
          :class:`~vivaglint._factor_analysis._VivaGlintFA` instance.

    Raises
    ------
    ValueError
        On invalid rotation/fm or n_factors out of range.
    """
    from vivaglint._factor_analysis import _VivaGlintFA

    valid_rotations = ("oblimin", "varimax", "promax", "quartimin", "quartimax", "equamax", "none")
    if rotation not in valid_rotations:
        raise ValueError(
            f"rotation must be one of: {', '.join(repr(r) for r in valid_rotations)}"
        )

    valid_fm = ("minres", "ml", "pa", "wls", "gls", "uls")
    if fm not in valid_fm:
        raise ValueError(
            f"fm must be one of: {', '.join(repr(f) for f in valid_fm)}"
        )

    data, questions, _ = _resolve_survey(survey)
    response_data = data[questions].copy()

    # Remove rows where ALL values are missing (mirrors R's rowSums(!is.na()) > 0)
    response_data = response_data[response_data.notna().any(axis=1)]
    if response_data.shape[0] < 2:
        raise ValueError(
            "extract_survey_factors() requires at least 2 respondents with "
            "non-missing responses, but fewer were found after removing "
            "rows with no question responses."
        )

    if n_factors is None:
        complete_rows = response_data.dropna().shape[0]
        if complete_rows < 2:
            raise ValueError(
                "extract_survey_factors() requires at least 2 complete rows "
                "of question responses for parallel analysis."
            )
        logger.info(
            "Determining optimal number of factors using parallel analysis..."
        )
        n_factors = _parallel_analysis(response_data, n_iter=20)
        logger.info("Parallel analysis suggests %d factor(s)", n_factors)

    if n_factors < 1 or n_factors > len(questions):
        raise ValueError(
            f"n_factors must be between 1 and {len(questions)} (number of questions)"
        )

    # rotation="none" -> None for _VivaGlintFA
    fa_rotation = None if rotation == "none" else rotation

    fa = _VivaGlintFA(
        n_factors=n_factors,
        rotation=fa_rotation,
        method=fm,
    )
    fa.fit(response_data)

    loadings_matrix = fa.loadings_          # shape (n_questions, n_factors)
    factor_names = [f"MR{i + 1}" for i in range(n_factors)]
    communalities = fa.get_communalities()   # array (n_questions,)

    # get_factor_variance() returns (SS loadings, proportion var, cumulative var)
    variance_info = fa.get_factor_variance()
    factor_variance_pct = variance_info[1] * 100  # proportion -> percentage

    rows = []
    for q_idx, question in enumerate(questions):
        communality = float(communalities[q_idx])
        for f_idx, factor_name in enumerate(factor_names):
            loading = float(loadings_matrix[q_idx, f_idx])
            var_pct = float(factor_variance_pct[f_idx])

            abs_l = abs(loading)
            # Mirrors R's case_when exactly:
            #   abs(loading) >= 0.75 ~ "Strong"
            #   abs(loading) <  0.60 ~ "Weak"
            #   TRUE                 ~ "Medium"
            if abs_l >= 0.75:
                loading_label = "Strong"
            elif abs_l < 0.60:
                loading_label = "Weak"
            else:
                loading_label = "Medium"

            rows.append(
                {
                    "question": question,
                    "factor": factor_name,
                    "loading": loading,
                    "loading_label": loading_label,
                    "communality": communality,
                    "factor_variance_pct": var_pct,
                }
            )

    factor_summary = pd.DataFrame(rows)

    # Apply min_loading filter (mirrors R: filter(abs(loading) >= min_loading))
    factor_summary = factor_summary[
        factor_summary["loading"].abs() >= min_loading
    ].copy()

    # Sort: factor asc, then descending abs(loading) — mirrors R's
    # arrange(factor, desc(abs(loading)))
    factor_summary["_abs_loading"] = factor_summary["loading"].abs()
    factor_summary = (
        factor_summary.sort_values(
            ["factor", "_abs_loading"], ascending=[True, False]
        )
        .drop(columns=["_abs_loading"])
        .reset_index(drop=True)
    )

    # Final column order matches R's select()
    factor_summary = factor_summary[
        ["question", "factor", "loading", "loading_label", "communality",
         "factor_variance_pct"]
    ]

    Vaccount = pd.DataFrame(
        [variance_info[0], variance_info[1], variance_info[2]],
        index=["SS loadings", "Proportion Var", "Cumulative Var"],
        columns=factor_names,
    )

    result = {
        "factor_summary": factor_summary,
        "fa_object": fa,
        "Vaccount": Vaccount,
    }

    if plot:
        from vivaglint.plots import _plot_survey_factors
        fig = _plot_survey_factors(factor_summary)
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# analyze_by_attributes
# ---------------------------------------------------------------------------

def analyze_by_attributes(
    survey: Union[GlintSurvey, pd.DataFrame],
    attribute_file: Optional[Union[str, Path]] = None,
    scale_points: Optional[int] = None,
    attribute_cols: Optional[Union[str, list[str]]] = None,
    emp_id_col: Optional[str] = None,
    min_group_size: int = 5,
    plot: bool = False,
) -> pd.DataFrame:
    """Aggregate survey responses by employee attribute segments.

    Calculates the same metrics as :func:`summarize_survey` for each
    combination of attribute-group values. Groups smaller than *min_group_size*
    are suppressed.

    Ported from ``R/analyze.R::analyze_by_attributes``.

    Group suppression mirrors R's ``dplyr::filter(group_size >= min_group_size)``.
    Question column detection excludes ALL joined attribute columns (not just
    the ones used for grouping in the current call), matching the R comment:
    "excluding standard columns AND ALL joined attribute columns".

    Parameters
    ----------
    survey:
        A :class:`~vivaglint.import_.GlintSurvey` or plain DataFrame. If
        attributes have already been joined via ``join_attributes()``,
        *attribute_file* may be omitted.
    attribute_file:
        Optional path to a CSV file containing employee attributes to join.
        If ``None``, the survey must already contain the columns in
        *attribute_cols*.
    scale_points:
        Number of scale points (2-11).
    attribute_cols:
        Column name or list of column names to group by (e.g. ``"Department"``
        or ``["Department", "Gender"]``).
    emp_id_col:
        Employee ID column name.
    min_group_size:
        Minimum group size for inclusion (default 5).
    plot:
        If ``True``, display a faceted dot plot of Glint Scores by attribute.

    Returns
    -------
    pd.DataFrame
        One row per attribute-group-question combination. Columns: attribute
        cols, ``group_size``, then the same columns as :func:`summarize_survey`.
        Empty DataFrame if no groups meet the size threshold.

    Raises
    ------
    ValueError
        On missing required arguments or columns not found.
    """
    if scale_points is None:
        raise ValueError("scale_points must be specified")
    if scale_points not in range(2, 12):
        raise ValueError("scale_points must be an integer between 2 and 11")
    if attribute_cols is None:
        raise ValueError("attribute_cols must be specified")

    if isinstance(attribute_cols, str):
        attribute_cols = [attribute_cols]

    if isinstance(survey, GlintSurvey):
        emp_id_col = emp_id_col or survey.metadata.get("emp_id_col")

    if emp_id_col is None:
        raise ValueError(
            "emp_id_col must be specified when survey is a plain DataFrame"
        )

    # Join attributes if a file was provided.
    # Use deepcopy() to avoid mutating the caller's GlintSurvey in-place.
    # copy.copy() is not sufficient: it creates a shallow copy that shares the
    # metadata dict, so join_attributes would still modify the original
    # survey.metadata["attribute_cols"] (Bug 2 partial fix — deepcopy required).
    if attribute_file is not None:
        import copy
        from vivaglint.import_ import join_attributes
        survey = join_attributes(copy.deepcopy(survey), attribute_file, emp_id_col=emp_id_col)

    if isinstance(survey, GlintSurvey):
        survey_data = survey.data
        # Use ALL known attribute columns so question detection excludes them all
        all_attr_cols: list[str] = survey.metadata.get("attribute_cols") or attribute_cols
    else:
        survey_data = survey
        all_attr_cols = attribute_cols

    # Validate requested attribute columns exist
    missing_cols = [c for c in attribute_cols if c not in survey_data.columns]
    if missing_cols:
        raise ValueError(
            "Attribute column(s) not found in survey data: "
            + ", ".join(missing_cols)
            + "\nDid you forget to supply attribute_file or call join_attributes() first?"
        )

    standard_cols = get_standard_columns(emp_id_col)
    excluded = set(standard_cols) | set(all_attr_cols)
    question_cols = [c for c in survey_data.columns if c not in excluded]

    # Build attribute group table (distinct employee-group rows → count)
    group_df = (
        survey_data[attribute_cols + [emp_id_col]]
        .drop_duplicates()
        .groupby(attribute_cols, as_index=False)
        .size()
        .rename(columns={"size": "group_size"})
    )
    # Group suppression: filter(group_size >= min_group_size)
    group_df = group_df[group_df["group_size"] >= min_group_size].reset_index(drop=True)

    if len(group_df) == 0:
        warnings.warn(
            "No attribute groups meet the minimum size threshold",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame()

    results_frames = []
    for _, group_row in group_df.iterrows():
        # Filter survey_data to this group
        mask = pd.Series([True] * len(survey_data), index=survey_data.index)
        for col in attribute_cols:
            mask &= survey_data[col] == group_row[col]

        # Build a slim df with standard + question cols for summarize_survey
        available_standard = [c for c in standard_cols if c in survey_data.columns]
        available_question = [c for c in question_cols if c in survey_data.columns]
        slim_df = survey_data.loc[mask, available_standard + available_question].copy()

        group_summary = summarize_survey(
            slim_df,
            scale_points=scale_points,
            questions="all",
            emp_id_col=emp_id_col,
        )

        # Prepend attribute values and group_size (mirrors R's bind_cols)
        for i, col in enumerate(attribute_cols):
            group_summary.insert(i, col, group_row[col])
        group_summary.insert(
            len(attribute_cols), "group_size", int(group_row["group_size"])
        )

        results_frames.append(group_summary)

    results = pd.concat(results_frames, ignore_index=True)

    if plot:
        from vivaglint.plots import _plot_by_attributes
        fig = _plot_by_attributes(results, attribute_cols)
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# analyze_attrition
# ---------------------------------------------------------------------------

def analyze_attrition(
    survey: Union[GlintSurvey, pd.DataFrame],
    attrition_file: Union[str, Path, pd.DataFrame],
    emp_id_col: Optional[str] = None,
    term_date_col: Optional[str] = None,
    scale_points: Optional[int] = None,
    time_periods: Optional[list[int]] = None,
    attribute_cols: Optional[Union[str, list[str]]] = None,
    min_group_size: int = 5,
    plot: bool = False,
) -> pd.DataFrame:
    """Analyze the relationship between survey responses and employee attrition.

    Calculates how much more (or less) likely employees are to leave within
    specified time periods based on whether they responded favorably or
    unfavorably to survey questions.

    Ported from ``R/analyze.R::analyze_attrition``.

    The attrition ratio is ``unfavorable_attrition / favorable_attrition``,
    matching the R source exactly:
    - ``inf`` when favorable_attrition == 0 and unfavorable_attrition > 0
    - ``NaN`` when both are 0 or either is missing

    Parameters
    ----------
    survey:
        A :class:`~vivaglint.import_.GlintSurvey` or plain DataFrame.
    attrition_file:
        Path to a CSV, a file-like CSV object, or a pandas DataFrame with at
        minimum an employee ID column and a termination date column.
    emp_id_col:
        Employee ID column name.
    term_date_col:
        Name of the termination date column in *attrition_file*.
    scale_points:
        Number of scale points (2-11).
    time_periods:
        List of time periods in days (default ``[90, 180, 365]``).
    attribute_cols:
        Optional column name(s) to segment results by. When ``None``,
        results cover the whole population.
    min_group_size:
        Minimum group size for attribute segments (default 5).
    plot:
        If ``True``, display a grouped bar chart of attrition ratios.

    Returns
    -------
    pd.DataFrame
        One row per (attribute group)-question-time period. Columns:
        (optional attribute cols), (optional ``group_size``), ``question``,
        ``days``, ``favorable_n``, ``favorable_attrition``, ``unfavorable_n``,
        ``unfavorable_attrition``, ``attrition_ratio``.

    Raises
    ------
    FileNotFoundError
        If *attrition_file* does not exist.
    ValueError
        On missing arguments or column validation failures.
    """
    if time_periods is None:
        time_periods = [90, 180, 365]

    if scale_points is None:
        raise ValueError("scale_points must be specified")
    if scale_points not in range(2, 12):
        raise ValueError("scale_points must be an integer between 2 and 11")
    if term_date_col is None:
        raise ValueError("term_date_col must be specified")

    favorability = get_favorability_map(scale_points)
    fav_set = set(favorability["favorable"])
    unf_set = set(favorability["unfavorable"])

    data, questions, resolved_emp_id_col = _resolve_survey(survey, emp_id_col)
    emp_id_col = resolved_emp_id_col

    if emp_id_col is None:
        raise ValueError(
            "emp_id_col must be specified when survey is a plain DataFrame"
        )

    if isinstance(attrition_file, pd.DataFrame):
        attrition_data = attrition_file.copy()
    else:
        attrition_path = Path(attrition_file)
        if not attrition_path.exists():
            raise FileNotFoundError(f"Attrition file not found: '{attrition_file}'")

        attrition_data = pd.read_csv(attrition_path)

    if emp_id_col not in attrition_data.columns:
        raise ValueError(f"Column '{emp_id_col}' not found in attrition file")
    if term_date_col not in attrition_data.columns:
        raise ValueError(f"Column '{term_date_col}' not found in attrition file")
    if emp_id_col not in data.columns:
        raise ValueError(f"Column '{emp_id_col}' not found in survey data")

    # Parse termination dates — mirrors R's lubridate::parse_date_time with
    # multiple order formats (ymd, mdy, dmy, ...)
    try:
        attrition_data[term_date_col] = pd.to_datetime(
            attrition_data[term_date_col]
        ).dt.date
    except Exception as exc:
        raise ValueError(
            f"Error parsing termination dates in column '{term_date_col}': {exc}"
        ) from exc

    if "Survey Cycle Completion Date" not in data.columns:
        raise ValueError(
            "Survey data must contain 'Survey Cycle Completion Date' column"
        )

    # Left-join attrition to survey (mirrors R's dplyr::left_join)
    combined_data = data.merge(
        attrition_data[[emp_id_col, term_date_col]],
        on=emp_id_col,
        how="left",
    )

    combined_data["_survey_date"] = pd.to_datetime(
        combined_data["Survey Cycle Completion Date"]
    ).dt.date

    combined_data["_days_to_term"] = combined_data.apply(
        lambda row: (row[term_date_col] - row["_survey_date"]).days
        if pd.notna(row[term_date_col]) and pd.notna(row["_survey_date"])
        else float("nan"),
        axis=1,
    )

    if attribute_cols is not None and isinstance(attribute_cols, str):
        attribute_cols = [attribute_cols]

    if attribute_cols is not None:
        missing_attr = [c for c in attribute_cols if c not in combined_data.columns]
        if missing_attr:
            raise ValueError(
                "Attribute column(s) not found in survey data: "
                + ", ".join(missing_attr)
                + "\nDid you forget to supply attribute_file or call join_attributes() first?"
            )

    def _run_attrition_core(data_slice: pd.DataFrame) -> pd.DataFrame:
        """Core logic: compute attrition metrics for all questions x time periods.

        Mirrors R's run_attrition_core() inner function exactly.
        """
        core_rows = []
        for question_text in questions:
            if question_text not in data_slice.columns:
                continue
            responses = data_slice[question_text]

            # Classify responses (mirrors R's dplyr::case_when)
            resp_class = responses.map(
                lambda r: "favorable"
                if r in fav_set
                else ("unfavorable" if r in unf_set else "neutral")
            )

            for days in time_periods:
                # "left within period" means days_to_term in (0, days]
                left_within = (
                    data_slice["_days_to_term"].notna()
                    & (data_slice["_days_to_term"] > 0)
                    & (data_slice["_days_to_term"] <= days)
                )

                fav_mask = (resp_class == "favorable") & responses.notna()
                favorable_n = int(fav_mask.sum())
                if favorable_n > 0:
                    favorable_attrition = round(
                        float(left_within[fav_mask].sum()) / favorable_n, 4
                    )
                else:
                    favorable_attrition = float("nan")

                unf_mask = (resp_class == "unfavorable") & responses.notna()
                unfavorable_n = int(unf_mask.sum())
                if unfavorable_n > 0:
                    unfavorable_attrition = round(
                        float(left_within[unf_mask].sum()) / unfavorable_n, 4
                    )
                else:
                    unfavorable_attrition = float("nan")

                # Attrition ratio — mirrors R's conditional logic exactly
                if math.isnan(favorable_attrition) or math.isnan(unfavorable_attrition):
                    attrition_ratio: float = float("nan")
                elif favorable_attrition == 0 and unfavorable_attrition == 0:
                    attrition_ratio = float("nan")
                elif favorable_attrition == 0 and unfavorable_attrition > 0:
                    attrition_ratio = float("inf")
                else:
                    attrition_ratio = round(
                        unfavorable_attrition / favorable_attrition, 2
                    )

                core_rows.append(
                    {
                        "question": question_text,
                        "days": days,
                        "favorable_n": favorable_n,
                        "favorable_attrition": favorable_attrition,
                        "unfavorable_n": unfavorable_n,
                        "unfavorable_attrition": unfavorable_attrition,
                        "attrition_ratio": attrition_ratio,
                    }
                )
        return pd.DataFrame(core_rows)

    # Overall analysis (no attribute segmentation)
    if attribute_cols is None:
        results = _run_attrition_core(combined_data)
        if plot:
            from vivaglint.plots import _plot_attrition
            fig = _plot_attrition(results, None)
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except Exception:
                pass
        return results

    # Attribute-segmented analysis
    group_df = (
        combined_data[attribute_cols + [emp_id_col]]
        .drop_duplicates()
        .groupby(attribute_cols, as_index=False)
        .size()
        .rename(columns={"size": "group_size"})
    )
    group_df = group_df[
        group_df["group_size"] >= min_group_size
    ].reset_index(drop=True)

    if len(group_df) == 0:
        warnings.warn(
            "No attribute groups meet the minimum size threshold",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame()

    results_frames = []
    for _, group_row in group_df.iterrows():
        mask = pd.Series([True] * len(combined_data), index=combined_data.index)
        for col in attribute_cols:
            mask &= combined_data[col] == group_row[col]

        group_data = combined_data[mask]
        core = _run_attrition_core(group_data)

        # Prepend attribute values and group_size (mirrors R's dplyr::bind_cols)
        for i, col in enumerate(attribute_cols):
            core.insert(i, col, group_row[col])
        core.insert(len(attribute_cols), "group_size", int(group_row["group_size"]))
        results_frames.append(core)

    results = pd.concat(results_frames, ignore_index=True)

    if plot:
        from vivaglint.plots import _plot_attrition
        fig = _plot_attrition(results, attribute_cols)
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# search_comments
# ---------------------------------------------------------------------------

def search_comments(
    survey: Union[GlintSurvey, pd.DataFrame],
    query: str,
    exact: bool = False,
    max_distance: float = 0.2,
) -> pd.DataFrame:
    """Search through survey comment text and return matching responses.

    Supports exact substring matching (case-sensitive) and fuzzy approximate
    matching using ``rapidfuzz`` (replacing R's ``agrep``).

    Ported from ``R/analyze.R::search_comments``.

    Matching strategy mirrors R exactly:
    - Exact: ``grepl(query, comments, fixed=TRUE)`` — case-sensitive literal.
    - Fuzzy: ``grepl(query, ignore.case=TRUE)`` (partial match) OR
      ``agrep(max.distance=max_distance, ignore.case=TRUE)`` (approx match).
      Both conditions are OR'd together, matching R's ``partial_match | fuzzy_match``.

    Parameters
    ----------
    survey:
        A :class:`~vivaglint.import_.GlintSurvey` or plain DataFrame.
    query:
        Non-empty string to search for within comment text.
    exact:
        If ``True``, case-sensitive literal substring match.
        If ``False`` (default), case-insensitive approximate matching.
    max_distance:
        Fuzzy tolerance (0-1). Higher values allow more differences.
        Mirrors R's ``agrep(max.distance=...)``. Default ``0.2``.

    Returns
    -------
    pd.DataFrame
        One row per matching comment. Columns: ``question``, ``response``,
        ``comment``, ``topics``. Empty DataFrame (same schema) if no matches.

    Raises
    ------
    ValueError
        If *query* is empty or *max_distance* is out of range.
    TypeError
        If *exact* is not bool.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty character string")
    if not isinstance(exact, bool):
        raise TypeError("exact must be True or False")
    if not isinstance(max_distance, (int, float)) or not (0 <= max_distance <= 1):
        raise ValueError("max_distance must be a number between 0 and 1")

    _empty = pd.DataFrame(
        columns=["question", "response", "comment", "topics"]
    ).astype({"response": "Float64", "comment": str, "topics": str})

    data, questions, _ = _resolve_survey(survey)

    _rapidfuzz_available = False
    if not exact:
        try:
            import rapidfuzz.fuzz as _rf_fuzz  # noqa: F401
            _rapidfuzz_available = True
        except ImportError:
            warnings.warn(
                "rapidfuzz is not installed; falling back to case-insensitive "
                "substring matching only (no approximate matching). "
                "Install with: pip install rapidfuzz",
                ImportWarning,
                stacklevel=2,
            )

    rows = []
    for question_text in questions:
        comment_col = f"{question_text}_COMMENT"
        topics_col = f"{question_text}_COMMENT_TOPICS"

        if comment_col not in data.columns:
            continue

        raw_comments = data[comment_col]
        responses = (
            data[question_text]
            if question_text in data.columns
            else pd.Series([float("nan")] * len(data), index=data.index)
        )
        topics_series = (
            data[topics_col]
            if topics_col in data.columns
            else pd.Series([None] * len(data), index=data.index)
        )

        # Rows with a non-empty comment (mirrors R's nchar(trimws(comments)) > 0)
        str_comments = raw_comments.astype(str)
        has_comment = (
            raw_comments.notna()
            & (str_comments.str.strip() != "")
            & (str_comments != "nan")
        )

        if not has_comment.any():
            continue

        if exact:
            # Case-sensitive literal substring match (R: grepl(fixed=TRUE))
            matched = has_comment & str_comments.str.contains(
                query, regex=False, na=False
            )
        else:
            # Case-insensitive partial (R: grepl(ignore.case=TRUE))
            partial_match = has_comment & str_comments.str.contains(
                query, case=False, regex=False, na=False
            )

            # Approximate match via rapidfuzz (R: agrep with max.distance)
            fuzzy_match = pd.Series(
                [False] * len(data), index=data.index, dtype=bool
            )
            if _rapidfuzz_available:
                import rapidfuzz.fuzz as rf_fuzz

                # R uses max.distance as an edit-distance ratio (0-1).
                # rapidfuzz partial_ratio returns 0-100 similarity.
                # Convert: score >= (1 - max_distance) * 100
                threshold = (1.0 - max_distance) * 100
                query_lower = query.lower()

                candidate_idx = has_comment[has_comment].index
                for idx in candidate_idx:
                    score = rf_fuzz.partial_ratio(
                        query_lower, str_comments[idx].lower()
                    )
                    if score >= threshold:
                        fuzzy_match[idx] = True

            # OR the two conditions (mirrors R: partial_match | fuzzy_match)
            matched = partial_match | fuzzy_match

        if not matched.any():
            continue

        for idx in data[matched].index:
            rows.append(
                {
                    "question": question_text,
                    "response": responses[idx],
                    "comment": str_comments[idx],
                    "topics": topics_series[idx],
                }
            )

    if not rows:
        return _empty

    return pd.DataFrame(rows)[["question", "response", "comment", "topics"]]
