"""Smoke tests for the analysis / reshape / hierarchy functions."""

from __future__ import annotations

import pandas as pd
import pytest

from vivaglint import (
    aggregate_by_manager,
    analyze_attrition,
    analyze_by_attributes,
    compare_cycles,
    get_correlations,
    get_response_dist,
    join_attributes,
    pivot_long,
    search_comments,
    split_survey_data,
    summarize_survey,
)


def test_summarize_survey(survey):
    out = summarize_survey(survey, scale_points=5)
    assert set(["question", "mean", "sd", "glint_score", "n_responses"]).issubset(
        out.columns
    )
    assert len(out) == 3
    # glint_score respects the R formula for a known column mean
    assert out["glint_score"].notna().all()


def test_summarize_survey_bad_scale(survey):
    with pytest.raises(ValueError):
        summarize_survey(survey, scale_points=99)


def test_get_response_dist(survey):
    out = get_response_dist(survey)
    assert "question" in out.columns
    assert any(c.startswith("count_") for c in out.columns)
    assert any(c.startswith("pct_") for c in out.columns)


def test_pivot_long_all(survey):
    out = pivot_long(survey, data_type="all")
    assert {"question", "response", "comment"}.issubset(out.columns)
    # 12 respondents x 3 questions
    assert len(out) == 36


def test_pivot_long_comments(survey):
    out = pivot_long(survey, data_type="comments")
    assert (out["comment"].fillna("") != "").all()


def test_split_survey_data(survey):
    parts = split_survey_data(survey)
    assert "quantitative" in parts and "qualitative" in parts
    assert "Employee ID" in parts["qualitative"].columns


def test_get_correlations_long(survey):
    out = get_correlations(survey)
    assert {"question1", "question2", "correlation", "p_value", "n"}.issubset(
        out.columns
    )


def test_get_correlations_matrix(survey):
    out = get_correlations(survey, format="matrix")
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == out.shape[1] == 3


def test_compare_cycles(survey):
    out = compare_cycles(survey, survey, scale_points=5, cycle_names=["Q1", "Q2"])
    assert "cycle" in out.columns
    assert "change_from_previous" in out.columns


def test_aggregate_by_manager(survey):
    out = aggregate_by_manager(survey, scale_points=5)
    assert "manager_id" in out.columns
    assert "team_size" in out.columns


def test_analyze_by_attributes(survey, attributes_path):
    enriched = join_attributes(survey, attributes_path)
    out = analyze_by_attributes(
        enriched, scale_points=5, attribute_cols="Department", min_group_size=2
    )
    assert "Department" in out.columns
    assert "group_size" in out.columns


def test_analyze_attrition(survey, attrition_path):
    out = analyze_attrition(
        survey,
        attrition_file=str(attrition_path),
        emp_id_col="Employee ID",
        term_date_col="Termination Date",
        scale_points=5,
    )
    assert {"question", "days", "favorable_n", "unfavorable_n"}.issubset(out.columns)


def test_search_comments_fuzzy(survey):
    # "flexibility" appears verbatim and as a typo ("flexability"); fuzzy search
    # should catch both.
    out = search_comments(survey, "flexibility")
    assert len(out) >= 3
    assert {"question", "response", "comment", "topics"}.issubset(out.columns)


def test_search_comments_exact(survey):
    out = search_comments(survey, "flexibility", exact=True)
    # Exact, case-sensitive: only literal "flexibility"
    assert (out["comment"].str.contains("flexibility")).all()
