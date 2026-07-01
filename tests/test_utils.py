"""Tests for vivaglint.utils helpers."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from vivaglint.utils import (
    get_favorability_map,
    get_question_stem,
    get_standard_columns,
    mean_to_glint_score,
    parse_comment_topics,
)


def test_get_standard_columns_includes_emp_id():
    cols = get_standard_columns("Employee ID")
    assert "Employee ID" in cols
    assert cols[0] == "First Name"
    assert "Manager ID" in cols


@pytest.mark.parametrize(
    "col,expected",
    [
        ("My work is meaningful", "My work is meaningful"),
        ("My work is meaningful_COMMENT", "My work is meaningful"),
        ("My work is meaningful_COMMENT_TOPICS", "My work is meaningful"),
        ("My work is meaningful_SENSITIVE_COMMENT_FLAG", "My work is meaningful"),
    ],
)
def test_get_question_stem(col, expected):
    assert get_question_stem(col) == expected


def test_mean_to_glint_score_matches_r_formula():
    # round(((mean - 1) / (scale - 1)) * 100)
    assert mean_to_glint_score(4.0, 5) == 75
    assert mean_to_glint_score(1.0, 5) == 0
    assert mean_to_glint_score(5.0, 5) == 100


def test_mean_to_glint_score_handles_nan():
    assert mean_to_glint_score(float("nan"), 5) is None
    assert mean_to_glint_score(None, 5) is None


def test_get_favorability_map_5point():
    fav = get_favorability_map(5)
    assert fav["favorable"] == [4, 5]
    assert fav["neutral"] == [3]
    assert fav["unfavorable"] == [1, 2]


def test_get_favorability_map_invalid():
    with pytest.raises(ValueError):
        get_favorability_map(99)


def test_parse_comment_topics_list():
    s = pd.Series(["A, B", "", None, "C"])
    out = parse_comment_topics(s, return_format="list")
    assert out == [["A", "B"], [], [], ["C"]]


def test_parse_comment_topics_tidy():
    s = pd.Series(["A, B", None, "C"])
    out = parse_comment_topics(s, return_format="tidy")
    assert list(out.columns) == ["index", "topic"]
    assert out["topic"].tolist() == ["A", "B", "C"]
