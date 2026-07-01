"""Tests for vivaglint.import_ (read/validate/build/extract/join)."""

from __future__ import annotations

import pandas as pd
import pytest

from vivaglint import extract_questions, join_attributes, read_glint_survey
from vivaglint.import_ import (
    GlintSurvey,
    build_glint_survey,
    validate_glint_structure,
)


def test_read_glint_survey_returns_glint_survey(survey):
    assert isinstance(survey, GlintSurvey)
    assert survey.metadata["emp_id_col"] == "Employee ID"
    assert survey.metadata["n_respondents"] == 12
    assert survey.metadata["n_questions"] == 3


def test_read_glint_survey_parses_dates(survey):
    assert pd.api.types.is_datetime64_any_dtype(
        survey.data["Survey Cycle Completion Date"]
    )


def test_read_glint_survey_missing_file():
    with pytest.raises(FileNotFoundError):
        read_glint_survey("does_not_exist.csv", emp_id_col="Employee ID")


def test_extract_questions(survey):
    q = extract_questions(survey)
    assert set(q["question"]) == {
        "My work is meaningful",
        "I feel valued",
        "I have work-life balance",
    }
    assert list(q.columns) == [
        "question",
        "response_col",
        "comment_col",
        "topics_col",
        "flag_col",
    ]


def test_build_glint_survey_without_sent_date():
    """API-style export: no 'Survey Cycle Sent Date' column.

    build_glint_survey with sent_date_col=None must succeed (only a warning,
    no error), mirroring the current R behaviour.
    """
    df = pd.DataFrame(
        {
            "Employment ID": ["A1", "A2"],
            "First Name": ["Ann", "Bob"],
            "Last Name": ["A", "B"],
            "Email": ["a@x.com", "b@x.com"],
            "Status": ["ACTIVE", "ACTIVE"],
            "Manager ID": ["M1", "M1"],
            "Survey Cycle Completion Date": ["26-03-2024 09:34", "26-03-2024 09:34"],
            "My work is meaningful": [4, 3],
            "My work is meaningful_COMMENT": ["Good", ""],
            "My work is meaningful_COMMENT_TOPICS": ["Culture", ""],
            "My work is meaningful_SENSITIVE_COMMENT_FLAG": ["", ""],
        }
    )
    survey = build_glint_survey(
        df, emp_id_col="Employment ID", sent_date_col=None
    )
    assert isinstance(survey, GlintSurvey)
    assert survey.metadata["n_questions"] == 1
    assert survey.metadata["standard_column_map"]["sent_date"] is None


def test_validate_requires_emp_id_column():
    df = pd.DataFrame({"Q": [1], "Q_COMMENT": [""], "Q_COMMENT_TOPICS": [""], "Q_SENSITIVE_COMMENT_FLAG": [""]})
    with pytest.raises(ValueError, match="Missing required standard column"):
        validate_glint_structure(df, emp_id_col="Employee ID")


def test_extra_metadata_columns_are_not_questions():
    """API exports include extra non-question metadata columns.

    Columns like 'Survey UUID' / 'Survey Cycle Title' have no _COMMENT etc.
    siblings and must NOT be treated as questions (regression: previously they
    tripped the 'incomplete question set' validation).
    """
    df = pd.DataFrame(
        {
            "Employment ID": ["A1", "A2"],
            "First Name": ["Ann", "Bob"],
            "Last Name": ["A", "B"],
            "Email": ["a@x.com", "b@x.com"],
            "Status": ["ACTIVE", "ACTIVE"],
            "Manager ID": ["M1", "M1"],
            "Survey Cycle Completion Date": ["26-03-2024 09:34", "26-03-2024 09:34"],
            # Extra metadata columns present in real API exports:
            "Survey UUID": ["u", "u"],
            "Survey Cycle UUID": ["c", "c"],
            "Survey Cycle Title": ["Pulse", "Pulse"],
            "Survey Cycle Creation Date": ["01-03-2024 00:00", "01-03-2024 00:00"],
            # One real question:
            "My work is meaningful": [4, 3],
            "My work is meaningful_COMMENT": ["Good", ""],
            "My work is meaningful_COMMENT_TOPICS": ["Culture", ""],
            "My work is meaningful_SENSITIVE_COMMENT_FLAG": ["", ""],
        }
    )
    survey = build_glint_survey(df, emp_id_col="Employment ID", sent_date_col=None)
    assert isinstance(survey, GlintSurvey)
    # Exactly one real question detected — metadata columns ignored.
    assert survey.metadata["n_questions"] == 1
    assert list(survey.metadata["questions"]["question"]) == ["My work is meaningful"]
    # Metadata columns remain available in the data.
    assert "Survey UUID" in survey.data.columns


def test_extract_questions_ignores_metadata_columns():
    df = pd.DataFrame(
        {
            "Employment ID": ["A1"],
            "Survey UUID": ["u"],
            "Q": [4],
            "Q_COMMENT": [""],
            "Q_COMMENT_TOPICS": [""],
            "Q_SENSITIVE_COMMENT_FLAG": [""],
        }
    )
    q = extract_questions(df, emp_id_col="Employment ID")
    assert list(q["question"]) == ["Q"]


def test_join_attributes(survey, attributes_path):
    enriched = join_attributes(survey, attributes_path)
    assert "Department" in enriched.data.columns
    assert set(enriched.metadata["attribute_cols"]) == {
        "Department",
        "Gender",
        "Tenure Group",
    }
