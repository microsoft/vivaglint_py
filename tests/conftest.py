"""Shared pytest fixtures for the vivaglint test suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vivaglint import read_glint_survey

FIXTURES = Path(__file__).parent / "fixtures"
SURVEY_CSV = FIXTURES / "sample_survey.csv"
ATTRIBUTES_CSV = FIXTURES / "sample_attributes.csv"
ATTRITION_CSV = FIXTURES / "sample_attrition.csv"
EMP_ID_COL = "Employee ID"


@pytest.fixture
def survey_path() -> Path:
    return SURVEY_CSV


@pytest.fixture
def attributes_path() -> Path:
    return ATTRIBUTES_CSV


@pytest.fixture
def attrition_path() -> Path:
    return ATTRITION_CSV


@pytest.fixture
def survey():
    """A loaded GlintSurvey from the sample CSV."""
    return read_glint_survey(SURVEY_CSV, emp_id_col=EMP_ID_COL)


@pytest.fixture
def survey_df() -> pd.DataFrame:
    """The raw sample survey as a DataFrame."""
    return pd.read_csv(SURVEY_CSV)
