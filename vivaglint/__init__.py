"""
vivaglint — Python port of the Microsoft Viva Glint R package.

Provides functions for importing, validating, and analyzing Viva Glint
survey data exports. All data is processed locally and no data is
transmitted to any external service.

Quick start
-----------
>>> from vivaglint import read_glint_survey, summarize_survey
>>> survey = read_glint_survey("survey_export.csv", emp_id_col="Employee ID")
>>> summary = summarize_survey(survey, scale_points=5)
"""

from vivaglint.import_ import (
    read_glint_survey,
    extract_questions,
    join_attributes,
)

from vivaglint.api import (
    glint_setup,
    read_glint_survey_api,
)

from vivaglint.reshape import (
    pivot_long,
    split_survey_data,
)

from vivaglint.analyze import (
    summarize_survey,
    get_response_dist,
    compare_cycles,
    get_correlations,
    extract_survey_factors,
    analyze_by_attributes,
    analyze_attrition,
    search_comments,
)

from vivaglint.hierarchy import (
    aggregate_by_manager,
)

__version__ = "0.1.1"

__all__ = [
    # import
    "read_glint_survey",
    "extract_questions",
    "join_attributes",
    # api
    "glint_setup",
    "read_glint_survey_api",
    # reshape
    "pivot_long",
    "split_survey_data",
    # analyze
    "summarize_survey",
    "get_response_dist",
    "compare_cycles",
    "get_correlations",
    "extract_survey_factors",
    "analyze_by_attributes",
    "analyze_attrition",
    "search_comments",
    # hierarchy
    "aggregate_by_manager",
]
