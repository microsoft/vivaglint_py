"""
vivaglint.import_
-----------------
Functions for reading and validating Viva Glint survey export CSV files.

Ported from R/import.R.  The module is named ``import_`` (with a trailing
underscore) to avoid shadowing Python's built-in ``import`` statement.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from vivaglint.utils import (
    get_standard_columns,
    get_question_stem,
    _QUESTION_SUFFIXES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class GlintSurvey:
    """Container for a loaded Viva Glint survey export.

    Mirrors the R ``glint_survey`` S3 object so that downstream functions can
    accept either a ``GlintSurvey`` or a plain ``pd.DataFrame``.

    Attributes
    ----------
    data : pd.DataFrame
        All survey responses with parsed date columns.
    metadata : dict
        Keys:
        - ``standard_columns`` (list[str])
        - ``emp_id_col`` (str)
        - ``questions`` (pd.DataFrame — one row per question)
        - ``n_respondents`` (int)
        - ``n_questions`` (int)
        - ``file_path`` (str)
        - ``attribute_cols`` (list[str], populated by ``join_attributes``)
    """

    def __init__(self, data: pd.DataFrame, metadata: dict) -> None:
        self.data = data
        self.metadata = metadata

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GlintSurvey("
            f"n_respondents={self.metadata.get('n_respondents')}, "
            f"n_questions={self.metadata.get('n_questions')})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_survey(
    survey: Union[GlintSurvey, pd.DataFrame],
    emp_id_col: Optional[str],
) -> tuple[pd.DataFrame, str]:
    """Return (data_frame, emp_id_col) from either a GlintSurvey or DataFrame."""
    if isinstance(survey, GlintSurvey):
        resolved_id_col = emp_id_col or survey.metadata.get("emp_id_col")
        return survey.data, resolved_id_col
    if isinstance(survey, pd.DataFrame):
        return survey, emp_id_col
    raise TypeError("survey must be a GlintSurvey object or a pandas DataFrame")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_glint_survey(
    file_path: Union[str, Path],
    emp_id_col: str,
    encoding: str = "UTF-8",
) -> GlintSurvey:
    """Read a Viva Glint survey export CSV file.

    Validates the file structure, parses date columns from the
    ``DD-MM-YYYY HH:MM`` format used by Glint, and returns a
    :class:`GlintSurvey` object ready for analysis.

    Ported from ``R/import.R::read_glint_survey``.

    Parameters
    ----------
    file_path:
        Path to the CSV export file.
    emp_id_col:
        Name of the employee ID column in this export (e.g. ``"Employee ID"``).
        Stored in ``survey.metadata["emp_id_col"]`` so downstream functions
        can resolve it automatically.
    encoding:
        File encoding (default ``"utf-8"``).

    Returns
    -------
    GlintSurvey
        Object with ``.data`` (DataFrame) and ``.metadata`` (dict).

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ValueError
        If the CSV cannot be read or fails structure validation.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: '{file_path}'\n"
            "Please check that the file path is correct."
        )

    try:
        data = pd.read_csv(path, encoding=encoding)
    except Exception as exc:
        raise ValueError(
            f"Error reading CSV file: {exc}\n"
            "Please ensure the file is a valid CSV format."
        ) from exc

    validate_glint_structure(data, emp_id_col)

    standard_cols = get_standard_columns(emp_id_col)

    # Parse date columns — Glint exports dates as "DD-MM-YYYY HH:MM"
    date_cols = ["Survey Cycle Completion Date", "Survey Cycle Sent Date"]
    for col in date_cols:
        if col in data.columns:
            try:
                data[col] = pd.to_datetime(data[col], format="%d-%m-%Y %H:%M")
            except Exception as exc:
                raise ValueError(
                    f"Error parsing date column '{col}': {exc}\n"
                    "Expected format: DD-MM-YYYY HH:MM (e.g. '26-03-2024 09:34')"
                ) from exc

    questions_df = extract_questions(data, emp_id_col)

    metadata: dict = {
        "standard_columns": standard_cols,
        "emp_id_col": emp_id_col,
        "questions": questions_df,
        "n_respondents": len(data),
        "n_questions": len(questions_df),
        "file_path": str(path),
        "attribute_cols": [],
    }

    logger.info(
        "Loaded survey: %d respondents, %d questions from '%s'",
        len(data),
        len(questions_df),
        path.name,
    )

    return GlintSurvey(data=data, metadata=metadata)


def extract_questions(
    data: Union[GlintSurvey, pd.DataFrame],
    emp_id_col: Optional[str] = None,
) -> pd.DataFrame:
    """Parse column names to extract survey questions.

    Returns a DataFrame with one row per question and columns describing
    which CSV columns carry that question's data.

    Ported from ``R/import.R::extract_questions``.

    Parameters
    ----------
    data:
        A :class:`GlintSurvey` or a plain DataFrame.
    emp_id_col:
        Required when *data* is a plain DataFrame.  When *data* is a
        :class:`GlintSurvey` the value is resolved from metadata automatically.

    Returns
    -------
    pd.DataFrame
        Columns: ``question``, ``response_col``, ``comment_col``,
        ``topics_col``, ``flag_col``.

    Raises
    ------
    ValueError
        If *emp_id_col* cannot be determined.
    """
    if isinstance(data, GlintSurvey):
        emp_id_col = data.metadata.get("emp_id_col") or emp_id_col
        df = data.data
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError("data must be a GlintSurvey or a pandas DataFrame")

    if emp_id_col is None:
        raise ValueError(
            "emp_id_col must be specified. When loading with read_glint_survey(), "
            "pass emp_id_col to store it automatically."
        )

    standard_cols = set(get_standard_columns(emp_id_col))
    question_cols = [c for c in df.columns if c not in standard_cols]
    question_stems = list(dict.fromkeys(get_question_stem(c) for c in question_cols))

    rows = []
    for stem in question_stems:
        rows.append(
            {
                "question": stem,
                "response_col": stem,
                "comment_col": f"{stem}_COMMENT",
                "topics_col": f"{stem}_COMMENT_TOPICS",
                "flag_col": f"{stem}_SENSITIVE_COMMENT_FLAG",
            }
        )

    return pd.DataFrame(rows, columns=["question", "response_col", "comment_col", "topics_col", "flag_col"])


def join_attributes(
    survey: Union[GlintSurvey, pd.DataFrame],
    attribute_source: Union[str, Path, pd.DataFrame],
    emp_id_col: Optional[str] = None,
) -> Union[GlintSurvey, pd.DataFrame]:
    """Join employee attribute data to a survey object.

    Reads attribute data from a CSV file or DataFrame and left-joins it to
    the survey by employee ID.  Returns an enriched :class:`GlintSurvey`
    (or plain DataFrame) ready for ``analyze_by_attributes()`` etc.

    Ported from ``R/import.R::join_attributes``.

    Parameters
    ----------
    survey:
        A :class:`GlintSurvey` or plain DataFrame.
    attribute_source:
        Either a file path to a CSV or an already-loaded DataFrame.
        All columns are coerced to ``str`` to avoid type conflicts during
        joining (mirrors R behaviour).
    emp_id_col:
        Employee ID column name.  Defaults to ``survey.metadata["emp_id_col"]``
        when *survey* is a :class:`GlintSurvey`.

    Returns
    -------
    GlintSurvey or pd.DataFrame
        Enriched object with attribute columns appended.  When the input is a
        :class:`GlintSurvey`, the names of all joined attribute columns are
        stored in ``survey.metadata["attribute_cols"]``.

    Raises
    ------
    TypeError / ValueError / FileNotFoundError
        On invalid inputs.
    """
    if isinstance(survey, GlintSurvey):
        emp_id_col = emp_id_col or survey.metadata.get("emp_id_col")
        data = survey.data.copy()
    elif isinstance(survey, pd.DataFrame):
        data = survey.copy()
    else:
        raise TypeError("survey must be a GlintSurvey object or a pandas DataFrame")

    if emp_id_col is None:
        raise ValueError(
            "emp_id_col must be specified when survey is a plain DataFrame "
            "(it cannot be resolved from metadata)"
        )

    if emp_id_col not in data.columns:
        raise ValueError(f"Column '{emp_id_col}' not found in survey data")

    # Load attribute source
    if isinstance(attribute_source, (str, Path)):
        attr_path = Path(attribute_source)
        if not attr_path.exists():
            raise FileNotFoundError(f"Attribute file not found: '{attribute_source}'")
        attributes = pd.read_csv(attr_path, dtype=str)
    elif isinstance(attribute_source, pd.DataFrame):
        attributes = attribute_source.copy().astype(str)
    else:
        raise TypeError(
            "attribute_source must be a file path (str/Path) or a DataFrame"
        )

    if emp_id_col not in attributes.columns:
        raise ValueError(f"Column '{emp_id_col}' not found in attribute data")

    # Coerce the attribute emp_id column to match the survey's dtype so that
    # the merge key aligns correctly (Bug 7).  Without this, a numeric survey
    # emp_id (e.g. int64) would never match the string values produced by
    # dtype=str, silently filling every attribute column with NaN.
    try:
        attributes[emp_id_col] = attributes[emp_id_col].astype(data[emp_id_col].dtype)
    except (ValueError, TypeError):
        pass  # leave as-is if coercion fails; merge will produce NaN with a log entry

    new_attr_cols = [c for c in attributes.columns if c != emp_id_col]

    # Warn about and remove overlapping columns
    overlap = [c for c in new_attr_cols if c in data.columns]
    if overlap:
        import warnings
        warnings.warn(
            "The following columns already exist in the survey data and will be "
            f"overwritten: {', '.join(repr(c) for c in overlap)}",
            UserWarning,
            stacklevel=2,
        )
        data = data.drop(columns=overlap)

    joined = data.merge(attributes, on=emp_id_col, how="left")

    n_unmatched = (~data[emp_id_col].isin(attributes[emp_id_col])).sum()
    if n_unmatched > 0:
        logger.info(
            "%d respondent(s) had no match in the attribute data and will have "
            "NaN for all attribute columns.",
            n_unmatched,
        )

    if isinstance(survey, GlintSurvey):
        existing_attr_cols: list[str] = survey.metadata.get("attribute_cols") or []
        survey.data = joined
        survey.metadata["attribute_cols"] = list(
            dict.fromkeys(existing_attr_cols + new_attr_cols)
        )
        return survey

    return joined


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def validate_glint_structure(data: pd.DataFrame, emp_id_col: str) -> None:
    """Validate that *data* conforms to the expected Viva Glint export structure.

    Checks for required standard columns, at least one question column set,
    complete question column sets (all four suffixes present), and no
    orphaned suffix columns without their base.

    Ported from ``R/import.R::validate_glint_structure``.

    Parameters
    ----------
    data:
        DataFrame to validate.
    emp_id_col:
        Name of the employee ID column.

    Raises
    ------
    ValueError
        With a descriptive message if any validation check fails.
    """
    required_cols = [
        "First Name", "Last Name", "Email", "Status",
        emp_id_col,
        "Survey Cycle Completion Date",
        "Survey Cycle Sent Date",
    ]
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        missing_str = ", ".join(f"'{c}'" for c in missing_cols)
        required_str = "\n".join(f"  - {c}" for c in required_cols)
        raise ValueError(
            f"Missing required standard column(s): {missing_str}\n\n"
            "Your CSV file must contain all of the following standard Viva Glint columns:\n"
            f"{required_str}\n\n"
            "Please ensure you are using a complete Viva Glint survey export."
        )

    all_standard_cols = set(get_standard_columns(emp_id_col))
    question_cols = [c for c in data.columns if c not in all_standard_cols]

    if not question_cols:
        raise ValueError(
            "No question columns found in the data.\n\n"
            "A Viva Glint export should contain at least one question with its associated columns:\n"
            "  - [Question Text]\n"
            "  - [Question Text]_COMMENT\n"
            "  - [Question Text]_COMMENT_TOPICS\n"
            "  - [Question Text]_SENSITIVE_COMMENT_FLAG\n\n"
            "Please check that you are using a complete survey export file."
        )

    question_stems = list(dict.fromkeys(get_question_stem(c) for c in question_cols))
    expected_suffixes = ["", "_COMMENT", "_COMMENT_TOPICS", "_SENSITIVE_COMMENT_FLAG"]

    error_messages: list[str] = []

    # Check each stem for incomplete column sets
    incomplete_questions: list[str] = []
    for stem in question_stems:
        expected = [stem + sfx for sfx in expected_suffixes]
        found = [c in data.columns for c in expected]
        if any(found) and not all(found):
            missing_for_q = [c for c, f in zip(expected, found) if not f]
            incomplete_questions.append(
                f"  - Question '{stem}' is missing: "
                + ", ".join(f"'{c}'" for c in missing_for_q)
            )

    if incomplete_questions:
        error_messages.append(
            "Incomplete question column sets found:\n"
            + "\n".join(incomplete_questions)
            + "\n\nEach question must have all four columns:\n"
            "  - [Question Text]\n"
            "  - [Question Text]_COMMENT\n"
            "  - [Question Text]_COMMENT_TOPICS\n"
            "  - [Question Text]_SENSITIVE_COMMENT_FLAG"
        )

    # Check for orphaned suffix columns without a base column
    orphaned: list[str] = []
    for col in question_cols:
        stem = get_question_stem(col)
        # Only flag columns that ARE a suffix variant (not the base)
        if col != stem and stem not in data.columns:
            orphaned.append(col)

    if orphaned:
        orphaned_str = "\n".join(f"  - '{c}'" for c in orphaned)
        error_messages.append(
            f"Orphaned columns found that don't belong to a complete question set:\n"
            f"{orphaned_str}\n\n"
            "Please check your CSV export for incomplete or corrupted question columns."
        )

    if error_messages:
        raise ValueError("\n\n".join(error_messages))
