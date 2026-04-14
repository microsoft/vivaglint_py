"""
vivaglint.utils
---------------
Internal utility functions shared across the package.

These functions are direct Python ports of the helpers in R/utils.R.
They are NOT part of the public API (no export); import them with:
    from vivaglint.utils import mean_to_glint_score, get_favorability_map, ...
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standard column helpers
# ---------------------------------------------------------------------------

def get_standard_columns(emp_id_col: str) -> list[str]:
    """Return the list of standard column names expected in a Viva Glint export.

    The employee ID column name is configurable because customers use different
    names. "Manager ID" is included for question-detection exclusion but is not
    required to be present.

    Parameters
    ----------
    emp_id_col:
        Name of the employee ID column in this particular export.

    Returns
    -------
    list[str]
        Ordered list of standard column names.
    """
    return [
        "First Name",
        "Last Name",
        "Email",
        "Status",
        emp_id_col,
        "Manager ID",
        "Survey Cycle Completion Date",
        "Survey Cycle Sent Date",
    ]


# ---------------------------------------------------------------------------
# Question stem extraction
# ---------------------------------------------------------------------------

# Suffixes must be removed in this specific order (longest-first so the strip
# is unambiguous — matches R's stringr::str_remove order).
_QUESTION_SUFFIXES = (
    "_COMMENT_TOPICS",
    "_SENSITIVE_COMMENT_FLAG",
    "_COMMENT",
)


def get_question_stem(col_name: str) -> str:
    """Extract the base question text from a column name.

    Removes the standard suffixes ``_COMMENT_TOPICS``,
    ``_SENSITIVE_COMMENT_FLAG``, and ``_COMMENT`` (in that order, matching
    R's ``utils.R::get_question_stem``).

    Parameters
    ----------
    col_name:
        Raw column name from the CSV.

    Returns
    -------
    str
        Base question text with no suffix.
    """
    for suffix in _QUESTION_SUFFIXES:
        if col_name.endswith(suffix):
            return col_name[: -len(suffix)]
    return col_name


# ---------------------------------------------------------------------------
# Glint Score conversion
# ---------------------------------------------------------------------------

def mean_to_glint_score(mean_val: float, scale_points: int) -> Optional[int]:
    """Convert a raw mean response to the 100-point Glint Score.

    Transforms a raw mean on a rating scale to the 0-100 Glint Score as
    displayed in the Viva Glint UI. Rounded to the nearest whole number,
    consistent with Glint's rounding convention.

    Formula (ported exactly from R/utils.R)::

        round(((mean_val - 1) / (scale_points - 1)) * 100)

    Parameters
    ----------
    mean_val:
        Raw mean response value. If ``None`` / ``NaN`` returns ``None``.
    scale_points:
        Number of scale points (2-11).

    Returns
    -------
    int or None
        Glint Score on a 0-100 scale, or ``None`` if *mean_val* is missing.
    """
    import math
    if mean_val is None or (isinstance(mean_val, float) and math.isnan(mean_val)):
        return None
    if scale_points < 2:
        raise ValueError(f"scale_points must be >= 2, got {scale_points!r}")
    return round(((mean_val - 1) / (scale_points - 1)) * 100)


# ---------------------------------------------------------------------------
# Favorability map
# ---------------------------------------------------------------------------

# Ported verbatim from R/utils.R::get_favorability_map.
_FAVORABILITY_MAP: dict[int, dict[str, list[int]]] = {
    2:  {"favorable": [2],           "neutral": [],              "unfavorable": [1]},
    3:  {"favorable": [3],           "neutral": [2],             "unfavorable": [1]},
    4:  {"favorable": [4],           "neutral": [2, 3],          "unfavorable": [1]},
    5:  {"favorable": [4, 5],        "neutral": [3],             "unfavorable": [1, 2]},
    6:  {"favorable": [4, 5, 6],     "neutral": [],              "unfavorable": [1, 2, 3]},
    7:  {"favorable": [6, 7],        "neutral": [4, 5],          "unfavorable": [1, 2, 3]},
    8:  {"favorable": [6, 7, 8],     "neutral": [4, 5],          "unfavorable": [1, 2, 3]},
    9:  {"favorable": [7, 8, 9],     "neutral": [4, 5, 6],       "unfavorable": [1, 2, 3]},
    10: {"favorable": [8, 9, 10],    "neutral": [4, 5, 6, 7],    "unfavorable": [1, 2, 3]},
    11: {"favorable": [10, 11],      "neutral": [8, 9],          "unfavorable": [1, 2, 3, 4, 5, 6, 7]},
}


def get_favorability_map(scale_points: int) -> dict[str, list[int]]:
    """Return the favorability classification for a given scale-point count.

    Ported from ``R/utils.R::get_favorability_map``.

    Parameters
    ----------
    scale_points:
        Number of scale points (2–11).

    Returns
    -------
    dict
        Keys ``"favorable"``, ``"neutral"``, ``"unfavorable"``, each mapping
        to a list of integer scale values in that category.

    Raises
    ------
    ValueError
        If *scale_points* is not in the range 2–11.
    """
    if scale_points not in _FAVORABILITY_MAP:
        raise ValueError(
            f"scale_points must be an integer between 2 and 11, got {scale_points!r}"
        )
    return _FAVORABILITY_MAP[scale_points]


# ---------------------------------------------------------------------------
# Comment topic parser
# ---------------------------------------------------------------------------

def parse_comment_topics(
    topics: pd.Series,
    return_format: str = "list",
) -> "list[list[str]] | pd.DataFrame":
    """Split comma-separated topic strings into lists or a tidy DataFrame.

    Ported from ``R/utils.R::parse_comment_topics``.

    Parameters
    ----------
    topics:
        pandas Series of comma-separated topic strings (may contain NaN/empty).
    return_format:
        ``"list"`` (default) — returns a list of string lists, one per row.
        ``"tidy"`` — returns a DataFrame with columns ``index`` (0-based) and
        ``topic``.

    Returns
    -------
    list[list[str]] or pd.DataFrame
    """
    if return_format not in ("list", "tidy"):
        raise ValueError("return_format must be either 'list' or 'tidy'")

    def _split(val: str) -> list[str]:
        if pd.isna(val) or str(val).strip() == "":
            return []
        return [t.strip() for t in str(val).split(",") if t.strip()]

    topic_list = [_split(v) for v in topics]

    if return_format == "list":
        return topic_list

    rows = []
    for idx, tpcs in enumerate(topic_list):
        for t in tpcs:
            rows.append({"index": idx, "topic": t})
    return pd.DataFrame(rows, columns=["index", "topic"])
