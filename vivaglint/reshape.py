"""
vivaglint.reshape
-----------------
Functions for reshaping Viva Glint survey data.

Ported from R/reshape.R.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import pandas as pd

from vivaglint.import_ import GlintSurvey, extract_questions
from vivaglint.utils import get_standard_columns

logger = logging.getLogger(__name__)


def pivot_long(
    survey: Union[GlintSurvey, pd.DataFrame],
    data_type: str = "all",
    include_empty: bool = False,
    include_standard_cols: Optional[Union[bool, list[str]]] = True,
) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Transform survey data from wide format to long format.

    One row per respondent-question combination. Can return all responses,
    only those with comments, or both depending on *data_type*.

    Ported from ``R/reshape.R::pivot_long``.

    Parameters
    ----------
    survey:
        A :class:`~vivaglint.import_.GlintSurvey` or plain DataFrame.
    data_type:
        ``"all"``      — all responses including those without comments.
        ``"comments"`` — only responses with non-empty comments.
        ``"both"``     — returns a ``dict`` with keys ``"all"`` and
                         ``"comments"``.
    include_empty:
        When *data_type* is ``"comments"`` or ``"both"``, whether to include
        rows where the comment is empty/NaN (default ``False``).
    include_standard_cols:
        Whether to include the standard columns (emp ID, name, etc.) in the
        output (default ``True``).  If a ``list`` of column names is passed,
        only those standard columns will be included.

    Returns
    -------
    pd.DataFrame or dict[str, pd.DataFrame]
        Long-format DataFrame(s). Columns when *include_standard_cols* is
        ``True``: standard columns + ``question``, ``response``, ``comment``,
        ``comment_topics``, ``sensitive_flag``.

    Raises
    ------
    ValueError
        If *data_type* is not one of the three valid values, or if
        *emp_id_col* cannot be determined.
    """
    if data_type not in ("all", "comments", "both"):
        raise ValueError("data_type must be one of: 'all', 'comments', or 'both'")

    if isinstance(survey, GlintSurvey):
        emp_id_col: Optional[str] = survey.metadata.get("emp_id_col")
        data = survey.data
        questions = survey.metadata.get("questions")
        if questions is None:
            questions = extract_questions(data, emp_id_col)
    else:
        emp_id_col = None
        data = survey
        questions = extract_questions(data)

    if emp_id_col is None:
        raise ValueError(
            "emp_id_col could not be determined. "
            "Load your survey with read_glint_survey() and specify emp_id_col."
        )

    standard_cols = get_standard_columns(emp_id_col)
    present_standard_cols = [c for c in standard_cols if c in data.columns]
    if include_standard_cols is True:
        selected_standard_cols = present_standard_cols
    elif include_standard_cols is False:
        selected_standard_cols = []
    elif isinstance(include_standard_cols, (list, tuple)):
        selected_standard_cols = [c for c in include_standard_cols if c in present_standard_cols]
    else:
        raise TypeError(
            "include_standard_cols must be a bool or a list of standard column names"
        )
    standard_data = data[selected_standard_cols].reset_index(drop=True)

    n = len(data)
    long_frames: list[pd.DataFrame] = []
    for _, row in questions.iterrows():
        q_frame = pd.DataFrame(
            {
                "question": row["question"],
                "response": data[row["response_col"]].values
                if row["response_col"] in data.columns
                else [pd.NA] * n,
                "comment": data[row["comment_col"]].values
                if row["comment_col"] in data.columns
                else [pd.NA] * n,
                "comment_topics": data[row["topics_col"]].values
                if row["topics_col"] in data.columns
                else [pd.NA] * n,
                "sensitive_flag": data[row["flag_col"]].values
                if row["flag_col"] in data.columns
                else [pd.NA] * n,
            }
        )

        if include_standard_cols:
            q_frame = pd.concat(
                [standard_data.reset_index(drop=True), q_frame.reset_index(drop=True)],
                axis=1,
            )

        long_frames.append(q_frame)

    long_data = pd.concat(long_frames, ignore_index=True)

    def _filter_comments(df: pd.DataFrame) -> pd.DataFrame:
        """Keep rows where comment is non-null and non-empty."""
        mask = df["comment"].notna() & (df["comment"].astype(str).str.strip() != "")
        return df[mask].reset_index(drop=True)

    if data_type == "all":
        return long_data

    if data_type == "comments":
        if not include_empty:
            long_data = _filter_comments(long_data)
        return long_data

    # data_type == "both"
    comments_data = long_data.copy()
    if not include_empty:
        comments_data = _filter_comments(comments_data)

    return {"all": long_data, "comments": comments_data}


def split_survey_data(
    survey: Union[GlintSurvey, pd.DataFrame],
    emp_id_col: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Separate numeric responses from qualitative comment columns.

    Returns a dict with two DataFrames:

    * ``"quantitative"`` — standard columns + numeric response columns.
    * ``"qualitative"``  — employee ID + comment, topic, and flag columns.

    Ported from ``R/analyze.R::split_survey_data`` (the function lives in
    analyze.R in the R source but is grouped here with reshape utilities as
    it is logically a reshape operation).

    Parameters
    ----------
    survey:
        A :class:`~vivaglint.import_.GlintSurvey` or plain DataFrame.
    emp_id_col:
        Required when *survey* is a plain DataFrame.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys ``"quantitative"`` and ``"qualitative"``.

    Raises
    ------
    ValueError
        If *emp_id_col* cannot be determined.
    """
    if isinstance(survey, GlintSurvey):
        emp_id_col = emp_id_col or survey.metadata.get("emp_id_col")
        data = survey.data
    elif isinstance(survey, pd.DataFrame):
        data = survey
    else:
        raise TypeError("survey must be a GlintSurvey or a pandas DataFrame")

    if emp_id_col is None:
        raise ValueError(
            "emp_id_col could not be determined. "
            "Load your survey with read_glint_survey() and specify emp_id_col."
        )

    standard_cols = get_standard_columns(emp_id_col)
    questions = extract_questions(data, emp_id_col)

    response_cols = [c for c in questions["response_col"] if c in data.columns]
    comment_cols  = [c for c in questions["comment_col"]   if c in data.columns]
    topics_cols   = [c for c in questions["topics_col"]    if c in data.columns]
    flag_cols     = [c for c in questions["flag_col"]      if c in data.columns]

    quant_cols = [c for c in standard_cols if c in data.columns] + response_cols
    qual_cols  = ([emp_id_col] if emp_id_col in data.columns else []) + \
                 comment_cols + topics_cols + flag_cols

    # Deduplicate while preserving order
    seen: set[str] = set()
    quant_cols_dedup: list[str] = []
    for c in quant_cols:
        if c not in seen:
            quant_cols_dedup.append(c)
            seen.add(c)

    seen = set()
    qual_cols_dedup: list[str] = []
    for c in qual_cols:
        if c not in seen:
            qual_cols_dedup.append(c)
            seen.add(c)

    return {
        "quantitative": data[quant_cols_dedup].copy(),
        "qualitative":  data[qual_cols_dedup].copy(),
    }
