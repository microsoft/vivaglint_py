"""
vivaglint.hierarchy
-------------------
Organisational hierarchy analysis functions.

Ported from R/hierarchy.R.  Both functions mirror their R counterparts exactly
in traversal logic, team membership resolution, and output column order.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import pandas as pd

from vivaglint.import_ import GlintSurvey, extract_questions
from vivaglint.analyze import summarize_survey

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# get_all_reports  (public, exported — mirrors R's exported function)
# ---------------------------------------------------------------------------

def get_all_reports(
    manager_id,
    data: pd.DataFrame,
    emp_id_col: str,
    manager_id_col: str,
) -> list:
    """Recursively collect all direct and indirect reports for a manager.

    Ported from ``R/hierarchy.R::get_all_reports``.

    The R source uses a plain recursive for-loop over direct reports, then
    calls ``unique()`` on the accumulated vector.  This Python port mirrors
    that logic exactly — including the deduplication step.

    A cycle-guard (``_visited`` set) is added as a defensive measure for
    malformed org charts with circular reporting relationships; this does not
    change behaviour for well-formed data.

    Parameters
    ----------
    manager_id:
        Employee ID of the manager whose reports are to be collected.
    data:
        Full survey DataFrame containing at least *emp_id_col* and
        *manager_id_col*.
    emp_id_col:
        Name of the employee ID column.
    manager_id_col:
        Name of the manager ID column.

    Returns
    -------
    list
        Deduplicated list of employee IDs for all direct and indirect reports.
        Returns an empty list if the manager has no direct reports.
    """
    return _get_all_reports_impl(manager_id, data, emp_id_col, manager_id_col,
                                  _visited=None)


def _get_all_reports_impl(
    manager_id,
    data: pd.DataFrame,
    emp_id_col: str,
    manager_id_col: str,
    _visited: Optional[set] = None,
) -> list:
    """Internal recursive implementation with cycle guard."""
    if _visited is None:
        _visited = set()

    # Guard against cycles in the org chart
    if manager_id in _visited:
        return []
    _visited.add(manager_id)

    direct_reports = (
        data.loc[data[manager_id_col] == manager_id, emp_id_col]
        .dropna()
        .unique()
        .tolist()
    )

    # Base case: no direct reports (mirrors R's `if (length(direct_reports) == 0)`)
    if not direct_reports:
        return []

    all_reports = list(direct_reports)

    # Recursively get reports of reports (mirrors R's for loop)
    for report in direct_reports:
        indirect_reports = _get_all_reports_impl(
            report, data, emp_id_col, manager_id_col, _visited=_visited
        )
        all_reports.extend(indirect_reports)

    # Deduplicate, preserving first-occurrence order (mirrors R's unique())
    return list(dict.fromkeys(all_reports))


# ---------------------------------------------------------------------------
# aggregate_by_manager
# ---------------------------------------------------------------------------

def aggregate_by_manager(
    survey: Union[GlintSurvey, pd.DataFrame],
    scale_points: int,
    emp_id_col: Optional[str] = None,
    manager_id_col: str = "Manager ID",
    full_tree: bool = False,
    plot: bool = False,
) -> pd.DataFrame:
    """Roll up survey responses to the manager level.

    Calculates the same metrics as :func:`~vivaglint.analyze.summarize_survey`
    for each manager's team.

    Ported from ``R/hierarchy.R::aggregate_by_manager``.

    Parameters
    ----------
    survey:
        A :class:`~vivaglint.import_.GlintSurvey` or plain DataFrame.
    scale_points:
        Number of scale points (2-11).
    emp_id_col:
        Employee ID column name. Resolved from survey metadata when omitted.
    manager_id_col:
        Name of the manager ID column (default ``"Manager ID"``).
    full_tree:
        If ``True``, includes all indirect reports (full subtree via
        :func:`get_all_reports`). If ``False`` (default), only direct
        reports are included.
    plot:
        If ``True``, display a ranked lollipop chart of team Glint Scores
        by manager and return the DataFrame.

    Returns
    -------
    pd.DataFrame
        One row per manager-question combination with columns:
        ``manager_id``, ``manager_name``, ``question``, ``team_size``,
        ``mean``, ``sd``, ``glint_score``, ``n_responses``, ``n_skips``,
        ``n_total``, ``pct_favorable``, ``pct_neutral``, ``pct_unfavorable``.
        When *plot* is ``True``, the same DataFrame is returned after
        displaying the plot.

    Raises
    ------
    ValueError
        If *emp_id_col* cannot be determined or required columns are missing.
    """
    if isinstance(survey, GlintSurvey):
        emp_id_col = emp_id_col or survey.metadata.get("emp_id_col")
        data = survey.data
        questions = survey.metadata.get("questions")
        if questions is None:
            questions = extract_questions(data, emp_id_col)
    elif isinstance(survey, pd.DataFrame):
        data = survey
        questions = extract_questions(data, emp_id_col)
    else:
        raise TypeError("survey must be a GlintSurvey or a pandas DataFrame")

    if emp_id_col is None:
        raise ValueError(
            "emp_id_col must be specified when survey is a plain DataFrame"
        )

    if manager_id_col not in data.columns:
        raise ValueError(
            f"Column '{manager_id_col}' not found in survey data. "
            "Ensure your export contains a Manager ID column."
        )

    # All distinct manager IDs referenced in the data
    # (employees who appear as someone else's manager)
    managers = data[manager_id_col].dropna().unique().tolist()

    # Build manager-name lookup: emp_id -> "First Name Last Name"
    # Mirrors R's mutate(manager_name = paste(`First Name`, `Last Name`))
    name_lookup: dict = {}
    if "First Name" in data.columns and "Last Name" in data.columns:
        mgr_name_df = (
            data[data[emp_id_col].isin(managers)]
            [[emp_id_col, "First Name", "Last Name"]]
            .drop_duplicates(subset=[emp_id_col])
        )
        for _, row in mgr_name_df.iterrows():
            first = str(row["First Name"]) if pd.notna(row.get("First Name")) else ""
            last = str(row["Last Name"]) if pd.notna(row.get("Last Name")) else ""
            name_lookup[row[emp_id_col]] = f"{first} {last}".strip()

    frames = []
    for mgr_id in managers:
        if full_tree:
            # Include full reporting subtree
            team_members = get_all_reports(
                mgr_id, data, emp_id_col, manager_id_col
            )
        else:
            # Direct reports only
            team_members = (
                data.loc[data[manager_id_col] == mgr_id, emp_id_col]
                .dropna()
                .unique()
                .tolist()
            )

        if not team_members:
            continue

        team_data = data[data[emp_id_col].isin(team_members)]
        if team_data.empty:
            continue

        team_summary = summarize_survey(
            team_data,
            scale_points=scale_points,
            questions="all",
            emp_id_col=emp_id_col,
        )
        team_summary["manager_id"] = mgr_id
        team_summary["manager_name"] = name_lookup.get(mgr_id, "")
        team_summary["team_size"] = len(team_members)
        frames.append(team_summary)

    if not frames:
        return pd.DataFrame(
            columns=[
                "manager_id", "manager_name", "question", "team_size",
                "mean", "sd", "glint_score", "n_responses", "n_skips",
                "n_total", "pct_favorable", "pct_neutral", "pct_unfavorable",
            ]
        )

    results = pd.concat(frames, ignore_index=True)

    # Reorder columns to match R's select(manager_id, manager_name, question,
    # team_size, everything())
    front_cols = ["manager_id", "manager_name", "question", "team_size"]
    other_cols = [c for c in results.columns if c not in front_cols]
    results = results[front_cols + other_cols]

    if plot:
        from vivaglint.plots import _plot_manager
        fig = _plot_manager(results)
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass

    return results
