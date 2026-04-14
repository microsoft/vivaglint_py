"""
vivaglint.plots
---------------
Internal plotting helpers.  Each function is called when the corresponding
public function is invoked with ``plot=True``.  All functions return a
``matplotlib.figure.Figure`` so the caller can ``print()`` / ``show()`` it
or save it without side effects.

These are stubs — full implementations are added incrementally.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

logger = logging.getLogger(__name__)


def _require_matplotlib():
    """Raise an ImportError with a helpful message if matplotlib is missing."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        import seaborn as sns            # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install them with:  pip install matplotlib seaborn"
        ) from exc


# ---------------------------------------------------------------------------
# Stub implementations
# ---------------------------------------------------------------------------

def _plot_survey_summary(data: pd.DataFrame):
    """Horizontal bar chart of Glint Scores per question.

    Parameters
    ----------
    data:
        Output of ``summarize_survey()`` — one row per question with at
        least columns ``question`` and ``glint_score``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, max(4, len(data) * 0.5)))
    sns.barplot(
        data=data,
        y="question",
        x="glint_score",
        orient="h",
        ax=ax,
        color="#0078D4",
    )
    ax.set_xlabel("Glint Score (0–100)")
    ax.set_ylabel("")
    ax.set_title("Survey Summary — Glint Scores")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    return fig


def _plot_response_dist(data: pd.DataFrame):
    """Bar chart of response value distributions per question.

    Parameters
    ----------
    data:
        Output of ``get_response_dist()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Response Distribution")
    ax.set_xlabel("Response Value")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def _plot_compare_cycles(data: pd.DataFrame, cycle_names: list[str] | None = None):
    """Line chart comparing Glint Scores across survey cycles.

    Parameters
    ----------
    data:
        Output of ``compare_cycles()``.
    cycle_names:
        Optional list of cycle labels for the legend.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 5))
    if "cycle" in data.columns and "glint_score" in data.columns:
        sns.lineplot(
            data=data,
            x="cycle",
            y="glint_score",
            hue="question" if "question" in data.columns else None,
            marker="o",
            ax=ax,
        )
    ax.set_title("Glint Score by Cycle")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Glint Score (0–100)")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    return fig


def _plot_correlations(data: pd.DataFrame):
    """Heatmap of question-to-question correlations.

    Parameters
    ----------
    data:
        Long-format correlation DataFrame from ``get_correlations()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    if {"question1", "question2", "correlation"}.issubset(data.columns):
        matrix = data.pivot(index="question1", columns="question2", values="correlation")
    else:
        matrix = pd.DataFrame()

    fig, ax = plt.subplots(figsize=(max(6, len(matrix)), max(5, len(matrix))))
    if not matrix.empty:
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax,
            square=True,
        )
    ax.set_title("Question Correlations")
    fig.tight_layout()
    return fig


def _plot_survey_factors(data: pd.DataFrame):
    """Dot-plot of factor loadings per question.

    Parameters
    ----------
    data:
        ``factor_summary`` DataFrame from ``extract_survey_factors()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, max(4, len(data) * 0.4)))
    if not data.empty and {"question", "loading", "factor"}.issubset(data.columns):
        sns.stripplot(
            data=data,
            x="loading",
            y="question",
            hue="factor",
            dodge=True,
            ax=ax,
        )
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("Factor Loadings")
    ax.set_xlabel("Loading")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def _plot_attrition(data: pd.DataFrame, attribute_cols=None):
    """Bar chart of attrition ratios by question and time period.

    Parameters
    ----------
    data:
        Output of ``analyze_attrition()``.
    attribute_cols:
        Optional list of attribute columns (used for faceting).

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 5))
    if not data.empty and "attrition_ratio" in data.columns:
        plot_data = data[data["attrition_ratio"].notna()].copy()
        if not plot_data.empty:
            sns.barplot(
                data=plot_data,
                x="days",
                y="attrition_ratio",
                hue="question" if "question" in plot_data.columns else None,
                ax=ax,
            )
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8,
               label="No difference (ratio = 1)")
    ax.set_title("Attrition Ratio: Unfavorable vs. Favorable Responders")
    ax.set_xlabel("Days Since Survey")
    ax.set_ylabel("Attrition Ratio")
    fig.tight_layout()
    return fig


def _plot_by_attributes(data: pd.DataFrame, attribute_cols=None):
    """Grouped bar chart of Glint Scores by attribute segment.

    Parameters
    ----------
    data:
        Output of ``analyze_by_attributes()``.
    attribute_cols:
        Optional list of attribute columns used for grouping labels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 5))
    if not data.empty and "glint_score" in data.columns:
        hue_col = attribute_cols[0] if attribute_cols else None
        sns.barplot(
            data=data,
            x="question" if "question" in data.columns else data.columns[0],
            y="glint_score",
            hue=hue_col,
            ax=ax,
        )
    ax.set_title("Glint Scores by Attribute")
    ax.set_xlabel("")
    ax.set_ylabel("Glint Score (0–100)")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig


def _plot_manager(data: pd.DataFrame):
    """Ranked lollipop chart of team Glint Scores by manager.

    Parameters
    ----------
    data:
        Output of ``aggregate_by_manager()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if data.empty or "glint_score" not in data.columns:
        fig, ax = plt.subplots()
        ax.set_title("Manager Glint Scores (no data)")
        return fig

    questions = data["question"].unique() if "question" in data.columns else [""]
    n_q = len(questions)

    fig, axes = plt.subplots(
        1, n_q, figsize=(max(6, n_q * 5), max(4, len(data["manager_id"].unique()) * 0.4)),
        squeeze=False,
    )

    for i, q in enumerate(questions):
        ax = axes[0][i]
        subset = (
            data[data["question"] == q].sort_values("glint_score")
            if "question" in data.columns
            else data.sort_values("glint_score")
        )
        label_col = "manager_name" if "manager_name" in subset.columns else "manager_id"
        labels = subset[label_col].fillna(subset["manager_id"])
        scores = subset["glint_score"]

        ax.hlines(range(len(labels)), 0, scores, color="lightgrey")
        ax.plot(scores, range(len(labels)), "o", color="#0078D4")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Glint Score (0–100)")
        ax.set_xlim(0, 100)
        ax.set_title(q if q else "Manager Glint Scores")

    fig.tight_layout()
    return fig
