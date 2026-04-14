"""
vivaglint.factor_analysis
-------------------------
Standalone EFA matching ``psych::fa()`` from R.

Background — Viva Glint use
----------------------------
Factor analysis answers: "How much does each survey item contribute to the
underlying engagement construct?"

A single-factor solution is the primary use case.  Loading strength:

  Strong  (>= 0.75) — Outcome variables (eSat, Recommend).  Greatest lift
                       on the engagement index if actioned.
  Medium (0.60–0.74) — Solid drivers.  A top-3 medium loader that is also
                        a top Key Driver gives a double benefit.
  Weak    (< 0.60)   — Less variance in the engagement construct.
                        Candidates for item-reduction analysis.

Usage
-----
::

    from vivaglint.factor_analysis import factor_analysis

    survey = read_glint_survey("survey.csv", emp_id_col="Employee ID")
    result = factor_analysis(survey, n_factors=1, rotation="oblimin")

    print(result["factor_summary"])
    print(result["Vaccount"])

Output keys (matching psych::fa naming)
----------------------------------------
  ``loadings``        DataFrame  (n_questions × n_factors) — pattern matrix
  ``communality``     Series     per-question h²
  ``uniquenesses``    Series     per-question u² = 1 − h²
  ``values``          ndarray    eigenvalues of the full correlation matrix
  ``Vaccount``        DataFrame  SS loadings / Proportion Var / Cumulative Var
  ``factor_summary``  DataFrame  long-format, filtered to abs(loading) >= min_loading
  ``n_factors``       int
  ``rotation``        str
  ``fm``              str

Reference: https://www.rdocumentation.org/packages/psych/topics/fa
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from vivaglint.import_ import GlintSurvey, extract_questions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def factor_analysis(
    survey: Union[GlintSurvey, pd.DataFrame],
    n_factors: int = 1,
    rotation: str = "oblimin",
    fm: str = "minres",
    min_loading: float = 0.3,
) -> dict:
    """Perform EFA on survey question responses, matching ``psych::fa()``.

    Parameters
    ----------
    survey:
        A :class:`~vivaglint.import_.GlintSurvey` or plain ``pd.DataFrame``.
    n_factors:
        Number of latent factors to extract.  Default ``1``.
    rotation:
        ``"oblimin"`` (default), ``"varimax"``, ``"promax"``,
        ``"quartimax"``, ``"quartimin"``, ``"equamax"``, or ``"none"``.
    fm:
        Factoring method.  ``"minres"``/``"uls"`` use MINRES optimisation
        (matches ``psych::fa(fm="minres")``).  All others use iterated PAF.
    min_loading:
        Minimum absolute loading included in ``factor_summary`` (default 0.3).

    Returns
    -------
    dict — see module docstring for full key list.
    """
    _VALID_ROTATIONS = {
        "oblimin", "varimax", "promax",
        "quartimax", "quartimin", "equamax", "none",
    }
    _VALID_FM = {"minres", "ml", "pa", "wls", "gls", "uls"}

    if rotation not in _VALID_ROTATIONS:
        raise ValueError(
            f"rotation must be one of: {', '.join(sorted(_VALID_ROTATIONS))}"
        )
    if fm not in _VALID_FM:
        raise ValueError(
            f"fm must be one of: {', '.join(sorted(_VALID_FM))}"
        )

    data, questions = _resolve_survey(survey)

    response_data = data[questions].copy()
    response_data = response_data[response_data.notna().any(axis=1)]

    if response_data.shape[0] < 2:
        raise ValueError(
            "factor_analysis() requires at least 2 respondents with "
            "non-missing responses."
        )

    n_questions = len(questions)
    if not (1 <= n_factors <= n_questions):
        raise ValueError(
            f"n_factors must be between 1 and {n_questions}."
        )

    arr = response_data.to_numpy(dtype=float)
    R = _pairwise_corr(arr)

    if np.isnan(R).any():
        raise ValueError(
            "Cannot compute the correlation matrix — one or more question "
            "pairs have insufficient overlapping responses or are constant."
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    if fm in ("minres", "uls"):
        loadings_arr, uniquenesses_arr = _extract_minres(R, n_factors)
    else:
        loadings_arr, uniquenesses_arr = _extract_pa(R, n_factors)

    communalities_arr = 1.0 - uniquenesses_arr

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------
    factor_prefix = _factor_prefix(fm)
    factor_names = [f"{factor_prefix}{i + 1}" for i in range(n_factors)]

    if rotation != "none" and n_factors > 1:
        loadings_arr = _rotate(loadings_arr, rotation)
    elif rotation != "none" and n_factors == 1:
        logger.debug("Rotation has no effect for a single-factor solution.")

    # psych sign convention: flip column if its largest absolute loading
    # is negative (matches psych::fa behaviour exactly)
    for j in range(loadings_arr.shape[1]):
        idx = np.argmax(np.abs(loadings_arr[:, j]))
        if loadings_arr[idx, j] < 0:
            loadings_arr[:, j] *= -1

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    values = np.sort(np.linalg.eigvalsh(R))[::-1]

    p = n_questions
    ss_loadings = np.sum(loadings_arr ** 2, axis=0)
    proportion_var = ss_loadings / p
    cumulative_var = np.cumsum(proportion_var)

    Vaccount = pd.DataFrame(
        [ss_loadings, proportion_var, cumulative_var],
        index=["SS loadings", "Proportion Var", "Cumulative Var"],
        columns=factor_names,
    )

    loadings_df = pd.DataFrame(loadings_arr, index=questions, columns=factor_names)
    communality_s = pd.Series(communalities_arr, index=questions, name="communality")
    uniqueness_s = pd.Series(uniquenesses_arr, index=questions, name="uniquenesses")

    factor_summary = _build_factor_summary(loadings_df, communality_s, Vaccount, min_loading)

    return {
        "loadings": loadings_df,
        "communality": communality_s,
        "uniquenesses": uniqueness_s,
        "values": values,
        "Vaccount": Vaccount,
        "factor_summary": factor_summary,
        "n_factors": n_factors,
        "rotation": rotation,
        "fm": fm,
    }


# ---------------------------------------------------------------------------
# Survey input
# ---------------------------------------------------------------------------

def _resolve_survey(survey: Union[GlintSurvey, pd.DataFrame]) -> tuple:
    if isinstance(survey, GlintSurvey):
        return survey.data, list(survey.metadata["questions"]["question"])
    if isinstance(survey, pd.DataFrame):
        return survey, list(extract_questions(survey)["question"])
    raise TypeError("survey must be a GlintSurvey object or a pandas DataFrame.")


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def _pairwise_corr(arr: np.ndarray) -> np.ndarray:
    """Pearson correlation with pairwise complete observations.

    Matches ``cor(data, use="pairwise.complete.obs")`` in R.
    """
    p = arr.shape[1]
    R = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            mask = ~np.isnan(arr[:, i]) & ~np.isnan(arr[:, j])
            if mask.sum() < 2:
                return np.full((p, p), np.nan)
            xi = arr[mask, i] - arr[mask, i].mean()
            xj = arr[mask, j] - arr[mask, j].mean()
            denom = np.sqrt(np.dot(xi, xi)) * np.sqrt(np.dot(xj, xj))
            R[i, j] = R[j, i] = 0.0 if denom == 0.0 else np.dot(xi, xj) / denom
    return R


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract_minres(R: np.ndarray, n_factors: int) -> tuple:
    """MINRES via uniqueness optimisation — matches ``psych::fa(fm="minres")``.

    Parameterises over uniquenesses (Psi), the same approach psych uses
    internally, ensuring communalities stay in [0, 1]::

        S = R − diag(Psi)
        L = top-k eigenvectors of S × sqrt(eigenvalue)
        minimise F(Psi) = 0.5 * ||offdiag(R − LL')||²
        Psi ∈ [0.005, 1.0]   (matching psych bounds)

    Returns (loadings, uniquenesses).
    """
    p = R.shape[0]
    n_factors = max(1, min(n_factors, p))

    def _loadings(Psi):
        S = R.copy()
        np.fill_diagonal(S, 1.0 - Psi)
        vals, vecs = np.linalg.eigh(S)
        idx = np.argsort(vals)[::-1]
        ev = np.maximum(vals[idx[:n_factors]], 0.0)
        return vecs[:, idx[:n_factors]] * np.sqrt(ev)

    def _obj(Psi):
        L = _loadings(Psi)
        res = R - L @ L.T
        np.fill_diagonal(res, 0.0)
        return 0.5 * np.sum(res ** 2)

    try:
        smc = np.clip(1.0 - 1.0 / np.diag(np.linalg.inv(R)), 0.0, 1.0)
        Psi0 = np.clip(1.0 - smc, 0.005, 0.995)
    except np.linalg.LinAlgError:
        Psi0 = np.full(p, 0.5)

    result = minimize(
        _obj, Psi0, method="L-BFGS-B",
        bounds=[(0.005, 1.0)] * p,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )

    Psi_opt = result.x
    return _loadings(Psi_opt), Psi_opt


def _extract_pa(R: np.ndarray, n_factors: int,
                max_iter: int = 1000, tol: float = 1e-6) -> tuple:
    """Iterated Principal Axis Factoring — matches ``psych::fa(fm="pa")``.

    Returns (loadings, uniquenesses).
    """
    p = R.shape[0]
    n_factors = max(1, min(n_factors, p))

    try:
        smc = np.clip(1.0 - 1.0 / np.diag(np.linalg.inv(R)), 0.0, 1.0)
        h2 = np.nan_to_num(smc, nan=0.5)
    except np.linalg.LinAlgError:
        h2 = np.full(p, 0.5)

    loadings = np.zeros((p, n_factors))
    for iteration in range(max_iter):
        h2_prev = h2.copy()
        S = R.copy()
        np.fill_diagonal(S, h2)
        vals, vecs = np.linalg.eigh(S)
        idx = np.argsort(vals)[::-1]
        vals, vecs = vals[idx], vecs[:, idx]
        k = max(1, min(n_factors, int(np.sum(vals > 1e-10))))
        ev = np.maximum(vals[:k], 0.0)
        L = vecs[:, :k] * np.sqrt(ev)
        if k < n_factors:
            L = np.hstack([L, np.zeros((p, n_factors - k))])
        loadings = L
        h2 = np.clip(np.sum(loadings ** 2, axis=1), 0.0, 1.0)
        if np.max(np.abs(h2 - h2_prev)) < tol:
            break

    return loadings, 1.0 - h2


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------

def _rotate(loadings: np.ndarray, rotation: str) -> np.ndarray:
    fns = {
        "varimax":   _varimax,
        "equamax":   _equamax,
        "quartimax": _quartimax,
        "oblimin":   _oblimin,
        "quartimin": _quartimin,
        "promax":    _promax,
    }
    fn = fns.get(rotation)
    return fn(loadings) if fn is not None else loadings


def _varimax(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
    p, k = A.shape
    T = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        d_old = d
        L = A @ T
        B = A.T @ (L ** 3 - (L @ np.diag(np.sum(L ** 2, axis=0))) / p)
        U, s, Vt = np.linalg.svd(B)
        T = U @ Vt
        d = np.sum(s)
        if d_old != 0.0 and d / d_old < 1.0 + tol:
            break
    return A @ T


def _equamax(A: np.ndarray, gamma: Optional[float] = None,
             max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
    p, k = A.shape
    if gamma is None:
        gamma = k / (2.0 * p)
    T = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        d_old = d
        L = A @ T
        h2 = np.sum(L ** 2, axis=1, keepdims=True)
        B = A.T @ (
            L ** 3 - gamma * h2 * L
            - (1.0 - gamma) * (L @ np.diag(np.sum(L ** 2, axis=0))) / p
        )
        U, s, Vt = np.linalg.svd(B)
        T = U @ Vt
        d = np.sum(s)
        if d_old != 0.0 and d / d_old < 1.0 + tol:
            break
    return A @ T


def _quartimax(A: np.ndarray) -> np.ndarray:
    return _equamax(A, gamma=0.0)


def _oblimin(A: np.ndarray, gamma: float = 0.0,
             max_iter: int = 1000, tol: float = 1e-5) -> np.ndarray:
    p, k = A.shape
    T = np.eye(k)
    N = np.ones((k, k)) - gamma * np.eye(k)
    step = 0.05
    for _ in range(max_iter):
        Ti = np.linalg.pinv(T)
        if not np.isfinite(Ti).all():
            break
        L = A @ Ti.T
        Gq = L * (L ** 2 @ N)
        grad = -(Ti.T @ (Gq.T @ L) @ Ti.T)
        T_new = T - step * grad
        col_norms = np.linalg.norm(T_new, axis=0)
        col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
        T_new /= col_norms
        if np.max(np.abs(T_new - T)) < tol:
            T = T_new
            break
        T = T_new
    L = A @ np.linalg.pinv(T).T
    col_scale = np.maximum(np.max(np.abs(L), axis=0), 1.0)
    return L / col_scale


def _quartimin(A: np.ndarray) -> np.ndarray:
    return _oblimin(A, gamma=0.0)


def _promax(A: np.ndarray, power: int = 3) -> np.ndarray:
    from scipy import linalg as sp_linalg
    L_v = _varimax(A)
    P = np.sign(L_v) * np.abs(L_v) ** power
    T, _, _, _ = sp_linalg.lstsq(L_v, P)
    col_norms = np.linalg.norm(T, axis=0)
    col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
    T /= col_norms
    L = A @ np.linalg.pinv(T).T
    col_scale = np.maximum(np.max(np.abs(L), axis=0), 1.0)
    return L / col_scale


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _factor_prefix(fm: str) -> str:
    return {"minres": "MR", "uls": "MR", "pa": "PA", "ml": "ML"}.get(fm, "F")


def _build_factor_summary(loadings_df, communality_s, Vaccount, min_loading):
    rows = []
    for question in loadings_df.index:
        for factor in loadings_df.columns:
            loading = float(loadings_df.loc[question, factor])
            if abs(loading) < min_loading:
                continue
            abs_l = abs(loading)
            label = "Strong" if abs_l >= 0.75 else ("Medium" if abs_l >= 0.60 else "Weak")
            rows.append({
                "question": question,
                "factor": factor,
                "loading": loading,
                "loading_label": label,
                "communality": float(communality_s[question]),
                "factor_variance_pct": float(
                    Vaccount.loc["Proportion Var", factor] * 100
                ),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "question", "factor", "loading", "loading_label",
            "communality", "factor_variance_pct",
        ])

    df = pd.DataFrame(rows)
    df["_abs"] = df["loading"].abs()
    df = (
        df.sort_values(["factor", "_abs"], ascending=[True, False])
        .drop(columns=["_abs"])
        .reset_index(drop=True)
    )
    return df
