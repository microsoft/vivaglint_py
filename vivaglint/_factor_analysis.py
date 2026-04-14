"""
vivaglint._factor_analysis
--------------------------
Pure NumPy/SciPy Exploratory Factor Analysis (EFA) that matches the behaviour
of ``psych::fa()`` from R's psych package.

Background — how Viva Glint uses factor analysis
-------------------------------------------------
Factor analysis answers the question: "How much does each survey item contribute
to the underlying engagement construct the survey is designed to measure?"

A single-factor solution (the default) is the primary use case. A well-designed
engagement survey should produce one dominant factor onto which all items load.
Items are interpreted by their loading strength:

  * **Strong (>= 0.75)** — Outcome variables (eSat, Recommend, etc.) are
    expected to load here. These items directly measure engagement and,
    if actioned, produce the greatest lift in the engagement index.

  * **Medium (0.60 – 0.74)** — Solid drivers. Improving a top-3 medium-loading
    driver typically has a "spillover" effect, lifting other items as well.
    A driver that is also a top key driver (from Driver Analysis) gives a
    double benefit: it improves engagement *and* raises other items.

  * **Weak (< 0.60)** — Items that explain less variance in the engagement
    construct. Weak-loading items are candidates for removal in item-reduction
    analysis.

When a non-outcome driver (e.g. Belonging, Prospects) loads as strongly as an
outcome variable, the model may be measuring a broader construct than intended.
Those items should be treated analytically as outcomes in that context.

Algorithm — matching psych::fa
-------------------------------
Extraction:
  ``fm="minres"`` (default) — Minimum Residual / Unweighted Least Squares (ULS).
  Minimises the sum of squared off-diagonal residuals of ``R - LL'`` using
  L-BFGS-B, warm-started from an iterated PAF solution.  Matches
  ``psych::fa(fm="minres")``.

  ``fm="pa"`` and all other codes — Iterated Principal Axis Factoring (PAF).
  Matches ``psych::fa(fm="pa")``.

Rotations (all match psych / GPArotation defaults):
  * ``"oblimin"``  — direct oblimin via gradient descent (oblique, gamma=0)
  * ``"varimax"``  — SVD-based Kaiser varimax (orthogonal)
  * ``"promax"``   — varimax + power target regression (oblique)
  * ``"equamax"``  — Crawford-Ferguson with gamma = k / (2p) (orthogonal)
  * ``"quartimax"`` / ``"quartimin"`` — Crawford-Ferguson gamma=0 (orthogonal)
  * ``"none"``     — unrotated solution

Interface
---------
``_VivaGlintFA`` exposes the methods that ``analyze.py`` calls:

  * ``fit(X)``               — fit the model; stores ``loadings_``
  * ``get_communalities()``  — array of per-variable h² (pre-rotation)
  * ``get_eigenvalues()``    — (corr_eigenvalues, factor_SS_loadings)
  * ``get_factor_variance()`` — (SS_loadings, proportion_var, cumulative_var)

Reference: https://www.rdocumentation.org/packages/psych/topics/fa
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)


class _VivaGlintFA:
    """EFA implementation matching ``psych::fa()`` from R.

    Parameters
    ----------
    n_factors:
        Number of latent factors to extract.
    rotation:
        One of ``"oblimin"``, ``"varimax"``, ``"promax"``, ``"equamax"``,
        ``"quartimax"``, ``"quartimin"``, ``"none"``, or ``None`` (same as
        ``"none"``).
    method:
        Extraction method.  ``"minres"`` / ``"uls"`` use MINRES optimisation
        (matches ``psych::fa(fm="minres")``).  All other values use iterated
        PAF (matches ``psych::fa(fm="pa")``).
    """

    def __init__(
        self,
        n_factors: int,
        rotation: Optional[str] = None,
        method: str = "minres",
    ) -> None:
        self.n_factors = n_factors
        self.rotation = rotation
        self.method = method

        self.loadings_: np.ndarray = np.empty((0, 0))
        self._corr_eigenvalues: np.ndarray = np.empty(0)
        self._communalities: np.ndarray = np.empty(0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X) -> "_VivaGlintFA":
        """Fit the factor model to data *X* (DataFrame or ndarray).

        Computes the correlation matrix (pairwise complete observations when
        missing values are present, matching R's ``use="pairwise.complete.obs"``),
        extracts factors, then applies the requested rotation.
        """
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Input data must be 2-dimensional.")

        n_rows, p = arr.shape
        if n_rows < 2:
            raise ValueError("Factor analysis requires at least 2 rows of data.")

        corr = (
            self._pairwise_corrcoef(arr)
            if np.isnan(arr).any()
            else np.corrcoef(arr, rowvar=False)
        )

        if np.isnan(corr).any():
            raise ValueError(
                "Cannot compute the correlation matrix — one or more question "
                "pairs have insufficient overlapping responses or are constant."
            )

        # Full-matrix eigenvalues (used by Kaiser criterion / parallel analysis)
        self._corr_eigenvalues = np.sort(np.linalg.eigvalsh(corr))[::-1]

        k = min(self.n_factors, p)
        if self.method in ("minres", "uls"):
            loadings = self._extract_minres(corr, k)
        else:
            loadings = self._extract_pa(corr, k)

        # Communalities are a property of extraction, not rotation.
        # Store pre-rotation h² so they are always in [0, 1].
        self._communalities = np.clip(np.sum(loadings ** 2, axis=1), 0.0, 1.0)

        rot = self.rotation
        if rot is not None and rot != "none":
            loadings, _ = self._rotate(loadings, rot)

        # psych sign convention: flip column if its largest absolute loading
        # is negative (matches psych::fa behaviour exactly)
        for j in range(loadings.shape[1]):
            idx = np.argmax(np.abs(loadings[:, j]))
            if loadings[idx, j] < 0:
                loadings[:, j] *= -1

        self.loadings_ = loadings
        return self

    def get_communalities(self) -> np.ndarray:
        """Per-variable communalities h² (pre-rotation).

        Communality is the proportion of each item's variance explained by the
        factor model.  For a single-factor engagement model, items with h²
        close to their squared loading indicate the factor captures most of
        their systematic variance.
        """
        return self._communalities

    def get_eigenvalues(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (correlation-matrix eigenvalues, factor SS-loadings).

        The first element mirrors ``psych::fa()$values`` (eigenvalues of the
        full correlation matrix) and is used for the Kaiser criterion.
        """
        factor_ev = np.sum(self.loadings_ ** 2, axis=0)
        return self._corr_eigenvalues, factor_ev

    def get_factor_variance(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (SS loadings, proportion variance, cumulative proportion).

        Mirrors ``psych::fa()$Vaccount`` for multi-factor solutions and
        ``fa_result$values[1] / ncol(data)`` for single-factor solutions
        (which are numerically equivalent when loadings are correctly extracted).
        """
        p = self.loadings_.shape[0]
        ss = np.sum(self.loadings_ ** 2, axis=0)
        proportion = ss / p
        cumulative = np.cumsum(proportion)
        return ss, proportion, cumulative

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pairwise_corrcoef(self, arr: np.ndarray) -> np.ndarray:
        """Pairwise Pearson correlations with missing-value handling.

        Equivalent to R's ``cor(data, use="pairwise.complete.obs")``.
        Returns an all-NaN matrix if any pair has fewer than 2 complete rows.
        """
        p = arr.shape[1]
        corr = np.eye(p, dtype=float)

        for i in range(p):
            xi = arr[:, i]
            for j in range(i + 1, p):
                xj = arr[:, j]
                mask = ~np.isnan(xi) & ~np.isnan(xj)
                n = mask.sum()
                if n < 2:
                    return np.full((p, p), np.nan)

                xi_c = xi[mask] - xi[mask].mean()
                xj_c = xj[mask] - xj[mask].mean()
                denom = np.sqrt(np.dot(xi_c, xi_c)) * np.sqrt(np.dot(xj_c, xj_c))
                r = 0.0 if denom == 0.0 else np.dot(xi_c, xj_c) / denom
                corr[i, j] = corr[j, i] = r

        return corr

    # ------------------------------------------------------------------
    # Factor extraction
    # ------------------------------------------------------------------

    def _extract_minres(
        self,
        R: np.ndarray,
        n_factors: int,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Minimum Residual (MINRES / ULS) extraction — matches psych::fa(fm="minres").

        Minimises the sum of squared off-diagonal residuals of ``R - LL'``
        using L-BFGS-B, warm-started from a PAF solution.

        Objective:  ``F = (1/4) * ||offdiag(R - L @ L.T)||_F^2``
        Gradient:   ``dF/dL = -offdiag(R - L @ L.T) @ L``
        """
        from scipy.optimize import minimize

        p = R.shape[0]
        n_factors = max(1, min(n_factors, p))
        L0 = self._extract_pa(R, n_factors)

        def _f(params):
            L = params.reshape(p, n_factors)
            res = R - L @ L.T
            np.fill_diagonal(res, 0.0)
            return 0.25 * np.sum(res ** 2)

        def _g(params):
            L = params.reshape(p, n_factors)
            res = R - L @ L.T
            np.fill_diagonal(res, 0.0)
            return (-res @ L).ravel()

        opt = minimize(
            _f,
            L0.ravel(),
            jac=_g,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": tol ** 2, "gtol": tol},
        )
        return opt.x.reshape(p, n_factors)

    def _extract_pa(
        self,
        R: np.ndarray,
        n_factors: int,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Iterated Principal Axis Factoring — matches psych::fa(fm="pa").

        Algorithm
        ---------
        1. Initialise communalities h² via squared multiple correlations
           (``SMC_i = 1 - 1 / (R⁻¹)_ii``).
        2. Replace the diagonal of R with h² to form the reduced matrix.
        3. Eigendecompose the reduced matrix; form loadings from the top k
           eigenvectors scaled by sqrt(eigenvalue).
        4. Update h² = row-wise sum of squared loadings.
        5. Repeat until max(|Δh²|) < tol.
        """
        p = R.shape[0]
        n_factors = max(1, min(n_factors, p))

        # Initial communalities: squared multiple correlations
        try:
            R_inv = linalg.inv(R)
            diag_inv = np.diag(R_inv)
            if not np.isfinite(diag_inv).all():
                raise linalg.LinAlgError
            h2 = np.clip(1.0 - 1.0 / diag_inv, 0.0, 1.0)
            h2 = np.nan_to_num(h2, nan=0.5)
        except linalg.LinAlgError:
            try:
                diag_inv = np.diag(linalg.pinv(R))
                h2 = np.clip(1.0 - 1.0 / diag_inv, 0.0, 1.0)
                h2 = np.nan_to_num(h2, nan=0.5)
            except (linalg.LinAlgError, ValueError):
                h2 = np.full(p, 0.5)

        loadings = np.zeros((p, n_factors))

        for iteration in range(max_iter):
            h2_prev = h2.copy()

            R_red = R.copy()
            np.fill_diagonal(R_red, h2)

            eigenvalues, eigenvectors = linalg.eigh(R_red)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            k = max(1, min(n_factors, int(np.sum(eigenvalues > 1e-10))))
            ev_k = np.maximum(eigenvalues[:k], 0.0)
            L = eigenvectors[:, :k] * np.sqrt(ev_k)

            if k < n_factors:
                L = np.hstack([L, np.zeros((p, n_factors - k))])

            loadings = L
            h2 = np.clip(np.sum(loadings ** 2, axis=1), 0.0, 1.0)

            if np.max(np.abs(h2 - h2_prev)) < tol:
                logger.debug("PAF converged after %d iterations.", iteration + 1)
                break

        return loadings

    # ------------------------------------------------------------------
    # Rotations
    # ------------------------------------------------------------------

    def _rotate(
        self, loadings: np.ndarray, rotation: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dispatch to the requested rotation algorithm."""
        dispatch = {
            "varimax": self._varimax,
            "equamax": self._equamax,
            "quartimax": self._quartimax,
            "oblimin": lambda A: self._oblimin(A, gamma=0.0),
            "quartimin": lambda A: self._oblimin(A, gamma=0.0),
            "promax": self._promax,
        }
        fn = dispatch.get(rotation)
        if fn is None:
            return loadings, np.eye(loadings.shape[1])
        return fn(loadings)

    def _varimax(
        self, A: np.ndarray, max_iter: int = 1000, tol: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """SVD-based Kaiser varimax (orthogonal) — matches psych/GPArotation."""
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

        return A @ T, T

    def _equamax(
        self,
        A: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-6,
        gamma: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Equamax rotation (orthogonal) — Crawford-Ferguson gamma = k/(2p)."""
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
                L ** 3
                - gamma * h2 * L
                - (1.0 - gamma) * (L @ np.diag(np.sum(L ** 2, axis=0))) / p
            )
            U, s, Vt = np.linalg.svd(B)
            T = U @ Vt
            d = np.sum(s)
            if d_old != 0.0 and d / d_old < 1.0 + tol:
                break

        return A @ T, T

    def _quartimax(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quartimax (orthogonal) — Crawford-Ferguson gamma = 0."""
        return self._equamax(A, gamma=0.0)

    def _oblimin(
        self,
        A: np.ndarray,
        gamma: float = 0.0,
        max_iter: int = 500,
        tol: float = 1e-5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Direct oblimin rotation (oblique) — matches psych/GPArotation.

        Minimises the oblimin criterion via gradient descent.
        ``gamma=0`` gives quartimin (most oblique);
        ``gamma=0.5`` gives biquartimin.
        """
        p, k = A.shape
        T = np.eye(k)
        step = 0.05
        N = np.ones((k, k)) - gamma * np.eye(k)

        for iteration in range(max_iter):
            Ti = np.linalg.pinv(T)
            if not np.isfinite(Ti).all():
                break

            L = A @ Ti.T
            Gq = L * (L ** 2 @ N)
            grad = -(Ti.T @ (Gq.T @ L) @ Ti.T)

            T_new = T - step * grad
            col_norms = np.linalg.norm(T_new, axis=0)
            col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
            T_new = T_new / col_norms

            if np.max(np.abs(T_new - T)) < tol:
                logger.debug("Oblimin converged after %d iterations.", iteration + 1)
                T = T_new
                break
            T = T_new

        L = A @ np.linalg.pinv(T).T
        col_scale = np.maximum(np.max(np.abs(L), axis=0), 1.0)
        return L / col_scale, T

    def _promax(
        self, A: np.ndarray, power: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Promax rotation (oblique) — matches psych default power=3.

        1. Varimax orthogonal reference solution L_v.
        2. Target P = sign(L_v) * |L_v|^power.
        3. Solve L_v @ T ≈ P via least squares; normalise columns of T.
        4. Pattern matrix L = A @ inv(T)'.
        """
        L_v, _ = self._varimax(A)
        P = np.sign(L_v) * np.abs(L_v) ** power
        T, _, _, _ = linalg.lstsq(L_v, P)

        col_norms = np.linalg.norm(T, axis=0)
        col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
        T = T / col_norms

        L = A @ np.linalg.pinv(T).T
        col_scale = np.maximum(np.max(np.abs(L), axis=0), 1.0)
        return L / col_scale, T
