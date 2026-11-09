"""Python-specific, local modeling APIs for analyst-approved survey experiments.

These are exploratory/predictive tools, not causal estimators or employee-risk
scoring APIs. scikit-learn is optional and imported only when an experiment runs.
"""

from __future__ import annotations

import copy
import re
import time
from typing import Any

import numpy as np
import pandas as pd


METHODS = {"dummy", "linear", "random_forest", "gradient_boosting", "extra_trees"}
UNSAFE = re.compile(
    r"(^|[_\W])(id|name|email|address|comment|age|gender|race|ethnicity|"
    r"religion|disability|salary|attrition|exit|termination|risk)([_\W]|$)",
    re.IGNORECASE,
)


def _validate(frame: pd.DataFrame, config: dict[str, Any]) -> None:
    features = config["features"]
    if not features or len(set(features)) != len(features):
        raise ValueError("Select distinct, explicitly approved features.")
    if config["task"] not in {"regression", "classification", "svd"}:
        raise ValueError("Unsupported modeling task.")
    protected = {config.get("target"), config.get("group"), config.get("time")}
    protected.update(config.get("exclude", []))
    for column in features:
        if column in protected or UNSAFE.search(column) or column.casefold() == "manager":
            raise ValueError("A feature is identifying, sensitive, excluded, or target-derived.")
        if column not in frame or frame[column].isna().all():
            raise ValueError("A selected feature is missing or entirely empty.")
        if not pd.api.types.is_numeric_dtype(frame[column]):
            if frame[column].nunique() > 20:
                raise ValueError("High-cardinality categorical features are not supported.")
    target = config.get("target")
    if config["task"] != "svd" and (
        not target or target not in frame or UNSAFE.search(target)
    ):
        raise ValueError("A safe, explicit survey target is required.")
    if config["task"] != "svd":
        for column in features:
            if frame[column].equals(frame[target]):
                raise ValueError("A predictor is an exact copy of the target.")
    if config["task"] == "svd" and any(
        not pd.api.types.is_numeric_dtype(frame[col]) for col in features
    ):
        raise ValueError("Centered SVD requires numeric survey items.")
    if config["min_group_size"] < 5:
        raise ValueError("The minimum group size must be at least five.")
    budget = config["budget"]
    if not 1 <= budget["threads"] <= 8 or not 1 <= budget["max_trials"] <= 50:
        raise ValueError("Thread/trial budget is outside the supported bounds.")
    if len(frame) > budget["max_rows"]:
        raise ValueError("Input exceeds the approved row budget.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate input column names are not supported.")


def _splits(frame: pd.DataFrame, config: dict[str, Any]):
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit, TimeSeriesSplit

    mode = config["validation"]["kind"]
    folds = config["validation"]["folds"]
    fraction = config["validation"]["holdout_fraction"]
    if not 2 <= folds <= 10 or not 0.1 <= fraction <= 0.4:
        raise ValueError("Use 2-10 folds and a holdout fraction between 0.1 and 0.4.")
    group = config.get("group")
    if not group or group not in frame or frame[group].isna().any():
        raise ValueError("A complete respondent/group key is required for leakage checks.")
    if mode == "group":
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=fraction, random_state=config.get("split_seed", config["seed"])
        )
        development, holdout = next(splitter.split(frame, groups=frame[group]))
        cv = list(GroupKFold(n_splits=folds).split(
            frame.iloc[development], groups=frame.iloc[development][group]
        ))
    elif mode == "time":
        column = config.get("time")
        if not column or column not in frame:
            raise ValueError("Chronological validation requires a confirmed time column.")
        dates = pd.to_datetime(frame[column], errors="raise", utc=True)
        if dates.isna().any():
            raise ValueError("Chronological validation cannot contain missing dates.")
        unique = np.sort(dates.unique())
        boundary = max(1, int(len(unique) * (1 - fraction)))
        if boundary >= len(unique):
            raise ValueError("Not enough distinct dates to reserve a future holdout.")
        development = np.flatnonzero(dates < unique[boundary])
        holdout = np.flatnonzero(dates >= unique[boundary])
        dev_dates = dates.iloc[development]
        date_folds = TimeSeriesSplit(n_splits=folds).split(np.sort(dev_dates.unique()))
        cv = []
        dev_unique = np.sort(dev_dates.unique())
        for train_dates, valid_dates in date_folds:
            cv.append((
                np.flatnonzero(dev_dates.isin(dev_unique[train_dates])),
                np.flatnonzero(dev_dates.isin(dev_unique[valid_dates])),
            ))
        if config["validation"].get("unseen_groups", True):
            if set(frame.iloc[development][group]) & set(frame.iloc[holdout][group]):
                raise ValueError("Future holdout overlaps groups; confirm known-person forecasting or change design.")
            for train, valid in cv:
                if set(frame.iloc[development[train]][group]) & set(frame.iloc[development[valid]][group]):
                    raise ValueError("Chronological folds overlap groups.")
    else:
        raise ValueError("Validation must be group-aware or chronological.")
    minimum = config["min_group_size"]
    for indices in [development, holdout]:
        if frame.iloc[indices][group].nunique() < minimum:
            raise ValueError("Development/holdout has too few independent groups.")
    for train, valid in cv:
        for indices in (development[train], development[valid]):
            if frame.iloc[indices][group].nunique() < minimum:
                raise ValueError("A validation fold has too few independent groups.")
    return development, holdout, cv


def _pipeline(frame: pd.DataFrame, config: dict[str, Any], model: dict[str, Any]):
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import (
        ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
        GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = [col for col in config["features"] if pd.api.types.is_numeric_dtype(frame[col])]
    categorical = [col for col in config["features"] if col not in numeric]
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler(with_std=config.get("scale", True))),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    if config["task"] == "svd":
        return Pipeline([
            ("prepare", preprocessing),
            ("model", PCA(n_components=config["components"], svd_solver="full")),
        ])
    classification = config["task"] == "classification"
    method = model["method"]
    choices = {
        "dummy": DummyClassifier if classification else DummyRegressor,
        "linear": LogisticRegression if classification else Ridge,
        "random_forest": RandomForestClassifier if classification else RandomForestRegressor,
        "gradient_boosting": GradientBoostingClassifier if classification else GradientBoostingRegressor,
        "extra_trees": ExtraTreesClassifier if classification else ExtraTreesRegressor,
    }
    if method not in choices:
        raise ValueError("Unsupported candidate method.")
    estimator = choices[method]()
    params = model.get("parameters", {})
    forbidden = {"random_state", "n_jobs", "warm_start", "n_iter_no_change", "validation_fraction"}
    if forbidden & set(params):
        raise ValueError("Seeds, threads, and internal validation are controlled by the approved design.")
    if any(key not in estimator.get_params() for key in params):
        raise ValueError("Unknown estimator parameter.")
    estimator.set_params(**params)
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=config["seed"])
    if "n_jobs" in estimator.get_params() and method != "linear":
        estimator.set_params(n_jobs=config["budget"]["threads"])
    steps = [("prepare", preprocessing)]
    if model.get("components"):
        if categorical:
            raise ValueError("SVD predictor reduction currently requires numeric features.")
        steps.append(("svd", PCA(n_components=model["components"], svd_solver="full")))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _metrics(y, prediction, probability=None, classes=None) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, log_loss, mean_absolute_error,
        mean_squared_error, r2_score, roc_auc_score,
    )

    if classes is None:
        return {
            "mae": float(mean_absolute_error(y, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "r2": float(r2_score(y, prediction)),
        }
    result = {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "log_loss": float(log_loss(y, probability, labels=classes)),
    }
    if len(classes) == 2:
        result["roc_auc"] = float(roc_auc_score(y, probability[:, 1]))
    return result


def _check_classes(y, classes, minimum: int, groups) -> None:
    counts = pd.DataFrame({"target": np.asarray(y), "group": np.asarray(groups)}).groupby("target")["group"].nunique()
    if set(counts.index) != set(classes) or (counts < minimum).any():
        raise ValueError("Every training/evaluation partition needs minimum support for each class.")


def run_model_experiment(
    data: pd.DataFrame,
    config: dict[str, Any],
    *,
    tuning_authorized: bool = False,
    evaluate_holdout: bool = False,
) -> dict[str, Any]:
    """Run one explicitly configured survey experiment.

    Parameters
    ----------
    data : pandas.DataFrame
        Local respondent data. The caller owns consent and safe output handling.
    config : dict
        Confirmed features, task, validation design, candidates and resource limits.
        ``split_seed`` can fix the holdout independently of estimator ``seed``.
        Identifiers may define splits but cannot be features.
    tuning_authorized : bool
        Explicit permission for the finite parameter grids in this experiment.
    evaluate_holdout : bool
        Evaluate only a previously frozen ``selected_model``. The caller must
        enforce project-level single-use holdout state.

    Returns
    -------
    dict
        ``summary`` contains aggregate diagnostics; ``private`` contains split
        indices and must never be published. ``frozen_config`` replays selected
        configurations without rerunning a search. No fitted model is exported.

    Notes
    -----
    Python-specific API; there is no corresponding R function. This function
    does not establish causal effects or authorize employee-level decisions.
    """
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import ParameterGrid
    from threadpoolctl import threadpool_limits

    _validate(data, config)
    frame = data.copy().reset_index(drop=True)
    task = config["task"]
    if task != "svd":
        target = pd.to_numeric(frame[config["target"]], errors="raise")
        if target.isna().any() or not np.isfinite(target).all():
            raise ValueError("Resolve missing/nonfinite target values explicitly before fitting.")
        if task == "regression" and config.get("score_scale"):
            scale = config["score_scale"]
            if scale < 2 or not target.between(1, scale).all():
                raise ValueError("The target does not match the confirmed survey scale.")
            target = 100 * (target - 1) / (scale - 1)
        if task == "classification" and not np.equal(target, np.floor(target)).all():
            raise ValueError("Classification requires confirmed numeric class codes.")
    numeric = frame[config["features"]].select_dtypes(include=np.number)
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Resolve infinite predictor values explicitly.")
    development, holdout, folds = _splits(frame, config)
    X = frame[config["features"]]
    started = time.monotonic()
    deadline = started + config["budget"]["max_seconds"]
    private = {"development_indices": development.tolist(), "holdout_indices": holdout.tolist()}
    if task == "svd":
        if evaluate_holdout:
            raise ValueError("SVD exploration does not consume a predictive holdout.")
        with threadpool_limits(limits=config["budget"]["threads"]):
            pipeline = _pipeline(frame, config, {})
            pipeline.fit(X.iloc[development])
            transformed = pipeline.named_steps["prepare"].transform(X.iloc[development])
            pca = pipeline.named_steps["model"]
            if not np.isfinite(pca.explained_variance_ratio_).all():
                raise ValueError("SVD has no usable variance; do not interpret this solution.")
            reconstruction = pca.inverse_transform(pca.transform(transformed))
            summary = {
                "task": task, "models": [],
                "svd": {
                    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                    "components": pca.components_.tolist(),
                    "features": config["features"],
                    "reconstruction_rmse": float(np.sqrt(np.mean((transformed - reconstruction) ** 2))),
                },
                "warnings": ["SVD components are exploratory, sign-indeterminate, and not confirmed constructs."],
            }
        return {"summary": summary, "private": private, "frozen_config": copy.deepcopy(config)}
    classes = np.sort(target.unique()) if task == "classification" else None
    if classes is not None and len(classes) < 2:
        raise ValueError("Classification requires at least two supported classes.")
    minimum = config["min_group_size"]
    for train, valid in folds:
        if classes is not None:
            _check_classes(target.iloc[development[train]], classes, minimum, frame.iloc[development[train]][config["group"]])
            _check_classes(target.iloc[development[valid]], classes, minimum, frame.iloc[development[valid]][config["group"]])
    primary = config["metric"]
    valid_metrics = {"mae", "rmse", "r2"} if task == "regression" else {
        "accuracy", "balanced_accuracy", "log_loss", "roc_auc"
    }
    if primary not in valid_metrics or (primary == "roc_auc" and len(classes) != 2):
        raise ValueError("The selected metric is not supported for this task.")
    candidates = []
    for model in config["models"]:
        grid = model.get("grid")
        if grid and not tuning_authorized:
            raise PermissionError("Tuning requires explicit authorization, not just a budget.")
        for parameters in ParameterGrid(grid or {}):
            candidate = {key: value for key, value in model.items() if key != "grid"}
            candidate["parameters"] = {**model.get("parameters", {}), **parameters}
            candidates.append(candidate)
    if not candidates or len(candidates) > config["budget"]["max_trials"]:
        raise ValueError("Candidate count exceeds the approved trial budget.")
    if evaluate_holdout and (
        len(candidates) != 1 or config.get("selected_model") != candidates[0]
    ):
        raise ValueError("Holdout evaluation requires exactly the frozen selected candidate.")
    results = []
    with threadpool_limits(limits=config["budget"]["threads"]):
        for model in candidates:
            if time.monotonic() >= deadline:
                raise TimeoutError("Experiment budget expired between fits; no partial success claimed.")
            pipeline = _pipeline(frame, config, model)
            scores = []
            train_scores = []
            for train, valid in folds:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Experiment budget expired between fits.")
                fit = clone(pipeline).fit(X.iloc[development[train]], target.iloc[development[train]])
                for indices, destination in (
                    (development[valid], scores), (development[train], train_scores)
                ):
                    prediction = fit.predict(X.iloc[indices])
                    probability = fit.predict_proba(X.iloc[indices]) if classes is not None else None
                    destination.append(_metrics(target.iloc[indices], prediction, probability, classes))
            row = {
                "candidate": model, "status": "completed",
                "validation": {key: float(np.mean([score[key] for score in scores])) for key in scores[0]},
                "fold_std": {key: float(np.std([score[key] for score in scores])) for key in scores[0]},
                "training": {key: float(np.mean([score[key] for score in train_scores])) for key in scores[0]},
            }
            if evaluate_holdout:
                if classes is not None:
                    _check_classes(target.iloc[holdout], classes, minimum, frame.iloc[holdout][config["group"]])
                fitted = pipeline.fit(X.iloc[development], target.iloc[development])
                prediction = fitted.predict(X.iloc[holdout])
                probability = fitted.predict_proba(X.iloc[holdout]) if classes is not None else None
                row["holdout"] = _metrics(target.iloc[holdout], prediction, probability, classes)
                if classes is None:
                    residual = target.iloc[holdout].to_numpy() - prediction
                    row["residuals"] = {
                        "mean_error": float(np.mean(residual)),
                        "error_std": float(np.std(residual)),
                    }
                else:
                    from sklearn.metrics import confusion_matrix

                    matrix = confusion_matrix(target.iloc[holdout], prediction, labels=classes)
                    support = [
                        frame.iloc[holdout[
                            (target.iloc[holdout].to_numpy() == actual) & (prediction == predicted)
                        ]][config["group"]].nunique()
                        for actual in classes for predicted in classes
                    ]
                    # Suppress the entire matrix, including margins, if any cell is small.
                    row["confusion_matrix"] = matrix.tolist() if min(support) >= minimum else None
                    row["calibration"] = None
                    if len(classes) == 2:
                        bins = np.minimum((probability[:, 1] * 5).astype(int), 4)
                        curve = []
                        for bucket in range(5):
                            mask = bins == bucket
                            if frame.iloc[holdout[mask]][config["group"]].nunique() < minimum:
                                break
                            curve.append({
                                "mean_probability": float(probability[mask, 1].mean()),
                                "observed_rate": float((target.iloc[holdout[mask]] == classes[1]).mean()),
                            })
                        else:
                            row["calibration"] = curve
                scoring = {
                    "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error",
                    "r2": "r2", "accuracy": "accuracy",
                    "balanced_accuracy": "balanced_accuracy", "log_loss": "neg_log_loss",
                    "roc_auc": "roc_auc",
                }[primary]
                importance = permutation_importance(
                    fitted, X.iloc[holdout], target.iloc[holdout], scoring=scoring,
                    n_repeats=5, random_state=config["seed"], n_jobs=config["budget"]["threads"],
                )
                row["importance"] = [
                    {"feature": feature, "mean": float(mean), "std": float(std)}
                    for feature, mean, std in zip(
                        config["features"], importance.importances_mean, importance.importances_std
                    )
                ]
            results.append(row)
    reverse = primary not in {"mae", "rmse", "log_loss"}
    winner = sorted(results, key=lambda row: row["validation"][primary], reverse=reverse)[0]
    frozen = copy.deepcopy(config)
    frozen["models"] = [winner["candidate"]]
    frozen["selected_model"] = winner["candidate"]
    summary = {
        "task": task, "metric": primary, "models": results,
        "selected_model": winner["candidate"],
        "score_units": "0-100 points" if config.get("score_scale") else "native",
        "holdout_evaluated": evaluate_holdout,
        "class_codes": classes.tolist() if classes is not None else None,
        "warnings": [
            "Prediction and permutation importance do not establish causal effects.",
            "Fold standard deviations are not confidence intervals.",
            "Correlated predictors can share or mask permutation importance.",
            "Budget deadlines are checked between fits; one fit may exceed the remaining time.",
            "Internal early stopping, sample weights and custom decision thresholds are not supported in this initial API.",
        ],
    }
    return {"summary": summary, "private": private, "frozen_config": frozen}
