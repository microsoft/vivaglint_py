import copy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
from vivaglint.modeling import _splits, run_model_experiment


@pytest.fixture
def experiment():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        "person": np.arange(240),
        "Q_work": rng.uniform(1, 5, 240),
        "Q_resources": rng.uniform(1, 5, 240),
    })
    frame["Q_outcome"] = 1 + 0.6 * (frame["Q_work"] - 1) + rng.uniform(0, 0.4, 240)
    config = {
        "features": ["Q_work", "Q_resources"], "target": "Q_outcome",
        "group": "person", "task": "regression", "seed": 42, "score_scale": 5,
        "min_group_size": 5, "metric": "mae",
        "validation": {"kind": "group", "folds": 3, "holdout_fraction": 0.2},
        "models": [
            {"id": "baseline", "method": "dummy"},
            {"id": "linear", "method": "linear"},
            {"id": "forest", "method": "random_forest", "parameters": {"n_estimators": 8}},
            {"id": "boost", "method": "gradient_boosting", "parameters": {"n_estimators": 8}},
        ],
        "budget": {"threads": 1, "max_trials": 8, "max_rows": 500, "max_seconds": 60},
    }
    return frame, config


def test_comparison_frozen_replay_and_holdout(experiment):
    frame, config = experiment
    result = run_model_experiment(frame, config)
    assert len(result["summary"]["models"]) == 4
    assert result["summary"]["selected_model"]["method"] != "dummy"
    assert result["summary"]["score_units"] == "0-100 points"
    assert not result["summary"]["holdout_evaluated"]
    frozen = result["frozen_config"]
    first = run_model_experiment(frame, frozen)
    second = run_model_experiment(frame, frozen)
    assert first["summary"] == second["summary"]
    heldout = run_model_experiment(frame, frozen, evaluate_holdout=True)
    assert heldout["summary"]["models"][0]["holdout"]["mae"] < 10
    assert len(heldout["summary"]["models"][0]["importance"]) == 2
    assert "development_indices" not in result["summary"]


def test_grid_requires_independent_tuning_consent_and_budget(experiment):
    frame, config = experiment
    config["models"] = [{
        "id": "forest", "method": "random_forest",
        "parameters": {"n_estimators": 5}, "grid": {"max_depth": [2, 4]},
    }]
    with pytest.raises(PermissionError, match="explicit authorization"):
        run_model_experiment(frame, config)
    assert len(run_model_experiment(frame, config, tuning_authorized=True)["summary"]["models"]) == 2
    config["budget"]["max_trials"] = 1
    with pytest.raises(ValueError, match="trial budget"):
        run_model_experiment(frame, config, tuning_authorized=True)


@pytest.mark.parametrize("feature", ["Q_outcome", "employee_id", "age", "Q_comment"])
def test_unsafe_and_target_features_are_rejected(experiment, feature):
    frame, config = experiment
    frame[feature] = np.arange(len(frame))
    config["features"].append(feature)
    with pytest.raises(ValueError, match="feature"):
        run_model_experiment(frame, config)


def test_repeated_people_do_not_cross_folds(experiment):
    frame, config = experiment
    frame["person"] = np.arange(240) // 2
    dev, holdout, cv = _splits(frame, config)
    assert not set(frame.iloc[dev]["person"]) & set(frame.iloc[holdout]["person"])
    for train, valid in cv:
        assert not set(frame.iloc[dev[train]]["person"]) & set(frame.iloc[dev[valid]]["person"])


def test_chronological_folds_preserve_dates_and_reject_group_overlap(experiment):
    frame, config = experiment
    frame["date"] = pd.date_range("2020-01-01", periods=len(frame))
    config["time"] = "date"
    config["validation"]["kind"] = "time"
    dev, holdout, cv = _splits(frame, config)
    assert frame.iloc[dev]["date"].max() < frame.iloc[holdout]["date"].min()
    for train, valid in cv:
        assert frame.iloc[dev[train]]["date"].max() < frame.iloc[dev[valid]]["date"].min()
    frame["person"] = np.arange(240) % 20
    with pytest.raises(ValueError, match="overlap"):
        _splits(frame, config)


def test_model_seed_can_change_without_moving_the_reserved_holdout(experiment):
    frame, config = experiment
    expected = _splits(frame, config)[1]
    config["split_seed"] = config["seed"]
    config["seed"] += 1
    assert np.array_equal(expected, _splits(frame, config)[1])


def test_classification_metrics_and_rare_class_gates(experiment):
    frame, config = experiment
    frame["Q_outcome"] = (frame["Q_work"] > 3).astype(int)
    config.update(task="classification", metric="balanced_accuracy")
    config.pop("score_scale")
    result = run_model_experiment(frame, config)
    assert "roc_auc" in result["summary"]["models"][0]["validation"]
    frame["Q_outcome"] = 0
    frame.loc[0, "Q_outcome"] = 1
    with pytest.raises(ValueError, match="support"):
        run_model_experiment(frame, config)


def test_svd_is_separate_and_reconstructs_full_rank(experiment):
    frame, config = experiment
    config.update(task="svd", components=2)
    config["models"] = []
    result = run_model_experiment(frame, config)
    assert result["summary"]["models"] == []
    assert sum(result["summary"]["svd"]["explained_variance_ratio"]) == pytest.approx(1)
    assert result["summary"]["svd"]["reconstruction_rmse"] < 1e-10
    assert result["summary"] == run_model_experiment(frame, copy.deepcopy(config))["summary"]


def test_holdout_requires_frozen_candidate(experiment):
    frame, config = experiment
    with pytest.raises(ValueError, match="frozen"):
        run_model_experiment(frame, config, evaluate_holdout=True)


def test_preprocessing_is_fitted_only_on_training_rows(experiment, monkeypatch):
    from sklearn.preprocessing import StandardScaler

    frame, config = experiment
    config["models"] = [{"id": "linear", "method": "linear"}]
    sizes = []
    original = StandardScaler.fit

    def observe(self, X, *args, **kwargs):
        sizes.append(len(X))
        return original(self, X, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", observe)
    run_model_experiment(frame, config)
    assert sizes and max(sizes) < len(frame) * 0.8


def test_categorical_preparation_and_missing_inputs(experiment):
    frame, config = experiment
    frame["work_mode"] = np.where(frame["Q_work"] > 3, "onsite", "remote")
    config["features"].append("work_mode")
    assert run_model_experiment(frame, config)["summary"]["models"]
    frame["Q_resources"] = np.nan
    with pytest.raises(ValueError, match="empty"):
        run_model_experiment(frame, config)


def test_target_copy_and_sparse_confusion_are_not_shared(experiment):
    frame, config = experiment
    frame["Q_copy"] = frame["Q_outcome"]
    config["features"].append("Q_copy")
    with pytest.raises(ValueError, match="copy"):
        run_model_experiment(frame, config)
    config["features"].remove("Q_copy")
    frame["Q_outcome"] = (frame["Q_work"] > 3).astype(int)
    config.update(task="classification", metric="accuracy")
    config.pop("score_scale")
    config["models"] = [{"id": "tree", "method": "random_forest", "parameters": {"n_estimators": 20}}]
    frozen = run_model_experiment(frame, config)["frozen_config"]
    result = run_model_experiment(frame, frozen, evaluate_holdout=True)
    assert result["summary"]["models"][0]["confusion_matrix"] is None
    assert result["summary"]["models"][0]["calibration"] is None


def test_manager_survey_item_is_not_confused_with_manager_identity(experiment):
    frame, config = experiment
    frame["Q_manager_support"] = frame["Q_work"] + 0.1
    config["features"].append("Q_manager_support")
    assert run_model_experiment(frame, config)["summary"]["models"]
