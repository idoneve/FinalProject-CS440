import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from specmine.diagnostics import (
    InsufficientFoldSupport,
    logistic_tuning_curve,
    training_loss_curve,
)


@pytest.fixture
def diagnostic_data():
    labels = np.tile([0, 1], 12)
    features = pd.DataFrame({"signal": labels * 2.0, "noise": np.arange(24) % 3})
    groups = np.repeat([f"repo-{i}" for i in range(12)], 2)
    confidence = np.arange(24) + 70
    estimator = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression())
    estimator.steps[-1] = ("classifier", estimator.steps[-1][1])
    return features, labels, groups, confidence, estimator


def test_loss_curve_reports_real_weighted_epoch_values(diagnostic_data):
    features, labels, _, confidence, _ = diagnostic_data
    first = training_loss_curve(features, labels, confidence, seed=7, epochs=8)
    second = training_loss_curve(features, labels, confidence, seed=7, epochs=8)
    pd.testing.assert_frame_equal(first, second)
    assert first["epoch"].tolist() == list(range(1, 9))
    assert np.isfinite(first["training_log_loss"]).all()
    assert first["training_log_loss"].iloc[-1] < first["training_log_loss"].iloc[0]


def test_tuning_uses_only_group_disjoint_training_folds(diagnostic_data):
    features, labels, groups, confidence, estimator = diagnostic_data
    results, splits = logistic_tuning_curve(
        features, labels, groups, confidence, estimator, seed=7, c_values=(0.1, 1.0)
    )
    assert results["C"].tolist() == [0.1, 1.0]
    assert results["mean_f1"].between(0, 1).all()
    assert all(len(scores) == 3 for scores in results["fold_scores"])
    for training, validation in splits:
        assert not set(groups[training]) & set(groups[validation])
        assert set(labels[training]) == {0, 1}
        assert set(labels[validation]) == {0, 1}


def test_tuning_reports_insufficient_group_support(diagnostic_data):
    features, labels, groups, confidence, estimator = diagnostic_data
    with pytest.raises(InsufficientFoldSupport):
        logistic_tuning_curve(features, labels, np.zeros_like(groups), confidence, estimator)
