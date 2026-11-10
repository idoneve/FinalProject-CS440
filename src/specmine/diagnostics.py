"""Training-only, repository-aware diagnostic analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


class InsufficientFoldSupport(ValueError):
    """The training data cannot support the requested grouped folds."""


def training_loss_curve(features, labels, confidence, *, seed=42, epochs=30):
    """Track training log loss for a separate, incremental SGD classifier."""
    labels = np.asarray(labels, dtype=int)
    weights = np.asarray(confidence, dtype=float) / 100
    weights = weights * compute_sample_weight("balanced", labels)
    if len(labels) != len(features) or len(weights) != len(labels):
        raise ValueError("features, labels, and confidence must align")
    if epochs < 1 or np.unique(labels[weights > 0]).size != 2:
        raise ValueError("Positive-weight examples from both classes are required")
    preprocessing = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    transformed = preprocessing.fit_transform(features)
    model = SGDClassifier(
        loss="log_loss", random_state=seed, learning_rate="constant", eta0=0.01,
    )
    records = []
    for epoch in range(1, epochs + 1):
        model.partial_fit(transformed, labels, classes=np.array([0, 1]), sample_weight=weights)
        loss = log_loss(labels, model.predict_proba(transformed), labels=[0, 1],
                        sample_weight=weights)
        records.append({"epoch": epoch, "training_log_loss": loss})
    return pd.DataFrame(records)


def logistic_tuning_curve(features, labels, groups, confidence, estimator, *,
                          seed=42, c_values=(0.01, 0.1, 1.0, 10.0), folds=3):
    """Score numeric logistic C values on repository-disjoint training folds."""
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups)
    weights = np.asarray(confidence, dtype=float) / 100
    if not (len(features) == len(labels) == len(groups) == len(weights)):
        raise ValueError("features, labels, groups, and confidence must align")
    if folds < 2 or len(np.unique(groups)) < folds:
        raise InsufficientFoldSupport("Not enough distinct training repositories for grouped CV")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(features, labels, groups))
    for train_index, validation_index in splits:
        if (np.unique(labels[train_index][weights[train_index] > 0]).size < 2
                or np.unique(labels[validation_index]).size < 2):
            raise InsufficientFoldSupport("Grouped CV folds need both classes and positive training weights")
        if set(groups[train_index]) & set(groups[validation_index]):
            raise RuntimeError("Repository leakage detected in tuning folds")
    records = []
    for c_value in c_values:
        scores = []
        for train_index, validation_index in splits:
            model = clone(estimator).set_params(classifier__C=c_value)
            model.fit(features.iloc[train_index], labels[train_index],
                      classifier__sample_weight=weights[train_index])
            predicted = model.predict(features.iloc[validation_index])
            scores.append(f1_score(labels[validation_index], predicted, zero_division=0))
        records.append({"C": c_value, "mean_f1": float(np.mean(scores)),
                        "std_f1": float(np.std(scores)), "fold_scores": scores})
    return pd.DataFrame(records), splits
