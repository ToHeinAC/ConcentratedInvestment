"""RandomForest forecaster with time-series cross-validation.

Predicts the probability that a proposed action (buy/sell at a leverage tier, given
the current market snapshot) is profitable. The probability doubles as the forecast
``confidence``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from .dataset import FEATURE_COLS


@dataclass
class TrainedModel:
    clf: RandomForestClassifier
    cv_scores: list[float] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)

    @property
    def mean_cv(self) -> float:
        return float(np.mean(self.cv_scores)) if self.cv_scores else float("nan")

    def predict_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """P(profitable) for each row, aligned to FEATURE_COLS."""
        return self.clf.predict_proba(X[FEATURE_COLS])[:, 1]


def train(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 200,
    n_splits: int = 5,
    seed: int = 42,
) -> TrainedModel:
    """Fit a RandomForest with TimeSeriesSplit CV and feature importances."""
    X = X[FEATURE_COLS]
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
    )

    cv_scores: list[float] = []
    if len(X) >= (n_splits + 1) and y.nunique() > 1:
        splitter = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = cross_val_score(clf, X, y, cv=splitter, scoring="roc_auc").tolist()

    clf.fit(X, y)
    importance = dict(
        sorted(
            zip(FEATURE_COLS, clf.feature_importances_),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    return TrainedModel(clf=clf, cv_scores=cv_scores, feature_importance=importance)
