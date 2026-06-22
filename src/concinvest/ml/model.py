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


# Small TimeSeriesSplit grid for Phase 3 tuning (kept tight so live runs stay fast).
PARAM_GRID: tuple[dict, ...] = (
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5},
    {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 10},
    {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 20},
)


@dataclass
class TrainedModel:
    clf: RandomForestClassifier
    cv_scores: list[float] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    @property
    def mean_cv(self) -> float:
        return float(np.mean(self.cv_scores)) if self.cv_scores else float("nan")

    def predict_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """P(profitable) for each row, aligned to FEATURE_COLS."""
        return self.clf.predict_proba(X[FEATURE_COLS])[:, 1]


def _cv_auc(clf, X: pd.DataFrame, y: pd.Series, n_splits: int) -> list[float]:
    """TimeSeriesSplit ROC-AUC scores (empty if data too small / single-class)."""
    if len(X) < (n_splits + 1) or y.nunique() <= 1:
        return []
    splitter = TimeSeriesSplit(n_splits=n_splits)
    return cross_val_score(clf, X, y, cv=splitter, scoring="roc_auc").tolist()


def tune(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[dict, list[float]]:
    """Pick the ``PARAM_GRID`` entry with the best mean TimeSeriesSplit ROC-AUC.

    Returns ``(best_params, best_cv_scores)``; falls back to the grid's first entry
    when the data is too small to cross-validate.
    """
    X = X[FEATURE_COLS]
    best: tuple[float, dict, list[float]] = (-1.0, dict(PARAM_GRID[0]), [])
    for params in PARAM_GRID:
        clf = RandomForestClassifier(n_jobs=-1, random_state=seed, **params)
        scores = _cv_auc(clf, X, y, n_splits)
        mean = float(np.mean(scores)) if scores else -1.0
        if mean > best[0]:
            best = (mean, dict(params), scores)
    return best[1], best[2]


def train(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 200,
    n_splits: int = 5,
    seed: int = 42,
    params: dict | None = None,
) -> TrainedModel:
    """Fit a RandomForest with TimeSeriesSplit CV and feature importances.

    ``params`` (e.g. from :func:`tune`) overrides the default hyperparameters.
    """
    X = X[FEATURE_COLS]
    params = params or {"n_estimators": n_estimators, "max_depth": None, "min_samples_leaf": 5}
    clf = RandomForestClassifier(n_jobs=-1, random_state=seed, **params)

    cv_scores = _cv_auc(clf, X, y, n_splits)
    clf.fit(X, y)
    importance = dict(
        sorted(
            zip(FEATURE_COLS, clf.feature_importances_),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    return TrainedModel(
        clf=clf, cv_scores=cv_scores, feature_importance=importance, params=params
    )


def tune_and_train(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42) -> TrainedModel:
    """Convenience: TSCV-tune hyperparameters, then fit on all of ``(X, y)``."""
    best_params, _ = tune(X, y, n_splits=n_splits, seed=seed)
    return train(X, y, n_splits=n_splits, seed=seed, params=best_params)
