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

from .dataset import ACTION_FEATURES, FEATURE_COLS

# Features below this RandomForest importance are pruned (action encoding is kept).
# The effective cutoff also scales with the feature count (``KEEP_UNIFORM_FRAC`` of the
# uniform 1/n share), so adding many correlated features — e.g. the momentum lags —
# can't dilute every feature below an absolute cutoff and prune the whole market signal.
MIN_IMPORTANCE: float = 0.02
KEEP_UNIFORM_FRAC: float = 0.5  # keep features >= this fraction of the uniform share


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
    features: list[str] = field(default_factory=lambda: list(FEATURE_COLS))

    @property
    def mean_cv(self) -> float:
        return float(np.mean(self.cv_scores)) if self.cv_scores else float("nan")

    def predict_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """P(profitable) for each row, over the model's (possibly pruned) features."""
        return self.clf.predict_proba(X[self.features])[:, 1]


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
    features: list[str] | None = None,
) -> TrainedModel:
    """Fit a RandomForest with TimeSeriesSplit CV and feature importances.

    ``params`` (e.g. from :func:`tune`) overrides the default hyperparameters;
    ``features`` restricts the column set (default all ``FEATURE_COLS``).
    """
    features = features or list(FEATURE_COLS)
    X = X[features]
    params = params or {"n_estimators": n_estimators, "max_depth": None, "min_samples_leaf": 5}
    clf = RandomForestClassifier(n_jobs=-1, random_state=seed, **params)

    cv_scores = _cv_auc(clf, X, y, n_splits)
    clf.fit(X, y)
    importance = dict(
        sorted(zip(features, clf.feature_importances_), key=lambda kv: kv[1], reverse=True)
    )
    return TrainedModel(
        clf=clf, cv_scores=cv_scores, feature_importance=importance,
        params=params, features=features,
    )


def select_features(trained: TrainedModel, min_importance: float = MIN_IMPORTANCE) -> list[str]:
    """Keep features at/above the prune cutoff plus the action encoding, in
    ``FEATURE_COLS`` order (so the model contract stays a stable superset).

    The cutoff is ``min(min_importance, KEEP_UNIFORM_FRAC / n_features)`` — the absolute
    floor for a small feature set (≤17 features keeps the historical 0.02 behaviour), but
    relaxed for a large one so a wide, correlated set (the momentum lags) doesn't push
    every market feature below an absolute cutoff and collapse the model to action-only."""
    imp = trained.feature_importance
    cutoff = min(min_importance, KEEP_UNIFORM_FRAC / max(len(imp), 1))
    keep = {f for f, v in imp.items() if v >= cutoff}
    keep.update(ACTION_FEATURES)
    return [f for f in FEATURE_COLS if f in keep]


def tune_and_train(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42, prune: bool = True
) -> TrainedModel:
    """TSCV-tune hyperparameters, then fit; optionally prune low-importance features
    and refit on the reduced set."""
    best_params, _ = tune(X, y, n_splits=n_splits, seed=seed)
    full = train(X, y, n_splits=n_splits, seed=seed, params=best_params)
    if not prune:
        return full
    feats = select_features(full)
    if 0 < len(feats) < len(full.features):
        return train(X, y, n_splits=n_splits, seed=seed, params=best_params, features=feats)
    return full
