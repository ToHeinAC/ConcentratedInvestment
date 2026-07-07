"""News sentiment scoring.

Two backends, both aggregated to the Story.md -3..+3 scale:
- **VADER** (default, light): NLTK compound in [-1, 1] scaled by 3.
- **FinBERT** (opt-in via the ``sentiment`` extra): financial transformer; the
  signed probability ``P(positive) - P(negative)`` is scaled by 3.

The backend is chosen by ``config.SENTIMENT_MODEL`` (overridable per call). FinBERT
is imported and loaded lazily on first use so the light path stays dependency-free.
"""

from __future__ import annotations

import threading

from .. import config

_lock = threading.Lock()
_analyzer = None
_finbert = None


def _get_analyzer():
    """Lazily build a shared VADER analyzer, downloading the lexicon if needed."""
    global _analyzer
    if _analyzer is not None:
        return _analyzer
    with _lock:
        if _analyzer is None:
            from nltk.sentiment import SentimentIntensityAnalyzer

            try:
                _analyzer = SentimentIntensityAnalyzer()
            except LookupError:
                import nltk

                nltk.download("vader_lexicon", quiet=True)
                _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def _get_finbert():
    """Lazily build a shared FinBERT sentiment pipeline (heavy, opt-in)."""
    global _finbert
    if _finbert is not None:
        return _finbert
    with _lock:
        if _finbert is None:
            from transformers import pipeline

            _finbert = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                top_k=None,
            )
    return _finbert


def _score_finbert_each(headlines: list[str]) -> list[float]:
    """Per-headline signed FinBERT sentiment (P(pos) - P(neg)) scaled to [-3, 3]."""
    clf = _get_finbert()
    out = []
    for result in clf(headlines):
        scores = {d["label"].lower(): d["score"] for d in result}
        out.append(3.0 * (scores.get("positive", 0.0) - scores.get("negative", 0.0)))
    return out


def score_texts(headlines: list[str], model: str | None = None) -> list[float]:
    """Per-headline sentiment on the [-3, 3] scale; empty list if no headlines.

    ``model`` selects the backend ("vader" / "finbert"); defaults to
    ``config.SENTIMENT_MODEL``.
    """
    if not headlines:
        return []
    if (model or config.SENTIMENT_MODEL).lower() == "finbert":
        return _score_finbert_each(headlines)
    analyzer = _get_analyzer()
    return [3.0 * analyzer.polarity_scores(h)["compound"] for h in headlines]


def score_headlines(headlines: list[str], model: str | None = None) -> float:
    """Mean sentiment of ``headlines`` scaled to [-3, 3]; 0.0 if empty."""
    scores = score_texts(headlines, model=model)
    return sum(scores) / len(scores) if scores else 0.0
