"""News sentiment scoring.

Phase 1 baseline: NLTK VADER on headlines, aggregated to a daily score on the
Story.md -3..+3 integer-ish scale (VADER compound in [-1, 1] scaled by 3). FinBERT
and German-source scraping are layered in during Phase 2.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_analyzer = None


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


def score_headlines(headlines: list[str]) -> float:
    """Mean VADER compound of ``headlines`` scaled to [-3, 3]; 0.0 if empty."""
    if not headlines:
        return 0.0
    analyzer = _get_analyzer()
    compounds = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    if not compounds:
        return 0.0
    return 3.0 * (sum(compounds) / len(compounds))
