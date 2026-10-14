"""Baselines: forecasting rules with no learning, only counting.

Every forecaster has the same signature: it receives a History (only data
known at 18:00 that evening) and returns a probability for tonight.
"""
from functools import partial


def all_history_rate(history) -> float:
    """Baseline A: share of alert nights over all past nights."""
    return history.labels.mean()


def recent_rate(history, n: int, smoothing: bool) -> float:
    """Baseline B: share of alert nights in the last n nights.

    With smoothing we add one imaginary alert night and one imaginary quiet
    night, so the result is never exactly 0% or 100%.
    """
    window = history.labels.iloc[-n:]
    if smoothing:
        return (window.sum() + 1) / (len(window) + 2)
    return window.mean()


def make_recent_rate(n: int, smoothing: bool):
    """Build a Baseline B forecaster with fixed settings."""
    return partial(recent_rate, n=n, smoothing=smoothing)
