"""Features computed at 18:00 (Kyiv time) for the night that starts that evening.

Every feature reads ONLY a History object (see src/evaluate.py), which already
contains nothing from after the cutoff. Each docstring states its leakage guard.
Decisions: ADR-0001 D11 (feature sets) and D17 (exact definitions).
"""
import numpy as np
import pandas as pd

from src.data import REGION, TZ
from src.evaluate import History, history_at

RATE_WINDOW = 30          # feature 1: nights
QUIET_CAP_HOURS = 168     # feature 5: cap (ADR D17)
DAYTIME_START_H = 6       # features 3 and 6: window 06:00 .. cutoff (18:00)

FEATURE_SET_A = ["rate_30", "last_night", "kyiv_daytime_alert"]
FEATURE_SET_B = FEATURE_SET_A + ["hours_since_kyiv_start", "other_regions_daytime"]
# Feature 4 ("Kyiv alert active at 18:00") was removed: it depends on end times,
# which are guessed for some alerts (ADR D4, D18).


def _day_start_local(history: History) -> pd.Timestamp:
    """Midnight (Kyiv time) of the forecast day."""
    return history.cutoff.tz_convert(TZ).normalize()


def _daytime_window_utc(history: History) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = (_day_start_local(history) + pd.Timedelta(hours=DAYTIME_START_H)).tz_convert("UTC")
    return start, history.cutoff


def _kyiv(history: History) -> pd.DataFrame:
    return history.alerts[history.alerts["region"] == REGION]


def rate_30(history: History) -> float:
    """1. Share of alert nights among the last 30 finished nights.
    Guard: History.labels holds only nights that ended before the cutoff.
    NaN if fewer than 30 past nights (such rows are dropped from training, D17)."""
    if len(history.labels) < RATE_WINDOW:
        return np.nan
    return float(history.labels.iloc[-RATE_WINDOW:].mean())


def last_night(history: History) -> float:
    """2. Outcome of last night (ended 06:00 today, i.e. before the 18:00 cutoff).
    Guard: read from History.labels only. NaN if that night is missing."""
    prev = _day_start_local(history).tz_localize(None) - pd.Timedelta(days=1)
    return float(history.labels.get(prev, np.nan))


def kyiv_daytime_alert(history: History) -> float:
    """3. 1 if a Kyiv City alert STARTED between 06:00 and 18:00 today.
    Guard: History.alerts holds only alerts started before the cutoff."""
    start, end = _daytime_window_utc(history)
    k = _kyiv(history)
    return float(((k["started_at"] >= start) & (k["started_at"] < end)).any())


def hours_since_kyiv_start(history: History) -> float:
    """5. Hours since the last Kyiv City alert STARTED, capped at 168 h.
    Guard: start times only (end times may be unknown or guessed, D4)."""
    k = _kyiv(history)
    if k.empty:
        return float(QUIET_CAP_HOURS)
    hours = (history.cutoff - k["started_at"].max()).total_seconds() / 3600
    return float(min(hours, QUIET_CAP_HOURS))


def other_regions_daytime(history: History) -> float:
    """6. Number of regions OTHER than Kyiv City with an alert started 06:00–18:00 today.
    Guard: alerts started before the cutoff only."""
    start, end = _daytime_window_utc(history)
    a = history.alerts
    mask = (a["region"] != REGION) & (a["started_at"] >= start) & (a["started_at"] < end)
    return float(a.loc[mask, "region"].nunique())


FEATURES = {
    "rate_30": rate_30,
    "last_night": last_night,
    "kyiv_daytime_alert": kyiv_daytime_alert,
    "hours_since_kyiv_start": hours_since_kyiv_start,
    "other_regions_daytime": other_regions_daytime,
}


def compute_features(history: History) -> dict:
    return {name: f(history) for name, f in FEATURES.items()}


def build_feature_table(labels: pd.Series, alerts: pd.DataFrame) -> pd.DataFrame:
    """One row per night; each row is computed from THAT night's own 18:00 history."""
    rows = {night: compute_features(history_at(night, labels, alerts)) for night in labels.index}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "night"
    table["truth"] = labels
    return table
