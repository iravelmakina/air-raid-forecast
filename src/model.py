"""Logistic regression forecaster, retrained every evening (ADR D12, D13).

It uses a precomputed feature table (src/features.build_feature_table): each row
was computed from that night's own 18:00 history, so rows never contain later data.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data import TZ

SLIDING_DAYS = 365


class LogisticForecaster:
    def __init__(self, table: pd.DataFrame, features: list[str], window: str):
        assert window in ("expanding", "sliding")
        self.table = table
        self.features = features
        self.window = window
        self.last_fit = None  # (scaler, model) of the most recent evening, for inspection

    def training_rows(self, night: pd.Timestamp) -> pd.DataFrame:
        rows = self.table[self.table.index < night]                     # only nights before tonight
        if self.window == "sliding":
            rows = rows[rows.index >= night - pd.Timedelta(days=SLIDING_DAYS)]
        rows = rows.dropna(subset=self.features)                         # drops nights with < 30 past nights (D17)
        assert rows.index.max() < night, "LEAK: training includes tonight or later"
        return rows

    def __call__(self, history) -> float:
        night = history.cutoff.tz_convert(TZ).normalize().tz_localize(None)
        train = self.training_rows(night)

        scaler = StandardScaler().fit(train[self.features])               # scaling learned from training rows only
        model = LogisticRegression().fit(scaler.transform(train[self.features]), train["truth"])
        self.last_fit = (scaler, model)

        tonight = self.table.loc[[night], self.features]                   # tonight's row: computed at 18:00 tonight
        return float(model.predict_proba(scaler.transform(tonight))[0, 1])

    def weights(self) -> pd.Series:
        """Weights of the last fitted evening (on scaled inputs, so they are comparable)."""
        scaler, model = self.last_fit
        return pd.Series(model.coef_[0], index=self.features)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def recalibrate(p_tonight: float, recent_preds, recent_truth) -> float:
    """Shift tonight's probability on the log-odds scale so that the model's average on
    recent nights would match their smoothed actual alert rate (ADR D21)."""
    p_bar = float(np.mean(recent_preds))
    r = (float(np.sum(recent_truth)) + 1) / (len(recent_truth) + 2)
    return _sigmoid(_logit(p_tonight) + _logit(r) - _logit(p_bar))


class RecalibratedForecaster:
    """Post-hoc v2 (ADR D21): locked model ranks nights, the last N nights set the level."""

    def __init__(self, base: LogisticForecaster, n_recent: int):
        self.base = base
        self.n_recent = n_recent

    def __call__(self, history) -> float:
        p = self.base(history)                                       # trains tonight's model
        night = history.cutoff.tz_convert(TZ).normalize().tz_localize(None)
        recent = self.base.table[self.base.table.index < night].dropna(subset=self.base.features).iloc[-self.n_recent:]
        assert recent.index.max() < night, "LEAK: recalibration uses tonight or later"
        scaler, model = self.base.last_fit
        recent_preds = model.predict_proba(scaler.transform(recent[self.base.features]))[:, 1]
        return recalibrate(p, recent_preds, recent["truth"].to_numpy())
