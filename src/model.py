"""Logistic regression forecaster, retrained every evening (ADR D12, D13).

It uses a precomputed feature table (src/features.build_feature_table): each row
was computed from that night's own 18:00 history, so rows never contain later data.
"""
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
