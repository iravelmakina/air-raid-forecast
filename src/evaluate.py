"""Walk-forward evaluation and metrics."""
from dataclasses import dataclass

import pandas as pd

from src.data import forecast_time, night_window


@dataclass
class History:
    """Everything a forecaster may see at 18:00 on the forecast evening."""
    labels: pd.Series      # outcomes of nights that had fully ended before the cutoff
    alerts: pd.DataFrame   # alerts that started before the cutoff (end unknown if still running)
    cutoff: pd.Timestamp


def history_at(night, labels, alerts) -> History:
    cutoff = forecast_time(night)

    # Past nights: only those whose window ended before the cutoff.
    past_labels = labels[labels.index < night]
    if len(past_labels):
        _, last_end = night_window(past_labels.index[-1])
        assert last_end <= cutoff, f"LEAK: night {past_labels.index[-1]} ends after cutoff"

    # Past alerts: started before the cutoff. If one was still running at 18:00,
    # its end time is not known yet, so we erase it.
    past_alerts = alerts[alerts["started_at"] < cutoff].copy()
    still_running = past_alerts["finished_at"] > cutoff
    past_alerts.loc[still_running, "finished_at"] = pd.NaT
    assert past_alerts["started_at"].max() < cutoff, "LEAK: alert after cutoff"

    return History(labels=past_labels, alerts=past_alerts, cutoff=cutoff)


def walk_forward(forecaster, nights, labels, alerts) -> pd.DataFrame:
    """For each night: build the 18:00 history, ask for a probability, store it with the truth."""
    rows = []
    for night in nights:
        history = history_at(night, labels, alerts)
        p = float(forecaster(history))
        rows.append({"night": night, "p": p, "truth": labels[night]})
    return pd.DataFrame(rows).set_index("night")


def brier(results: pd.DataFrame) -> float:
    """Average of (probability - outcome)^2. 0 = perfect, 0.25 = always 50%, lower is better."""
    return ((results["p"] - results["truth"]) ** 2).mean()


def decision_counts(results: pd.DataFrame, threshold: float = 0.5) -> dict:
    """What happens if someone acts on the forecast: warn when p >= threshold."""
    warn = results["p"] >= threshold
    alert = results["truth"] == 1
    hits = int((warn & alert).sum())
    false_alarms = int((warn & ~alert).sum())
    misses = int((~warn & alert).sum())
    return {
        "hits": hits,
        "false_alarms": false_alarms,
        "misses": misses,
        "correct_quiet": int((~warn & ~alert).sum()),
        "recall": hits / (hits + misses) if hits + misses else float("nan"),
        "precision": hits / (hits + false_alarms) if hits + false_alarms else float("nan"),
    }


def skill_score(model_brier: float, reference_brier: float) -> float:
    """Improvement over the reference baseline. > 0 better, 0 same, < 0 worse."""
    return 1 - model_brier / reference_brier
