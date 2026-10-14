"""Step 1: choose Baseline B settings on the validation period, then score both
baselines on the test period (walk-forward, last 120 nights)."""
import pandas as pd

from src.data import load_alerts, build_night_labels
from src.baselines import all_history_rate, make_recent_rate
from src.evaluate import walk_forward, brier, decision_counts

TEST_NIGHTS = 120
VALIDATION_START = "2024-10-01"   # ADR-0001 D9-C: long validation incl. busy periods
B_WINDOWS = [3, 7, 14, 30]
B_SMOOTHING = [False, True]

alerts = load_alerts()
labels = build_night_labels(alerts)

test_nights = labels.index[-TEST_NIGHTS:]
val_nights = labels.index[(labels.index >= VALIDATION_START) & (labels.index < test_nights[0])]
print(f"Validation: {val_nights[0].date()} .. {val_nights[-1].date()} ({len(val_nights)} nights)  "
      f"(alert-night rate {labels[val_nights].mean():.0%})")
print(f"Test:       {test_nights[0].date()} .. {test_nights[-1].date()}  "
      f"(alert-night rate {labels[test_nights].mean():.0%})\n")

# 1) Choose Baseline B settings using ONLY the validation period.
grid = []
for n in B_WINDOWS:
    for smoothing in B_SMOOTHING:
        res = walk_forward(make_recent_rate(n, smoothing), val_nights, labels, alerts)
        grid.append({"n": n, "smoothing": smoothing, "val_brier": brier(res)})
grid = pd.DataFrame(grid).sort_values("val_brier")
print("Baseline B settings on VALIDATION (lower Brier is better):")
print(grid.round(4).to_string(index=False), "\n")
best = grid.iloc[0]
print(f"Locked: n={best.n}, smoothing={best.smoothing}\n")

# 2) Score both baselines on the TEST period, once.
forecasters = {
    "A: all-history rate": all_history_rate,
    f"B: last {best.n} nights{' (smoothed)' if best.smoothing else ''}":
        make_recent_rate(int(best.n), bool(best.smoothing)),
}
rows = []
for name, f in forecasters.items():
    res = walk_forward(f, test_nights, labels, alerts)
    rows.append({"forecaster": name, "brier": brier(res), **decision_counts(res)})
print("TEST results (threshold 50%):")
print(pd.DataFrame(rows).round(3).to_string(index=False))
