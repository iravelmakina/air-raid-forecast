"""Step 2: choose settings on the VALIDATION period only (ADR D8, D9, D11–D13, D17).

Never touches test nights. Writes results/selection.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import DATA_COMMIT, load_alerts, build_night_labels
from src.baselines import all_history_rate, make_recent_rate
from src.evaluate import walk_forward, brier, decision_counts, skill_score
from src.features import FEATURE_SET_A, FEATURE_SET_B, build_feature_table
from src.model import LogisticForecaster

TEST_NIGHTS = 120
VALIDATION_START = "2024-10-01"
B_GRID = [(n, s) for n in (3, 7, 14, 30) for s in (False, True)]
# Ordered from simplest to most complex, for the tie rule (D17).
CANDIDATES = [("A", "expanding"), ("A", "sliding"), ("B", "expanding"), ("B", "sliding")]
FEATURE_SETS = {"A": FEATURE_SET_A, "B": FEATURE_SET_B}
TIE = 0.001
OUT = Path(__file__).resolve().parent.parent / "results" / "selection.json"

alerts = load_alerts()
labels = build_night_labels(alerts)
test_nights = labels.index[-TEST_NIGHTS:]
val_nights = labels.index[(labels.index >= VALIDATION_START) & (labels.index < test_nights[0])]
assert val_nights.max() < test_nights.min(), "validation overlaps test"
print(f"Validation {val_nights[0].date()} .. {val_nights[-1].date()}: {len(val_nights)} nights, "
      f"alert-night rate {labels[val_nights].mean():.0%}")
print(f"Test (NOT used here) {test_nights[0].date()} .. {test_nights[-1].date()}\n")

# 1) Baseline B settings
b_rows = []
for n, smooth in B_GRID:
    res = walk_forward(make_recent_rate(n, smooth), val_nights, labels, alerts)
    b_rows.append({"n": n, "smoothing": smooth, "val_brier": brier(res)})
b_table = pd.DataFrame(b_rows).sort_values("val_brier")
best_b = b_table.iloc[0]
b_forecaster = make_recent_rate(int(best_b.n), bool(best_b.smoothing))
b_val = walk_forward(b_forecaster, val_nights, labels, alerts)
a_val = walk_forward(all_history_rate, val_nights, labels, alerts)
print("Baseline B grid (validation):\n" + b_table.round(4).to_string(index=False))
print(f"-> B locked: last {int(best_b.n)} nights, smoothing={bool(best_b.smoothing)}\n")

# 2) Model candidates
print("Building feature table (each row from its own 18:00 history)...")
table = build_feature_table(labels, alerts)
m_rows, m_results, weights = [], {}, {}
for fset, window in CANDIDATES:
    f = LogisticForecaster(table, FEATURE_SETS[fset], window)
    res = walk_forward(f, val_nights, labels, alerts)
    m_results[(fset, window)] = res
    weights[f"{fset}/{window}"] = f.weights().round(3).to_dict()
    m_rows.append({"features": fset, "window": window, "val_brier": brier(res),
                   "skill_vs_B": skill_score(brier(res), brier(b_val))})
m_table = pd.DataFrame(m_rows)
best_val = m_table["val_brier"].min()
eligible = m_table[m_table["val_brier"] <= best_val + TIE]          # tie rule: simplest within 0.001
chosen = eligible.iloc[0]                                            # CANDIDATES order = simplest first
print("\nModel candidates (validation):\n" + m_table.round(4).to_string(index=False))
print(f"-> Model locked: features {chosen.features}, window {chosen.window}\n")

summary = []
for name, res in [("A: all-history", a_val), (f"B: last {int(best_b.n)}", b_val),
                  ("Model", m_results[(chosen.features, chosen.window)])]:
    summary.append({"forecaster": name, "brier": brier(res), **decision_counts(res)})
print("Validation summary (threshold 50%):\n" + pd.DataFrame(summary).round(3).to_string(index=False))
print("\nLast-evening weights per candidate (scaled inputs):")
for k, w in weights.items():
    print(f"  {k}: {w}")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({
    "data_commit": DATA_COMMIT,
    "validation": [str(val_nights[0].date()), str(val_nights[-1].date())],
    "test": [str(test_nights[0].date()), str(test_nights[-1].date())],
    "baseline_B": {"n": int(best_b.n), "smoothing": bool(best_b.smoothing)},
    "model": {"features": chosen.features, "window": chosen.window},
    "baseline_B_grid": b_table.round(5).to_dict(orient="records"),
    "model_candidates": m_table.round(5).to_dict(orient="records"),
    "validation_summary": pd.DataFrame(summary).round(4).to_dict(orient="records"),
}, indent=2))
print(f"\nWrote {OUT.relative_to(OUT.parent.parent)}")
