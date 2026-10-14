"""Step 6b: POST-HOC experiment v2 - recalibration (ADR D21). NOT a clean test.

1) Choose N from {14, 30, 60} on the VALIDATION period only.
2) Run v2 once on the test period and compare with v1 and baseline B.
The official v1 results (results/test_*) are never modified. Refuses to run twice.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_alerts, build_night_labels
from src.baselines import make_recent_rate
from src.evaluate import walk_forward, brier, decision_counts, skill_score, paired_bootstrap
from src.features import FEATURE_SET_A, FEATURE_SET_B, FEATURE_SET_B_PLUS, build_feature_table
from src.model import LogisticForecaster, RecalibratedForecaster

RES = Path(__file__).resolve().parent.parent / "results"
SUMMARY = RES / "posthoc_test_summary.json"
if SUMMARY.exists():
    sys.exit(f"REFUSED: {SUMMARY.name} already exists. One post-hoc round only (ADR D21).")

selection = json.loads((RES / "selection.json").read_text())
final = json.loads((RES / "revision.json").read_text())["final_model"]
v1_pred = pd.read_csv(RES / "test_predictions.csv", index_col="night", parse_dates=["night"])
SETS = {"A": FEATURE_SET_A, "B": FEATURE_SET_B, "B+approach": FEATURE_SET_B_PLUS}
N_GRID = [14, 30, 60]
TIE = 0.001

alerts = load_alerts()
labels = build_night_labels(alerts)
table = build_feature_table(labels, alerts)
v0, v1_ = selection["validation"]
val = labels.index[(labels.index >= v0) & (labels.index <= v1_)]
test = labels.index[-120:]
assert [str(test[0].date()), str(test[-1].date())] == selection["test"]


def make_v2(n):
    return RecalibratedForecaster(LogisticForecaster(table, SETS[final["features"]], final["window"]), n)


# 1) choose N on validation
grid = pd.DataFrame([{"n": n, "val_brier": brier(walk_forward(make_v2(n), val, labels, alerts))} for n in N_GRID])
best = grid["val_brier"].min()
n_locked = int(grid[grid["val_brier"] <= best + TIE]["n"].max())     # tie rule: larger N
print("N on validation:\n" + grid.round(4).to_string(index=False) + f"\n-> N locked: {n_locked}\n")

# 2) one post-hoc run on the test period
b_cfg = selection["baseline_B"]
v2 = walk_forward(make_v2(n_locked), test, labels, alerts)
b = walk_forward(make_recent_rate(b_cfg["n"], b_cfg["smoothing"]), test, labels, alerts)
v1 = v1_pred[["p_model", "truth"]].rename(columns={"p_model": "p"})
assert v1.index.equals(v2.index) and (v1["truth"] == v2["truth"]).all()

metrics = {name: {"brier": float(brier(r)), "avg_predicted": float(r["p"].mean()), **decision_counts(r)}
           for name, r in {"v2_recalibrated": v2, "v1_model": v1, "baseline_B": b}.items()}
skill = float(skill_score(metrics["v2_recalibrated"]["brier"], metrics["baseline_B"]["brier"]))
boot = paired_bootstrap(v2, b)

pd.DataFrame({"truth": v2["truth"], "p_v2": v2["p"], "p_v1": v1["p"], "p_baseline_B": b["p"]}).to_csv(RES / "posthoc_test_predictions.csv")
SUMMARY.write_text(json.dumps({
    "label": "POST-HOC, NOT A CLEAN TEST: v2 was designed after seeing v1's test errors (ADR D21)",
    "n_grid_validation": grid.round(5).to_dict(orient="records"),
    "n_locked": n_locked,
    "metrics": metrics,
    "skill_v2_vs_B": skill,
    "bootstrap": boot,
    "would_pass_90pct_rule": bool(skill > 0 and boot["share_model_ahead"] >= 0.90),
}, indent=2))
print("POST-HOC test (not a clean test):")
print(pd.DataFrame(metrics).T.round(3).to_string())
print(f"\nSkill v2 vs B: {skill:+.2%} (90% range {boot['skill_p05']:+.2%} .. {boot['skill_p95']:+.2%}; "
      f"v2 ahead in {boot['share_model_ahead']:.0%} of resamples)")
