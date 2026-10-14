"""Step 4: the ONE official test run (ADR D14). Refuses to run twice.

Locked settings come from results/selection.json and results/revision.json.
Writes results/test_predictions.csv, results/test_weights.csv, results/test_summary.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import DATA_COMMIT, load_alerts, build_night_labels
from src.baselines import all_history_rate, make_recent_rate
from src.evaluate import walk_forward, brier, decision_counts, skill_score, paired_bootstrap
from src.features import FEATURE_SET_A, FEATURE_SET_B, FEATURE_SET_B_PLUS, build_feature_table
from src.model import LogisticForecaster

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SUMMARY = RES / "test_summary.json"
if SUMMARY.exists():
    sys.exit(f"REFUSED: {SUMMARY.name} already exists. The test is run once (ADR D14).")

selection = json.loads((RES / "selection.json").read_text())
revision = json.loads((RES / "revision.json").read_text())
final = revision["final_model"]
SETS = {"A": FEATURE_SET_A, "B": FEATURE_SET_B, "B+approach": FEATURE_SET_B_PLUS}
SUCCESS_SHARE = 0.90

alerts = load_alerts()
labels = build_night_labels(alerts)
test_nights = labels.index[-120:]
assert [str(test_nights[0].date()), str(test_nights[-1].date())] == selection["test"], "test window drifted"

table = build_feature_table(labels, alerts)
model = LogisticForecaster(table, SETS[final["features"]], final["window"])
weights_per_evening = {}


def model_with_weights(history):
    p = model(history)
    night = history.cutoff.tz_convert("Europe/Kyiv").normalize().tz_localize(None)
    weights_per_evening[night] = model.weights()
    return p


b_cfg = selection["baseline_B"]
runs = {
    "model": walk_forward(model_with_weights, test_nights, labels, alerts),
    "baseline_A": walk_forward(all_history_rate, test_nights, labels, alerts),
    "baseline_B": walk_forward(make_recent_rate(b_cfg["n"], b_cfg["smoothing"]), test_nights, labels, alerts),
}

metrics = {name: {"brier": float(brier(r)), **decision_counts(r)} for name, r in runs.items()}
skill = skill_score(metrics["model"]["brier"], metrics["baseline_B"]["brier"])
boot = paired_bootstrap(runs["model"], runs["baseline_B"])
beats_b = bool(skill > 0 and boot["share_model_ahead"] >= SUCCESS_SHARE)  # plain bool for JSON

RES.mkdir(exist_ok=True)
pd.DataFrame({"truth": runs["model"]["truth"], **{f"p_{k}": r["p"] for k, r in runs.items()}}).to_csv(RES / "test_predictions.csv")
pd.DataFrame(weights_per_evening).T.rename_axis("night").to_csv(RES / "test_weights.csv")
SUMMARY.write_text(json.dumps({
    "data_commit": DATA_COMMIT,
    "test": selection["test"],
    "alert_night_rate": float(labels[test_nights].mean()),
    "model": final,
    "baseline_B": b_cfg,
    "metrics": metrics,
    "skill_model_vs_B": float(skill),
    "skill_A_vs_B": float(skill_score(metrics["baseline_A"]["brier"], metrics["baseline_B"]["brier"])),
    "bootstrap": boot,
    "success_criterion": f"skill > 0 and model ahead in >= {SUCCESS_SHARE:.0%} of resamples",
    "model_beats_B": beats_b,
}, indent=2))

print(f"TEST {selection['test'][0]} .. {selection['test'][1]}: {len(test_nights)} nights, "
      f"alert-night rate {labels[test_nights].mean():.0%}\n")
print(pd.DataFrame(metrics).T.round(3).to_string())
print(f"\nSkill model vs B: {skill:+.2%}   (90% bootstrap range {boot['skill_p05']:+.2%} .. {boot['skill_p95']:+.2%}; "
      f"model ahead in {boot['share_model_ahead']:.0%} of resamples)")
print(f"Pre-registered verdict: {'MODEL BEATS B' if beats_b else 'NO RELIABLE IMPROVEMENT OVER B'}")
