"""Step 5: error analysis on the SAVED test outputs (ADR D15). Never re-runs the model.
Feature values for the worst nights are recomputed from data (features only, no model).
Writes results/analysis.md."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis import md_table, weights_summary, calibration_table, worst_nights, halves
from src.data import load_alerts, build_night_labels
from src.features import FEATURE_SET_A, build_feature_table

RES = Path(__file__).resolve().parent.parent / "results"
pred = pd.read_csv(RES / "test_predictions.csv", index_col="night", parse_dates=["night"])
weights = pd.read_csv(RES / "test_weights.csv", index_col="night", parse_dates=["night"])
summary = json.loads((RES / "test_summary.json").read_text())

alerts = load_alerts()
features = build_feature_table(build_night_labels(alerts), alerts).loc[pred.index, FEATURE_SET_A]

overall = pd.DataFrame({
    "avg_predicted": [pred["p_model"].mean(), pred["p_baseline_B"].mean()],
    "actual_alert_rate": [pred["truth"].mean()] * 2,
}, index=pd.Index(["model", "baseline B"], name="forecaster"))

worst = worst_nights(pred, features)
worst.index = worst.index.date
worst.index.name = "night"

sections = [
    "# Error analysis (test period, ADR D15)\n",
    f"Test {summary['test'][0]} .. {summary['test'][1]}; model = set {summary['model']['features']}, "
    f"{summary['model']['window']} window. Findings are indicative only (120 nights).\n",
    "## 1. Weights (scaled inputs)\n", md_table(weights_summary(weights)),
    "\n## 2. Calibration\n", md_table(overall), "", md_table(calibration_table(pred)),
    "\n## 3. Worst 10 nights (model)\n", md_table(worst),
    "\n## 4. Test halves\n", md_table(halves(pred)),
]
(RES / "analysis.md").write_text("\n".join(sections) + "\n")
print("\n".join(sections))
