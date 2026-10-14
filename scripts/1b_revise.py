"""Step 3b: ONE pre-registered revision round on validation (ADR D20).

Candidate: feature set B + approach-region feature, sliding window.
It replaces the locked model only if its Brier is lower by >= 0.001 on BOTH
validation halves. Otherwise the locked model stays. No further rounds.
Never touches test nights. Writes results/revision.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_alerts, build_night_labels
from src.baselines import make_recent_rate
from src.evaluate import walk_forward, brier
from src.features import FEATURE_SET_A, FEATURE_SET_B, FEATURE_SET_B_PLUS, build_feature_table
from src.model import LogisticForecaster

ROOT = Path(__file__).resolve().parent.parent
selection = json.loads((ROOT / "results" / "selection.json").read_text())
SETS = {"A": FEATURE_SET_A, "B": FEATURE_SET_B}
HALF_SPLIT = "2026-01-01"
MARGIN = 0.001

alerts = load_alerts()
labels = build_night_labels(alerts)
v_start, v_end = selection["validation"]
val = labels.index[(labels.index >= v_start) & (labels.index <= v_end)]
assert str(val.max().date()) < selection["test"][0], "validation overlaps test"
halves = {"half 1": val[val < HALF_SPLIT], "half 2": val[val >= HALF_SPLIT]}

table = build_feature_table(labels, alerts)
locked = selection["model"]
forecasters = {
    f"locked: set {locked['features']}, {locked['window']}":
        LogisticForecaster(table, SETS[locked["features"]], locked["window"]),
    "candidate: set B + approach regions, sliding":
        LogisticForecaster(table, FEATURE_SET_B_PLUS, "sliding"),
    f"baseline B: last {selection['baseline_B']['n']}":
        make_recent_rate(selection["baseline_B"]["n"], selection["baseline_B"]["smoothing"]),
}
results = {name: walk_forward(f, val, labels, alerts) for name, f in forecasters.items()}

rows = {}
for name, res in results.items():
    rows[name] = {h: brier(res.loc[nights]) for h, nights in halves.items()}
    rows[name]["full"] = brier(res)
names = list(rows)
locked_name, cand_name = names[0], names[1]
accepted = all(rows[cand_name][h] <= rows[locked_name][h] - MARGIN for h in halves)

print(f"Validation halves: {halves['half 1'][0].date()}..{halves['half 1'][-1].date()} "
      f"({len(halves['half 1'])}) | {halves['half 2'][0].date()}..{halves['half 2'][-1].date()} ({len(halves['half 2'])})\n")
print(f"{'forecaster':48s} {'half 1':>8s} {'half 2':>8s} {'full':>8s}")
for name, r in rows.items():
    print(f"{name:48s} {r['half 1']:8.4f} {r['half 2']:8.4f} {r['full']:8.4f}")
w = forecasters[cand_name].weights().round(3).to_dict()
print(f"\nCandidate last-evening weights: {w}")
print(f"\nAcceptance rule (lower by >= {MARGIN} on both halves): {'ACCEPTED' if accepted else 'REJECTED'}")

final = {"features": "B+approach", "window": "sliding"} if accepted else locked
(ROOT / "results" / "revision.json").write_text(json.dumps({
    "rule": f"candidate must beat locked model by >= {MARGIN} Brier on both validation halves (ADR D20)",
    "halves": {h: [str(n[0].date()), str(n[-1].date())] for h, n in halves.items()},
    "brier": {k: {h: round(v, 5) for h, v in r.items()} for k, r in rows.items()},
    "candidate_last_weights": w,
    "accepted": accepted,
    "final_model": final,
}, indent=2))
print(f"Final model for the test: {final}")
