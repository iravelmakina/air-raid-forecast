# Implementation Plan

- **Status:** Approved 2026-10-14
- **Decisions:** all *why* lives in [ADR-0001](adr/0001-forecast-setup.md). This document covers only *how* and *in what order*.

## Repository structure

```
kyiv-night-alerts/
├── README.md                  # written last (ADR D16)
├── requirements.txt           # exact versions pinned
├── docs/
│   ├── adr/0001-forecast-setup.md
│   └── implementation-plan.md
├── src/
│   ├── data.py        # load pinned data, build night labels (ADR D1–D4)
│   ├── baselines.py   # A all-history, B last N nights, optional smoothing (D7, D8)
│   ├── evaluate.py    # walk-forward, cutoff assertions, Brier, counts, skill, paired bootstrap (D5, D6, D14)
│   ├── features.py    # features 1, 2, 3, 5, 6 (feature 4 dropped, D18), one function each, leakage guard documented (D11, D17)
│   ├── model.py       # logistic regression forecaster: scaling + training per evening (D12, D13)
│   └── analysis.py    # four fixed error-analysis checks (D15)
├── scripts/
│   ├── 1_select.py    # validation only -> results/selection.json
│   ├── 1b_revise.py   # one pre-registered revision round (D20) -> results/revision.json
│   ├── 2_test.py      # single test run -> results/test_*; exits if results already exist
│   ├── 3_analyze.py   # D15 checks on saved test results -> results/analysis.md
│   └── 4_posthoc.py   # post-hoc v2: recalibration (D21) -> results/posthoc_*; refuses to run twice
├── tests/test_leakage.py
└── results/                   # committed outputs
```
`run_baselines.py` (first prototype) is removed; its logic moves into `1_select.py`.

## Mechanics

- **Feature table:** one row per night, computed once from that night's own 18:00 history. The model at evening *d* trains only on rows of nights before *d*.
- **Model per evening:** select training rows by window (expanding, or the last 365 nights) → fit scaling on those rows → fit logistic regression (library defaults) → predict tonight.
- **Selection (`1_select.py`):** (a) B grid N ∈ {3, 7, 14, 30} × smoothing on/off; (b) 4 model candidates (feature set A/B × window expanding/sliding), all walk-forward over the validation period (1 Oct 2024 – 14 Jun 2026). Lowest Brier wins; tie rule per D17. Writes `results/selection.json`.
- **Test (`2_test.py`):** locked model + baselines A and B, walk-forward over 15 Jun – 12 Oct 2026. Bootstrap: 1,000 paired resamples, seed 42. Writes per-night predictions (CSV) and summary metrics (JSON). **Exits with an error if test results already exist.**
- **Analysis (`3_analyze.py`):** reads the saved test outputs only; never re-runs the model.

## Automated checks (`tests/test_leakage.py`)

1. An alert from 21:50 to 22:10 labels that night 1.
2. An alert starting at 18:01 is invisible to that evening's features.
3. An alert running at 18:00 has its end erased.
4. Walk-forward never trains on the night it predicts.
5. A night that ended < 24 h before the snapshot is excluded (added with D19).

## Environment

Python 3.12.3, pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0, pytest. All pinned in `requirements.txt`.
Data: downloaded on the first run from the pinned commit; it is not committed (`.gitignore`).

## Run order and gates

| Step | Work | Gate |
|---|---|---|
| 1 | Pin versions; `features.py` + leakage tests | All tests pass |
| 2 | `model.py` + `1_select.py`; run validation | — |
| 3 | **Review selection with the author** | Author approval before any test run |
| 3b | One pre-registered revision round (ADR D20): `scripts/1b_revise.py` → `results/revision.json` | Acceptance rule applied mechanically |
| 4 | `2_test.py` with bootstrap; run once | — |
| 5 | `analysis.py` + `3_analyze.py` | — |
| 6 | README (the author writes the reflection) | — |
| 6b | Post-hoc v2 (ADR D21): N chosen on validation, then one labelled run on the test period; README section | Pre-registered in the ADR before running |
| 7 | Publish the organized repo to the author's GitHub | Author review of the structure |

## Delivery workflow

- **First run:** done by the AI in its sandbox; each step's files and real outputs are handed off for review.
- **Publishing (option A):** the AI builds the repo locally with one commit per step and hands it over as a git bundle. The author clones it, adds their GitHub remote, and pushes. No credentials are shared in the conversation.
- **Commit authorship:** placeholder identity, replaced by the author before pushing.
