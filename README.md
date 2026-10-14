# Will Kyiv have an air raid alert tonight?

A small Python time-series project on Ukrainian air raid alerts. Every evening at **18:00 (Kyiv time)** it gives a
**probability that Kyiv City will have an air raid alert between 22:00 and 06:00** that night, using only alert
history known at 18:00.

**Short answer from the results:** with alert history alone, a simple rule, *"how often were there night alerts in
the last 30 nights?"*, beats the machine-learning model. The model is too slow to follow the sharp rise in alerts
in 2026. Details below.

All decisions and their reasons are in [`docs/adr/0001-forecast-setup.md`](docs/adr/0001-forecast-setup.md); the
build plan is in [`docs/implementation-plan.md`](docs/implementation-plan.md). The project was built with an AI
assistant (Claude) as the main engineering tool; the full conversation log is submitted separately.

---

## 1. The question and why it is framed this way

- **What:** one probability per night: "alert in Kyiv City at any moment 22:00–06:00?"
- **When:** forecast made at 18:00, so a person can decide where to sleep tonight.
- **Why not "next hour"?** Alert apps already show that live, and "the alert is on now" predicts it almost trivially.
- **Why Kyiv City?** In 2025 most regions switched to district-level alerts, which changed what "region under
  alert" means in the data. Kyiv City is recorded as one unit before and after, so its history is consistent.

## 2. Data

- **Source:** volunteer dataset from [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
  (`volunteer_data_en.csv`), **pinned to commit `7be3358`** so results are reproducible. It is downloaded automatically
  on the first run and is not stored in this repo.
- **Covered:** 26 Feb 2022 – 12 Oct 2026, about 1,690 nights.

Problems found along the way, and what was done about them:

| Problem | What we did |
|---|---|
| The repo's "official" file stopped on 7 Sep 2026 (despite daily repo updates), and 1,874 of its 4,141 Kyiv rows were exact duplicates | Used the volunteer file instead (current to 13 Oct, no duplicates, 97.5% agreement with the official file on nights) |
| Some alerts have no recorded end; the dataset guesses "start + 30 min" (21% of Kyiv alerts in 2026) | Night labels barely depend on end times (tested), so they are kept; the model uses no duration-based inputs |
| The newest night in the file may be missing a late "all clear" message | A night only counts if it ended at least 24 h before the data snapshot |
| Alert levels changed a lot: 27% of nights had alerts in 2023, 57–58% in 2025–26, ~90% in Sep–Oct 2026 | Evaluation is designed around this (see below) |

## 3. How the forecasts are evaluated

- **Replay, night by night ("walk-forward").** For each test night, the forecaster gets only what was known at 18:00
  that evening, makes a forecast, and is scored; then the replay moves one day forward. Code assertions and
  automated tests stop the program if any later information slips in.
- **Periods:**
  - training: everything before each forecast evening;
  - validation, used to choose settings: 1 Oct 2024 – 14 Jun 2026 (622 nights);
  - test, used **once** for the final result: 15 Jun – 12 Oct 2026 (120 nights, 71% with alerts).
- **Score: Brier score.** The average of (predicted probability − what happened)², where what happened is 1 or 0.
  0 is perfect, 0.25 is "always 50%", lower is better. It punishes confident mistakes.
- **Skill score:** improvement over the reference baseline (above 0 = better).
- **Decision view:** if you act on "≥ 50%", how many alert nights do you catch, and how many false alarms do you get?
- **Luck check:** the 120 test nights are resampled 1,000 times to see how much the result could move by chance.
- **Success rule, written before the test:** the model "beats" the baseline only if its skill is above 0 **and** it
  is ahead in at least 90% of resamples.

### The forecasters

| Name | What it does |
|---|---|
| Baseline A | Share of alert nights in all history since 2022 |
| **Baseline B (reference)** | Share of alert nights in the last 30 nights (smoothed so it is never exactly 0% or 100%); window chosen on validation |
| Model | Logistic regression (a weighted formula turned into a probability), retrained every evening on the last 12 months. Inputs: alert rate over the last 30 nights, alert last night, any Kyiv alert today 06:00–18:00 |

The model was chosen on validation from 4 candidates (3 or 5 inputs × all history or last 12 months). One extra idea
(alerts in regions around Kyiv, 15:00–18:00) was tested in a single pre-registered round and **rejected** by its
rule before the test.

## 4. Results (test: 15 Jun – 12 Oct 2026)

| Forecaster | Brier ↓ | Alert nights caught (of 85) | False alarms (of 35 quiet nights) |
|---|---|---|---|
| Baseline A | 0.275 | 0 | 0 |
| **Baseline B** | **0.200** | 85 | 35 |
| Model | 0.226 | 40 | 5 |

- **Skill of the model vs baseline B: −12.9%** (90% range from resampling: −25.7% to −3.1%; the model is ahead in only
  2% of resamples).
- **Verdict (pre-registered): no improvement over the baseline. The model is reliably worse on this period.**
- Baseline B at 50% simply warns every night in this busy period. The model warns rarely but is right 89% of the
  time when it does; it misses too many alert nights.

## 5. Why the model lost (error analysis)

Full tables: [`results/analysis.md`](results/analysis.md).

1. **Its probabilities were too low.** It predicted 53% on average while 71% of nights had alerts. It never said
   ≥ 80%, even when ~90% of nights had alerts. It learns its level from the past 12 months (~50%) and cannot catch
   up with a sudden rise; the 30-night rule follows the level directly.
2. **Its main input was unstable.** The weight on "alert rate over the last 30 nights" changed sign during the test
   (from −0.23 to +0.33). The link between past and future alerts in the recent year is weak and inconsistent.
3. **It got worse as alerts increased:** −6% vs baseline in Jun–Aug (58% alert nights), −24% in Aug–Oct (83%).
4. **Some nights cannot be predicted from history:** several quiet nights in a very busy September fooled every
   forecaster.

**Takeaway:** at 18:00, alert history mainly tells you *the current level of danger*. Knowing the last month's level
is as good as it gets with this data; predicting *which* nights will be quiet needs information about the attacks
themselves.

## 6. Post-hoc experiment (not a clean test)

After the official test, we tried **one** fix for the diagnosed problem, written down before running it
([ADR D21](docs/adr/0001-forecast-setup.md)). The model still decides which nights look riskier than average, but its
overall level is adjusted every evening to match the alert rate of the last N nights. N was chosen on validation
(N = 60).

**This fix was designed after seeing where the model failed on the test nights, so its score on those same nights is
optimistic.** It is not a fair estimate of future performance and does not replace the official result above.

| Forecaster (test nights) | Brier ↓ | Average forecast (actual: 71%) |
|---|---|---|
| Model v1 (official) | 0.226 | 53% |
| **Model v2 (recalibrated, post-hoc)** | **0.199** | 63% |
| Baseline B | 0.200 | 66% |

- v2 ties baseline B: skill +0.8%, 90% range −3.5% to +4.8%; v2 is ahead in 64% of resamples, far from the 90% bar.
- On the calmer validation period, recalibration made the model *worse* (0.244 vs 0.241). It helps when the level jumps,
  not in general.
- **Conclusion:** fixing the level brings the model up to the simple 30-night rule, but not past it. History at 18:00
  carries little information about *which* nights will be attacked. A clean check of v2 needs new nights after
  12 Oct 2026.

## 7. Limitations

- One city, one forecast time, one data source.
- The test is short (120 nights) and covers one sharp rise in attacks.
- Baseline scores on the test were looked at once early in the project, before some settings were fixed. This is
  disclosed in the ADR; the model itself touched the test only once.
- The first official test run crashed while saving results. Nothing was viewed; the bug was fixed and the rerun was
  verified to be identical (fixed data, code, and random seed). See the git history.
- The replay assumes alerts up to 18:00 are known at 18:00. That is true live, but this dataset updates once a day.

## 8. Future work

- **Warning signals** (e.g. Ukrainian Air Force Telegram posts about launches or bomber activity). They were left out
  on purpose to keep scope small. Note: Telegram keeps only the last edit time of a post, so posts edited after 18:00
  must be dropped to avoid leakage.
- **A later forecast time** (e.g. 21:00), when attacks may already be visible. This trades warning time for accuracy.
- **Check v2 on new nights** after 12 Oct 2026. That is the only clean way to know if recalibration really helps.

## 9. How to run

Requires Python 3.12.

```bash
pip install -r requirements.txt
pytest                         # 6 leakage/label/math tests
python scripts/1_select.py     # validation: choose baseline B settings and the model -> results/selection.json
python scripts/1b_revise.py    # one pre-registered revision round -> results/revision.json
python scripts/2_test.py       # the single test run -> results/test_*.{csv,json}
python scripts/3_analyze.py    # error analysis -> results/analysis.md
python scripts/4_posthoc.py    # post-hoc v2 (recalibration), labelled not-clean -> results/posthoc_*
```

Each script takes from about a minute to a few minutes. `2_test.py` and `4_posthoc.py` refuse to run if their summary files exist; to reproduce, run them in a fresh
clone after deleting `results/test_*` and `results/posthoc_*`.

## 10. Repository map

```
src/data.py         load pinned data, build night labels
src/features.py     inputs computed at 18:00, each with its leakage guard
src/baselines.py    baselines A and B
src/model.py        logistic regression, retrained each evening
src/evaluate.py     walk-forward replay, scores, resampling
src/analysis.py     error-analysis checks
scripts/            the four steps above, in order
tests/              leakage and label tests
results/            all outputs used in this README
docs/               decision record (ADR) and implementation plan
```
