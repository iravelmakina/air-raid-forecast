# How this project was built: a summary of the AI conversation

This is a readable summary of the full conversation between the author (a backend engineer new to ML) and the AI
assistant (Claude). The full log is submitted separately; the decisions themselves are recorded in
[ADR-0001](adr/0001-forecast-setup.md). This file was drafted by the AI from the conversation and reviewed by the author.

**Working agreement set at the start:** one step at a time; at each step 2–3 options with downsides and costs; the
author decides; the AI labels guesses as `[ASSUMPTION]` or `[UNSURE]`, explains every new term, pushes back, and
does not expand scope.

---

## 1. Timeline

### Phase 1: Framing the question
- **The AI's first proposals:** predict alert yes/no for the next hour, alert yes/no for the next day at 00:00, or
  minutes under alert. It recommended "yes/no for tomorrow, at 00:00".
- **The author challenged it twice:** "why 00:00 and not 18:00?" and "is 'at least X hours tomorrow' actually useful?"
  The AI admitted 00:00 was chosen for convenience, and that "X hours" was a proxy nobody would act on.
- **Final framing:** at 18:00, the probability of an alert in **Kyiv City** during **22:00–06:00** that night. It maps
  to a real decision: where to sleep tonight.

### Phase 2: Data
- **Source:** Vadimkin's air-raid dataset on GitHub, later pinned to one commit.
- **The district-level alert reform:** the AI first described it as happening "around August 2025". Checking news
  sources showed it was staggered by region, from Feb 2025 (Kharkiv city) to a nationwide rollout in Dec 2025. Kyiv
  City was chosen partly because its records stayed city-level.
- **A public repo from another applicant** to the same program was found. It was flagged so the framing would not
  copy it.
- **Data checks run by the AI, with findings:**
  - The repo's "official" file had stopped on **7 Sep 2026**, although the AI had repeated the README's claim that
    it is "updated daily".
  - **1,874 of 4,141** Kyiv rows in the official file were duplicates.
  - The volunteer file was current to 13 Oct and agreed with the official file on **97.5%** of nights, so the author
    chose it.
  - The share of nights with alerts changed sharply: 27% (2023) → 57–58% (2025–26) → **87–92%** (Sep–Oct 2026).
  - Some alerts have a guessed end time (start + 30 min): **21%** of Kyiv alerts in 2026. Dropping them would corrupt
    18 of 120 test labels; stretching them changes none. So they were kept, with no duration-based inputs.

### Phase 3: Evaluation design
- **Split:** walk-forward replay over the last 120 nights. Each evening, train on everything before 18:00 and
  predict that night. Chosen over a single frozen split because of the regime change.
- **Metric:** after several rounds of "explain this simpler", the author chose the **Brier score + skill score** to
  judge probability quality, plus **hits / false alarms / misses at 50%** to show what happens if someone acts on the
  forecast. Accuracy was dropped: with ~70–90% alert nights, "always yes" looks good on accuracy while knowing nothing.
- **Baselines:** A (all-history rate) and B (rate over the last N nights). Another rule (last night's outcome) was
  dropped. The AI found a flaw in its own B: it gave exactly 0% or 100% on 20 of 120 nights. The fix was smoothing,
  with N chosen on a validation period.
- **Validation period:**
  - **The author noticed the AI had chosen the validation dates without asking.** The AI explained its reasoning
    and opened it as a decision.
  - The AI then recommended a 240-night window, claiming it covered busier periods. A data check showed this was
    **wrong** (only 48% alert nights).
  - Final choice: a **long validation, 1 Oct 2024 – 14 Jun 2026**, which includes busy periods similar to the test.
- **Test seen early:** the AI had used test nights to illustrate baselines before the settings were fixed. The
  author asked whether this was forbidden. Conclusion: it doesn't break the brief's leakage rule, but it is a
  good-practice issue. It was disclosed in the ADR, with a rule that later changes need a reason other than test
  scores.
- **Text/Telegram data:** skipped to keep scope small; recorded as future work.

### Phase 4: Decisions first, then a plan
- **The author set a rule:** no code until all decisions are made and a plan is approved.
- **Decisions made:**
  - features: two candidate sets;
  - training window: all history vs last 12 months;
  - model: logistic regression only;
  - test protocol: run once, bootstrap luck check, pre-registered success rule "skill > 0 and ahead in ≥ 90% of
    resamples";
  - four fixed error-analysis checks;
  - README structure.
- **The author caught a made-up number:** the AI's proposed 720-hour cap for "hours since last alert" was not
  based on data. A check showed the longest gap ever was 372 h, so the cap was set to **168 h**.
- **The author asked to keep the implementation plan separate from the ADR.** The ADR holds only decisions.
- **Publishing:** no GitHub connector was available. Pushing with a token pasted in chat would have put a secret
  into the graded log, so the repo is handed over as a **git bundle** for the author to push.

### Phase 5: Implementation (first run by the AI, step by step)

| Step | What happened |
|---|---|
| 1. Features + tests | The leakage tests **caught a real bug** on first run: a cutoff assertion failed on evenings with no alerts at all. Two more issues were found and decided by the author: drop the feature "alert active at 18:00" (it depends on guessed end times), and count a night only if it ended ≥ 24 h before the data snapshot (the newest night might miss a late "all clear"). |
| 2. Validation selection | All 4 model candidates were slightly **worse** than baseline B. A negative model weight looked like a bug but was explained by the data: in the latest year, past alerts barely correlated with tonight's outcome. |
| 3b. One revision round | Pre-registered in git **before running:** add "alerts in regions around Kyiv, 15:00–18:00". **Rejected** by its own rule (it didn't win on both validation halves). |
| 4. Official test | Run once with a code guard against re-runs. The first run **crashed** while saving results. Outputs were set aside unseen, the bug was fixed, and the rerun was proven identical. |
| 5. Error analysis | The model's probabilities were **too low**: 53% average forecast vs 71% actual. It couldn't follow the 2026 rise. |
| 6. README | A plain-language write-up of the whole project. |
| 6b. Post-hoc "second look" | One fix (recalibrating the level to the last N nights), pre-registered, with N chosen on validation (60). Clearly labelled **not a clean test**. Result: it ties baseline B. |
| Stop decision (ADR D22) | Adding Telegram as a further round was discussed. **The author decided to stop iterating**, with this reasoning: it is not established that relevant warnings are published before 18:00 (the examples found were posted at 19:47 and 22:27), so the real limit may be the chosen forecast time, not only the missing data. Adding Telegram at 18:00 would test the wrong thing. |

---

## 2. Final results (test: 15 Jun – 12 Oct 2026, 120 nights, 71% with alerts)

| Forecaster | Brier ↓ | Note |
|---|---|---|
| Baseline A (all history) | 0.275 | Never warned |
| **Baseline B (last 30 nights)** | **0.200** | Reference |
| Model v1 (official) | 0.226 | Skill −12.9%; reliably worse than B |
| Model v2 (post-hoc, not clean) | 0.199 | Ties B (+0.8%, ahead in only 64% of resamples) |

**Conclusion:** at 18:00, alert history mostly tells you the *current level* of danger, and a simple 30-night rule
captures that best. Predicting *which* nights will be attacked needs information about the attacks themselves:
warning signals, or a later forecast time.

---

## 3. Mistakes and corrections

| What went wrong | Caught by | What changed |
|---|---|---|
| The first framing (00:00, "at least X hours") wasn't useful for real decisions | Author | 18:00 night-window forecast |
| The AI repeated "updated daily" for a file that had stopped on 7 Sep | AI (data check) | Switched to the volunteer file |
| The AI described the 2025 alert reform as one date | AI (news check) | Staggered dates documented |
| The AI chose the validation dates without asking | Author | Reopened as a decision |
| The AI claimed a 240-night validation covered busy periods | AI (data check) | Long validation chosen |
| The AI suggested accuracy as a "readable side number" | AI (on reflection) | Replaced by hit / miss counts |
| Baseline B could output exactly 0% or 100% | AI | Smoothing; settings chosen on validation |
| Test scores were seen before some settings were fixed | Author asked; AI disclosed | Disclosure + rule for later changes |
| A 720 h cap not based on data | Author | 168 h, based on observed gaps |
| Cutoff assertion failed on evenings with no alerts | Automated tests | Bug fixed |
| A feature depended on guessed end times | AI (during implementation) | Feature dropped |
| The newest night could be mislabelled | AI (during implementation) | 24 h completeness rule |
| The test script crashed while saving results | AI | Fixed; identical rerun verified |

---

## 4. Concepts covered along the way

Target, forecast time and horizon; base rate; leakage; walk-forward evaluation; training / validation / test sets;
Brier score, skill score, precision and recall; baselines; overfitting and validation overfitting; hyperparameters
and smoothing; regime change (concept drift); logistic regression, weights, scaling and regularization;
pre-registration; bootstrap resampling; calibration and under-forecasting; post-hoc analysis.

## 5. Left for the future

- Check v2 on new nights after 12 Oct 2026, the only clean test of it.
- Warning signals and forecast time, **together** (ADR D22). First measure *when* warnings (e.g. Air Force Telegram
  posts) are usually published. Then choose the forecast time (e.g. 21:00, trading warning time for accuracy). Only
  then add the signals, dropping posts edited after the forecast time, and judge the result on fresh nights.
