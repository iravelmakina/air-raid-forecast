# ADR-0001: Forecasting setup for Kyiv night air raid alerts

- **Status:** Accepted (D1–D19)
- **Date:** 2026-10-14
- **Living document:** updated after each decision step
- **Working rule:** all decisions are made and a full implementation plan ([`docs/implementation-plan.md`](../implementation-plan.md)) is approved before any further code is written or run.

## Context and Problem Statement

University admissions take-home: a Python time-series project on Ukrainian air raid alerts.
The brief requires the most recent data, evaluation on the latest period, and no leakage
(nothing used for a forecast may come from after the forecast time). The AI conversation log
is the main graded signal. The author is a backend engineer with no prior ML experience.

## Decision Drivers

- **DR1 Brief compliance:** most recent data, evaluation on the latest period, zero leakage.
- **DR2 Real-world usefulness:** the forecast should support a decision a civilian actually makes.
- **DR3 Honest evaluation:** results must be reproducible and must not be flattered by tuning on the test.
- **DR4 Time budget:** about one working day in total.
- **DR5 Simplicity:** no prior ML experience, so every component must be explainable and checkable.
- **DR6 Data reality:** the 2025 district-level alert reform and a 2026 high-alert regime change the meaning and level of the data.

## Considered Options

### D1: What is predicted, and when?

**A — Hourly yes/no, next hour.** At each hour, predict whether an alert happens in the next hour. Wins on many test points; loses on usefulness (DR2), because the current state almost decides it and alert apps already show it live.

**B — Daily yes/no at 00:00.** At midnight, predict whether tomorrow has any alert. Wins on simplicity; loses on usefulness, and midnight splits night attacks across two days.

**C — Daily hours under alert.** Predict the number of alert hours tomorrow. Wins on information content; loses on evaluation difficulty for a beginner (DR5), and the total says nothing about when alerts happen.

**D — Night window, forecast at 18:00.** At 18:00, predict whether any alert overlaps 22:00–06:00 that night. Wins on usefulness (where to sleep tonight) and a clean cutoff; loses on a possibly high base rate.

**E — Work hours, forecast at 18:00.** At 18:00, predict an alert during 09:00–18:00 the next day. Wins on business usefulness; loses on a 15-hour gap before the window. Data check: equally saturated recently (Sep 90%, Oct 92%), so it is no fallback.

### D2: Which region?

**A — Kyiv City.** One city-level unit. Wins on stability: the data holds only city-level records, so the 2025 district reform never split it (verified) (DR6). Loses on a high recent base rate.

**B — A frontline oblast.** Wins on alert volume; loses on near-constant alerts, and the reform redefined "oblast under alert" (DR6).

**C — All of Ukraine.** Wins on scope; loses on roughly 25× the work (DR4).

### D3: Which data source?

**A — Official file (Vadimkin repo).** Wins on official origin; loses on recency: it ends on 7 Sep 2026 despite daily repo commits, and 1,874 of 4,141 Kyiv rows are exact duplicates (DR1).

**B — Volunteer file (Vadimkin repo, eTryvoga).** Wins on recency (to 13 Oct 2026), no duplicates, and 97.5% night-level agreement with the official file; loses on guessed end times for some alerts (see D4).

**C — alerts.in.ua API.** Wins on being the live source; loses on needing a token, with unknown history depth (DR4).

**D — Scrape Telegram ourselves.** Wins on rawness; loses on a full day of parser work (DR4).

### D4: Alerts with guessed end times (`naive=True`, end = start + 30 min)

**A — Keep as is, use no duration-based inputs.** Wins because the night label barely depends on the end time (stretching all guessed alerts to 3 hours changes 0 of the last 120 labels); loses duration as a possible input.

**B — Fill end times from the official file.** Wins on realism; loses because the official file only covers data up to 7 Sep and needs an extra join.

**C — Drop guessed alerts.** Wins on nothing; loses by corrupting 18 of the last 120 labels.

### D5: How is the forecast evaluated over time?

**A — Single split, last ~60 nights.** Wins on simplicity; loses on a tiny, noisy test almost entirely in the ~90% regime.

**B — Single split, last ~120 nights.** Wins on simplicity; loses because a model frozen in June never sees the regime change.

**C — Walk-forward over the last 120 nights.** Each evening: train on everything before 18:00, predict the night, move one day forward. Wins on realism, adaptation to regime change, and an explicit per-night cutoff (DR1, DR3); loses on more code and the risk of off-by-one leakage, which is mitigated by assertions in the code.

### D6: Metric

**A — Accuracy.** Wins on readability; loses because "always yes" scores about the base rate (~71%), and it discards the probability.

**B — Brier score + skill score vs. reference baseline.** Wins on judging probability quality, penalising confident mistakes, and being a proper scoring rule (DR3); loses on intuitiveness.

**C — Precision and recall.** Wins on describing the consequences of acting on the forecast; loses because it needs a threshold and can be gamed in isolation.

### D7: Baselines

**A — All-history rate.** Wins on showing the regime-change effect; loses on usefulness. Test: Brier 0.275, never warns.

**B — Recent rate (last N nights).** Wins as the strongest common-sense rule, so it is the honest bar; loses on needing settings (see D8).

**C — Last night's outcome.** Wins on simplicity; loses on outputting only 0% or 100%. Test: Brier 0.383, worst of the three.

### D8: Configuring baseline B

**A — Keep N=7, no smoothing.** Wins on zero effort; loses because it outputs 0% or 100% on 20 of 120 test nights (3 of them wrong, each costing 1.0).

**B — N=7 + smoothing, by judgment.** Wins on fixing that flaw; loses because N is untested.

**C — Choose N ∈ {3, 7, 14, 30} and smoothing on/off on the validation period, then lock them.** Wins on honesty and reusability for the model (DR3); loses about 30 minutes.

### D9: Validation period

**A — 120 nights (16 Feb – 15 Jun 2026).** Wins on recency; loses on regime mismatch (45% alert nights vs. 71% in the test).

**B — 240 nights (19 Oct 2025 – 15 Jun 2026).** Wins on more nights; loses because it averages 48%, with no busy periods. The AI's first claim that it covered busier periods was wrong and was corrected after a data check.

**C — Long: 1 Oct 2024 – 15 Jun 2026 (623 nights).** Wins on including busy periods similar to the test (Nov 2024 at 83%, Jan–Feb 2025 at 81–86%); loses on older nights getting the same say as recent ones.

### D10: Optional text/media data

**A — Skip.** Wins on time (DR4) and on avoiding a second leakage surface; loses on a possible extra signal.

**B — Include Telegram or news.** Wins on a possible extra signal; loses about a day of work plus publish-time leakage risk.

### D11: Features (inputs computed at 18:00)

Candidate features, each computable from the alert history before 18:00:
(1) alert-night rate over the last 30 nights; (2) alert last night; (3) any Kyiv alert today 06:00–18:00 (by start time);
(4) Kyiv alert active at 18:00 (its end is not used); (5) hours since the last Kyiv alert *started* (start times only, per D4);
(6) number of regions with an alert today before 18:00 (by start time); (7) day of week.

**A — Minimal set (1–3).** Kyiv-only recent state. Wins on simplicity; loses on adding little beyond what baseline B already knows.

**B — Medium set (1–6; feature 4 later dropped by D18).** Adds the current state, quiet time, and nationwide activity. Wins on information beyond B (especially 6); loses on more leakage guards to check.

**C — Large set (1–7 plus several window lengths and lags).** Wins on breadth; loses on overfitting risk with ~1,000–1,700 training nights.

### D12: Training window

**A — Expanding.** All nights since 26 Feb 2022, growing each evening. Wins on the most examples; loses because calm periods (2023: 27% alert nights) weigh as much as busy ones.

**B — Sliding, last 12 months.** Wins on recency; loses on fewer examples, so the weights are noisier.

**C — All history, recent nights weighted more.** Wins on combining both; loses on a decay rate that is one more setting to tune and explain (DR5).

### D13: Model

**A — Logistic regression.** A weighted sum of inputs squashed into a probability. Wins on small-data robustness, readable weights, and reasonable probabilities (DR5); loses on not learning thresholds or interactions by itself.

**B — Gradient-boosted trees.** Many small learned if/else rules. Wins on learning thresholds and interactions; loses on overfitting with this little data, many settings, and hard explanation. It is also the model used by another public applicant repo.

**C — Both as candidates.** Wins on letting validation decide; loses on doubling the grid and the chance of a lucky winner.

### D14: Final test protocol

**A — Selected model only.** Wins on simplicity; loses because one score from 120 nights cannot separate skill from luck.

**B — Selected model + bootstrap uncertainty.** Resample the test nights with replacement 1,000 times, recompute the skill score, and report its range and the share of resamples where the model beats B. Wins on honesty about sampling noise (DR3); loses on one more concept.

**C — B + all candidates shown on the test.** Wins on transparency; loses on the temptation to change the selection after seeing test scores.

### D15: Error analysis

**A — None.** Wins on time; loses on insight and bug detection.

**B — Four fixed checks.** Weights, calibration table, worst nights, test halves. Wins on answering what the model learned, whether its probabilities can be trusted, where it breaks, and whether it survives the regime change; loses because every check is noisy at 120 nights.

**C — B + more slicing.** Wins on richer findings; loses on the multiple comparisons problem (some slice looks impressive by chance) and on time.

### D16: Write-up

**A — Minimal README.** How to run plus the results table. Wins on speed; loses by hiding the reasoning that is graded.

**B — Standard README.** Problem and framing, data and issues found, evaluation design, results with uncertainty, error analysis, limitations and future work, how to run; links to this ADR and the log. Wins on telling the full story without duplicating the ADR; loses about an hour of writing.

**C — B + many charts.** Wins on visual appeal; loses on time and scope creep.

### D17: Feature-level definitions

Sub-choices fixed before any model result:

**Feature 5 cap: A — no cap.** Uses the raw hours since the last Kyiv alert started. Wins on simplicity; loses because rare 2023 gaps (max 372 h, Oct 2023) become extreme values after scaling and can distort the weight.

**Feature 5 cap: B — cap at 168 h (7 days).** Wins on taming old extremes while never affecting validation or test evenings (2025–2026 max gap ~99 h); loses by treating 9- and 15-day quiet spells as equal.

**Feature 5 cap: C — log transform.** Wins on smooth compression; loses on one more concept to explain (DR5).

An earlier AI proposal of a 720 h cap was not based on data (it would never trigger) and was rejected after a data check.

### D18: Feature 4 ("Kyiv alert active at 18:00")

Found during implementation step 1: whether an alert is still running at 18:00 depends on its end time, which is guessed (+30 min) for some alerts. This contradicts D4.

**A — Keep it.** Wins on a possibly useful "attack under way" signal; loses on consistency with D4, because part of its values would be invented.

**B — Drop it.** Wins on consistency with D4; the "an alert just started" information is largely kept by feature 5 (start times only). Loses on a small amount of information.

### D19: When a night counts as complete

Found during implementation step 1: the last night in the file (13 Oct) was labelled quiet, but its only Kyiv alert started at 17:55 with a guessed end and was the last Kyiv alert in the file. Its "all clear" may simply not have been in the snapshot yet.

**A — Night complete once it ended before the snapshot.** Wins on keeping the newest night; loses on possible mislabels from late messages.

**B — Night complete only if it ended ≥ 24 h before the snapshot.** Wins as a general data-completeness rule that also protects future data refreshes; loses one night of recency. The test window shifts to 15 Jun – 12 Oct 2026.

## Decision

- **D1 → D:** night window 22:00–06:00 Kyiv time, forecast at 18:00, output is a probability. This is the clearest real decision (DR2).
- **D2 → A:** Kyiv City. It is the only well-known unit unaffected by the reform (DR6).
- **D3 → B:** volunteer file, pinned to commit `7be33582f840bf99b14f92118b40f6131b492e7c`. It is the only current source that fits the budget (DR1, DR4).
- **D4 → A:** keep guessed end times; no duration-based inputs. The labels are robust to them.
- **D5 → C:** walk-forward over the last 120 complete nights, **15 Jun – 12 Oct 2026** (DR1, DR3). Amended by D19; originally 16 Jun – 13 Oct.
- **D6 → B + C:** Brier and skill score are the verdict. Hits, false alarms, misses, precision and recall at a **fixed 50% threshold** are reported for the model and the baselines, to show the consequences of acting on the forecast.
- **D7 → A + B:** B is the reference for the skill score; C is dropped.
- **D8 → C:** settings for B are chosen on validation only. The re-run on the D9-C validation picked **N=30, smoothed** (validation Brier 0.241; test Brier 0.203 on the pre-D19 test window, to be recomputed in the official test run). This supersedes the earlier pick on the D9-A validation (N=14, smoothed). At the 50% threshold, B30 warns on every test night (85 hits, 35 false alarms, 0 misses), so as a decision rule it equals "always warn".
- **D9 → C:** validation is 1 Oct 2024 – 14 Jun 2026 (end moved by one night by D19). Training always starts on 26 Feb 2022 and grows by one night each evening.
- **D10 → A:** skip text/media data. Accepted trade-off: possibly the strongest signal is left out, in exchange for time and a simpler setup suited to learning the core concepts first. Recorded as future work in the README.
- **D11 → A and B as candidates:** the choice between them is made on validation. Day of week (7) is excluded.
- **D12 → A and B as candidates:** chosen on validation. Allowed even though the baselines have already run, because the window is a model setting and the model has not touched the test.
- **D13 → A:** logistic regression with the library-default regularization (not tuned). Inputs are scaled using statistics fitted on each evening's training data only. The full grid is **2 feature sets × 2 windows = 4 candidates**.
- **D14 → B:** the lowest validation Brier among the 4 candidates is locked. Then the locked model and baselines A and B30 run walk-forward over the test **once**. Reported: Brier, skill score vs. B30, hits / false alarms / misses / precision / recall at 50%, and the bootstrap range (1,000 resamples). **Pre-registered success criterion:** the model beats B only if skill > 0 **and** it is ahead in ≥ 90% of resamples; otherwise the conclusion is "no reliable improvement over B". A losing model is reported as is, with no re-runs.
- **D15 → B, with settings fixed in advance:** (1) the last evening's weights plus each weight's min–max across the 120 test evenings; (2) calibration groups < 50%, 50–80%, ≥ 80%, with night counts; (3) the top 10 nights by squared error; (4) test nights 1–60 vs. 61–120, model vs. B within each half. Findings go to the README only and never trigger changes or re-runs.
- **D16 → B:** standard README. The 100-word reflection is written by the author, not the AI. No charts, to keep scope small.
- **D18 → B:** feature 4 dropped. Feature set B is now features 1, 2, 3, 5, 6.
- **D19 → B:** a night counts only if it ended ≥ 24 h before the snapshot. Both D18 and D19 were decided on principle (data consistency), before any model result existed.
- **D17:** feature 5 is capped at **168 h** (B). Other fixed definitions (Kyiv local time, cutoff 18:00): feature 3 = a Kyiv City alert *started* 06:00–18:00 today; feature 6 = the number of *other* regions (excluding Kyiv City) with an alert started 06:00–18:00 today. Training rows with fewer than 30 past nights are dropped (feature 1 is undefined there). Tie rule in selection: a validation Brier difference < 0.001 goes to the simpler candidate (set A before B, expanding before sliding).

### Expected Benefits

- **DR1:** the test is the latest period, and every forecast has a code-enforced 18:00 cutoff. An alert still running at 18:00 has its end time hidden.
- **DR2:** a probability for "alert tonight" maps directly to a sleeping decision.
- **DR3:** settings are chosen on validation only, the data version is pinned, the test is run once, and the success criterion and analysis settings are fixed before any model result is seen.
- **DR5:** a single, explainable model with a 4-candidate grid; every weight can be read and checked.
- **DR6:** the target unit is stable across the reform, and validation contains periods similar to the test.

### Accepted Downsides

- **Small test:** 120 nights is about 7% of history (common guidance is about 20%), so scores contain sampling noise. Differences of about 0.01–0.02 in Brier may be luck.
- **Test seen early:** test scores for baselines A, B (N=7, no smoothing) and C were seen on the test nights *before* D8/D9 were fixed. This does not break the brief's leakage rule (no future data entered any forecast), but it is a good-practice issue because it could bias later choices. Mitigation: B's settings are chosen by the validation run, D9 was argued from base rates rather than scores, the model touches the test once, and every later change is recorded here with a reason that is not a test score.
- **Old data in validation:** settings may fit 2024–25 dynamics that differ from 2026.
- **Weak signal:** on the D9-A validation every B setting scored worse than a coin flip (Brier > 0.25); on the D9-C validation the best is only slightly better (0.241). The model may not beat B; if so, that is reported as a finding.
- **50% threshold is arbitrary:** fixed in advance and never tuned on the test.
- **No thresholds or interactions:** logistic regression cannot learn them by itself; accepted for simplicity and robustness (D13).
- **Overlapping inputs:** features 1 and 2 carry similar information, so their individual weights may split unpredictably (D15 check 1).
- **Validation selection luck:** even 4 candidates leave some chance of picking a winner that was only lucky on validation.
- **Real-time assumption:** the replay assumes alerts up to 18:00 are known at 18:00. That is true live, but the volunteer file updates only once a day, so a deployment would need a live source.

---

## Open Decisions (next steps)

- None. Execution follows the plan above.

## Future Decisions

- **F1:** fresh out-of-sample check on nights after 12 Oct 2026 (note for the README).
- **F2:** Telegram or other warning signals could be added later as a separate iteration. They are not added now, to reduce scope (see D10).
