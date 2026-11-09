# Actual execution results

Run date: November 6, 2026. Core environment: Python 3.14.7, Apple Silicon, CPU. Exact dependency versions are in `requirements.lock`. No GPU or LLM API key was used.

## Completed work

- Downloaded five completed seasons (2021-22 through 2025-26) and available 2026-27 regular-season player/team logs through nba_api.
- Retrieved 134,072 real player-game rows. Built 132,920 feature rows; 1,152 observations lacked usable prior played-game history.
- Trained numerical candidates, selected by past validation, calibrated on the next period, and evaluated the frozen later period.
- Ran two predeclared input ablations and three earlier expanding-prefix backtests.
- Downloaded an official historical injury PDF and extracted its text. It was excluded from backtests because its observed retrieval was after those games.
- Ran the offline synthetic end-to-end workflow, generated 79 grounded training examples and 80 separate later evaluation bundles, and checked the Streamlit forecast flow with AppTest.
- Offline safety tests and lint checks pass. See `checks.json` for the final check counts.

## Real-data evaluation

**This is a retrospective date-proxy simulation, not a fully archived exact two-hours-before-tipoff replay.** Source revisions, historical schedule changes and verified historical publication times are unavailable. No historical injury or lineup knowledge was added. The final test dates were December 15, 2025 through November 5, 2026; eligible predictions begin after the checkpoint boundary purge.

Selected model: **ridge**, by validation MAE. Final evaluated population: **20,643 player-game rows in 966 games over 125 UTC dates**. Final test IDs were frozen before model selection. No parameter was selected using final-test scores.

| Point metric | Result | 95% date-block bootstrap interval |
|---|---:|---:|
| MAE, points | 4.57 | 4.51–4.66 |
| RMSE, points | 5.94 | 5.84–6.08 |
| Within ±3 points | 41.18% | 40.45–41.77% |
| Within ±5 points | 64.09% | 63.43–64.72% |
| Within ±7 points | 79.63% | 78.83–80.32% |

**The goal of at least 65% within ±5 points was NOT achieved**: 64.09%, and the date-block bootstrap interval is approximately 63.43–64.72%. Do not present this as “65% accurate.”

| Interval target | Observed coverage | Mean width, points | Mean interval score |
|---|---:|---:|---:|
| 65% | 67.09% | 10.67 | 16.73 |
| 80% | 89.82% | 18.30 | 22.24 |
| 90% | 94.82% | 21.07 | 24.92 |

The 65% intervals covered 67.09% of outcomes (bootstrap interval 66.30–67.81%). This exceeds the minimum coverage target but is not exact nominal calibration. The 80% and 90% intervals materially overcover and are broad; do not call them precisely calibrated. Nonnegative conformal expansion, quantile misspecification, nesting, discrete scoring, and temporal distribution changes are plausible contributors, not experimentally established causes.

| Baseline/candidate | Test MAE | Within ±5 |
|---|---:|---:|
| ridge | 4.5738 | 64.09% |
| boosted_7 | 4.5959 | 64.02% |
| boosted_15 | 4.5816 | 64.07% |
| season_average | 4.7252 | 63.23% |
| recent_average | 4.8581 | 62.30% |
| minutes_rate | 4.5950 | 64.27% |

The selected Ridge model improved MAE over the season and recent averages. This does not establish significance of small candidate differences; a paired comparison would be needed. Selection optimized MAE, so another model can win a different metric.

| Points threshold | Brier score |
|---|---:|
| More than 10.5 | 0.1592 |
| More than 20.5 | 0.0782 |
| More than 30.5 | 0.0211 |

Reliability plots and binned frequencies are in `reports/real/reliability.png` and `evaluation.json`. Brier scores across thresholds are not directly comparable because event frequencies differ. Probabilities come from an approximate CDF and are not separately calibrated classifiers.

## Ablations and earlier time periods

| Predeclared input removal | Test MAE |
|---|---:|
| without_minutes | 4.5946 |
| without_opponent | 4.5863 |

These small MAE changes are descriptive and do not establish a causal or statistically significant contribution. The minutes ablation removes lagged minutes features but retains the candidate family, including minutes/rate targets.

| Earlier prefix endpoint | Evaluated rows | MAE | Within ±5 |
|---|---:|---:|---:|
| 2024-03-13 | 10,797 | 4.5676 | 65.10% |
| 2025-01-24 | 13,326 | 4.5183 | 64.92% |
| 2025-12-14 | 16,824 | 4.6441 | 63.93% |

## Exclusions and likely failure modes

- Final test before exclusions: 20,968 rows. DNP/missing or zero-minute observations: 56. Fewer than three prior played games: 170. Checkpoint-boundary purge among eligible rows: 99. Final evaluated rows: 20,643.
- Provider logs may omit DNPs entirely. Those absences cannot be counted from this feed, so this is not a full-roster availability evaluation.
- Role changes and unexpected minutes drive scoring opportunity but are only imperfectly represented by past minutes. Verified injuries, starting roles and teammate absences were unavailable.
- Conservative historical result-availability proxies deliberately lose some recent games and back-to-back information. Rest is measured from the most recent allowed prior appearance, not a fully known schedule.
- A single frozen model spans a long test period; role/team changes and season shifts can degrade transfer. These are plausible failure mechanisms, not a completed causal diagnosis.
- The bootstrap resamples dates (200 repetitions) and preserves within-date dependence. It does not account for all serial dependence, selection uncertainty or source revision error.

## Synthetic demonstration

The separate artificial league evaluated 554 synthetic rows, MAE 3.8747, within ±5 70.76%, and 65%-interval coverage 62.82%. **These are software demonstrations, not evidence about NBA performance.**

## Access limits and work not completed

- NBA schedule JSON returned **HTTP 403 Forbidden**. The exact attempted URL/error is in `reports/provider_access.json`. `nba-predictor schedule` retries from a permitted environment; `nba-predictor import-schedule /absolute/path/to/authorized_schedule.json` imports an observed timestamped feed. Capture the snapshot before the two-hour cutoff. No real upcoming prediction is claimed.
- Official PDF download/text extraction succeeded. Reliable entity tagging and verification of publication times remain explicit steps; no automatic historical injury or lineup dataset was reconstructed.
- No optional LLM inference, weight download, LoRA optimization, or base-versus-adapter comparison was run. Implemented code and generated examples are not a completed training experiment. Exact remaining commands and hardware/license notes are in `docs/FINETUNING.md`.
- The local Git repository is initialized. No GitHub repository was created and nothing was published.

## Reproduction

Follow `README.md` from a fresh environment. Raw and cleaned sports data and model weights are ignored by Git, so a fresh clone needs provider access to reproduce real results; the synthetic workflow and tests run offline after dependencies are installed. Aggregate reports, configuration, split spans, model metadata and frozen-test hash are saved. Individual real player-game prediction files remain local and ignored.
