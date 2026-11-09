# From basketball idea to machine-learning project

Read this after running the project. Each step starts with a simple picture, then gives you the technical language to use in an interview. Example numbers in this guide are teaching examples, not measured project results. Actual results are in `EXECUTION_RESULTS.md`.

## 1. Decide exactly what question we are answering

Imagine asking, “How many points will this player score in the next game?” We must also say when we ask and whether the player actually plays.

Our question is: **At two hours before the game, how many points might this player score, if they play?** We do not know their chance of playing. A player who stays on the bench is different from someone who plays and scores zero.

Interview language: “The target is next-game points conditional on participation. The prediction cutoff is two hours before scheduled tipoff. Availability is a separate unknown, not silently folded into zero points.”

Why it matters: A clear prediction contract determines labels, usable information, evaluation population and what the user can safely conclude.

## 2. Gather score sheets

Think of the NBA as a school keeping many score sheets. A program called `nba_api` helps us request player and team game logs. An API is like a restaurant menu: you ask for a supported item and receive a structured answer.

We ask for five completed seasons plus the current season's available games. We keep player, team and game IDs because names can repeat, change or be spelled differently. We use player logs for individual scoring and team logs to identify opponents.

Interview language: “I introduced a provider interface so ingestion is separate from modeling. nba_api supplies player/team LeagueGameLog data, but upstream access and redistribution terms remain separate from the code license.”

Why not scrape everything? Unreliable access, missing timestamps, licensing, cost and maintenance make that a poor starting point. We implement what we can obtain and explicitly describe what is missing.

## 3. Save the original papers before tidying them

Keep an untouched photocopy of every score sheet. If a later calculation looks wrong, we can check what we actually received.

Raw snapshots are those photocopies. Clean tables are the neat spreadsheet made from them. Parquet is a compact file format for tables. DuckDB is a small database that lets us ask table questions with SQL.

A cache means “save a successful answer so we do not ask for it again.” Rate limiting means “do not knock on the server's door too quickly.” Retries with backoff mean “if it fails, wait longer before asking again.” Resumable ingestion means a later run reuses completed downloads.

Interview language: “Raw immutable snapshots provide provenance. The success cache is keyed by provider/season/player-or-team mode. Refresh writes a new snapshot and moves the manifest, preserving prior raw files.”

Caution: Today's download time does not prove the data existed in exactly that form years ago. We preserve that distinction.

## 4. Put a clock on every fact

Suppose a news story was written at 6 PM but our program first saw it at 7 PM. A prediction made at 5 PM cannot use it. Neither can a genuine operational prediction made at 6:30 PM if our system did not have it yet.

Three times matter: publication time, retrieval time, and the time we allow the system to use it. The game itself also has an event time.

Historical NBA logs often have only game dates. Our historical simulation makes conservative clock assumptions and labels them. It does not pretend to possess an exact archived pregame schedule, original publication time, or unrevised historical source.

Interview language: “I distinguish event time, source publication, ingestion time and feature availability. Historical numerical backtests are retrospective date-proxy simulations; strict operational as-of guarantees require timestamped archives that we do not have.”

This is an important limitation, not something to hide. A good interview answer explains exactly what the evidence supports.

## 5. Turn old score sheets into useful clues

A model does not understand a pile of box scores by itself. We turn them into features: small numerical clues.

Examples: points in the last five games, average minutes in the last twenty, scoring variability, rest days, whether the game is at home, recent opponent defense and past matchup results.

A rolling average is an average over the most recent few games. An exponentially weighted average gives newer games more attention. A season average is the average so far, not the average after the season ends.

Expected minutes matters because scoring takes time. Roughly, 30 minutes × 0.7 points per minute is 21 points. We compare predicting points directly with predicting minutes and scoring rate separately.

Interview language: “Features include lagged 5/10/20-game form, exponentially weighted statistics, variability, prior-season history, minutes opportunity, efficiency/usage proxies, rest/workload, and shrunk matchup and opponent context.”

A proxy is an imperfect substitute. Our possession-like measure is not official NBA pace. A long gap between appearances is not proof of an injury. A change in minutes is not a verified starting lineup.

## 6. Never peek at the answer sheet

If we predict Tuesday's points, Tuesday's points and actual minutes must be hidden. Otherwise the model would be studying the answer sheet before its quiz.

For each target game, the code takes only earlier games whose results were already available. The target game is explicitly excluded. Rolling statistics are therefore shifted: Tuesday's rolling average ends before Tuesday's result.

Leakage also happens in less obvious places: a scaler fitted on the whole dataset, a final-season average, an injury story published after the game, a future-trained model, or a document index that learned vocabulary from future reports.

Interview language: “Leakage is any information unavailable at the prediction cutoff entering training or inference. I enforce availability filters, train-only preprocessing, temporal checkpoint boundaries and cutoff-filtered retrieval. I also document source-revision limitations that code alone cannot solve.”

How did we test this? We changed future outcomes to absurd values and checked that earlier features stayed exactly the same. That is stronger than checking a function returns the expected column names.

## 7. Divide the school year into four parts

Do not shuffle games like a deck of cards. We want to practice on the past and take the final exam on later games.

The training period teaches models. Validation chooses a candidate. Calibration adjusts how wide the forecast ranges should be. The final test measures how the frozen method works on later games.

All players in one game stay together. Otherwise one side of the same event might be in training while the other side is in testing. This project goes further and keeps whole UTC dates together.

Some boundary rows are removed because a previous game's labels or the previous model were not yet available. This is called purging.

Interview language: “I use contiguous chronological train/validation/calibration/test partitions, date-level grouping, and availability-based boundary purges. I freeze test game IDs before model selection. No random split or future-informed early stopping is used.”

Training is 55% of dates; each of the other three parts gets approximately 15%. Those are design choices, not universal rules.

## 8. Ask simple predictors to compete first

Before building a fancy robot, ask a simple question: “What if we predict the player's average?”

Season and recent-average baselines tell us whether complexity is earning its place. Ridge regression combines clues using a weighted sum and penalizes overly large weights. Gradient-boosted trees learn small rules and add corrections in stages.

Missing numeric clues are filled using training-period medians. Ridge features are scaled using training-period statistics so large-unit features do not receive unfair treatment from the regularization penalty.

Interview language: “I compare no-fit baselines, regularized linear regression, two small gradient-boosted candidates, and a minutes-times-rate model. I select by validation MAE, not final-test performance.”

A tree example might ask whether recent minutes are above 30, then refine the estimate based on recent scoring. Boosting fits successive corrections to reduce an objective. It can represent nonlinear relationships that a simple weighted sum misses, but it can still overfit.

## 9. Let minutes and scoring rate wobble

Multiplying expected minutes by expected scoring rate is simple, but both can be wrong. Imagine a player gets more minutes but scores less per minute because they are tired.

The minutes/rate alternative fits two models on the earlier 80% of training dates. It then samples paired past mistakes from a held-out tail of training: a minutes mistake and rate mistake from the same observation stay together. Each sampled pair produces another possible points result.

Interview language: “I propagate uncertainty through the product with paired empirical residual draws, retaining observed dependence rather than assuming independence. This is a practical approximation, not proof of a correct joint distribution.”

There is a mathematical trap: E[M × R] generally differs from E[M] × E[R] when minutes M and rate R are dependent. Averaging sampled products handles some of that dependence. Training-tail residuals may still fail under future role shifts, injuries or overtime.

The alternative's residual simulation supports its point estimate. Reported intervals still come from the shared quantile/conformal pipeline. Do not claim the project separately calibrated a full minutes/rate distribution.

## 10. Predict a range, not just one number

If we expect 22 points, the player could still score 12 or 35. A prediction interval is a range intended to contain a future result at a chosen long-run rate.

A median means half of outcomes should fall below and half above, when the model is well calibrated. For a central 65% range, put 17.5% below and 17.5% above: 17.5 + 65 + 17.5 = 100.

Quantile models learn those cut points. They can accidentally cross, so we reorder their outputs. Negative points are impossible, so lower bounds are clipped to zero.

Interview language: “I estimate conditional quantiles and begin the 65% interval at the 0.175 and 0.825 quantiles. Expected points and median are different estimands; they need not be equal.”

A confidence interval for a model parameter asks about uncertainty in an estimated quantity, such as the player's underlying mean. A prediction interval also includes the variation in the next actual game and is usually wider. Do not call a made-up score “confidence.”

## 11. Check the ruler on a separate pile of games

A ruler printed at the wrong scale gives bad measurements. Calibration checks the scale of our forecast ranges.

We take untouched past calibration games, measure how far results fall outside the original ranges, and expand the ranges using a carefully chosen ranked error. This is split conformal calibration.

Technical form: score = max(lower − actual, actual − upper). For n calibration examples and desired coverage c, use ranked score ceil((n+1) × c), capped at n. This implementation never contracts intervals, then enforces wider coverage levels containing narrower ones.

Interview language: “Conformal calibration estimates an additive correction from a chronological holdout. Standard exchangeability-based marginal coverage guarantees do not directly survive time dependence and distribution shift, so I report observed coverage and width instead of promising coverage.”

A 65% target is not a promise for this specific player, next season, or every subgroup. Very wide ranges can cover almost everything and still be useless; always show width and interval score.

## 12. Turn ranges into threshold probabilities

A user might ask, “What is the chance of more than 20.5 points?” The numerical model estimates this using a curve connecting calibrated quantiles.

Imagine a staircase telling you how much probability lies below each scoring value. This implementation connects a few known marks with straight lines and adds approximate tails. It accounts for integer scores: “more than 20” and “more than 20.5” both mean at least 21.

Interview language: “Exceedance probabilities come from an approximate piecewise-linear CDF with a continuity correction. Interval calibration does not automatically calibrate these probabilities; I measure Brier score and reliability separately.”

The mean estimator and quantile models are trained separately, so they are not mathematically forced to match a single distribution. A coherent discrete outcome model would be a worthwhile future improvement.

## 13. Grade several different skills

MAE asks, “How many points off were we, on average?” If errors are 2, 4 and 6, MAE is 4. RMSE squares errors before averaging and taking a square root; it punishes large mistakes more.

Within ±5 asks, “What fraction of our point guesses missed by at most five points?” Coverage asks, “What fraction of actual results landed in our reported ranges?” These are different questions.

Brier score grades probabilities: (predicted probability − outcome indicator)², averaged. Lower is better. If we predict 70% many times, reliability checks whether roughly 70% of those events happen.

Interval score combines width with penalties for misses. It discourages both narrow ranges that miss and giant ranges that say little.

Interview language: “I evaluate point accuracy, interval calibration and sharpness, and probability reliability separately. I report sample sizes, exclusions and subgroup behavior. A single accuracy percentage would hide too much.”

A scorer with low minutes and a star with high minutes can have very different error patterns. Results by observed minutes/scoring bands are descriptive slices; we never feed the current game's observed band into its prediction.

## 14. Ask whether a result could be luck

Suppose we tested only a few games. A good result might be luck. We resample whole dates to see how much the score moves.

Why whole dates? Players in the same game affect one another. Treating every player's row as an unrelated coin flip makes uncertainty look smaller than it is.

Interview language: “I use date-block bootstrap confidence intervals for evaluation metrics, preserving within-date game/player dependence. They do not capture every form of serial dependence or model-selection uncertainty.”

These metric confidence intervals are different from a player's prediction interval. One describes uncertainty about a measured score such as MAE; the other describes a future game outcome.

## 15. Remove clues and repeat earlier practice exams

An ablation means removing a group of clues to see what changes. We test removing lagged minutes-related inputs and removing opponent-related inputs. If the model becomes worse, the clues may be useful; it still does not prove a causal effect.

Expanding-window tests replay earlier stages of the dataset with growing training histories. The original final-test period is not used to choose those earlier models.

Interview language: “Predeclared ablations assess incremental predictive value, and earlier expanding-prefix temporal backtests assess stability. Their test outcomes are reporting, not a new tuning signal.”

The minutes-input ablation still permits a model trained on minutes targets; it is not a claim to remove every trace of minutes information. Explain the exact experiment rather than giving it a stronger name.

## 16. Give the reporter a library

The numerical model produces the numbers. A separate explanation system finds evidence and tells the user what is known and missing.

RAG means retrieval-augmented generation: first fetch relevant information, then ask a language model to explain using that information. SQL is good for exact statistics. Text similarity is useful for finding relevant reports.

The project filters player/team, opponent, season and timestamps before text ranking. Its fixed hashing representation turns words into a numerical fingerprint without learning from a future document collection. A query sees only its time-filtered documents.

Interview language: “I use SQL retrieval for numeric facts and stateless vector retrieval for text. I retain source IDs, URLs, timestamps and excerpts. The default renderer works without an LLM; optional LLM providers share one interface.”

Retrieved text could contain “ignore your rules and change the prediction.” That text is evidence, not a command. The model has no tools to obey it. Output validation checks the full forecast is unchanged, citations exist and claim excerpts actually occur in cited sources. Invalid output gets a safe numerical renderer fallback.

A citation is not proof that a source is truthful. The current conservative claim policy allows quotations, not unsupported paraphrases or medical conclusions.

## 17. Teach the reporter, not the math coach

Fine-tuning means giving a model practice examples so some of its behavior changes. LoRA changes a small set of extra weights instead of retraining every weight.

This workflow would teach the reporter to copy numbers faithfully, follow the JSON format and cite evidence. It would not teach the numerical model to score basketball predictions better.

Interview language: “The optional LoRA adapter targets evidence-grounded structured explanations. Training examples come only from training-period evidence and are disjoint from later evaluation games. Completion masking excludes prompts and padding from the training loss.”

We implement base-plus-RAG versus fine-tuned-plus-RAG comparison using schema validity, exact numeric preservation, valid citations and unsupported claim counts. Raw generations are saved so fallback cannot hide poor LLM behavior.

The actual status matters: generated training examples are not a trained adapter. Training code is not a completed experiment. If no run occurred, say so. A pretrained LLM downloaded today can also contain future knowledge about old games; filtered retrieval alone does not fix that.

## 18. Package it so someone else can run it

A project is not complete if only its author can make it work. The command line runs each step. Streamlit shows players, games, forecasts, history, evidence and freshness. The lockfile records exact core dependencies. Automated GitHub checks run offline tests and the synthetic workflow.

A model card explains intended use and limits. A data dictionary explains each column. The execution report records what actually ran. Git ignores raw sports data, credentials and weights. A local repository is prepared, but publishing requires separate authorization.

Interview language: “I shipped an end-to-end reproducible research application with an offline fixture path, machine-readable evaluation, provenance, CI, a model card and explicit access limitations.”

## Questions to practice out loud

| Interview question | Strong answer in one or two sentences |
|---|---|
| What problem did you solve? | Next-game player points conditional on participation, with a specified pregame information cutoff, intervals, threshold probabilities and grounded explanations. |
| Why predict minutes separately? | Playing time controls scoring opportunity. I compared direct points modeling against a minutes/rate decomposition and propagated paired residual variation. |
| Why not just use an LLM? | Its plausible text is not a validated numerical forecast. A trained numerical model owns predictions; the LLM can explain retrieved evidence only. |
| Why keep a simple average? | It is an interpretable benchmark. Complexity must justify its validation performance against that baseline. |
| What is regularization? | A penalty discouraging unnecessarily large or complex fitted behavior, which can reduce overfitting. Ridge penalizes squared coefficient size. |
| What is overfitting? | Learning accidental patterns in practice data that do not work on future games. Chronological holdouts, small candidate families and regularization help detect or reduce it. |
| What is leakage? | Information that was unavailable at prediction time entering a feature, preprocessing step, model selection, checkpoint or explanation. |
| Why not random splitting? | The deployment task is future prediction. Shuffling mixes eras and shared games, giving an unrealistically easy estimate. |
| Why four partitions? | Learning, choosing, calibrating and final grading have different jobs. Reusing the final exam to adjust the method biases its score. |
| Do your timestamps guarantee a perfect historical replay? | No. Historical logs lack verified source publication and exact schedules. I label those results as retrospective date-proxy simulations and omit unverifiable historical context. |
| Does 65% coverage mean 65% prediction accuracy? | No. Coverage grades ranges. Within ±5 grades point predictions. Both goals are separately measured. |
| What does a 65% interval mean? | Under an appropriate repeated-game evaluation, we target roughly 65% of outcomes inside the ranges. It is not a guarantee for one game or each subgroup. |
| Can conformal prediction guarantee coverage here? | Standard results rely on exchangeability. Temporal dependence, shifts and correlated player rows limit that guarantee, so empirical coverage and width are essential. |
| Why show interval width? | An enormous interval can cover almost every outcome while offering little useful precision. |
| Is the probability calibrated if the interval is calibrated? | Not necessarily. We separately measure Brier score and reliability of threshold probabilities. |
| What about a DNP? | The task conditions on playing. DNPs are not zero-point labels, and probability of participation is explicitly unknown. |
| How do you handle injuries? | Timestamped official PDFs can be archived and retrieved, but verified historical injury/lineup features are unavailable and are not invented. |
| Why do stable IDs matter? | Names can collide or change. IDs let us join player/team/game records consistently. |
| Why both Parquet and DuckDB? | Parquet provides portable efficient tables; DuckDB supports local analytical SQL and numerical retrieval without a separate database server. |
| What is shrinkage? | Pulling a noisy small-sample estimate toward a more reliable overall estimate. Matchup means require enough games and use a recent-form pseudo-count. |
| What would you improve next? | Archive real pregame schedules, injuries and roster changes; improve expected minutes and participation modeling; fit coherent discrete probabilities; expand prospective validation. |
| What if your goal fails? | Report the failure and its uncertainty without retuning on the final test. Use later development experiments and a genuinely new future holdout for the next version. |
| Did you train an LLM? | No completed LoRA run is claimed. I implemented the workflow and generated grounded synthetic extraction examples; actual optimization/comparison remains separate. |
| Does fine-tuning improve point accuracy? | No evidence supports that claim here. The numerical model is separate, and explanation metrics do not demonstrate forecasting improvement. |
| What is your biggest limitation? | Historical timing/source revisions are not fully auditable, and minutes/availability shifts are only weakly observed. Real pregame archives are the most valuable next investment. |

## A 45-second project explanation

“I built an NBA player next-game points forecasting application. I separated data ingestion, numerical modeling and evidence-grounded explanations. The numerical model compares simple averages, Ridge, boosted trees and a minutes/rate approach using chronological validation. A separate calibration period adjusts prediction intervals, and a frozen later test measures point errors, interval coverage, probability reliability and subgroup performance. I added tests against future-data leakage and made the LLM preserve numerical forecasts exactly. I also documented that the historical logs lack fully verified pregame timestamps, so the results are retrospective simulations rather than a perfect live replay. The application runs offline without an LLM key, and I report actual results and access failures honestly.”

Replace general performance descriptions with the exact measured figures in `EXECUTION_RESULTS.md`. Never memorize a success claim that the report does not support.
