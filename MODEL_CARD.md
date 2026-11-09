# Model card

## Task and population
Predict a selected NBA player's points conditional on participating in the next game, using only prior played-game results and known matchup context. Regular season only. This is an educational research project. It does not estimate probability of playing, model playoffs, or promise betting returns.

Expected points are an estimate of the conditional mean; median is the conditional 50th quantile. Prediction intervals describe a future game's outcome, not confidence intervals for parameters. The mean model and quantile models are fitted separately: exact distributional consistency between the mean and the approximate CDF is not imposed.

## Candidates and selection
Season average, recent 5-game average, scaled Ridge (alpha 30), histogram gradient boosting (7 or 15 leaves), and minutes × scoring-rate prediction. Selection is by validation MAE only. Trees have 65 boosting iterations, minimum 25 observations per leaf, L2 penalty 5, and no internal random early-stopping split. The scaler and median imputer see training rows only. Quantile trees predict .05, .10, .175, .50, .825, .90, .95, independently of the selected conditional-mean estimator.

The minutes/rate alternative fits two trees on the first 80% of training dates and takes paired held-out training-tail residual draws (up to 400). Each draw perturbs minutes and scoring rate together, clips minutes to [1,48] and rate to nonnegative values, and multiplies the two. This propagates their empirical joint variation, although it assumes residual behavior transfers across time. The mean across draws is its point forecast. The minutes estimator is not an actual-minute feature. Regulation clipping underrepresents overtime.

## Time protocol
Four contiguous UTC-date partitions: 55% training, 15% selection validation, 15% calibration, and 15% final test. All players in a game and all games on a date share a partition. Validation, calibration and testing purge rows whose cutoff precedes the previous model/label availability boundary. Final test game IDs are written and hashed before fitting. Seeds and exact split spans are saved. Training only uses observed labels from prior partitions.

For a historical backtest the model is retrained as a simulation at an historical date. Metadata preserves both this simulated availability and the actual wall-clock training date. Live prediction rejects a model actually trained after the requested historical cutoff; use the explicit backtest command for simulations. LLM checkpoints must likewise exist before any claimed operational prediction; historical evaluation in this repository uses the deterministic renderer only, so no future pretrained LLM knowledge is injected.

Important evidence limitation: nba_api game logs do not provide verified historical tipoff, original publication, roster announcements or source revisions. We use 12:00 UTC on the game date as a conservative event-time proxy, with results usable at 00:00 UTC two days later. This deliberately sacrifices recent information, including some back-to-back games. It is NOT a fully archived, exact two-hours-before-tipoff evaluation. Historical home/opponent/team assignment comes from game logs and may reflect schedule changes that were not yet known. Those aspects cannot be fully audited without archived schedules. Null publication is preserved. Timestamped text is omitted unless publication, availability and observed retrieval are all before cutoff. These limitations prevent a strict operational as-of accuracy claim.

## Intervals and probabilities
For 65%, start at conditional quantiles 17.5% and 82.5%. Calculate `max(lower-y, y-upper)` on separate past calibration observations. Use order statistic `ceil((n+1)*coverage)`, capped at n, and a nonnegative expansion. Perform the same process for 80% and 90%. Rearrange crossed raw quantiles first, clip lower bounds to zero, and enforce nesting by expansion only. This can make intervals conservative. Finite-sample marginal conformal coverage relies on exchangeability, which time series, shared games and distribution shifts violate; empirical coverage and width are always reported.

Exceedance probability interpolates a CDF through calibrated quantile endpoints and median, assuming bounded approximate tails. Apply a half-point continuity correction to integer scoring thresholds. Repeated anchors use rightmost mass. This is an approximation, not a separately calibrated probability classifier; Brier scores and reliability plots reveal its shortcomings. Mean-model selection does not select the quantile model. A more principled discrete distribution is future work.

## Evaluation and exclusions
Untouched held-out future games: MAE, RMSE, within ±3/5/7, interval coverage/width/score, threshold Brier scores and reliability. Bootstrap complete UTC-date blocks for 95% metric uncertainty, retaining game/player correlations within each date. This does not account for all serial dependence, model-selection uncertainty, or team-level long-range dependence. Groups by actual minutes and observed scoring are descriptive outcome strata, not input features. Injuries remain an explicitly unavailable group; role change is only a lagged proxy.

DNP, missing points, zero minutes and insufficient history are excluded explicitly. Missing entire box-score rows cannot be counted as DNPs. Evaluated population favors established players and those who play, creating selection limitations. Rows with no prior game vanish at feature creation and are counted in the feature-build report. Boundary-purge exclusions are counted separately.

Predeclared ablations remove minutes-related or opponent-related inputs. The minutes ablation still uses the same candidate family (including minutes/rate targets), so interpret it as removal of lagged minutes features, not removal of all minutes information. Earlier expanding-prefix backtests exclude the original final test. Do not tune based on ablation test outcomes.

## Explanation model and fine-tuning
Numerical output is authoritative. SQL retrieves recent statistics. A fixed hashing vectorizer retrieves text after entity and time filtering. Unlike a learned embedding/index, hashing fits no future vocabulary. Each historical query builds its own time-filtered view before ranking. Retrieved prose is untrusted. Structured output must preserve the entire forecast and cite known IDs. Claims are restricted to verbatim source excerpts; this sacrifices fluency for a checkable grounding rule. Citations alone do not establish that the source is true. Excerpts can still contain misleading text; the deterministic fallback only emits known numerical statistics.

Ridge attributions are additive contributions relative to training feature means; other model types use one-feature replacement with training medians. These are model associations, not causes; correlated inputs limit interpretation.

The default offline renderer is not an LLM. Optional compatible API and local Transformers providers are implemented. LoRA training targets extraction and structured explanations, not points prediction. Code existence is not a completed training experiment. Base-versus-adapter evaluation measures raw structured validity, numerical preservation, citation correctness and unsupported excerpts without hiding failures behind fallback. See the execution report for what was actually run.

## Unavailable factors
Verified starting role, teammate absences, medical restrictions, historical injury archives, coaching intentions, live roster changes, player tracking, geographic travel distances, official usage/pace, overtime and playoff adjustments, and player-specific availability probability. No claim to consider everything.
