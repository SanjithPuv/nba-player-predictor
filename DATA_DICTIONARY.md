# Data dictionary

All identifiers are strings. All timestamps use UTC. Each player/game row is unique.

| Table/field | Meaning | Timing and limitations |
|---|---|---|
| logs.game_id/player_id/team_id/opponent_id | Stable provider IDs; synthetic IDs start SYN | NBA IDs are preserved, including game leading zeros |
| logs.season | Season start/end label | Regular season only |
| logs.tipoff | Scheduled event time, or explicitly marked historical date proxy | nba_api date-only logs use game date at 12:00 UTC; not verified tipoff |
| logs.available_at | Earliest permitted use by this implementation | Real historical results: game date + 2 UTC days, a conservative assumption, not observed publication |
| logs.published_at | Source's known publication time | Null for date-only historical logs; never invented |
| logs.retrieved_at | Actual download time | Current retrieval does not establish historical publication or rule out revisions |
| logs.timestamp_quality | exact synthetic, observed snapshot, or historical proxy | Historical results must be reported as retrospective simulations |
| logs.points/minutes | Observed played-game labels | Not features of that same game |
| logs.played | Positive minutes and nonmissing points | No DNP-to-zero imputation; missing player rows are unknown |
| logs.fga/fgm/fta/ftm | Attempts and makes | Sources may revise statistics later |
| logs.assists/turnovers | Box-score counts | Available only after a prior game |
| logs.home | Home court indicator | Derived from matchup string in retrospective logs |
| schedule.tipoff | Timestamp supplied by schedule provider | Snapshot must exist by prediction cutoff |
| schedule.available_at/retrieved_at | Observed schedule snapshot time | Historical schedule cannot be reconstructed from today's file |
| features.cutoff | Target tipoff minus 2 hours | Historical tipoff proxy implies a conservative clock approximation |
| features.max_history_available_at | Latest historical result used | Must not exceed cutoff |
| *_5 / *_10 / *_20 | Means over previous played games | Target excluded; rows unavailable at cutoff excluded |
| points_ewm/minutes_ewm | Exponential recent-form average | Last 20 games, decay factor 0.8 |
| points_std/minutes_std | Population standard deviation | Last 20 played games |
| season_points | Season-to-date mean | Past history fallback at season start |
| prior_season_points | Mean over available previous seasons | Falls back to season_points for new players |
| expected_minutes | 0.6 × prior 5-game mean + 0.4 × prior 20-game mean | Feature; separate minutes model supplies output estimate |
| rate_20 | Total points / total minutes | Prior 20 games; not average of per-game ratios |
| efficiency_20 | Points / [2 × (FGA + 0.44 FTA)] | True-shooting-like proxy, not official advanced data |
| usage_proxy | FGA + 0.44 FTA + turnovers per game | Not official usage percentage; excludes team possession denominator |
| opponent_allowed | Opponent's recent points allowed | Sum of prior player box scores per opponent team/game |
| opponent_pace_proxy | Opposing FGA + 0.44 FTA + turnovers | Not official pace: lacks offensive rebounds and possessions adjustment |
| rest_days/back_to_back/workload_7d | Lagged rest and minutes workload | Real data uses date proxy; DNPs not played games |
| return_gap | More than 10 days since prior played game | Does not establish injury or reason for absence |
| team_changed | Current row team differs from latest prior team | Historical assignment from logs; live team inferred from last log |
| travel_team_change_proxy | Opponent changed since last appearance | Weak schedule-change indicator; not mileage, geography or flight distance |
| matchup_points/matchup_n | Shrunk matchup mean and number of games | At least 3 prior games, 10-game pseudo-count at recent mean |
| role_change_proxy | Absolute 5- vs 20-game minutes change ≥4 | Lagged proxy; not verified starting lineup |
| history_n | Prior played-game count | Model requires at least 3 |
| injury_group | unavailable | Never infer injuries from missing data |
| text_index.source_id/source_url | Citation identity and source | Text is untrusted evidence, not instructions |
| text_index.player_id/team_id/opponent_id/season | Required entity tags | Player OR team relevance, same season, compatible opponent |
| text_index.published_at/retrieved_at/available_at | Three independent timing checks | All must precede cutoff; unknown publication excludes document |

Raw snapshots live under `data/raw`; cleaned Parquet and DuckDB tables under `data/clean` and `data/nba.duckdb`. Raw sports data, model files, and individual real-data prediction rows are ignored by Git. Aggregated evaluation reports may be shared subject to source terms.
