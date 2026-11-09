"""Forecast contract and as-of upcoming-game construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import now, utc
from .features import build_features
from .rag import explain, retrieve

MISSING = ["Verified pregame injury status and probability of playing",
           "Timestamped starting lineup and teammate absences",
           "Exact travel distance and verified roster/rotation changes",
           "Historical source revisions and exact tipoff times for date-only logs"]


def upcoming_targets(logs, schedule, player_id, as_of=None):
    as_of = utc(as_of or now())
    history = logs[(logs.player_id.astype(str) == str(player_id)) & (utc(logs.available_at) <= as_of)]
    if history.empty:
        raise ValueError("No player history available at this time")
    last = history.sort_values("tipoff").iloc[-1]
    # Current team inferred from last available box score; caller must verify recent trades.
    team = str(last.team_id)
    available = schedule[(utc(schedule.available_at) <= as_of) & (utc(schedule.tipoff) > as_of)]
    if "status" in available:
        available = available[available.status.eq(1)]
    available = available[(available.home_team_id.astype(str) == team) | (available.away_team_id.astype(str) == team)]
    rows = []
    for game in available.sort_values("tipoff").to_dict("records"):
        home = str(game["home_team_id"]) == team
        game_time = utc(game["tipoff"])
        start_year = game_time.year if game_time.month >= 10 else game_time.year-1
        season = game.get("season") or f"{start_year}-{str(start_year+1)[-2:]}"
        rows.append(dict(game_id=str(game["game_id"]), player_id=str(player_id), player_name=last.player_name,
                         team_id=team, opponent_id=str(game["away_team_id"] if home else game["home_team_id"]),
                         tipoff=game["tipoff"], home=int(home), season=season,
                         played=None, points=np.nan, minutes=np.nan, synthetic=bool(last.synthetic),
                         source_url=game["source_url"], timestamp_quality=game["timestamp_quality"],
                         schedule_available_at=game["available_at"]))
    return pd.DataFrame(rows)


def forecast(model, logs, target, root, index, threshold=20.5, as_of=None, provider=None,
             *, cutoff_at=None):
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    as_of = utc(as_of or now())
    tipoff = utc(target["tipoff"])
    cutoff = utc(cutoff_at) if cutoff_at is not None else tipoff-pd.Timedelta(hours=2)
    if pd.isna(cutoff) or cutoff >= tipoff or as_of >= tipoff:
        raise ValueError("Prediction and data cutoff must be valid times before tipoff")
    if as_of < cutoff:
        if cutoff_at is not None:
            raise ValueError("Explicit cutoff is in the future relative to the prediction time")
        raise ValueError("Two-hour cutoff has not occurred; forecast would use an unavailable future cutoff")
    if utc(target.get("schedule_available_at", cutoff)) > cutoff:
        raise ValueError("Schedule snapshot was unavailable at cutoff")
    if not model.synthetic and utc(max(model.trained_at, getattr(model, "calibrated_at", model.trained_at))) > cutoff:
        raise ValueError("Actual model execution was after this historical cutoff; use backtest command")
    features = build_features(logs, pd.DataFrame([target]), cutoff_at=cutoff)
    if features.empty or features.history_n.iloc[0] < 3:
        raise ValueError("Need at least three played historical games")
    d = model.distribution(features)
    row = features.iloc[0]
    values = {
        "player_id": str(row.player_id), "player_name": row.player_name, "game_id": str(row.game_id),
        "expected_points": round(float(d["expected"][0]), 4),
        "median_points": round(float(d["median"][0]), 4),
        "prediction_intervals": {str(int(c*100)): [round(float(x), 4) for x in a[0]] for c, a in d["intervals"].items()},
        "threshold": float(threshold), "probability_exceeding_threshold": round(float(model.exceedance(d, threshold)[0]), 6),
        "estimated_minutes": round(float(d["minutes"][0]), 4),
        "conditional_on_playing": True, "availability_probability": None,
        "availability_assumptions": ["Player participates; DNP probability is not estimated.",
                                     "Current team is inferred from the last available game; verify trades.",
                                     "Playing time follows historical role; no verified lineup or injury adjustment."],
        "prediction_timestamp": now(), "data_cutoff": str(cutoff), "scheduled_tipoff": str(target["tipoff"]),
        "cutoff_policy": "explicit_pregame_as_of" if cutoff_at is not None else "two_hours_before_tipoff",
        "calibration_scope": ("Historical date-proxy calibration; a custom cutoff has not been separately backtested."
                              if cutoff_at is not None else "Historical date-proxy calibration"),
        "model_available_at_simulated": model.available_at, "model_executed_at": model.trained_at,
        "latest_statistics_available_at": str(row.max_history_available_at),
        "synthetic": model.synthetic, "missing_information": MISSING,
        "model_attributions": model.contributions(features),
        "main_factors": {"recent_5_game_points": round(float(row.points_5), 3),
                         "recent_20_game_minutes": round(float(row.minutes_20), 3),
                         "expected_minutes_feature": round(float(row.expected_minutes), 3),
                         "rest_days": float(row.rest_days),
                         "note": "Descriptive model inputs, not causal explanations or formal feature attributions."},
        "probability_method": "Piecewise-linear CDF through calibrated quantiles; reliability is evaluated, not guaranteed.",
        "interval_note": "Prediction interval for a played game's points, not parameter confidence or accuracy."}
    evidence = retrieve(root, index, row.player_id, row.team_id, row.opponent_id, row.season, cutoff)
    explanation = explain(values, evidence, provider)
    return {"forecast": values, "explanation": explanation, "evidence": evidence}
