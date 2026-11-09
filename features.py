"""As-of features: all historical observations must be available before the cutoff."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import utc

RAW_STATS = ["points", "minutes", "fga", "fta", "assists", "turnovers"]
FEATURES = [f"{stat}_{window}" for stat in RAW_STATS for window in (5, 10, 20)] + [
    "points_ewm", "minutes_ewm", "points_std", "minutes_std", "season_points",
    "prior_season_points", "rate_20", "efficiency_20", "usage_proxy", "expected_minutes",
    "home", "rest_days", "back_to_back", "workload_7d", "return_gap", "team_changed",
    "matchup_points", "matchup_n", "opponent_allowed", "opponent_pace_proxy",
    "travel_team_change_proxy", "history_n", "role_change_proxy",
]


def _mean(a, default=np.nan):
    return float(np.mean(a)) if len(a) else default


def _team_history(logs):
    played = logs[logs.played & logs.minutes.gt(0)].copy()
    played["possessions"] = played.fga + .44 * played.fta + played.turnovers
    # Player sums omit offensive rebounds: explicitly a pace proxy, not official pace.
    sums = played.groupby(["game_id", "team_id", "opponent_id"], as_index=False).agg(
        points=("points", "sum"), possessions=("possessions", "sum"),
        tipoff=("tipoff", "first"), available_at=("available_at", "max"))
    result = {}
    for team, group in sums.groupby("opponent_id"):
        result[str(team)] = group.sort_values("available_at")
    return result


def build_features(logs, targets=None, cutoff_hours=2, *, cutoff_at=None):
    """targets may contain future rows with no outcomes. No target outcome is read."""
    logs = logs.copy()
    for column in ("tipoff", "available_at"):
        logs[column] = utc(logs[column])
    if logs.duplicated(["game_id", "player_id"]).any():
        raise ValueError("Duplicate player/game observations")
    targets = logs.copy() if targets is None else targets.copy()
    targets["tipoff"] = utc(targets.tipoff)
    groups = {str(p): g.sort_values("available_at") for p, g in logs.groupby("player_id")}
    teams = _team_history(logs)
    output = []
    for row in targets.to_dict("records"):
        cutoff = (utc(cutoff_at) if cutoff_at is not None
                  else row["tipoff"] - pd.Timedelta(hours=cutoff_hours))
        if pd.isna(cutoff) or cutoff >= row["tipoff"]:
            raise ValueError("Feature cutoff must be a valid time before tipoff")
        h = groups.get(str(row["player_id"]))
        if h is None:
            continue
        # Same-game exclusion is explicit even for a corrupted publication timestamp.
        h = h[(h.available_at <= cutoff) & (h.tipoff < cutoff) &
              (h.game_id != row["game_id"]) & h.played & h.minutes.gt(0)]
        if h.empty:
            continue
        f = {k: row.get(k) for k in ["game_id", "player_id", "player_name", "team_id", "opponent_id",
                                     "season", "tipoff", "points", "minutes", "played", "synthetic",
                                     "source_url", "timestamp_quality"]}
        f["cutoff"] = cutoff
        f["max_history_available_at"] = h.available_at.max()
        for stat in RAW_STATS:
            a = h[stat].to_numpy(float)
            for w in (5, 10, 20):
                f[f"{stat}_{w}"] = _mean(a[-w:])
        for stat in ("points", "minutes"):
            a = h[stat].to_numpy(float)
            weights = .8 ** np.arange(min(20, len(a)))[::-1]
            f[f"{stat}_ewm"] = float(np.average(a[-20:], weights=weights))
            f[f"{stat}_std"] = float(np.std(a[-20:]))
        cur = h[h.season == row["season"]]
        prior = h[h.season < row["season"]]
        f["season_points"] = _mean(cur.points.to_numpy(), _mean(h.points.to_numpy()))
        f["prior_season_points"] = _mean(prior.points.to_numpy(), f["season_points"])
        recent = h.tail(20)
        f["rate_20"] = float(recent.points.sum() / max(1, recent.minutes.sum()))
        f["efficiency_20"] = float(recent.points.sum() / max(1, 2*(recent.fga.sum()+.44*recent.fta.sum())))
        f["usage_proxy"] = f["fga_10"] + .44*f["fta_10"] + f["turnovers_10"]
        f["expected_minutes"] = .6*f["minutes_5"] + .4*f["minutes_20"]
        f["home"] = row["home"]
        gap = (row["tipoff"] - h.tipoff.iloc[-1]).total_seconds()/86400
        f["rest_days"], f["back_to_back"] = min(14, max(0, gap-1)), int(gap < 1.6)
        f["return_gap"] = int(gap > 10)
        f["workload_7d"] = float(h.loc[h.tipoff >= cutoff-pd.Timedelta(days=7), "minutes"].sum())
        f["team_changed"] = int(str(row["team_id"]) != str(h.team_id.iloc[-1]))
        f["travel_team_change_proxy"] = int(str(row["opponent_id"]) != str(h.opponent_id.iloc[-1]))
        match = h[h.opponent_id.astype(str) == str(row["opponent_id"])].tail(20)
        n = len(match)
        f["matchup_n"] = n
        # Less than three games: no matchup adjustment. Otherwise shrink toward form.
        f["matchup_points"] = ((match.points.sum()+10*f["points_20"])/(n+10)
                                  if n >= 3 else f["points_20"])
        opp = teams.get(str(row["opponent_id"]))
        if opp is not None:
            opp = opp[(opp.available_at <= cutoff) & (opp.tipoff < cutoff)].tail(20)
        f["opponent_allowed"] = _mean(opp.points.to_numpy()) if opp is not None else np.nan
        f["opponent_pace_proxy"] = _mean(opp.possessions.to_numpy()) if opp is not None else np.nan
        f["history_n"] = len(h)
        f["role_change_proxy"] = int(abs(f["minutes_5"]-f["minutes_20"]) >= 4)
        f["injury_group"] = "unavailable"
        output.append(f)
    return pd.DataFrame(output)


def temporal_partitions(frame, fractions=(.55, .15, .15, .15)):
    """Split by UTC date, stricter than game grouping; dates never cross boundaries."""
    if len(fractions) != 4 or not np.isclose(sum(fractions), 1):
        raise ValueError("Four split fractions must sum to one")
    dates = utc(frame.tipoff).dt.normalize()
    unique = np.sort(dates.unique())
    boundaries = np.cumsum([int(len(unique)*x) for x in fractions[:3]])
    if min(np.diff(np.r_[0, boundaries, len(unique)])) < 2:
        raise ValueError("Need at least two distinct dates per partition")
    names = ["train", "validation", "calibration", "test"]
    mapping = {date: names[int(np.searchsorted(boundaries, i, side="right"))]
               for i, date in enumerate(unique)}
    result = frame.copy()
    result["partition"] = dates.map(mapping)
    if result.groupby("game_id").partition.nunique().max() != 1:
        raise ValueError("Game split across partitions")
    return result
