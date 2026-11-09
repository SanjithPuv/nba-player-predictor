"""Providers, immutable raw snapshots, conservative historical availability, and storage."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import duckdb
import numpy as np
import pandas as pd
import requests


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc(value):
    return pd.to_datetime(value, utc=True)


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


class Provider(Protocol):
    def logs(self, season: str, kind: str) -> pd.DataFrame: ...
    def schedule(self, game_date: str | None = None) -> pd.DataFrame: ...


class NBAProvider:
    """nba_api is a community client, not a licensed data feed. Cache successes only."""

    def __init__(self, root="data", interval=1.5, retries=2, timeout=12):
        self.root = Path(root)
        self.interval, self.retries, self.timeout = interval, retries, timeout
        self.last_call = 0.0

    def _retry(self, function):
        for attempt in range(self.retries + 1):
            time.sleep(max(0, self.interval - (time.monotonic() - self.last_call)))
            self.last_call = time.monotonic()
            try:
                return function()
            except Exception:
                if attempt == self.retries:
                    raise
                time.sleep(min(30, 2 ** attempt * 2))

    def logs(self, season, kind="P", refresh=False):
        from nba_api.stats.endpoints import leaguegamelog

        folder = self.root / "raw" / "nba_api" / season / kind
        manifest = folder / "latest.json"
        if manifest.exists() and not refresh:
            return pd.read_parquet(folder / json.loads(manifest.read_text())["file"])
        frame = self._retry(lambda: leaguegamelog.LeagueGameLog(
            season=season, player_or_team_abbreviation=kind,
            season_type_all_star="Regular Season", timeout=self.timeout,
        ).get_data_frames()[0])
        if frame.empty:
            raise ValueError(f"No {kind} logs for {season}; empty response not cached")
        stamp = now()
        frame["source_url"] = "https://stats.nba.com/stats/leaguegamelog"
        frame["retrieved_at"] = stamp
        frame["season"] = season
        folder.mkdir(parents=True, exist_ok=True)
        filename = hashlib.sha256(stamp.encode()).hexdigest()[:16] + ".parquet"
        frame.to_parquet(folder / filename, index=False)
        dump_json(manifest, {"file": filename, "retrieved_at": stamp, "season": season})
        return frame

    def schedule(self, game_date=None):
        if game_date is not None:
            from nba_api.stats.endpoints import scoreboardv3

            game_date = pd.Timestamp(game_date).strftime("%Y-%m-%d")
            data = self._retry(lambda: scoreboardv3.ScoreboardV3(
                game_date=game_date, timeout=self.timeout).get_dict())
            stamp = now()
            dump_json(self.root / "raw" / "schedule" /
                      f"{game_date}_{stamp.replace(':', '-')}.json", data)
            url = f"https://stats.nba.com/stats/scoreboardv3?GameDate={game_date}&LeagueID=00"
            rows = [dict(game_id=str(g["gameId"]), tipoff=g["gameTimeUTC"],
                         home_team_id=str(g["homeTeam"]["teamId"]),
                         away_team_id=str(g["awayTeam"]["teamId"]),
                         status=g["gameStatus"], source_url=url, published_at=None,
                         retrieved_at=stamp, available_at=stamp,
                         timestamp_quality="observed_daily_scoreboard_snapshot")
                    for g in data["scoreboard"]["games"] if g.get("gameTimeUTC")]
            if not rows:
                raise ValueError(f"No scheduled games returned for {game_date}")
            return pd.DataFrame(rows)
        url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
        def fetch():
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        data = self._retry(fetch)
        stamp = now()
        dump_json(self.root / "raw" / "schedule" / (stamp.replace(":", "-") + ".json"), data)
        rows = []
        for day in data["leagueSchedule"]["gameDates"]:
            for g in day["games"]:
                if g.get("gameDateTimeUTC"):
                    rows.append(dict(game_id=str(g["gameId"]), tipoff=g["gameDateTimeUTC"],
                                     home_team_id=str(g["homeTeam"]["teamId"]),
                                     away_team_id=str(g["awayTeam"]["teamId"]),
                                     status=g.get("gameStatus"), source_url=url,
                                     published_at=None, retrieved_at=stamp,
                                     available_at=stamp, timestamp_quality="observed_snapshot"))
        return pd.DataFrame(rows)

    def injury(self, url, published_at):
        """Archive an official timestamped PDF; extract verbatim evidence, no roster inference."""
        from urllib.parse import urlparse
        from pypdf import PdfReader
        import io

        if urlparse(url).hostname != "ak-static.cms.nba.com":
            raise ValueError("Expected an official NBA injury PDF URL")
        published_at = utc(published_at)
        if published_at > utc(now()):
            raise ValueError("Publication timestamp is in the future")
        def fetch():
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.content
        raw = self._retry(fetch)
        stamp = now()
        digest = hashlib.sha256(raw).hexdigest()
        folder = self.root / "raw" / "injuries"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{digest}.pdf").write_bytes(raw)
        text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)
        return dict(source_id=digest, source_url=url, text=text, published_at=str(published_at),
                    retrieved_at=stamp, available_at=stamp, player_id=None, team_id=None,
                    opponent_id=None, season=None,
                    timestamp_quality="observed_snapshot_publication_user_supplied")


def clean_logs(players, teams):
    """Date-only logs lack tipoff/publication times. Mark conservative proxies explicitly."""
    p, t = players.copy(), teams.copy()
    for frame in (p, t):
        frame["GAME_ID"] = frame.GAME_ID.astype(str).str.zfill(10)
        frame["TEAM_ID"] = frame.TEAM_ID.astype(str)
    pairs = t[["GAME_ID", "TEAM_ID"]].drop_duplicates()
    opponents = pairs.merge(pairs, on="GAME_ID", suffixes=("", "_opp"))
    opponents = opponents[opponents.TEAM_ID != opponents.TEAM_ID_opp]
    if opponents.duplicated(["GAME_ID", "TEAM_ID"]).any():
        raise ValueError("Game does not have exactly two teams")
    p = p.merge(opponents, on=["GAME_ID", "TEAM_ID"], validate="many_to_one")
    dates = utc(p.GAME_DATE).dt.normalize()
    p["tipoff"] = dates + pd.Timedelta(hours=12)
    # 12:00 UTC minus two hours is before evening games, but is only a proxy, not exact.
    p["available_at"] = dates + pd.Timedelta(days=2)
    p["published_at"] = pd.NaT
    p["timestamp_quality"] = "historical_date_proxy_unverified_revisions"
    p["synthetic"] = False
    p["home"] = (~p.MATCHUP.str.contains("@", regex=False)).astype(int)
    mapping = {"GAME_ID": "game_id", "PLAYER_ID": "player_id", "PLAYER_NAME": "player_name",
               "TEAM_ID": "team_id", "TEAM_ID_opp": "opponent_id", "PTS": "points",
               "MIN": "minutes", "FGA": "fga", "FGM": "fgm", "FTA": "fta",
               "FTM": "ftm", "AST": "assists", "TOV": "turnovers"}
    p = p.rename(columns=mapping)
    p["player_id"] = p.player_id.astype(str)
    p["minutes"] = pd.to_numeric(p.minutes, errors="coerce")
    p["played"] = p.minutes.gt(0) & p.points.notna()
    return p[list(mapping.values()) + ["tipoff", "available_at", "published_at", "season", "home",
                                       "source_url", "retrieved_at", "timestamp_quality",
                                       "synthetic", "played"]]


def store_tables(root, **tables):
    root = Path(root)
    (root / "clean").mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(root / "nba.duckdb")) as con:
        for name, table in tables.items():
            if not name.replace("_", "").isalnum():
                raise ValueError("Invalid table name")
            table.to_parquet(root / "clean" / f"{name}.parquet", index=False)
            con.register("incoming", table)
            con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM incoming')
            con.unregister("incoming")


def synthetic_fixture(seed=41, days=190):
    """Small artificial league; deliberately includes DNPs and rotation changes."""
    rng = np.random.default_rng(seed)
    rows, games = [], []
    start = pd.Timestamp("2023-10-20T23:00:00Z")
    for day in range(days):
        tip = start + pd.Timedelta(days=day * 2)
        season = "2023-24" if day < 100 else "2024-25"
        for a, b in [(1, 2), (3, 4)]:
            game = f"SYN{day:04}{a}"
            games.append(dict(game_id=game, tipoff=tip, home_team_id=str(a), away_team_id=str(b),
                              available_at=tip-pd.Timedelta(days=40),
                              retrieved_at=tip-pd.Timedelta(days=40),
                              published_at=tip-pd.Timedelta(days=40), source_url="synthetic://schedule",
                              timestamp_quality="synthetic_exact", status=3))
            for team, opp in [(a, b), (b, a)]:
                for slot in range(5):
                    player = str(team * 10 + slot)
                    played = rng.random() > 0.055
                    rotation = 5 if day > 125 and slot == 2 else 0
                    minutes = float(np.clip(rng.normal(34 - slot * 3 + rotation, 3.4), 4, 43)) if played else 0.0
                    rate = 0.82 - slot * .105
                    points = int(max(0, rng.poisson(minutes * rate))) if played else None
                    fga = int(max(0, rng.poisson(minutes * rate * .8))) if played else 0
                    rows.append(dict(game_id=game, player_id=player, player_name=f"Synthetic Player {player}",
                                     team_id=str(team), opponent_id=str(opp), tipoff=tip,
                                     available_at=tip+pd.Timedelta(hours=4),
                                     published_at=tip+pd.Timedelta(hours=4),
                                     retrieved_at=tip+pd.Timedelta(hours=4), season=season,
                                     points=points, minutes=minutes, fga=fga, fgm=int(fga*.46),
                                     fta=int(rng.poisson(4)) if played else 0, ftm=3 if played else 0,
                                     assists=int(rng.poisson(3)), turnovers=int(rng.poisson(2)),
                                     home=int(team == a), played=played, synthetic=True,
                                     source_url="synthetic://fixture", timestamp_quality="synthetic_exact"))
    future = start + pd.Timedelta(days=days*2)
    games.append(dict(game_id="SYN_NEXT", tipoff=future, home_team_id="1", away_team_id="2",
                      available_at=future-pd.Timedelta(days=30), retrieved_at=future-pd.Timedelta(days=30),
                      published_at=future-pd.Timedelta(days=30), source_url="synthetic://schedule",
                      timestamp_quality="synthetic_exact", status=1))
    return pd.DataFrame(rows), pd.DataFrame(games)
