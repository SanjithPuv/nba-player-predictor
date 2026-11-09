"""CLI entrypoints. Run nba-predictor --help for the complete offline workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import NBAProvider, clean_logs, dump_json, now, store_tables, synthetic_fixture, utc
from .evaluation import evaluate, experiments
from .features import build_features
from .model import train_from_features
from .predict import forecast, upcoming_targets
from .rag import CompatibleLLM, index_documents


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data", help="Data directory (use a separate root for synthetic data)")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--reports", default="reports")
    p.add_argument("--config", default="config/default.json")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="Run full offline synthetic workflow")
    ingestion = sub.add_parser("ingest")
    ingestion.add_argument("--seasons", nargs="+")
    ingestion.add_argument("--refresh", action="store_true", help="Refresh immutable season snapshots")
    ingestion.add_argument("--fail-fast", action="store_true")
    daily = sub.add_parser("schedule")
    daily.add_argument("--date", help="Use NBA daily ScoreboardV3 for this YYYY-MM-DD date")
    schedule = sub.add_parser("import-schedule", help="Import timestamped schedule JSON records")
    schedule.add_argument("file")
    injury = sub.add_parser("injury", help="Archive official PDF and extract text; indexing requires entity tagging")
    injury.add_argument("url")
    injury.add_argument("--published-at", required=True)
    sub.add_parser("features")
    sub.add_parser("train")
    sub.add_parser("calibrate")
    backtest = sub.add_parser("backtest")
    backtest.add_argument("--experiments", action="store_true")
    backtest.add_argument("--thresholds", nargs="+", type=float)
    index = sub.add_parser("index")
    index.add_argument("file", help="JSON array of timestamped, entity-tagged documents")
    index.add_argument("--cutoff")
    pred = sub.add_parser("predict")
    pred.add_argument("--player-id", required=True)
    pred.add_argument("--game-id")
    pred.add_argument("--threshold", type=float, default=20.5)
    pred.add_argument("--as-of")
    pred.add_argument("--cutoff-now", action="store_true",
                      help="Explicit pregame forecast at --as-of/current time, rather than the two-hour cutoff")
    pred.add_argument("--llm", action="store_true")
    examples = sub.add_parser("llm-examples")
    examples.add_argument("bundles", nargs="?", help="JSON array of archived training-period forecast/evidence bundles")
    examples.add_argument("--output", default="data/llm/train.jsonl")
    fine = sub.add_parser("finetune")
    fine.add_argument("dataset")
    fine.add_argument("--output", default="artifacts/lora")
    fine.add_argument("--lora-config", default="config/lora.json")
    compare = sub.add_parser("llm-evaluate")
    compare.add_argument("bundles")
    compare.add_argument("--adapter", required=True)
    compare.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    return p


def run(args):
    root, artifacts, reports = Path(args.root), Path(args.artifacts), Path(args.reports)
    config = json.loads(Path(args.config).read_text())
    root.mkdir(parents=True, exist_ok=True)
    provider = NBAProvider(root, config["request_interval_seconds"], config["retries"], config["request_timeout_seconds"])
    command = args.command
    if command == "ingest":
        players, teams, status = [], [], []
        for season in (args.seasons or config["seasons"]):
            try:
                p, t = provider.logs(season, "P", args.refresh), provider.logs(season, "T", args.refresh)
                players.append(p)
                teams.append(t)
                status.append({"season": season, "status": "success", "player_rows": len(p), "team_rows": len(t)})
                print(f"{season}: {len(p)} player rows, {len(t)} team rows", flush=True)
            except Exception as exc:
                status.append({"season": season, "status": "error", "error": str(exc)})
                print(f"{season}: {type(exc).__name__}: {exc}", flush=True)
                if args.fail_fast:
                    dump_json(reports/"ingestion.json", status)
                    raise
        dump_json(reports/"ingestion.json", status)
        if players:
            logs = clean_logs(pd.concat(players, ignore_index=True), pd.concat(teams, ignore_index=True))
            store_tables(root, logs=logs)
        else:
            raise RuntimeError("No seasons downloaded; details in ingestion report")
    elif command == "schedule":
        store_tables(root, schedule=provider.schedule(args.date))
    elif command == "import-schedule":
        frame = pd.DataFrame(json.loads(Path(args.file).read_text()))
        required = {"game_id", "tipoff", "home_team_id", "away_team_id", "available_at", "retrieved_at", "published_at", "source_url", "timestamp_quality"}
        if not required.issubset(frame):
            raise ValueError(f"Missing schedule fields: {required-set(frame)}")
        for col in ("available_at", "retrieved_at", "tipoff"):
            frame[col] = utc(frame[col])
        if (frame.available_at < frame.retrieved_at).any():
            raise ValueError("Schedule availability cannot precede observed retrieval")
        store_tables(root, schedule=frame)
    elif command == "injury":
        record = provider.injury(args.url, args.published_at)
        dump_json(root/"clean"/"untagged_injury.json", [record])
        print("Archived PDF and extracted text. Tag player/team/opponent/season before indexing.")
    elif command == "features":
        logs = pd.read_parquet(root/"clean"/"logs.parquet")
        features = build_features(logs, cutoff_hours=config["cutoff_hours"])
        store_tables(root, features=features)
        dump_json(reports/"feature_build.json", {"raw_rows": len(logs), "feature_rows": len(features),
                                                 "no_prior_played_history": len(logs)-len(features),
                                                 "synthetic": bool(logs.synthetic.all()),
                                                 "built_at": now()})
    elif command == "train":
        features = pd.read_parquet(root/"clean"/"features.parquet")
        model, _ = train_from_features(features, artifacts, config["seed"])
        print(json.dumps(model.validation_scores, indent=2))
    elif command == "calibrate":
        model = joblib.load(artifacts/"model.joblib")
        split = pd.read_parquet(artifacts/"partitions.parquet")
        cal = split[(split.partition == "calibration") & (utc(split.cutoff) > utc(model.fit_available_at))]
        model.calibrate(cal).save(artifacts)
        print(json.dumps(model.corrections))
    elif command == "backtest":
        model = joblib.load(artifacts/"model.joblib")
        split = pd.read_parquet(artifacts/"partitions.parquet")
        report = evaluate(model, split[split.partition == "test"], reports,
                          args.thresholds or config["thresholds"], config["bootstrap_date_blocks"])
        if args.experiments:
            experiments(split.drop(columns="partition"), reports/"experiments", config["seed"])
        print(json.dumps({"data_kind": report["data_kind"], "metrics": report["metrics"], "goals": report["goals"]}, indent=2))
    elif command == "index":
        index_documents(json.loads(Path(args.file).read_text()), root/"text_index.parquet", args.cutoff)
    elif command == "predict":
        logs = pd.read_parquet(root/"clean"/"logs.parquet")
        schedule = pd.read_parquet(root/"clean"/"schedule.parquet")
        model = joblib.load(artifacts/"model.joblib")
        as_of = args.as_of or now()
        targets = upcoming_targets(logs, schedule, args.player_id, as_of)
        if args.game_id and not targets.empty:
            targets = targets[targets.game_id == args.game_id]
        if targets.empty:
            raise ValueError("No upcoming game in the available schedule snapshot")
        result = forecast(model, logs, targets.iloc[0].to_dict(), root, root/"text_index.parquet",
                          args.threshold, as_of, CompatibleLLM() if args.llm else None,
                          cutoff_at=as_of if args.cutoff_now else None)
        dump_json(reports/"prediction.json", result)
        print(json.dumps(result, indent=2, default=str))
    elif command == "llm-examples":
        from .finetune import make_examples, generate_grounded_bundles
        model = joblib.load(artifacts/"model.joblib")
        if args.bundles:
            count = make_examples(json.loads(Path(args.bundles).read_text()), args.output,
                                  model.training_game_ids, model.fit_data_cutoff)
        else:
            split = pd.read_parquet(artifacts/"partitions.parquet")
            count = generate_grounded_bundles(root, split, args.output)
        print(f"Wrote {count} training-only examples")
    elif command == "finetune":
        from .finetune import train_lora
        train_lora(args.dataset, args.output, args.lora_config)
    elif command == "llm-evaluate":
        from .finetune import compare_llms
        compare_llms(json.loads(Path(args.bundles).read_text()), reports/"llm_comparison.json", args.base_model, args.adapter)
    elif command == "demo":
        if root == Path("data") or artifacts == Path("artifacts"):
            raise ValueError("Use --root data/synthetic --artifacts artifacts/synthetic --reports reports/synthetic")
        logs, schedule = synthetic_fixture(config["seed"])
        store_tables(root, logs=logs, schedule=schedule)
        for step in ("features", "train", "calibrate", "backtest"):
            print(f"Running {step} (SYNTHETIC ONLY)", flush=True)
            local = argparse.Namespace(**vars(args))
            local.command, local.experiments, local.thresholds = step, False, None
            run(local)
        model = joblib.load(artifacts/"model.joblib")
        as_of = utc(schedule.tipoff.max())-pd.Timedelta(hours=1)
        target = upcoming_targets(logs, schedule, "10", as_of).iloc[0].to_dict()
        result = forecast(model, logs, target, root, root/"text_index.parquet", 20.5, as_of)
        dump_json(reports/"prediction.json", result)
        dump_json(reports/"experiment_config.json", config | {"synthetic": True})
        print(f"Offline workflow complete: {reports}")


def main():
    args = parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
