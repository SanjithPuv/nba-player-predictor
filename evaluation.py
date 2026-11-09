"""Untouched future-game evaluation, date-block uncertainty, and declared ablations."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import dump_json, utc
from .features import FEATURES, temporal_partitions
from .model import ForecastModel, eligible


def point_metrics(y, prediction):
    error = np.asarray(prediction)-np.asarray(y)
    return {"n": len(error), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            **{f"within_{k}": float(np.mean(np.abs(error) <= k)) for k in (3, 5, 7)}}


def metrics(y, distribution, thresholds):
    y = np.asarray(y, float)
    result = point_metrics(y, distribution["expected"])
    for cov, interval in distribution["intervals"].items():
        lo, hi = interval.T
        score = hi-lo + 2/(1-cov)*(lo-y)*(y < lo) + 2/(1-cov)*(y-hi)*(y > hi)
        result[f"coverage_{int(cov*100)}"] = float(np.mean((y >= lo) & (y <= hi)))
        result[f"width_{int(cov*100)}"] = float(np.mean(hi-lo))
        result[f"interval_score_{int(cov*100)}"] = float(np.mean(score))
    for threshold in thresholds:
        p = ForecastModel.exceedance(distribution, threshold)
        result[f"brier_{threshold}"] = float(np.mean((p-(y > threshold))**2))
    return result


def subset_distribution(d, mask):
    return {k: {c: a[mask] for c, a in v.items()} if k == "intervals" else v[mask]
            for k, v in d.items()}


def evaluate(model, test, out, thresholds=(10.5, 20.5, 30.5), bootstraps=200):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    eligible_test = eligible(test)
    usable = eligible_test[utc(eligible_test.cutoff) > utc(model.available_at)].copy()
    if usable.empty:
        raise ValueError("No eligible test rows after checkpoint availability")
    d = model.distribution(usable)
    y = usable.points.to_numpy(float)
    summary = metrics(y, d, thresholds)
    dates = utc(usable.tipoff).dt.strftime("%Y-%m-%d").to_numpy()
    unique = np.unique(dates)
    rng = np.random.default_rng(model.seed)
    samples = []
    for _ in range(bootstraps):
        blocks = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(dates == date) for date in blocks])
        samples.append(metrics(y[indices], subset_distribution(d, indices), thresholds))
    cis = {key: [float(np.quantile([s[key] for s in samples], q)) for q in (.025, .975)]
           for key in summary if key != "n"} if samples else {}
    baselines = {name: point_metrics(y, pred) for name, pred in model.point_predictions(usable).items()}
    groups = {}
    bins = {"minutes_band": pd.cut(usable.minutes, [0, 15, 25, 35, 100], include_lowest=True).astype(str),
            "scoring_band": pd.cut(usable.points, [-1, 10, 20, 30, 1000]).astype(str),
            "season": usable.season.astype(str), "injury_group": usable.injury_group.astype(str),
            "role_change_proxy": usable.role_change_proxy.astype(str)}
    for label, values in bins.items():
        groups[label] = {}
        for value in values.unique():
            mask = (values == value).to_numpy()
            groups[label][str(value)] = metrics(y[mask], subset_distribution(d, mask), thresholds)
    reliability = {}
    for threshold in thresholds:
        p = model.exceedance(d, threshold)
        entries = []
        for lo in np.arange(0, 1, .1):
            mask = (p >= lo) & ((p < lo+.1) if lo < .9 else (p <= 1))
            if mask.any():
                entries.append({"n": int(mask.sum()), "predicted": float(p[mask].mean()),
                                "observed": float((y[mask] > threshold).mean())})
        reliability[str(threshold)] = entries
    report = {"data_kind": "SYNTHETIC DEMONSTRATION ONLY" if model.synthetic else "REAL retrospective date-proxy simulation",
              "selected_model": model.selected, "test_frozen_before_selection": True,
              "frozen_test_hash": model.frozen_test_hash,
              "metrics": summary, "date_block_bootstrap_95_percent_intervals": cis,
              "bootstrap_repetitions": bootstraps, "independent_dates": len(unique),
              "goals": {"within_5_at_least_65_percent": summary["within_5"] >= .65,
                        "observed_65_interval_coverage_at_least_target": summary["coverage_65"] >= .65,
                        "coverage_is_not_guaranteed": True},
              "counts": {"test_rows_before_exclusions": len(test),
                         "dnp_or_missing_points": int((~test.played.fillna(False).astype(bool) | test.points.isna() | test.minutes.le(0)).sum()),
                         "insufficient_history": int((test.history_n < 3).sum()),
                         "checkpoint_boundary_purge": len(eligible_test)-len(usable),
                         "evaluated": len(usable), "observed_games": int(usable.game_id.nunique()),
                         "note": "Provider logs may omit DNPs entirely; absent players cannot be counted."},
              "baseline_comparisons": baselines, "groups": groups, "reliability": reliability,
              "limitations": ["Date-block bootstrap assumes sampled dates represent future variation; serial dependence can remain.",
                              "Role changes are lagged minutes proxies, not verified lineup or injury groups.",
                              "Intervals condition on playing; zero-minute DNPs are excluded.",
                              "Retrospective source revisions and exact historical tipoff times are not verified.",
                              "No numerical accuracy claim can be made from synthetic data."]}
    dump_json(out/"evaluation.json", report)
    predictions = usable[["game_id", "player_id", "tipoff", "points", "minutes"]].copy()
    predictions["expected"] = d["expected"]
    predictions["median"] = d["median"]
    for c, a in d["intervals"].items():
        predictions[f"lower_{int(c*100)}"], predictions[f"upper_{int(c*100)}"] = a.T
    predictions.to_parquet(out/"test_predictions.parquet", index=False)
    plot_reliability(reliability, out/"reliability.png", model.synthetic)
    return report


def plot_reliability(reliability, path, synthetic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Ideal")
    for threshold, entries in reliability.items():
        ax.plot([x["predicted"] for x in entries], [x["observed"] for x in entries], "o-", label=f">{threshold}")
    ax.set(xlabel="Predicted probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1),
           title="SYNTHETIC ONLY" if synthetic else "Retrospective probability reliability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def experiments(features, folder, seed=41):
    """Predeclared ablations and expanding-window periods; never select using final test."""
    folder = Path(folder)
    split = temporal_partitions(features)
    results = {"ablations": {}, "expanding_windows": []}
    sets = {
        "without_minutes": [f for f in FEATURES if "minutes" not in f and f not in {"rate_20", "workload_7d", "role_change_proxy"}],
        "without_opponent": [f for f in FEATURES if "opponent" not in f and "matchup" not in f],
    }
    for name, names in sets.items():
        model = ForecastModel(names, seed).fit(split)
        cal = split[(split.partition == "calibration") & (utc(split.cutoff) > utc(model.available_at))]
        model.calibrate(cal)
        report = evaluate(model, split[split.partition == "test"], folder/name, bootstraps=60)
        results["ablations"][name] = report["metrics"]
    # Earlier prefixes only, so original final test stays untouched by rolling selection.
    pretest = split[split.partition != "test"].drop(columns="partition")
    dates = np.sort(pretest.tipoff.unique())
    for fraction in (.65, .82, 1.):
        end = dates[int(len(dates)*fraction)-1]
        past = pretest[pretest.tipoff <= end]
        parts = temporal_partitions(past)
        model = ForecastModel(seed=seed).fit(parts)
        cal = parts[(parts.partition == "calibration") & (utc(parts.cutoff) > utc(model.available_at))]
        model.calibrate(cal)
        report = evaluate(model, parts[parts.partition == "test"], folder/f"window_{fraction}", bootstraps=60)
        results["expanding_windows"].append({"prefix_end": str(end), "metrics": report["metrics"]})
    dump_json(folder/"experiments.json", results)
    return results
