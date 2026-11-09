"""Past-only model selection, split-conformal quantiles, and a minutes/rate alternative."""
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import dump_json, now, utc
from .features import FEATURES, temporal_partitions

LEVELS = [.05, .1, .175, .5, .825, .9, .95]
COVERAGES = [.65, .8, .9]


def tree(loss="squared_error", quantile=None, leaf=15, seed=41):
    return HistGradientBoostingRegressor(loss=loss, quantile=quantile, max_iter=65,
                                        max_leaf_nodes=leaf, min_samples_leaf=25,
                                        l2_regularization=5, early_stopping=False,
                                        random_state=seed)


def conformal_correction(y, lower, upper, coverage):
    scores = np.maximum(lower - y, y - upper)
    n = len(scores)
    if not n:
        raise ValueError("Empty calibration data")
    rank = min(n, int(np.ceil((n+1)*coverage)))
    # Nonnegative correction avoids shrinking and preserves base quantile interpretation.
    return float(max(0, np.sort(scores)[rank-1]))


def eligible(frame):
    return frame[frame.played.fillna(False).astype(bool) & frame.minutes.gt(0) &
                 frame.points.notna() & frame.history_n.ge(3)].copy()


class ForecastModel:
    def __init__(self, features=None, seed=41):
        self.features = list(features or FEATURES)
        self.seed = seed
        self.calibrated = False

    def _x(self, frame):
        return frame[self.features].astype(float).replace([np.inf, -np.inf], np.nan)

    def fit(self, partitions):
        train = eligible(partitions[partitions.partition == "train"])
        val = eligible(partitions[partitions.partition == "validation"])
        val = val[utc(val.cutoff) > utc(train.tipoff.max())+pd.Timedelta(days=2)]
        if len(train) < 60 or len(val) < 20:
            raise ValueError("Insufficient train/validation observations")
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        x = self.imputer.fit_transform(self._x(train))
        xv = self.imputer.transform(self._x(val))
        y = train.points.to_numpy(float)
        self.models = {
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=30)),
            "boosted_7": tree(leaf=7, seed=self.seed),
            "boosted_15": tree(leaf=15, seed=self.seed),
        }
        scores = {}
        with threadpool_limits(limits=2):
            for name, model in self.models.items():
                model.fit(x, y)
                scores[name] = float(mean_absolute_error(val.points, np.maximum(0, model.predict(xv))))
            # Fit minutes and scoring rate only on played training rows. Paired residual draws
            # retain their empirical dependence rather than multiplying two point estimates.
            self.minutes_model, self.rate_model = tree(seed=self.seed), tree(seed=self.seed)
            # Reserve a chronological tail INSIDE training for paired residuals.
            # Selection validation labels must not fit this candidate's residual law.
            training_dates = np.sort(train.tipoff.unique())
            boundary = training_dates[max(0, int(len(training_dates)*.8)-1)]
            base_mask = (train.tipoff <= boundary).to_numpy()
            tail_mask = ~base_mask
            self.minutes_model.fit(x[base_mask], train.minutes.to_numpy()[base_mask])
            self.rate_model.fit(x[base_mask], (train.points/train.minutes).to_numpy()[base_mask])
            mv = np.clip(self.minutes_model.predict(x[tail_mask]), 1, 48)
            rv = np.maximum(0, self.rate_model.predict(x[tail_mask]))
            self.paired_residuals = np.c_[train.minutes.to_numpy()[tail_mask]-mv,
                                         (train.points/train.minutes).to_numpy()[tail_mask]-rv]
            self.residual_fit_partition = "chronological_training_tail_only"
            scores["minutes_rate"] = float(mean_absolute_error(val.points, self._minute_samples(xv).mean(axis=1)))
            self.quantiles = {q: tree("quantile", q, seed=self.seed).fit(x, y) for q in LEVELS}
        scores["season_average"] = float(mean_absolute_error(val.points, val.season_points))
        scores["recent_average"] = float(mean_absolute_error(val.points, val.points_5))
        self.validation_scores = scores
        self.selected = min(scores, key=scores.get)
        self.fit_data_cutoff = str(train.cutoff.max())
        self.selection_data_cutoff = str(val.cutoff.max())
        # Historical simulation artifact is available only after all selection labels are known.
        self.available_at = str(val.tipoff.max()+pd.Timedelta(days=2))
        self.fit_available_at = self.available_at
        self.trained_at = now()
        self.synthetic = bool(partitions.synthetic.fillna(False).all())
        self.split_manifest = {
            name: {"rows": len(g), "games": int(g.game_id.nunique()),
                   "start": str(g.tipoff.min()), "end": str(g.tipoff.max())}
            for name, g in partitions.groupby("partition")}
        test = partitions[partitions.partition == "test"]
        self.frozen_test_hash = hashlib.sha256(
            "\n".join(sorted(test.game_id.unique())).encode()).hexdigest()
        self.training_game_ids = set(train.game_id.astype(str))
        return self

    def _minute_samples(self, x):
        residual = self.paired_residuals
        # Deterministic empirical Monte Carlo with paired validation residuals, <=400 draws.
        idx = np.linspace(0, len(residual)-1, min(400, len(residual))).astype(int)
        m = np.clip(self.minutes_model.predict(x)[:, None] + residual[idx, 0], 1, 48)
        r = np.maximum(0, self.rate_model.predict(x)[:, None] + residual[idx, 1])
        return m*r

    def point_predictions(self, frame):
        with threadpool_limits(limits=2):
            x = self.imputer.transform(self._x(frame))
            pred = {name: np.maximum(0, model.predict(x)) for name, model in self.models.items()}
            pred.update(season_average=frame.season_points.to_numpy(float),
                        recent_average=frame.points_5.to_numpy(float),
                        minutes_rate=self._minute_samples(x).mean(axis=1))
        return pred

    def raw_quantiles(self, frame):
        with threadpool_limits(limits=2):
            x = self.imputer.transform(self._x(frame))
            raw = np.column_stack([self.quantiles[q].predict(x) for q in LEVELS])
        # Rearrangement resolves crossing before conformal scores are computed.
        return np.maximum(0, np.sort(raw, axis=1))

    def calibrate(self, frame):
        frame = eligible(frame)
        if len(frame) < 30:
            raise ValueError("Need at least 30 calibration observations")
        if utc(frame.cutoff.min()) <= utc(self.fit_available_at):
            raise ValueError("Calibration overlaps model selection or label availability")
        raw = self.raw_quantiles(frame)
        self.corrections = {}
        for coverage in COVERAGES:
            alpha = (1-coverage)/2
            lo, hi = LEVELS.index(round(alpha, 3)), LEVELS.index(round(1-alpha, 3))
            self.corrections[coverage] = conformal_correction(frame.points.to_numpy(), raw[:, lo], raw[:, hi], coverage)
        self.calibration_rows = len(frame)
        self.available_at = str(frame.tipoff.max()+pd.Timedelta(days=2))
        self.calibrated = True
        self.calibrated_at = now()
        return self

    def distribution(self, frame, enforce_cutoff=True):
        if not self.calibrated:
            raise ValueError("Run calibration first")
        if enforce_cutoff and (utc(frame.cutoff) <= utc(self.available_at)).any():
            raise ValueError("Checkpoint/calibration was not available at prediction cutoff")
        raw = self.raw_quantiles(frame)
        intervals = {}
        last_lo, last_hi = None, None
        for coverage in COVERAGES:
            alpha = round((1-coverage)/2, 3)
            lo = np.maximum(0, raw[:, LEVELS.index(alpha)]-self.corrections[coverage])
            hi = raw[:, LEVELS.index(round(1-alpha, 3))]+self.corrections[coverage]
            if last_lo is not None:
                lo, hi = np.minimum(lo, last_lo), np.maximum(hi, last_hi)
            intervals[coverage] = np.c_[lo, hi]
            last_lo, last_hi = lo, hi
        return dict(expected=self.point_predictions(frame)[self.selected], median=raw[:, 3],
                    minutes=np.clip(self.minutes_model.predict(self.imputer.transform(self._x(frame))), 1, 48),
                    intervals=intervals)

    @staticmethod
    def exceedance(distribution, threshold):
        # Piecewise-linear CDF through calibrated anchors; endpoint tail is an explicit
        # approximation. Probability reliability must be evaluated independently.
        intervals = distribution["intervals"]
        anchors = np.column_stack([np.zeros(len(distribution["median"])),
                                   intervals[.9][:, 0], intervals[.8][:, 0], intervals[.65][:, 0],
                                   distribution["median"], intervals[.65][:, 1],
                                   intervals[.8][:, 1], intervals[.9][:, 1],
                                   intervals[.9][:, 1] + np.maximum(5, intervals[.9][:, 1]-distribution["median"])])
        probs = [0, .05, .1, .175, .5, .825, .9, .95, 1]
        # A points line means P(points > threshold); use a half-point continuity correction
        # for integer outcomes. Duplicate anchors retain rightmost CDF mass.
        t = np.floor(threshold)+.5
        return np.array([1-np.interp(t, row, probs) for row in anchors]).clip(0, 1)

    def contributions(self, frame, k=5):
        """Local model associations, not causes. Reference values use training data only."""
        x = self.imputer.transform(self._x(frame))
        if self.selected == "ridge":
            pipeline = self.models["ridge"]
            scaled = pipeline.steps[0][1].transform(x)[0]
            effects = scaled * pipeline.steps[-1][1].coef_
            method = "Additive Ridge contributions relative to training feature means, before nonnegative clipping"
        elif self.selected in {"season_average", "recent_average"}:
            key = "season_points" if self.selected == "season_average" else "points_5"
            return {"method": "Selected average baseline", "effects": [{"feature": key, "value": float(frame[key].iloc[0])}],
                    "interpretation": "Predictive associations, not causal effects"}
        else:
            def estimate(a):
                if self.selected == "minutes_rate":
                    return self._minute_samples(a).mean(axis=1)
                return self.models[self.selected].predict(a)
            with threadpool_limits(limits=2):
                reference = float(estimate(x)[0])
                effects = []
                for j in range(x.shape[1]):
                    altered = x.copy()
                    altered[0, j] = self.imputer.statistics_[j]
                    effects.append(reference-float(estimate(altered)[0]))
            effects = np.array(effects)
            method = "One-feature substitution with training median; correlated features can distort attribution"
        ranked = np.argsort(np.abs(effects))[::-1][:k]
        return {"method": method,
                "effects": [{"feature": self.features[i], "signed_points": round(float(effects[i]), 4)} for i in ranked],
                "interpretation": "Predictive associations, not causal effects"}

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, folder / "model.joblib")
        dump_json(folder / "metadata.json", {
            "selected": self.selected, "validation_mae": self.validation_scores,
            "trained_at": self.trained_at, "available_at": self.available_at,
            "calibrated": self.calibrated, "synthetic": self.synthetic,
            "calibrated_at": getattr(self, "calibrated_at", None),
            "calibration_rows": getattr(self, "calibration_rows", 0),
            "fit_available_at": self.fit_available_at,
            "residual_fit_partition": self.residual_fit_partition,
            "features": self.features, "split_manifest": self.split_manifest,
            "frozen_test_hash": self.frozen_test_hash, "seed": self.seed,
            "corrections": getattr(self, "corrections", {}),
            "note": "Available-at is a simulated historical training date; actual execution is trained_at."})


def train_from_features(features, folder, seed=41):
    split = temporal_partitions(features)
    # Save the test identity BEFORE fitting/selection; no test outcomes guide selection.
    Path(folder).mkdir(parents=True, exist_ok=True)
    dump_json(Path(folder)/"frozen_test.json", sorted(split.loc[split.partition == "test", "game_id"].unique()))
    model = ForecastModel(seed=seed).fit(split)
    model.save(folder)
    split.to_parquet(Path(folder)/"partitions.parquet", index=False)
    return model, split
