"""Demand prediction: trips per hour for a taxi zone.

Input : data/curated/hourly_zone_demand.parquet   (produced by Spark)
Output: data/features/hourly_features.parquet
        data/models/demand_model_lags.txt   (LightGBM, most accurate, needs recent history)
        data/models/demand_model_calendar.txt (LightGBM, calendar-only, works for ANY future hour)
        data/models/model_metrics.json      (baselines vs models, chronological test)
        data/models/feature_importance.json
        data/models/test_predictions.parquet

Evaluation is strictly chronological: Jan-Apr 2026 train, May 2026 test.
All lag features are shifted so a row only sees the past -> no target leakage.
"""
from __future__ import annotations

import json
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from taxipulse import config
from taxipulse.logging_utils import log, stage

CALENDAR_FEATURES = ["zone_id", "hour", "dow", "is_weekend", "month", "day_of_month", "zone_hour_dow_mean"]
LAG_FEATURES = ["lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h", "roll_mean_24h", "roll_mean_168h"]
ALL_FEATURES = CALENDAR_FEATURES + LAG_FEATURES
TARGET = "trip_count"


# ---------------------------------------------------------------- features --
def build_feature_table(hz: pd.DataFrame) -> pd.DataFrame:
    """Turn the sparse Spark aggregate into a dense zone x hour grid with lags."""
    hz = hz.copy()
    hz["ts"] = pd.to_datetime(hz["pickup_date"]) + pd.to_timedelta(hz["pickup_hour"], unit="h")

    # Only model zones with real demand; the long tail of ~0 trips/hour zones is noise.
    zone_totals = hz.groupby("zone_id")[TARGET].sum()
    zones = zone_totals[zone_totals >= config.ML_MIN_ZONE_TRIPS].index
    hz = hz[hz["zone_id"].isin(zones)]
    log.info("Modelling %d zones (>= %d trips over the period)", len(zones), config.ML_MIN_ZONE_TRIPS)

    # Dense grid: an hour with no trips is a real observation of 0, not a missing value.
    hours = pd.date_range(hz["ts"].min().floor("D"), hz["ts"].max().floor("D") + pd.Timedelta(hours=23), freq="h")
    grid = pd.MultiIndex.from_product([zones, hours], names=["zone_id", "ts"]).to_frame(index=False)
    df = grid.merge(hz[["zone_id", "ts", TARGET]], on=["zone_id", "ts"], how="left").fillna({TARGET: 0})
    df[TARGET] = df[TARGET].astype(int)
    df = df.sort_values(["zone_id", "ts"]).reset_index(drop=True)

    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek          # 0=Mon ... 6=Sun
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df["ts"].dt.month
    df["day_of_month"] = df["ts"].dt.day

    g = df.groupby("zone_id")[TARGET]
    for h in (1, 2, 3, 24, 168):
        df[f"lag_{h}h"] = g.shift(h)
    # rolling windows are computed on the *shifted* series so the current hour is excluded
    df["roll_mean_24h"] = g.transform(lambda s: s.shift(1).rolling(24, min_periods=12).mean())
    df["roll_mean_168h"] = g.transform(lambda s: s.shift(1).rolling(168, min_periods=84).mean())
    return df


def add_target_encoding(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Historical mean demand per (zone, hour, dow) computed on TRAIN rows only.
    This is also our strongest naive baseline."""
    means = df[train_mask].groupby(["zone_id", "hour", "dow"])[TARGET].mean().rename("zone_hour_dow_mean")
    return df.merge(means, on=["zone_id", "hour", "dow"], how="left")


# ----------------------------------------------------------------- metrics --
def metrics(y_true, y_pred) -> dict:
    y_pred = np.clip(y_pred, 0, None)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def fit_lgbm(X: pd.DataFrame, y: pd.Series, seed: int) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression_l1",   # MAE objective: robust to demand spikes
        n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1,
    )
    model.fit(X, y, categorical_feature=["zone_id"])
    return model


# --------------------------------------------------------------------- run --
def run() -> dict:
    config.ensure_dirs()
    with stage("load hourly zone demand"):
        hz = pd.read_parquet(config.CURATED_DIR / "hourly_zone_demand.parquet")
    with stage("feature engineering"):
        df = build_feature_table(hz)
        train_mask = df["ts"] < pd.Timestamp(config.ML_TRAIN_END)
        df = add_target_encoding(df, train_mask)
        # rows whose lags are undefined (first week) cannot be used by the lag model
        model_rows = df.dropna(subset=LAG_FEATURES + ["zone_hour_dow_mean"]).copy()
        model_rows.to_parquet(config.FEATURES_DIR / "hourly_features.parquet", index=False)
        train = model_rows[model_rows["ts"] < pd.Timestamp(config.ML_TRAIN_END)]
        test = model_rows[model_rows["ts"] >= pd.Timestamp(config.ML_TRAIN_END)]
        log.info("train rows %d (%s -> %s) | test rows %d (%s -> %s)", len(train), train.ts.min(), train.ts.max(), len(test), test.ts.min(), test.ts.max())

    results: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target": "trips per hour per pickup zone",
        "zones_modelled": int(df["zone_id"].nunique()),
        "train_period": [str(train.ts.min()), str(train.ts.max())],
        "test_period": [str(test.ts.min()), str(test.ts.max())],
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "seed": config.ML_RANDOM_SEED,
        "models": {},
    }
    y_test = test[TARGET].values

    with stage("baselines"):
        results["models"]["baseline_same_hour_last_week"] = {"features": ["lag_168h"], **metrics(y_test, test["lag_168h"].values)}
        results["models"]["baseline_zone_hour_dow_mean"] = {"features": ["zone_hour_dow_mean"], **metrics(y_test, test["zone_hour_dow_mean"].values)}
        results["models"]["baseline_previous_hour"] = {"features": ["lag_1h"], **metrics(y_test, test["lag_1h"].values)}

    with stage("train calendar-only LightGBM"):
        m_cal = fit_lgbm(train[CALENDAR_FEATURES], train[TARGET], config.ML_RANDOM_SEED)
        pred_cal = np.clip(m_cal.predict(test[CALENDAR_FEATURES]), 0, None)
        results["models"]["lightgbm_calendar_only"] = {"features": CALENDAR_FEATURES, **metrics(y_test, pred_cal)}
        m_cal.booster_.save_model(str(config.MODELS_DIR / "demand_model_calendar.txt"))

    with stage("train LightGBM with lag features"):
        m_lag = fit_lgbm(train[ALL_FEATURES], train[TARGET], config.ML_RANDOM_SEED)
        pred_lag = np.clip(m_lag.predict(test[ALL_FEATURES]), 0, None)
        results["models"]["lightgbm_with_lags"] = {"features": ALL_FEATURES, **metrics(y_test, pred_lag)}
        m_lag.booster_.save_model(str(config.MODELS_DIR / "demand_model_lags.txt"))

    with stage("write artefacts"):
        imp = sorted(zip(ALL_FEATURES, m_lag.feature_importances_.tolist()), key=lambda kv: -kv[1])
        imp_cal = sorted(zip(CALENDAR_FEATURES, m_cal.feature_importances_.tolist()), key=lambda kv: -kv[1])
        (config.MODELS_DIR / "feature_importance.json").write_text(json.dumps({"lightgbm_with_lags": imp, "lightgbm_calendar_only": imp_cal}, indent=2))
        out = test[["zone_id", "ts", TARGET, "lag_168h", "zone_hour_dow_mean"]].copy()
        out["pred_calendar"] = pred_cal.round(1)
        out["pred_lags"] = pred_lag.round(1)
        out.to_parquet(config.MODELS_DIR / "test_predictions.parquet", index=False)
        # per-zone error so the dashboard can show where the model is weak
        out["abs_err"] = (out["pred_lags"] - out[TARGET]).abs()
        per_zone = out.groupby("zone_id").agg(mae=("abs_err", "mean"), mean_demand=(TARGET, "mean")).round(2).reset_index()
        per_zone.to_csv(config.MODELS_DIR / "per_zone_test_error.csv", index=False)
        base = results["models"]["baseline_same_hour_last_week"]["MAE"]
        results["improvement_over_last_week_baseline_pct"] = round(100 * (base - results["models"]["lightgbm_with_lags"]["MAE"]) / base, 1)
        (config.MODELS_DIR / "model_metrics.json").write_text(json.dumps(results, indent=2))
        log.info("Metrics: %s", json.dumps({k: {kk: v[kk] for kk in ("MAE", "RMSE", "R2")} for k, v in results["models"].items()}, indent=1))
    return results


if __name__ == "__main__":
    run()
