"""Serve predictions from the trained models.

Two modes, honestly labelled:
  * predict_calendar(zone, when)  -> works for ANY future date/hour, uses only
    calendar features + the zone's historical hour/dow mean. Less accurate.
  * predict_next_hours(zone, n)   -> rolls the lag model forward from the last
    hour of real history, feeding its own predictions back as lags. Accurate
    for the first few hours, degrades further out.
"""
from __future__ import annotations

import json
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd

from taxipulse import config
from taxipulse.ml.train import ALL_FEATURES, CALENDAR_FEATURES, TARGET


class ModelNotAvailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load():
    cal_path = config.MODELS_DIR / "demand_model_calendar.txt"
    lag_path = config.MODELS_DIR / "demand_model_lags.txt"
    feat_path = config.FEATURES_DIR / "hourly_features.parquet"
    if not (cal_path.exists() and lag_path.exists() and feat_path.exists()):
        raise ModelNotAvailable("Train the model first: python -m taxipulse.ml.train")
    feats = pd.read_parquet(feat_path)
    means = feats.drop_duplicates(["zone_id", "hour", "dow"])[["zone_id", "hour", "dow", "zone_hour_dow_mean"]]
    metrics = json.loads((config.MODELS_DIR / "model_metrics.json").read_text())
    return lgb.Booster(model_file=str(cal_path)), lgb.Booster(model_file=str(lag_path)), feats, means, metrics


def available_zones() -> list[int]:
    return sorted(_load()[2]["zone_id"].unique().tolist())


def model_metrics() -> dict:
    return _load()[4]


def predict_calendar(zone_id: int, when: pd.Timestamp) -> dict:
    cal, _, _, means, metrics = _load()
    when = pd.Timestamp(when).floor("h")
    row = pd.DataFrame([{
        "zone_id": zone_id, "hour": when.hour, "dow": when.dayofweek, "is_weekend": int(when.dayofweek >= 5),
        "month": when.month, "day_of_month": when.day,
    }])
    row = row.merge(means, on=["zone_id", "hour", "dow"], how="left")
    if row["zone_hour_dow_mean"].isna().any():
        raise ValueError(f"Zone {zone_id} is not in the modelled set (too few trips).")
    pred = float(np.clip(cal.predict(row[CALENDAR_FEATURES])[0], 0, None))
    return {
        "zone_id": zone_id, "timestamp": str(when), "predicted_trips": round(pred, 1),
        "historical_mean_same_hour_dow": round(float(row["zone_hour_dow_mean"][0]), 1),
        "model": "lightgbm_calendar_only",
        "expected_error_mae": metrics["models"]["lightgbm_calendar_only"]["MAE"],
    }


def predict_next_hours(zone_id: int, n_hours: int = 6) -> pd.DataFrame:
    """Recursive multi-step forecast beyond the last observed hour."""
    _, lag_model, feats, means, _ = _load()
    hist = feats[feats["zone_id"] == zone_id].sort_values("ts")
    if hist.empty:
        raise ValueError(f"Zone {zone_id} is not in the modelled set.")
    series = hist[TARGET].tolist()
    ts = hist["ts"].iloc[-1]
    rows = []
    for _ in range(n_hours):
        ts = ts + pd.Timedelta(hours=1)
        s = pd.Series(series)
        m = means[(means.zone_id == zone_id) & (means.hour == ts.hour) & (means.dow == ts.dayofweek)]["zone_hour_dow_mean"]
        x = {
            "zone_id": zone_id, "hour": ts.hour, "dow": ts.dayofweek, "is_weekend": int(ts.dayofweek >= 5),
            "month": ts.month, "day_of_month": ts.day, "zone_hour_dow_mean": float(m.iloc[0]) if len(m) else float(s.tail(168).mean()),
            "lag_1h": series[-1], "lag_2h": series[-2], "lag_3h": series[-3], "lag_24h": series[-24], "lag_168h": series[-168],
            "roll_mean_24h": float(s.tail(24).mean()), "roll_mean_168h": float(s.tail(168).mean()),
        }
        pred = float(np.clip(lag_model.predict(pd.DataFrame([x])[ALL_FEATURES])[0], 0, None))
        rows.append({"zone_id": zone_id, "timestamp": ts, "predicted_trips": round(pred, 1), "is_prediction": True})
        series.append(pred)
    return pd.DataFrame(rows)


def recent_history(zone_id: int, hours: int = 72) -> pd.DataFrame:
    feats = _load()[2]
    h = feats[feats["zone_id"] == zone_id].sort_values("ts").tail(hours)
    return h[["zone_id", "ts", TARGET]].rename(columns={"ts": "timestamp", TARGET: "actual_trips"})
