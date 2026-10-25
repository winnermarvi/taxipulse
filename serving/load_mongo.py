"""Load the curated analytical tables into MongoDB (NoSQL serving layer).

Why Mongo here: the dashboard/API needs fast key-based reads of small
aggregates ("give me zone 161", "give me the hour x weekday heatmap"), which
is exactly the document-store access pattern. Spark stays the compute engine;
Mongo is only a serving store for its outputs.

    python -m taxipulse.serving.load_mongo            # needs mongod running
    python -m taxipulse.serving.load_mongo --dry-run  # just show what would load

Collections mirror the curated parquet tables one-to-one.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from taxipulse import config
from taxipulse.logging_utils import log

INDEXES = {
    "zone_metrics": ["zone_id", "borough"],
    "route_metrics": ["PULocationID", "DOLocationID"],
    "hourly_zone_demand": [("zone_id", "pickup_date")],
    "daily_metrics": ["pickup_date"],
    "zone_hourly_profile": ["zone_id"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    tables = sorted(p.stem for p in config.CURATED_DIR.glob("*.parquet"))
    if not tables:
        raise SystemExit("No curated tables. Run the Spark pipeline first.")
    if args.dry_run:
        for t in tables:
            log.info("would load %-28s %8d docs", t, len(pd.read_parquet(config.CURATED_DIR / f"{t}.parquet")))
        return

    from pymongo import ASCENDING, MongoClient
    client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client[config.MONGO_DB]
    for t in tables:
        df = pd.read_parquet(config.CURATED_DIR / f"{t}.parquet")
        for c in df.select_dtypes(include=["datetime64[ns]", "object"]).columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]) or df[c].map(lambda v: hasattr(v, "isoformat")).any():
                df[c] = df[c].astype(str)
        docs = json.loads(df.to_json(orient="records"))
        db[t].drop()
        if docs:
            db[t].insert_many(docs, ordered=False)
        for idx in INDEXES.get(t, []):
            db[t].create_index([(f, ASCENDING) for f in (idx if isinstance(idx, tuple) else (idx,))])
        log.info("loaded %-28s %8d docs", t, len(docs))
    ov = json.loads((config.CURATED_DIR / "overview.json").read_text())
    db["overview"].drop(); db["overview"].insert_one(ov)
    for name in ("quality_report", "benchmark_results"):
        p = config.REPORTS_DIR / f"{name}.json"
        if p.exists():
            db[name].drop(); db[name].insert_one(json.loads(p.read_text()))
    mp = config.MODELS_DIR / "model_metrics.json"
    if mp.exists():
        db["model_metrics"].drop(); db["model_metrics"].insert_one(json.loads(mp.read_text()))
    log.info("Done. Database '%s' now has collections: %s", config.MONGO_DB, sorted(db.list_collection_names()))


if __name__ == "__main__":
    main()
