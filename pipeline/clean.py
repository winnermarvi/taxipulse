"""Stage 1-3: ingest raw TLC parquet, validate, clean, standardise, enrich.

Input : data/raw/yellow_tripdata_*.parquet  + taxi_zone_lookup.csv
Output: data/processed/trips/  (parquet, partitioned by year_month)
        data/reports/quality_report.json

Cleaning philosophy (see docs/DATA.md): every dropped record is counted under
the FIRST rule it violates, so the quality report explains exactly what
happened to the data. Nothing is deleted just for looking unusual.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from taxipulse import config
from taxipulse.logging_utils import log, stage

# Column names verified against the actual 2026 TLC parquet schema.
REQUIRED_COLUMNS = [
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
    "trip_distance", "RatecodeID", "PULocationID", "DOLocationID", "payment_type",
    "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
    "improvement_surcharge", "total_amount", "congestion_surcharge", "Airport_fee",
]


# ------------------------------------------------------------------ ingest --
def list_raw_files(raw_dir: Path = config.RAW_DIR, months: list[str] | None = None) -> list[Path]:
    files = sorted(raw_dir.glob(config.RAW_TRIP_GLOB))
    if months:
        files = [f for f in files if any(m in f.name for m in months)]
    return files


def read_raw_trips(spark: SparkSession, raw_dir: Path = config.RAW_DIR, months: list[str] | None = None) -> DataFrame:
    """Read monthly parquet files (all, or a subset). Spark unions the files
    automatically and tags each row with the source file for per-month accounting."""
    files = [str(f) for f in list_raw_files(raw_dir, months)]
    if not files:
        raise FileNotFoundError(f"No TLC parquet files found in {raw_dir}. Run scripts/download_data.py first.")
    log.info("Reading %d raw files: %s", len(files), [Path(f).name for f in files])
    df = spark.read.parquet(*files).withColumn("source_file", F.regexp_extract(F.input_file_name(), r"(yellow_tripdata_\d{4}-\d{2})", 1))
    # 2026 files store timestamps as TIMESTAMP_NTZ (no timezone). Casting to a
    # plain timestamp is lossless because the session timezone is pinned to UTC.
    for c in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
        df = df.withColumn(c, F.col(c).cast("timestamp"))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Source schema is missing expected columns: {missing}")
    return df


def read_zone_lookup(spark: SparkSession, path: Path = config.ZONE_LOOKUP_FILE) -> DataFrame:
    schema = T.StructType([
        T.StructField("LocationID", T.IntegerType()),
        T.StructField("Borough", T.StringType()),
        T.StructField("Zone", T.StringType()),
        T.StructField("service_zone", T.StringType()),
    ])
    return spark.read.csv(str(path), header=True, schema=schema)


# ---------------------------------------------------------------- validate --
def add_validation_flags(df: DataFrame) -> DataFrame:
    """Compute derived fields and a single `reject_reason` column (null = keep)."""
    df = df.withColumn(
        "duration_min",
        (F.col("tpep_dropoff_datetime").cast("long") - F.col("tpep_pickup_datetime").cast("long")) / 60.0,
    )
    df = df.withColumn("avg_speed_mph", F.when(F.col("duration_min") > 0, F.col("trip_distance") / (F.col("duration_min") / 60.0)))
    # The file month is authoritative: a January file containing a March pickup is a vendor error.
    df = df.withColumn("file_month", F.substring("source_file", 17, 7))
    df = df.withColumn("pickup_month", F.date_format("tpep_pickup_datetime", "yyyy-MM"))

    reason = (
        F.when(F.col("tpep_pickup_datetime").isNull() | F.col("tpep_dropoff_datetime").isNull(), "null_timestamp")
        .when(F.col("pickup_month") != F.col("file_month"), "pickup_outside_file_month")
        .when(F.col("duration_min") < config.MIN_DURATION_MIN, "duration_too_short_or_negative")
        .when(F.col("duration_min") > config.MAX_DURATION_MIN, "duration_too_long")
        .when(F.col("trip_distance").isNull() | (F.col("trip_distance") <= config.MIN_DISTANCE_MI), "zero_or_null_distance")
        .when(F.col("trip_distance") > config.MAX_DISTANCE_MI, "distance_implausible")
        .when(F.col("fare_amount").isNull() | (F.col("fare_amount") <= config.MIN_FARE), "non_positive_fare")
        .when(F.col("fare_amount") > config.MAX_FARE, "fare_implausible")
        .when(F.col("total_amount") <= 0, "non_positive_total")
        .when(F.col("avg_speed_mph") > config.MAX_SPEED_MPH, "speed_implausible")
        .when(~F.col("PULocationID").between(config.VALID_ZONE_MIN, config.VALID_ZONE_MAX), "unknown_pickup_zone")
        .when(~F.col("DOLocationID").between(config.VALID_ZONE_MIN, config.VALID_ZONE_MAX), "unknown_dropoff_zone")
    )
    return df.withColumn("reject_reason", reason)


# ------------------------------------------------------------------- enrich --
def enrich(df: DataFrame, zones: DataFrame) -> DataFrame:
    """Standardise types, add time features, join human-readable zone names."""
    pu = zones.select(F.col("LocationID").alias("PULocationID"), F.col("Borough").alias("pu_borough"), F.col("Zone").alias("pu_zone"))
    do = zones.select(F.col("LocationID").alias("DOLocationID"), F.col("Borough").alias("do_borough"), F.col("Zone").alias("do_zone"))

    out = (
        df.select(
            F.col("VendorID").cast("int").alias("vendor_id"),
            F.col("tpep_pickup_datetime").alias("pickup_ts"),
            F.col("tpep_dropoff_datetime").alias("dropoff_ts"),
            # passenger_count 0 is a data-entry artefact, treat as unknown
            F.when(F.col("passenger_count") > 0, F.col("passenger_count")).cast("int").alias("passenger_count"),
            F.col("trip_distance").alias("trip_distance_mi"),
            F.col("RatecodeID").cast("int").alias("rate_code"),
            F.col("PULocationID").cast("int").alias("PULocationID"),
            F.col("DOLocationID").cast("int").alias("DOLocationID"),
            F.col("payment_type").cast("int").alias("payment_type"),
            F.col("fare_amount"), F.col("extra"), F.col("mta_tax"), F.col("tip_amount"),
            F.col("tolls_amount"), F.col("improvement_surcharge"), F.col("total_amount"),
            F.col("congestion_surcharge"), F.col("Airport_fee").alias("airport_fee"),
            F.col("cbd_congestion_fee"),
            F.round("duration_min", 2).alias("duration_min"),
            F.round("avg_speed_mph", 2).alias("avg_speed_mph"),
            F.col("file_month").alias("year_month"),
        )
        .withColumn("pickup_date", F.to_date("pickup_ts"))
        .withColumn("pickup_hour", F.hour("pickup_ts"))
        .withColumn("pickup_dow", F.dayofweek("pickup_ts"))          # 1=Sunday ... 7=Saturday
        .withColumn("is_weekend", F.col("pickup_dow").isin(1, 7))
        .withColumn("pickup_month", F.month("pickup_ts"))
        # Tip % only means something for card payments (cash tips are not recorded)
        .withColumn("tip_pct", F.when((F.col("payment_type") == 1) & (F.col("fare_amount") > 0), F.round(100 * F.col("tip_amount") / F.col("fare_amount"), 2)))
        .join(F.broadcast(pu), "PULocationID", "left")
        .join(F.broadcast(do), "DOLocationID", "left")
    )
    return out


# --------------------------------------------------------------- reporting --
def build_quality_report(raw: DataFrame, flagged: DataFrame, kept_count: int) -> dict:
    raw_count = raw.count()
    per_month = {r["file_month"]: r["n"] for r in flagged.groupBy("file_month").agg(F.count("*").alias("n")).collect()}
    kept_per_month = {r["file_month"]: r["n"] for r in flagged.filter(F.col("reject_reason").isNull()).groupBy("file_month").agg(F.count("*").alias("n")).collect()}
    reasons = {r["reject_reason"]: r["n"] for r in flagged.filter(F.col("reject_reason").isNotNull()).groupBy("reject_reason").agg(F.count("*").alias("n")).orderBy(F.desc("n")).collect()}
    null_rates = raw.select([F.round(F.avg(F.col(c).isNull().cast("double")), 4).alias(c) for c in REQUIRED_COLUMNS]).collect()[0].asDict()
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_records": raw_count,
        "retained_records": kept_count,
        "removed_records": raw_count - kept_count,
        "retention_rate": round(kept_count / raw_count, 4),
        "raw_records_per_month": dict(sorted(per_month.items())),
        "retained_records_per_month": dict(sorted(kept_per_month.items())),
        "removed_by_reason": reasons,
        "null_rate_raw": null_rates,
        "rules": {
            "duration_min": [config.MIN_DURATION_MIN, config.MAX_DURATION_MIN],
            "trip_distance_mi": [config.MIN_DISTANCE_MI, config.MAX_DISTANCE_MI],
            "fare_amount": [config.MIN_FARE, config.MAX_FARE],
            "max_avg_speed_mph": config.MAX_SPEED_MPH,
            "valid_zone_ids": [config.VALID_ZONE_MIN, config.VALID_ZONE_MAX],
        },
    }


# --------------------------------------------------------------------- run --
def _month_of(path: Path) -> str:
    return path.stem.replace("yellow_tripdata_", "")


def run(spark: SparkSession, months: list[str] | None = None) -> dict:
    """Clean one month at a time. Each month overwrites only its own partition
    of data/processed/trips, so adding a new TLC file later means running this
    for that month only. The quality report is rebuilt from per-month reports."""
    config.ensure_dirs()
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    zones = read_zone_lookup(spark)
    out = config.PROCESSED_DIR / "trips"
    month_reports_dir = config.REPORTS_DIR / "quality_by_month"
    month_reports_dir.mkdir(exist_ok=True)

    for f in list_raw_files(months=months):
        m = _month_of(f)
        with stage(f"clean {m}"):
            raw = read_raw_trips(spark, months=[m])
            flagged = add_validation_flags(raw)
            clean = enrich(flagged.filter(F.col("reject_reason").isNull()), zones)
            clean.write.mode("overwrite").partitionBy("year_month").parquet(str(out))
            kept = spark.read.parquet(str(out)).filter(F.col("year_month") == m).count()
            rep = build_quality_report(raw, flagged, kept)
            (month_reports_dir / f"{m}.json").write_text(json.dumps(rep, indent=2))
            log.info("%s: raw %d -> kept %d (%.2f%%)", m, rep["raw_records"], kept, 100 * rep["retention_rate"])

    return merge_quality_reports(month_reports_dir)


def merge_quality_reports(month_reports_dir: Path) -> dict:
    """Combine per-month quality reports into one dataset-level report."""
    reps = [json.loads(p.read_text()) for p in sorted(month_reports_dir.glob("*.json"))]
    raw = sum(r["raw_records"] for r in reps)
    kept = sum(r["retained_records"] for r in reps)
    reasons: dict[str, int] = {}
    for r in reps:
        for k, v in r["removed_by_reason"].items():
            reasons[k] = reasons.get(k, 0) + v
    null_rates = {c: round(sum(r["null_rate_raw"][c] * r["raw_records"] for r in reps) / raw, 4) for c in reps[0]["null_rate_raw"]}
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_records": raw, "retained_records": kept, "removed_records": raw - kept,
        "retention_rate": round(kept / raw, 4),
        "raw_records_per_month": {k: v for r in reps for k, v in r["raw_records_per_month"].items()},
        "retained_records_per_month": {k: v for r in reps for k, v in r["retained_records_per_month"].items()},
        "removed_by_reason": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "null_rate_raw": null_rates,
        "rules": reps[0]["rules"],
    }
    path = config.REPORTS_DIR / "quality_report.json"
    path.write_text(json.dumps(report, indent=2))
    log.info("Dataset: raw %d -> kept %d (%.2f%%). Report: %s", raw, kept, 100 * report["retention_rate"], path)
    return report


if __name__ == "__main__":
    import sys
    from taxipulse.spark_session import get_spark
    run(get_spark("TaxiPulse-clean"), months=sys.argv[1:] or None)
