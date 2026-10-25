"""Stage 4: distributed aggregation of the processed trips into small
analytical datasets that the API / dashboard / ML layer consume.

Input : data/processed/trips/
Output: data/curated/<table>.parquet  (+ .csv for anything small enough to
        open in Excel, and overview.json)

Every table here is the answer to an analytical question (see docs/DATA.md).
Nothing downstream ever touches the 19M-row trip table again.
"""
from __future__ import annotations

import json
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from taxipulse import config
from taxipulse.logging_utils import log, stage

DOW_NAMES = {1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri", 7: "Sat"}


def _save(df: DataFrame, name: str, csv: bool = True, out_dir: Path = config.CURATED_DIR) -> int:
    """Persist a (small) aggregate as parquet and optionally csv. Returns row count."""
    pdf = df.toPandas()
    pdf.to_parquet(out_dir / f"{name}.parquet", index=False)
    if csv:
        pdf.to_csv(out_dir / f"{name}.csv", index=False)
    log.info("curated/%s: %d rows", name, len(pdf))
    return len(pdf)


def _trip_metrics():
    """Common measure set reused by several aggregations."""
    return [
        F.count("*").alias("trip_count"),
        F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
        F.round(F.avg("total_amount"), 2).alias("avg_total"),
        F.round(F.avg("trip_distance_mi"), 2).alias("avg_distance_mi"),
        F.round(F.avg("duration_min"), 2).alias("avg_duration_min"),
        F.round(F.avg("tip_pct"), 2).alias("avg_tip_pct"),
        F.round(F.sum("total_amount"), 2).alias("total_revenue"),
    ]


def run(spark: SparkSession) -> dict:
    config.ensure_dirs()
    trips = spark.read.parquet(str(config.PROCESSED_DIR / "trips"))
    dow_expr = F.create_map([F.lit(x) for kv in DOW_NAMES.items() for x in kv])
    summary: dict = {}

    # ---------------------------------------------------------- overview --
    with stage("overview KPIs"):
        row = trips.agg(
            F.count("*").alias("total_trips"),
            F.round(F.sum("total_amount"), 2).alias("total_revenue"),
            F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
            F.round(F.avg("total_amount"), 2).alias("avg_total"),
            F.round(F.avg("trip_distance_mi"), 2).alias("avg_distance_mi"),
            F.round(F.avg("duration_min"), 2).alias("avg_duration_min"),
            F.round(F.avg("tip_pct"), 2).alias("avg_tip_pct_card"),
            F.round(F.avg("passenger_count"), 2).alias("avg_passengers"),
            F.min("pickup_date").alias("date_from"),
            F.max("pickup_date").alias("date_to"),
            F.countDistinct("pickup_date").alias("days_covered"),
            F.countDistinct("PULocationID").alias("active_pickup_zones"),
        ).collect()[0].asDict()
        row["date_from"], row["date_to"] = str(row["date_from"]), str(row["date_to"])
        row["avg_trips_per_day"] = round(row["total_trips"] / row["days_covered"], 1)
        (config.CURATED_DIR / "overview.json").write_text(json.dumps(row, indent=2, default=str))
        summary["overview"] = row

    # ---------------------------------------------------------- temporal --
    with stage("temporal aggregates"):
        _save(trips.groupBy("pickup_date", "pickup_dow", "is_weekend").agg(*_trip_metrics()).orderBy("pickup_date"), "daily_metrics")
        _save(trips.groupBy("pickup_hour").agg(*_trip_metrics()).orderBy("pickup_hour"), "hourly_profile")
        _save(trips.groupBy("pickup_dow").agg(*_trip_metrics()).withColumn("dow_name", dow_expr[F.col("pickup_dow")]).orderBy("pickup_dow"), "dow_profile")
        _save(trips.groupBy("year_month").agg(*_trip_metrics()).orderBy("year_month"), "monthly_metrics")
        _save(
            trips.groupBy("pickup_dow", "pickup_hour").agg(F.count("*").alias("trip_count"), F.round(F.avg("fare_amount"), 2).alias("avg_fare"))
            .withColumn("dow_name", dow_expr[F.col("pickup_dow")]).orderBy("pickup_dow", "pickup_hour"),
            "hour_dow_heatmap",
        )
        _save(trips.groupBy("is_weekend", "pickup_hour").agg(*_trip_metrics()).orderBy("is_weekend", "pickup_hour"), "weekend_vs_weekday_hourly")

    # ------------------------------------------------------- geographic --
    with stage("zone + borough aggregates"):
        pickups = trips.groupBy("PULocationID", "pu_zone", "pu_borough").agg(
            F.count("*").alias("pickup_trips"),
            F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
            F.round(F.avg("total_amount"), 2).alias("avg_total"),
            F.round(F.avg("trip_distance_mi"), 2).alias("avg_distance_mi"),
            F.round(F.avg("duration_min"), 2).alias("avg_duration_min"),
            F.round(F.avg("tip_pct"), 2).alias("avg_tip_pct"),
            F.round(F.sum("total_amount"), 2).alias("pickup_revenue"),
        ).withColumnRenamed("PULocationID", "zone_id").withColumnRenamed("pu_zone", "zone").withColumnRenamed("pu_borough", "borough")
        dropoffs = trips.groupBy("DOLocationID").agg(F.count("*").alias("dropoff_trips")).withColumnRenamed("DOLocationID", "zone_id")
        zone_metrics = pickups.join(dropoffs, "zone_id", "outer").fillna(0, ["pickup_trips", "dropoff_trips"]).orderBy(F.desc("pickup_trips"))
        _save(zone_metrics, "zone_metrics")
        _save(trips.groupBy("pu_borough").agg(*_trip_metrics()).withColumnRenamed("pu_borough", "borough").orderBy(F.desc("trip_count")), "borough_metrics")
        _save(
            trips.groupBy("PULocationID", "pu_zone", "pu_borough", "pickup_hour").agg(F.count("*").alias("trip_count"))
            .withColumnRenamed("PULocationID", "zone_id").orderBy("zone_id", "pickup_hour"),
            "zone_hourly_profile", csv=False,
        )

    # ----------------------------------------------------------- routes --
    with stage("route / flow aggregates"):
        routes = (
            trips.groupBy("PULocationID", "pu_zone", "pu_borough", "DOLocationID", "do_zone", "do_borough")
            .agg(*_trip_metrics())
            .filter(F.col("trip_count") >= config.MIN_ROUTE_TRIPS)
            .orderBy(F.desc("trip_count"))
            .limit(config.TOP_ROUTES_N)
        )
        _save(routes, "route_metrics")
        _save(trips.groupBy("pu_borough", "do_borough").agg(*_trip_metrics()).orderBy(F.desc("trip_count")), "borough_flows")

    # -------------------------------------------------------- economics --
    with stage("economics aggregates"):
        pay_map = F.create_map([F.lit(x) for kv in config.PAYMENT_TYPES.items() for x in kv])
        rate_map = F.create_map([F.lit(x) for kv in config.RATE_CODES.items() for x in kv])
        _save(trips.groupBy("payment_type").agg(*_trip_metrics()).withColumn("payment_name", pay_map[F.col("payment_type")]).orderBy(F.desc("trip_count")), "payment_type_breakdown")
        _save(trips.groupBy("rate_code").agg(*_trip_metrics()).withColumn("rate_name", rate_map[F.col("rate_code")]).orderBy(F.desc("trip_count")), "rate_code_breakdown")
        _save(trips.groupBy("passenger_count").agg(*_trip_metrics()).orderBy("passenger_count"), "passenger_count_breakdown")
        # Fare vs distance relationship, 0.5-mile bins (capped at 30 mi for readability)
        _save(
            trips.filter(F.col("trip_distance_mi") <= 30)
            .withColumn("distance_bin_mi", F.floor(F.col("trip_distance_mi") * 2) / 2)
            .groupBy("distance_bin_mi").agg(F.count("*").alias("trip_count"), F.round(F.avg("fare_amount"), 2).alias("avg_fare"), F.round(F.avg("duration_min"), 2).alias("avg_duration_min"))
            .orderBy("distance_bin_mi"),
            "fare_by_distance",
        )
        _save(
            trips.filter(F.col("duration_min") <= 90).withColumn("duration_bin_min", (F.floor(F.col("duration_min") / 5) * 5).cast("int"))
            .groupBy("duration_bin_min").agg(F.count("*").alias("trip_count")).orderBy("duration_bin_min"),
            "duration_distribution",
        )
        _save(trips.groupBy("pickup_hour").agg(F.round(F.avg("tip_pct"), 2).alias("avg_tip_pct"), F.round(F.avg("tip_amount"), 2).alias("avg_tip_amount"), F.count("*").alias("trip_count")).orderBy("pickup_hour"), "tips_by_hour")

    # ------------------------------------------------ ML base table -------
    with stage("hourly zone demand (ML base)"):
        hz = (
            trips.groupBy("PULocationID", "pickup_date", "pickup_hour")
            .agg(F.count("*").alias("trip_count"), F.round(F.avg("fare_amount"), 2).alias("avg_fare"), F.round(F.sum("total_amount"), 2).alias("revenue"))
            .withColumnRenamed("PULocationID", "zone_id").orderBy("zone_id", "pickup_date", "pickup_hour")
        )
        n = _save(hz, "hourly_zone_demand", csv=False)
        summary["hourly_zone_demand_rows"] = n

    (config.CURATED_DIR / "_manifest.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    from taxipulse.spark_session import get_spark
    run(get_spark("TaxiPulse-aggregate"))
