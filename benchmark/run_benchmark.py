"""Big Data performance experiments. Nothing here is estimated: every number is
wall-clock measured on the machine that runs the script, and the environment
is recorded alongside the results.

    python -m taxipulse.benchmark.run_benchmark
    python -m taxipulse.benchmark.run_benchmark --repeats 3

Output: data/reports/benchmark_results.json / .csv

Experiments (same query everywhere: trips per pickup zone x hour):
  1. single_machine_pandas  vs  spark_dataframe          (full dataset)
  2. spark on Parquet  vs  spark on CSV                   (one month)
  3. spark RDD map/reduce (MapReduce style)  vs  DataFrame (one month)
  4. column pruning: read all 20 columns vs only 2        (full dataset)
  5. core scaling: local[1] vs local[*]                   (full dataset)
Comparisons that are not apples-to-apples are labelled in `note`.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from taxipulse import config
from taxipulse.logging_utils import log

RAW_FILES = sorted(config.RAW_DIR.glob(config.RAW_TRIP_GLOB))
ONE_MONTH = RAW_FILES[0]
COLS = ["PULocationID", "tpep_pickup_datetime"]


def timed(fn, repeats: int) -> tuple[float, list[float]]:
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        runs.append(round(time.perf_counter() - t0, 2))
    return statistics.median(runs), runs


def environment() -> dict:
    try:
        mem_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, AttributeError):
        mem_gb = None
    import pyspark
    return {
        "platform": platform.platform(), "python": platform.python_version(), "pyspark": pyspark.__version__,
        "cpu_cores": os.cpu_count(), "ram_gb": mem_gb, "driver_memory": config.SPARK_DRIVER_MEMORY,
        "dataset_files": [f.name for f in RAW_FILES],
        "dataset_rows": sum(pq.ParquetFile(f).metadata.num_rows for f in RAW_FILES),
        "dataset_size_mb": round(sum(f.stat().st_size for f in RAW_FILES) / 1e6, 1),
        "one_month_file": ONE_MONTH.name,
        "one_month_rows": pq.ParquetFile(ONE_MONTH).metadata.num_rows,
    }


def spark_session(master: str) -> SparkSession:
    s = (SparkSession.builder.appName("TaxiPulse-benchmark").master(master)
         .config("spark.driver.memory", config.SPARK_DRIVER_MEMORY)
         .config("spark.sql.shuffle.partitions", config.SPARK_SHUFFLE_PARTITIONS)
         .config("spark.sql.session.timeZone", "UTC").config("spark.ui.showConsoleProgress", "false").getOrCreate())
    # untimed warm-up so JVM/JIT start-up does not distort the first measurement
    s.read.parquet(str(ONE_MONTH)).select(*COLS).limit(1000).count()
    return s


# ------------------------------------------------------------ workloads --
def pandas_zone_hour(files: list[Path]) -> int:
    """Single-machine: read month by month with column pruning, concat, groupby."""
    parts = [pq.read_table(f, columns=COLS).to_pandas() for f in files]
    df = pd.concat(parts, ignore_index=True)
    out = df.groupby([df["PULocationID"], df["tpep_pickup_datetime"].dt.hour]).size()
    return len(out)


def spark_zone_hour(spark: SparkSession, paths: list[str], fmt: str = "parquet", all_columns: bool = False) -> int:
    reader = spark.read.format(fmt)
    if fmt == "csv":
        reader = reader.option("header", True).option("inferSchema", False)
    df = reader.load(paths)
    if not all_columns:
        df = df.select(*COLS)
    df = df.withColumn("h", F.hour(F.col("tpep_pickup_datetime").cast("timestamp")))
    return df.groupBy("PULocationID", "h").count().count()


def spark_rdd_mapreduce(spark: SparkSession, path: str) -> int:
    """Classic MapReduce: map -> (key, 1), reduceByKey -> sum. Same result as groupBy."""
    rdd = spark.read.parquet(path).select(*COLS).rdd
    return (rdd.map(lambda r: ((r[0], r[1].hour), 1)).reduceByKey(lambda a, b: a + b).count())


# ------------------------------------------------------------------ main --
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    config.ensure_dirs()
    env = environment()
    log.info("Environment: %s", env)
    results = []

    def record(experiment, approach, scope, rows, median, runs, note=""):
        results.append({"experiment": experiment, "approach": approach, "scope": scope, "rows": rows,
                        "median_seconds": round(median, 2), "runs_seconds": runs, "note": note})
        log.info("%-32s %-28s %-10s %8.2fs  %s", experiment, approach, scope, median, runs)

    full_paths = [str(f) for f in RAW_FILES]

    # 1. pandas vs spark (full dataset) ------------------------------------
    med, runs = timed(lambda: pandas_zone_hour(RAW_FILES), args.repeats)
    record("1_single_machine_vs_spark", "pandas (single process)", "full", env["dataset_rows"], med, runs,
           "pandas already benefits from column pruning; it is the strongest fair single-machine baseline")
    spark = spark_session("local[*]")
    med, runs = timed(lambda: spark_zone_hour(spark, full_paths), args.repeats)
    record("1_single_machine_vs_spark", "spark DataFrame local[*]", "full", env["dataset_rows"], med, runs,
           "includes Spark job scheduling overhead; on a 1-core machine Spark cannot parallelise")

    # 4. column pruning (full dataset) -------------------------------------
    med, runs = timed(lambda: spark_zone_hour(spark, full_paths, all_columns=True), args.repeats)
    record("4_column_pruning", "spark read all 20 columns", "full", env["dataset_rows"], med, runs,
           "Catalyst still prunes unused columns at plan time; difference shows optimiser effect")
    med, runs = timed(lambda: spark_zone_hour(spark, full_paths, all_columns=False), args.repeats)
    record("4_column_pruning", "spark read 2 columns", "full", env["dataset_rows"], med, runs)

    # 2. parquet vs csv (one month) ----------------------------------------
    csv_path = config.PROCESSED_DIR / "benchmark_one_month_csv"
    if not csv_path.exists():
        log.info("Writing one month as CSV for the format experiment (one-off)...")
        spark.read.parquet(str(ONE_MONTH)).write.mode("overwrite").option("header", True).csv(str(csv_path))
    csv_mb = round(sum(p.stat().st_size for p in csv_path.glob("*.csv")) / 1e6, 1)
    med, runs = timed(lambda: spark_zone_hour(spark, [str(ONE_MONTH)]), args.repeats)
    record("2_parquet_vs_csv", f"spark on Parquet ({round(ONE_MONTH.stat().st_size/1e6,1)} MB)", "1 month", env["one_month_rows"], med, runs)
    med, runs = timed(lambda: spark_zone_hour(spark, [str(csv_path)], fmt="csv"), args.repeats)
    record("2_parquet_vs_csv", f"spark on CSV ({csv_mb} MB)", "1 month", env["one_month_rows"], med, runs,
           "CSV must parse every byte of every row; Parquet reads only the 2 needed columns")

    # 3. RDD MapReduce vs DataFrame (one month) ----------------------------
    med, runs = timed(lambda: spark_rdd_mapreduce(spark, str(ONE_MONTH)), args.repeats)
    record("3_mapreduce_vs_dataframe", "RDD map + reduceByKey (MapReduce)", "1 month", env["one_month_rows"], med, runs,
           "Python lambdas serialise every row through the JVM<->Python bridge")
    med, runs = timed(lambda: spark_zone_hour(spark, [str(ONE_MONTH)]), args.repeats)
    record("3_mapreduce_vs_dataframe", "DataFrame groupBy (Catalyst/Tungsten)", "1 month", env["one_month_rows"], med, runs)
    spark.stop()

    # 5. core scaling (full dataset) --------------------------------------
    for master in ("local[1]", "local[*]"):
        s = spark_session(master)
        med, runs = timed(lambda: spark_zone_hour(s, full_paths), args.repeats)
        record("5_core_scaling", f"spark {master} ({s.sparkContext.defaultParallelism} cores)", "full", env["dataset_rows"], med, runs,
               "identical when the machine has a single core")
        s.stop()

    out = {"generated_at": datetime.now().isoformat(timespec="seconds"), "environment": env, "repeats": args.repeats,
           "query": "trips per (pickup zone, pickup hour)", "results": results}
    (config.REPORTS_DIR / "benchmark_results.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(results).drop(columns=["runs_seconds"]).to_csv(config.REPORTS_DIR / "benchmark_results.csv", index=False)
    log.info("Benchmark written to %s", config.REPORTS_DIR / "benchmark_results.json")


if __name__ == "__main__":
    main()
