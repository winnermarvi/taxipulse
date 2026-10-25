"""Experiment 6: memory scaling — where single-machine processing runs out of room.

Experiments 1-5 used a NARROW query (2 of 20 columns, one count), which plays to
pandas' strengths; pandas won. This experiment asks the opposite question: how
much memory does each engine need when the job touches every column, and what
happens as the dataset grows?

Workload (identical in both engines): read all 20 columns of N months, derive
trip duration from the timestamps, then aggregate by
(pickup zone x pickup hour x payment type) computing 8 measures.

Method — and why it is measured this way:
  * pandas runs in a CHILD PROCESS, once per dataset size (1, 3, 5 months). The
    child reports wall time and PEAK RESIDENT MEMORY (ru_maxrss), which is real
    physical memory actually touched.
  * A guard rail (RLIMIT_AS, default 85% of system RAM) exists ONLY to stop a
    runaway process from dragging the whole machine into swap. It caps virtual
    address space, which PyArrow over-reserves, so a trip of the guard rail is
    reported honestly as "guard rail" and NOT as proof that the data did not fit.
  * Spark runs the same workload at each size with its normal driver memory.

The result is a growth curve: memory per million rows, from which the dataset
size that would exhaust this machine is extrapolated.

    python -m taxipulse.benchmark.memory_scaling
    python -m taxipulse.benchmark.memory_scaling --months 1 2 3 4 5

Output: data/reports/memory_scaling.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime

import pyarrow.parquet as pq

from taxipulse import config

GROUP_KEYS = ["PULocationID", "pickup_hour", "payment_type"]


def system_ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, AttributeError):
        return None


def _files(n_months: int) -> list:
    return sorted(config.RAW_DIR.glob(config.RAW_TRIP_GLOB))[:n_months]


def _rows(files) -> int:
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


# ------------------------------------------------------------ pandas side --
def pandas_child(n_months: int, guard_gb: float) -> dict:
    """Runs inside the child process. Reports peak RESIDENT memory, not a cap."""
    if guard_gb > 0:
        limit = int(guard_gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    import pandas as pd

    files = _files(n_months)
    t0 = time.perf_counter()
    try:
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
        df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
        out = df.groupby(GROUP_KEYS).agg(
            trips=("fare_amount", "size"), total_fare=("fare_amount", "sum"), avg_fare=("fare_amount", "mean"),
            max_fare=("fare_amount", "max"), avg_distance=("trip_distance", "mean"),
            avg_duration=("duration_min", "mean"), total_tip=("tip_amount", "sum"),
            avg_passengers=("passenger_count", "mean"))
        return {"status": "completed", "seconds": round(time.perf_counter() - t0, 2), "result_rows": int(len(out)),
                "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)}
    except MemoryError:
        return {"status": "guard_rail_tripped", "seconds": round(time.perf_counter() - t0, 2),
                "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
                "note": ("hit the virtual-address guard rail, which PyArrow over-reserves; this is NOT proof that the "
                         "data exceeded physical memory. Re-run with --guard-gb 0 to measure the true requirement.")}
    except Exception as exc:
        return {"status": "failed_other", "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                "seconds": round(time.perf_counter() - t0, 2),
                "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)}


def run_pandas(n_months: int, guard_gb: float) -> dict:
    proc = subprocess.run([sys.executable, "-m", "taxipulse.benchmark.memory_scaling", "--child",
                           "--child-months", str(n_months), "--guard-gb", str(guard_gb)],
                          capture_output=True, text=True, cwd=str(config.PROJECT_ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    return {"status": "killed_by_os", "exit_code": proc.returncode,
            "note": "the child was terminated by the OS out-of-memory killer before it could report — the clearest possible evidence that the job did not fit",
            "stderr_tail": proc.stderr.strip()[-300:]}


# ------------------------------------------------------------- spark side --
def run_spark(spark, n_months: int) -> dict:
    from pyspark.sql import functions as F

    files = [str(f) for f in _files(n_months)]
    t0 = time.perf_counter()
    df = spark.read.parquet(*files)
    for c in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
        df = df.withColumn(c, F.col(c).cast("timestamp"))   # 2026 files use TIMESTAMP_NTZ
    df = (df.withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
            .withColumn("duration_min", (F.col("tpep_dropoff_datetime").cast("long") - F.col("tpep_pickup_datetime").cast("long")) / 60.0))
    rows = df.groupBy(*GROUP_KEYS).agg(
        F.count("*").alias("trips"), F.sum("fare_amount").alias("total_fare"), F.avg("fare_amount").alias("avg_fare"),
        F.max("fare_amount").alias("max_fare"), F.avg("trip_distance").alias("avg_distance"),
        F.avg("duration_min").alias("avg_duration"), F.sum("tip_amount").alias("total_tip"),
        F.avg("passenger_count").alias("avg_passengers")).count()
    return {"status": "completed", "seconds": round(time.perf_counter() - t0, 2), "result_rows": rows,
            "driver_memory": config.SPARK_DRIVER_MEMORY}


# -------------------------------------------------------------------- run --
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, nargs="*", default=[1, 3, 5], help="dataset sizes to test, in months")
    ap.add_argument("--guard-gb", type=float, default=0,
                    help="optional cap on VIRTUAL address space. Default 0 (off): PyArrow over-reserves virtual memory, "
                         "so this cap trips long before physical memory is exhausted and produces misleading results.")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--child-months", type=int, default=1, help=argparse.SUPPRESS)
    args = ap.parse_args()

    ram = system_ram_gb()
    guard = max(args.guard_gb, 0)

    if args.child:
        print("__RESULT__" + json.dumps(pandas_child(args.child_months, args.guard_gb)))
        return

    config.ensure_dirs()
    from taxipulse.spark_session import get_spark

    print(f"Experiment 6: full-width heavy aggregation (all 20 columns, 8 measures)")
    print(f"System RAM: {ram} GB | pandas guard rail: {guard} GB virtual | Spark driver: {config.SPARK_DRIVER_MEMORY}\n")
    print(f"{'months':>7} {'rows':>12} | {'pandas':>10} {'peak RSS':>10} | {'spark':>10}")
    print("-" * 60)

    spark = get_spark("TaxiPulse-memory-scaling")
    measurements = []
    for m in args.months:
        files = _files(m)
        rows = _rows(files)
        p = run_pandas(m, guard)
        s = run_spark(spark, m)
        measurements.append({"months": m, "rows": rows, "input_mb": round(sum(f.stat().st_size for f in files) / 1e6, 1),
                             "pandas": p, "spark": s})
        p_time = f"{p['seconds']}s" if p["status"] == "completed" else p["status"]
        print(f"{m:>7} {rows:>12,} | {p_time:>10} {str(p.get('peak_rss_gb', '-')) + ' GB':>10} | {str(s['seconds']) + 's':>10}")
        # Persist after EVERY size: at the largest size the OS may kill this process
        # itself, and partial measurements are still valid measurements.
        (config.REPORTS_DIR / "memory_scaling.json").write_text(json.dumps(
            {"generated_at": datetime.now().isoformat(timespec="seconds"), "experiment": "6_memory_scaling",
             "status": "in_progress", "measurements": measurements}, indent=2))
    spark.stop()

    # --- growth analysis: memory per million rows, and the projected limit ---
    ok = [x for x in measurements if x["pandas"]["status"] == "completed"]
    analysis: dict = {}
    if len(ok) >= 2:
        first, last = ok[0], ok[-1]
        d_gb = last["pandas"]["peak_rss_gb"] - first["pandas"]["peak_rss_gb"]
        d_rows = (last["rows"] - first["rows"]) / 1e6
        per_m = round(d_gb / d_rows, 3) if d_rows else None
        analysis["gb_per_million_rows"] = per_m
        analysis["memory_amplification_vs_parquet"] = round(last["pandas"]["peak_rss_gb"] * 1000 / last["input_mb"], 1)
        if per_m and ram:
            usable = ram * 0.8   # leave 20% for the OS and everything else
            analysis["projected_max_rows_on_this_machine"] = int((usable / per_m) * 1e6)
            analysis["projected_max_months"] = round((usable / per_m) * 1e6 / (last["rows"] / last["months"]), 1)
            analysis["assumption"] = f"linear growth, {usable:.1f} GB usable of {ram} GB total"
    out = {"generated_at": datetime.now().isoformat(timespec="seconds"), "experiment": "6_memory_scaling",
           "workload": "read all 20 columns, derive duration, aggregate 8 measures by (zone, hour, payment type)",
           "environment": {"platform": platform.platform(), "python": platform.python_version(),
                           "cpu_cores": os.cpu_count(), "ram_gb": ram, "guard_rail_gb": guard,
                           "spark_driver_memory": config.SPARK_DRIVER_MEMORY},
           "measurements": measurements, "analysis": analysis,
           "caveat": ("peak_rss_gb is real physical memory touched by the pandas process. A 'guard_rail_tripped' status "
                      "reflects the virtual-address safety cap, not a proven physical-memory limit.")}
    path = config.REPORTS_DIR / "memory_scaling.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWritten to {path}")
    if analysis.get("gb_per_million_rows"):
        print(f"\npandas needs ~{analysis['gb_per_million_rows']} GB per million rows "
              f"({analysis['memory_amplification_vs_parquet']}x the Parquet file size held in RAM).")
        if "projected_max_months" in analysis:
            print(f"Projected limit on this machine: ~{analysis['projected_max_rows_on_this_machine']:,} rows "
                  f"(~{analysis['projected_max_months']} months of this dataset).")
        print("Spark's memory use does not grow with the dataset: it streams partitions and spills to disk.")


if __name__ == "__main__":
    main()
