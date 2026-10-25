"""Run the whole Spark pipeline: raw -> processed -> curated.

    python -m taxipulse.pipeline.run_pipeline            # everything
    python -m taxipulse.pipeline.run_pipeline --skip-clean   # re-aggregate only
    python -m taxipulse.pipeline.run_pipeline --months 2026-06  # add a new month
"""
from __future__ import annotations

import argparse
import time

from taxipulse.logging_utils import log
from taxipulse.pipeline import aggregate, clean
from taxipulse.spark_session import get_spark


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-clean", action="store_true", help="reuse data/processed and only rebuild curated tables")
    ap.add_argument("--months", nargs="*", help="only clean these months, e.g. 2026-04 2026-05")
    args = ap.parse_args()

    t0 = time.perf_counter()
    spark = get_spark()
    log.info("Spark %s | master=%s | cores=%d", spark.version, spark.sparkContext.master, spark.sparkContext.defaultParallelism)
    if not args.skip_clean:
        clean.run(spark, months=args.months)
    aggregate.run(spark)
    log.info("Pipeline finished in %.1f minutes", (time.perf_counter() - t0) / 60)
    spark.stop()


if __name__ == "__main__":
    main()
