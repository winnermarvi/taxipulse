"""One place to build the SparkSession so every stage uses identical settings."""
from __future__ import annotations

from pyspark.sql import SparkSession

from taxipulse import config


def get_spark(app_name: str | None = None) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name or config.SPARK_APP_NAME)
        .master(config.SPARK_MASTER)
        .config("spark.driver.memory", config.SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", config.SPARK_SHUFFLE_PARTITIONS)
        # TLC timestamps are naive New York local time. Forcing the session to
        # UTC stops Spark from silently shifting them by the machine's timezone.
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
