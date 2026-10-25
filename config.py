"""Central configuration for NYC TaxiPulse.

Every path, threshold and Spark setting lives here so nothing machine-specific
is scattered through the code. Override anything with an environment variable
(e.g. TAXIPULSE_DATA_DIR, TAXIPULSE_SPARK_MASTER, TAXIPULSE_DRIVER_MEMORY).
"""
from __future__ import annotations

import os
from pathlib import Path

# Spark local mode must bind to loopback. Under WSL mirrored networking the
# machine's interfaces mirror Windows (Wi-Fi etc.), and letting Spark pick one
# makes it depend on the current network. Pinning 127.0.0.1 also silences the
# "hostname resolves to a loopback address" warning.
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

# ---------------------------------------------------------------- paths -----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("TAXIPULSE_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"              # untouched TLC parquet + zone lookup
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned, enriched trip-level parquet
CURATED_DIR = DATA_DIR / "curated"      # small analytical tables for API/UI
FEATURES_DIR = DATA_DIR / "features"    # ML feature tables
MODELS_DIR = DATA_DIR / "models"        # trained model + metrics
REPORTS_DIR = DATA_DIR / "reports"      # quality report, benchmark results

RAW_TRIP_GLOB = "yellow_tripdata_*.parquet"
ZONE_LOOKUP_FILE = RAW_DIR / "taxi_zone_lookup.csv"

# TLC public download base (official NYC TLC CloudFront bucket)
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net"
DEFAULT_MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]

# ---------------------------------------------------------------- spark -----
SPARK_APP_NAME = "NYC-TaxiPulse"
SPARK_MASTER = os.getenv("TAXIPULSE_SPARK_MASTER", "local[*]")
SPARK_DRIVER_MEMORY = os.getenv("TAXIPULSE_DRIVER_MEMORY", "4g")
SPARK_SHUFFLE_PARTITIONS = int(os.getenv("TAXIPULSE_SHUFFLE_PARTITIONS", "16"))

# ------------------------------------------------------- cleaning rules -----
# Each rule is documented in docs/DATA.md. Values are deliberately generous:
# we remove records that are physically impossible or unusable, not merely
# unusual. Long airport trips and expensive fares survive these filters.
MIN_DURATION_MIN = 1.0        # trips shorter than 1 minute are meter errors
MAX_DURATION_MIN = 180.0      # 3 hours; longer is almost always a stuck meter
MIN_DISTANCE_MI = 0.0         # strictly greater than 0
MAX_DISTANCE_MI = 100.0       # NYC to anywhere in the metro area is < 100 mi
MIN_FARE = 0.0                # strictly greater than 0 (negatives are refunds)
MAX_FARE = 500.0
MAX_SPEED_MPH = 80.0          # average speed above this is physically implausible in NYC
VALID_ZONE_MIN, VALID_ZONE_MAX = 1, 263   # 264 = Unknown, 265 = N/A in the TLC lookup

# ------------------------------------------------------------ analytics -----
TOP_ROUTES_N = 2000           # routes persisted for the flow analysis
MIN_ROUTE_TRIPS = 50          # ignore extremely rare zone pairs

# ------------------------------------------------------------------- ML -----
ML_TRAIN_END = "2026-05-01"   # rows with pickup_date < this are training
ML_RANDOM_SEED = 42
ML_MIN_ZONE_TRIPS = 5000      # only model zones with meaningful demand

# ---------------------------------------------------------------- mongo -----
MONGO_URI = os.getenv("TAXIPULSE_MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("TAXIPULSE_MONGO_DB", "taxipulse")

PAYMENT_TYPES = {0: "Flex Fare / Unknown", 1: "Credit card", 2: "Cash", 3: "No charge", 4: "Dispute", 5: "Unknown", 6: "Voided"}
RATE_CODES = {1: "Standard", 2: "JFK", 3: "Newark", 4: "Nassau/Westchester", 5: "Negotiated", 6: "Group ride", 99: "Unknown"}


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, CURATED_DIR, FEATURES_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
