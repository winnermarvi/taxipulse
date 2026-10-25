"""Tiny stage logger + timer so the pipeline is observable in a viva."""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("taxipulse")


@contextmanager
def stage(name: str):
    log.info("▶ START %s", name)
    t0 = time.perf_counter()
    yield
    log.info("✔ DONE  %s in %.1fs", name, time.perf_counter() - t0)
