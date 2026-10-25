"""Background job runner for the TaxiPulse Control Center.

Every button in the app runs one of the project's real command-line entry
points (the same commands documented in the runbook) as a background
subprocess. Output streams into a log file that the app tails live, so the
examiner sees genuine Spark / LightGBM output, not a simulation.

Rules that keep a live demo safe:
  * only ONE job runs at a time (two Spark jobs would fight over memory);
  * every job can be stopped from the UI;
  * job state lives on disk, so a browser refresh never loses a running job.
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from taxipulse import config

JOBS_DIR = config.DATA_DIR / "app_jobs"
STATE_FILE = JOBS_DIR / "state.json"
PY = sys.executable

# key -> (button label, [argv], rough duration shown to the user)
JOBS: dict[str, tuple[str, list[str], str]] = {
    "download":   ("Download TLC data",            [PY, "scripts/download_data.py"], "~30 s"),
    "tests":      ("Run unit tests",               [PY, "-m", "pytest", "tests", "-v", "-p", "no:cacheprovider"], "~10 s"),
    "pipeline":   ("Run Spark pipeline",           [PY, "-m", "taxipulse.pipeline.run_pipeline"], "~1-2 min"),
    "train":      ("Train demand model",           [PY, "-m", "taxipulse.ml.train"], "~1 min"),
    "bench1":     ("Run benchmarks 1-5 (quick)",  [PY, "-m", "taxipulse.benchmark.run_benchmark", "--repeats", "1"], "~1 min"),
    "bench3":     ("Run benchmarks 1-5 (3 repeats)", [PY, "-m", "taxipulse.benchmark.run_benchmark", "--repeats", "3"], "~2-3 min"),
    "mem3":       ("Memory scaling: months 1-3",   [PY, "-m", "taxipulse.benchmark.memory_scaling", "--months", "1", "2", "3", "--guard-gb", "0"], "~1 min"),
    "mem4":       ("Memory scaling: months 1-4",   [PY, "-m", "taxipulse.benchmark.memory_scaling", "--months", "1", "2", "3", "4", "--guard-gb", "0"], "~2 min"),
    "mongo_load": ("Load results into MongoDB",    [PY, "-m", "taxipulse.serving.load_mongo"], "~20 s"),
}
# The one-click demo: tests -> pipeline -> model -> benchmarks -> memory scaling
SEQUENCES: dict[str, tuple[str, list[str], str]] = {
    "full_demo": ("Run the full demo", ["tests", "pipeline", "train", "bench1", "mem3"], "~2-3 min"),
}

_procs: dict[str, subprocess.Popen] = {}   # survives Streamlit reruns (module-level)


def label(key: str) -> str:
    return (JOBS.get(key) or SEQUENCES[key])[0]


def duration(key: str) -> str:
    return (JOBS.get(key) or SEQUENCES[key])[2]


def pretty(argv: list[str]) -> str:
    """Command as a human would type it (venv python shown as `python`)."""
    return " ".join(["python" if a == PY else a for a in argv])


def commands_for(key: str) -> list[tuple[str, list[str]]]:
    steps = SEQUENCES[key][1] if key in SEQUENCES else [key]
    return [(JOBS[s][0], JOBS[s][1]) for s in steps]


# ------------------------------------------------------------------ state --
def _read_state() -> dict | None:
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split()[2] != "Z"   # zombie = finished, not yet reaped
    except (FileNotFoundError, IndexError):
        return False


def _exit_marker(log: Path) -> tuple[int, float] | None:
    try:
        with open(log, "rb") as fh:
            fh.seek(max(0, log.stat().st_size - 400))
            tail = fh.read().decode(errors="replace")
    except FileNotFoundError:
        return None
    for line in reversed(tail.splitlines()):
        if line.startswith("__EXIT__"):
            rc, ended = line[len("__EXIT__"):].split()
            return int(rc), float(ended)
    return None


def status() -> dict | None:
    """Current/last job: dict with state = running | succeeded | failed | stopped."""
    st = _read_state()
    if not st:
        return None
    log = Path(st["log"])
    proc = _procs.get(st["id"])
    if proc is not None:
        proc.poll()                              # reap if finished
    marker = _exit_marker(log)
    if marker:
        rc, ended = marker
        st.update(state="succeeded" if rc == 0 else ("stopped" if st.get("stopped") else "failed"),
                  returncode=rc, elapsed=round(ended - st["started"], 1))
    elif _pid_alive(st["pid"]):
        st.update(state="running", elapsed=round(time.time() - st["started"], 1))
    else:
        st.update(state="stopped" if st.get("stopped") else "failed", returncode=-9,
                  elapsed=round(time.time() - st["started"], 1),
                  note="process ended without reporting (killed by the OS or stopped)")
    return st


def is_running() -> bool:
    s = status()
    return bool(s and s["state"] == "running")


# ---------------------------------------------------------------- control --
def start(key: str) -> tuple[bool, str]:
    if is_running():
        return False, f"'{status()['label']}' is still running. Wait for it to finish or stop it first."
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    log = JOBS_DIR / f"{job_id}_{key}.log"
    steps = commands_for(key)

    lines = ["set -e"]
    for i, (lab, argv) in enumerate(steps, 1):
        if len(steps) > 1:
            lines.append(f"echo; echo '==================== STEP {i}/{len(steps)}: {lab} ===================='")
        lines.append(f"echo {shlex.quote('$ ' + pretty(argv))}")
        lines.append(shlex.join(argv))
    script = "( " + "\n".join(lines) + "\n ); rc=$?; echo; echo \"__EXIT__$rc $(date +%s.%N)\""

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(log, "w") as fh:
        proc = subprocess.Popen(["bash", "-c", script], stdout=fh, stderr=subprocess.STDOUT,
                                cwd=str(config.PROJECT_ROOT), env=env, start_new_session=True)
    _procs[job_id] = proc
    STATE_FILE.write_text(json.dumps({
        "id": job_id, "key": key, "label": label(key), "pid": proc.pid, "started": time.time(),
        "log": str(log), "commands": [pretty(a) for _, a in steps],
    }, indent=2))
    return True, "started"


def stop() -> None:
    st = _read_state()
    if not st or not _pid_alive(st["pid"]):
        return
    st["stopped"] = True
    STATE_FILE.write_text(json.dumps(st, indent=2))
    try:
        os.killpg(st["pid"], signal.SIGTERM)     # whole process group: bash + python + Spark JVM
    except ProcessLookupError:
        pass


def tail(n: int = 60, hide_warnings: bool = True) -> str:
    st = _read_state()
    if not st:
        return ""
    try:
        text = Path(st["log"]).read_text(errors="replace")
    except FileNotFoundError:
        return ""
    lines = [l for l in text.splitlines() if not l.startswith("__EXIT__")]
    if hide_warnings:
        noisy = (" WARN ", "setLogLevel", "Setting default log level", "SPARK_LOCAL_IP", "WARNING: Using incubator")
        lines = [l for l in lines if not any(x in l for x in noisy) and not l.startswith("\tat ")]
    return "\n".join(lines[-n:])
