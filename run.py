# -*- coding: utf-8 -*-
"""pipeline entry point"""

import argparse
import sys
import os
import json
import time
from datetime import datetime

from pipeline.gprot_processing import run_gprot_pipeline
from pipeline.phos_processing import run_phos_pipeline
from config.utils import load_config

cfg = load_config()

parser = argparse.ArgumentParser(
    prog="run.py",
    description="omics-pipeline",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
    python run.py                  run full pipeline (gprot + phos)
    python run.py --gprot-only     run gprot only
    python run.py --phos-only      run phos only

Outputs are written to: outputs/
    """
)
parser.add_argument("--gprot-only", action="store_true")
parser.add_argument("--phos-only",  action="store_true")
parser.add_argument("--version",    action="version", version="1.0")
args = parser.parse_args()

run_gprot = not args.phos_only
run_phos  = not args.gprot_only

gprot_stats = {}
phos_stats  = {}
start_time  = time.time()

try:
    if run_gprot:
        gprot_stats = run_gprot_pipeline(cfg=cfg) or {}

    if run_phos:
        phos_stats = run_phos_pipeline(cfg=cfg) or {}

    # build log entry
    entry = {
        "date": str(datetime.now()),
        "runtime_seconds": round(time.time() - start_time, 1),
        "gprot": gprot_stats,
        "phos": phos_stats,
        "summary": {
            "gprot_experiments":         len(gprot_stats),
            "phos_experiments":          len(phos_stats),
            "total_proteins":            sum(v.get("proteins", 0) for v in gprot_stats.values()),
            "total_peptides":            sum(v.get("peptides", 0) for v in gprot_stats.values()),
            "total_phosphosites":        sum(v.get("phosphosites", 0) for v in phos_stats.values()),
            "total_psms_raw_gprot":      sum(v.get("psms_raw", 0) for v in gprot_stats.values()),
            "total_psms_raw_phos":       sum(v.get("psms_raw", 0) for v in phos_stats.values()),
            "avg_psm_retention_gprot":   round(sum(v.get("psm_retention_pct", 0) for v in gprot_stats.values()) / len(gprot_stats), 1) if gprot_stats else 0,
            "avg_psm_retention_phos":    round(sum(v.get("psm_retention_pct", 0) for v in phos_stats.values()) / len(phos_stats), 1) if phos_stats else 0,
        }
    }

    # load existing log (supports old list-format files) and append this run
    log_path = "run_log.json"
    runs = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            existing = json.load(f)
        runs = existing if isinstance(existing, list) else existing.get("runs", [])
    runs.append(entry)

    # cumulative stats across all runs
    all_experiments = set()
    for r in runs:
        all_experiments.update(r.get("gprot", {}).keys())
        all_experiments.update(r.get("phos", {}).keys())

    total_runtime_sec = round(sum(r.get("runtime_seconds", 0) for r in runs), 1)
    cumulative = {
        "total_runs":              len(runs),
        "unique_experiments":      len(all_experiments),
        "total_gprot_experiments": sum(r["summary"].get("gprot_experiments", 0) for r in runs),
        "total_phos_experiments":  sum(r["summary"].get("phos_experiments", 0) for r in runs),
        "total_proteins":          sum(r["summary"].get("total_proteins", 0) for r in runs),
        "total_peptides":          sum(r["summary"].get("total_peptides", 0) for r in runs),
        "total_phosphosites":      sum(r["summary"].get("total_phosphosites", 0) for r in runs),
        "total_psms_raw_gprot":    sum(r["summary"].get("total_psms_raw_gprot", 0) for r in runs),
        "total_psms_raw_phos":     sum(r["summary"].get("total_psms_raw_phos", 0) for r in runs),
        "total_runtime_seconds":   total_runtime_sec,
        "total_runtime_hours":     round(total_runtime_sec / 3600, 2),
    }

    with open(log_path, "w") as f:
        json.dump({"runs": runs, "cumulative": cumulative}, f, indent=2)

except KeyboardInterrupt:
    print("\n[aborted] pipeline interrupted by user", file=sys.stderr)
    sys.exit(1)

except Exception as e:
    print(f"\n[error] pipeline failed: {e}", file=sys.stderr)
    sys.exit(1)