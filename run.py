# -*- coding: utf-8 -*-
"""
run.py - Entry point for the omics pipeline.

Author: Aakesh Yoganathan
Lab: Tamir Lab, UNC Chapel Hill
Dependencies: See requirements.txt
"""

import argparse
import sys
import os
import json
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

try:
    if run_gprot:
        gprot_stats = run_gprot_pipeline(cfg=cfg) or {}

    if run_phos:
        phos_stats = run_phos_pipeline(cfg=cfg) or {}

    # build log entry
    entry = {
        "date": str(datetime.now()),
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

    # load existing log and append
    log_path = "run_log.json"
    log = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
    log.append(entry)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    # cumulative stats
    total_runs          = len(log)
    cum_proteins        = sum(r["summary"].get("total_proteins", 0) for r in log)
    cum_peptides        = sum(r["summary"].get("total_peptides", 0) for r in log)
    cum_phosphosites    = sum(r["summary"].get("total_phosphosites", 0) for r in log)
    cum_psms_gprot      = sum(r["summary"].get("total_psms_raw_gprot", 0) for r in log)
    cum_psms_phos       = sum(r["summary"].get("total_psms_raw_phos", 0) for r in log)
    all_experiments     = set()
    for r in log:
        all_experiments.update(r.get("gprot", {}).keys())
        all_experiments.update(r.get("phos", {}).keys())

except KeyboardInterrupt:
    print("\n[aborted] pipeline interrupted by user", file=sys.stderr)
    sys.exit(1)

except Exception as e:
    print(f"\n[error] pipeline failed: {e}", file=sys.stderr)
    sys.exit(1)