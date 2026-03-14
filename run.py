# -*- coding: utf-8 -*-
"""
run.py - Entry point for the omics pipeline.

Author: Aakesh Yoganathan
Lab: Tamir Lab, UNC Chapel Hill
Dependencies: See requirements.txt
"""

import argparse
import sys

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
parser.add_argument("--phos-only", action="store_true")
parser.add_argument("--version", action="version", version="1.0")
args = parser.parse_args()

run_gprot = not args.phos_only
run_phos  = not args.gprot_only

try:
    if run_gprot:
        run_gprot_pipeline(cfg=cfg)

    if run_phos:
        run_phos_pipeline(cfg=cfg)

except KeyboardInterrupt:
    print("\n[aborted] pipeline interrupted by user", file=sys.stderr)
    sys.exit(1)

except Exception as e:
    print(f"\n[error] pipeline failed: {e}", file=sys.stderr)
    sys.exit(1)