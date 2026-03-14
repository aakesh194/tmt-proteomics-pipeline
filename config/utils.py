"""
utils.py - Configuration loader for the omics pipeline

Loads settings from config.yaml. Non-programmers should edit config.yaml only.

Author: Aakesh Yoganathan
Lab: Tamir Lab, UNC Chapel Hill
"""

import os
import yaml

_DEFAULTS = {
    "meta_prot_csv": "metaFiles_gProt_BHA_2.csv",
    "meta_phos_csv": "metaFiles_gphos_BHA_2.csv",

    "meta_index": "expType",
    "meta_cols": {
        "psms":     "PSMs",
        "mgf":      "mgf_file",
        "library":  "library",
        "pepts":    "pepts",
        "mods":     "mods",
        "sup_corr": "sup_corr",
    },

    "raw_dir": "Raw files",
    "lib_dir": "Library files",
    "out_dir": "outputs",

    "delta_ppm_max": 5,
    "expectation_max": 0.05,
    "ions_score_min": 15,
    "require_mod_contains": "TMT",

    "abundance_contains": "Abundance",
    "min_fraction_channels_present": 0.33,

    "pool_regex": r"BRG",
    "bridge_regex": r"B.*\d+",

    "phos_min_abundance": 1000,
    "phos_ptmrs_threshold": 50,
    "phos_filter_pY": True,
    "phos_site_csv": "Phosphosite Motifs_M.csv",

    "qc_top_n": 300,
    "qc_zscore": True,

    "notes": "",
}


def load_config(yaml_path=None):
    """
    Load configuration from config.yaml, filling in defaults for any missing
    keys. Also computes derived values so callers never have to.
    """
    #print(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml"))

    cfg = dict(_DEFAULTS)

    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")

    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        cfg.update(user_cfg)
    else:
        print(f"[config] Warning: '{yaml_path}' not found. Using built-in defaults.")

    # Derived values - always kept in sync here
    dmax = cfg["delta_ppm_max"]
    cfg["delta_ppm_range"] = (-dmax, dmax)
    cfg["post_bridge_dir"]    = os.path.join(cfg["out_dir"], "After bridging")
    cfg["post_bridge_prefix"] = "gProt"

    return cfg