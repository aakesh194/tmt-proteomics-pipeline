"""
Tests for pipeline/gprot_processing.py — functions unique to the gProt path:

  sup_Corrections (supernatant correction factors)
  sum_peps        (PSMs → protein-level sums)
  End-to-end:     sup_Corrections → sum_peps → apply_sup_correction → bridgeCenter_data
  Orchestration:  process_gprot_dataset — reads real files, writes stage CSVs
"""

import textwrap

import pandas as pd
import pytest
from pyteomics import mgf

from pipeline.gprot_processing import (
    _reset_run_state,
    process_gprot_dataset,
    sum_peps,
    sup_Corrections,
)
from pipeline.proteomics_core import (
    PSM_filter,
    apply_sup_correction,
    bridgeCenter_data,
    nan_imputation,
)


# ═══════════════════════════════════════════════════════════════════════
# Stage 03 — Sup corrections, gprot only (writes 03_peps.csv + sup factors)
# ═══════════════════════════════════════════════════════════════════════

def _sup_df(abund_126, abund_127):
    """Build a PSM DataFrame for sup_Corrections tests. Length inferred from lists."""
    n = len(abund_126)
    return pd.DataFrame({
        "Gene":                        ["G"] * n,
        "Master Protein Accessions":   ["P1"] * n,
        "Master Protein Descriptions": ["desc"] * n,
        "Sequence":                    [f"PEP{i}" for i in range(n)],
        "Abundance 126":               abund_126,
        "Abundance 127":               abund_127,
    })


@pytest.mark.parametrize(
    "abund_126,abund_127,cfg_extra,expected",
    [
        # Basic math: 2 asymmetric PSMs → factors hand-computed
        ([100.0, 300.0], [200.0, 400.0], {}, (0.7619, 1.2381)),
        # top-quantile filter: only top 25% used → survivors have equal channels → factor 1.0
        ([10.0, 20.0, 100.0, 200.0], [10.0, 20.0, 100.0, 200.0],
         {"gprot_use_top_quantile": True, "gprot_top_quantile": 0.75},
         (1.0, 1.0)),
        # low-variance filter: only stable PSMs used → survivors have equal channels → factor 1.0
        ([10.0, 20.0, 100.0, 200.0], [500.0, 1000.0, 100.0, 200.0],
         {"gprot_use_low_variance": True, "gprot_low_variance_quantile": 0.5},
         (1.0, 1.0)),
    ],
    ids=["math", "top_quantile", "low_variance"],
)
def test_03_sup_corrections(abund_126, abund_127, cfg_extra, expected):
    """
    Supernatant correction factors computed correctly under three configs:
      math:         no filter, factors derived from raw channel ratios (0.7619 / 1.2381)
      top_quantile: only top-25% abundance PSMs contribute → factor 1.0
      low_variance: only stable (low-CV) PSMs contribute → factor 1.0
    """
    df = _sup_df(abund_126, abund_127)
    cfg = {"abundance_contains": "Abundance", **cfg_extra}
    _, sup = sup_Corrections(df, cfg)
    assert sup["Abundance 126"].iloc[0] == pytest.approx(expected[0], abs=1e-4)
    assert sup["Abundance 127"].iloc[0] == pytest.approx(expected[1], abs=1e-4)


# ═══════════════════════════════════════════════════════════════════════
# Stage 04 — Sum peps → proteins, gprot only (writes 04_sum_peps.csv)
# ═══════════════════════════════════════════════════════════════════════

def test_04_sum_peps(gprot_psm_df):
    """Sums PSMs by (Gene, Accessions) and takes first accession before semicolon."""
    out = sum_peps(gprot_psm_df)

    # Index shape
    assert out.index.names == ["Gene", "Accessions"]
    # Values summed correctly
    assert out.loc[("FOO", "P1"), "Abundance 126"] == 300.0  # 100+200
    assert out.loc[("BAR", "P2"), "Abundance 127"] == 720.0  # 310+410
    # 'P1;P1b' → 'P1' (first before semicolon)
    assert ("FOO", "P1") in out.index
    assert ("FOO", "P1;P1b") not in out.index


# ═══════════════════════════════════════════════════════════════════════
# End-to-end composition (gProt)
#
# Chains sup_Corrections → sum_peps → apply_sup_correction → bridgeCenter_data
# on hand-computable input. Calls the same apply_sup_correction that
# process_gprot_dataset uses in production, so a regression in either the
# helper or the stage boundaries will surface here.
# ═══════════════════════════════════════════════════════════════════════

def test_end_to_end():
    """
    Correction factors = mean(ratio_channel_to_AbundAve) across the 3 grouped PSMs.
    Per-PSM AbundAve = row-mean of {BRG, Ab127, Ab128}:
      PSM1: AbundAve=200 → ratios [1/2, 1, 3/2]
      PSM2: AbundAve=400 → ratios [1/2, 1, 3/2]
      PSM3: AbundAve=200 → ratios [1/2, 3/2, 1]
    Factors: BRG=1/2, Ab127=7/6, Ab128=4/3
    Protein sums:
      GENE_A/P1: BRG=300, Ab127=600, Ab128=900
      GENE_B/P2: BRG=100, Ab127=300, Ab128=200
    After correction (divide by factor):
      GENE_A: BRG=600, Ab127=3600/7, Ab128=675
      GENE_B: BRG=200, Ab127=1800/7, Ab128=150
    After bridge (divide by BRG, drop BRG):
      GENE_A: Ab127=6/7, Ab128=9/8
      GENE_B: Ab127=9/7, Ab128=3/4
    """
    psms = pd.DataFrame({
        "Gene":                        ["GENE_A", "GENE_A", "GENE_B"],
        "Master Protein Accessions":   ["P1", "P1", "P2"],
        "Master Protein Descriptions": ["A_desc", "A_desc", "B_desc"],
        "Sequence":                    ["PEPA", "PEPB", "PEPC"],
        "Abundance BRG": [100.0, 200.0, 100.0],
        "Abundance 127": [200.0, 400.0, 300.0],
        "Abundance 128": [300.0, 600.0, 200.0],
    })
    cfg = {"abundance_contains": "Abundance"}

    _, sup = sup_Corrections(psms, cfg)
    prot = sum_peps(psms)
    corrected = apply_sup_correction(prot, sup)
    bridged = bridgeCenter_data(corrected, brg_regex="BRG")

    a = bridged.loc[("GENE_A", "P1")]
    b = bridged.loc[("GENE_B", "P2")]
    assert "Abundance BRG" not in bridged.columns
    assert a["Abundance 127"] == pytest.approx(6 / 7)
    assert a["Abundance 128"] == pytest.approx(9 / 8)
    assert b["Abundance 127"] == pytest.approx(9 / 7)
    assert b["Abundance 128"] == pytest.approx(3 / 4)


# ═══════════════════════════════════════════════════════════════════════
# Real-data integration (BHA15/16 subsets in tests/fixtures/)
#
# Exercises the full gprot chain against real Proteome Discoverer output.
# Catches breakage that synthetic tests miss: PD schema drift, semicolon-
# separated accessions, unusual gene names, real BRG channel patterns.
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.integration
def test_gprot_end_to_end_real(real_gprot, shipped_cfg, tmp_path):
    """
    Full gprot chain on real PSMs, invoked through the production entrypoint
    `process_gprot_dataset` — catches wiring regressions that a hand-chained
    test would silently paper over.
    """
    import os as _os
    fixture_dir = _os.path.dirname(real_gprot["psms"])

    # Override raw_dir/lib_dir to point at the fixture folder; keep the rest of
    # shipped_cfg (thresholds, meta_cols, abundance_contains, pool_regex).
    cfg = {**shipped_cfg,
           "raw_dir": fixture_dir,
           "lib_dir": fixture_dir,
           "out_dir": str(tmp_path)}
    meta_cols = cfg["meta_cols"]

    # Metadata: single row pointing at the fixture filenames
    meta_df = pd.DataFrame({
        meta_cols["psms"]:     [_os.path.basename(real_gprot["psms"])],
        meta_cols["mgf"]:      [_os.path.basename(real_gprot["mgf"])],
        meta_cols["library"]:  [_os.path.basename(real_gprot["library"])],
        meta_cols["sup_corr"]: ["realtest_corr.csv"],
    }, index=pd.Index(["REALTEST"], name="expType"))

    _reset_run_state()   # avoid pollution from earlier tests using the module globals
    result = process_gprot_dataset("REALTEST", meta_df, cfg)

    # Every stage CSV written
    run_dir = tmp_path / "REALTEST" / "gprot"
    for stage in ["01_psm_filtered.csv", "02_psm_imputed.csv", "03_peps.csv",
                  "04_sum_peps.csv", "06_corr_prot.csv", "06_corr_pept.csv"]:
        assert (run_dir / stage).exists(), f"missing {stage}"

    # Correction factors side-file written to out_dir under the meta-supplied name
    assert (tmp_path / "realtest_corr.csv").exists()

    # Stats block is shaped correctly and reflects real retention
    assert result["experiment"] == "REALTEST"
    assert result["psms_raw"] > 0
    assert 0 < result["psm_retention_pct"] <= 100.0
    assert result["proteins"] > 0


# ═══════════════════════════════════════════════════════════════════════
# Orchestration — process_gprot_dataset on synthetic on-disk inputs
#
# Reads meta.csv → PSMs.txt + MGF + library.csv, runs every stage, writes
# stage-labeled CSVs to run_dir. Catches wiring regressions (stages
# disconnected, output filenames drifted, sup_corr misrouted) that
# individual stage tests can't see.
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def gprot_run_env(tmp_path):
    """
    Build a minimal on-disk environment sufficient for process_gprot_dataset:
      raw_dir/psms.txt      — 3 PSMs across 2 genes, all pass PSM_filter
      raw_dir/spectra.mgf   — 2 scans matching the PSMs' Scan/Charge/mz keys
      lib_dir/library.csv   — headers→names mapping
      out_dir/              — where the stage CSVs will be written
    Returns (cfg, meta_df, out_dir) ready to hand to process_gprot_dataset.
    """
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    lib_dir = tmp_path / "lib"; lib_dir.mkdir()
    out_dir = tmp_path / "out"; out_dir.mkdir()

    # ── Synthetic MGF: 2 scans keyed by (scan, charge, pepmass) ─────────
    (raw_dir / "spectra.mgf").write_text(textwrap.dedent("""\
        BEGIN IONS
        TITLE=scan1
        SCANS=1
        CHARGE=2+
        PEPMASS=500.0
        100.0 50.0
        200.0 500.0
        END IONS
        BEGIN IONS
        TITLE=scan2
        SCANS=2
        CHARGE=3+
        PEPMASS=600.0
        150.0 25.0
        250.0 750.0
        END IONS
        """))

    # ── Library: every column we want kept post-PSM_filter ──────────────
    lib_df = pd.DataFrame({
        "headers": [
            "First Scan", "Charge", "mz in Da",
            "Sequence", "Master Protein Accessions", "Master Protein Descriptions",
            "Abundance BRG", "Abundance 127",
        ],
        "names": [
            "First Scan", "Charge", "mz in Da",
            "Sequence", "Master Protein Accessions", "Master Protein Descriptions",
            "BRG", "Sample_127",
        ],
    })
    lib_df.to_csv(lib_dir / "library.csv", index=False)

    # ── PSMs: 3 rows across 2 genes; all pass filters and merge with MGF ──
    psms = pd.DataFrame({
        "Search Engine Rank":          [1, 1, 1],
        "Delta M in ppm":              [1.0, 1.0, 1.0],
        "Expectation Value":           [0.01, 0.01, 0.01],
        "Modifications":               ["TMT6plex", "TMT6plex", "TMT6plex"],
        "Ions Score":                  [50, 50, 50],
        "Master Protein Descriptions": ["GN=GENE_A desc A", "GN=GENE_A desc A", "GN=GENE_B desc B"],
        "First Scan":                  [1, 1, 2],
        "Charge":                      [2, 2, 3],
        "mz in Da":                    [500.0, 500.0, 600.0],
        "Sequence":                    ["PEPA", "PEPB", "PEPC"],
        "Master Protein Accessions":   ["P1", "P1", "P2"],
        "Abundance BRG":               [100.0, 200.0, 100.0],
        "Abundance 127":               [200.0, 400.0, 300.0],
    })
    psms.to_csv(raw_dir / "psms.txt", sep="\t", index=False)

    # ── Metadata: one experiment row ────────────────────────────────────
    meta_df = pd.DataFrame({
        "expType":  ["TESTEXP"],
        "psms":     ["psms.txt"],
        "mgf":      ["spectra.mgf"],
        "library":  ["library.csv"],
        "sup_corr": ["testexp_corr.csv"],  # non-empty → passes the sup_corr check
    }).set_index("expType")

    cfg = {
        # Filter thresholds — permissive enough that all 3 PSMs survive
        "delta_ppm_range": (-5, 5),
        "expectation_max": 0.05,
        "require_mod_contains": "TMT",
        "ions_score_min": 0,
        "abundance_contains": "Abundance",
        "min_fraction_channels_present": 0.1,
        # Paths
        "raw_dir": str(raw_dir),
        "lib_dir": str(lib_dir),
        "out_dir": str(out_dir),
        "meta_cols": {
            "psms": "psms", "mgf": "mgf", "library": "library", "sup_corr": "sup_corr",
        },
        # sup_Corrections config (no filtering — use all PSMs)
        "gprot_use_top_quantile": False,
        "gprot_use_low_variance": False,
    }
    return cfg, meta_df, out_dir


def test_gprot_orchestration(gprot_run_env):
    """
    End-to-end wiring: process_gprot_dataset reads the synthetic inputs,
    runs every stage, and writes the expected stage-labeled CSVs to run_dir.
    """
    cfg, meta_df, out_dir = gprot_run_env

    result = process_gprot_dataset("TESTEXP", meta_df, cfg)

    # Every stage output present at the expected path
    run_dir = out_dir / "TESTEXP" / "gprot"
    for stage_csv in [
        "01_psm_filtered.csv",
        "02_psm_imputed.csv",
        "03_peps.csv",
        "04_sum_peps.csv",
        "06_corr_prot.csv",
        "06_corr_pept.csv",
    ]:
        assert (run_dir / stage_csv).exists(), f"missing {stage_csv}"

    # sup_corr side-file written to out_dir with the metadata-supplied name
    assert (out_dir / "testexp_corr.csv").exists()

    # Stats block has the right shape and plausible values
    assert result["experiment"] == "TESTEXP"
    assert result["psms_raw"] == 3
    assert result["psms_after_filter"] == 3      # permissive cfg keeps all
    assert result["proteins"] == 2                # GENE_A and GENE_B
    assert result["psm_retention_pct"] == 100.0

    # Correction actually ran — library rename put "Sample_127" in the output
    corr = pd.read_csv(run_dir / "06_corr_prot.csv")
    assert "Sample_127" in corr.columns
