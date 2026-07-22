"""
Phos pipeline tests — behavior unique to the phos path.

Shared functions (PSM_filter, nan_imputation, apply_sup_correction, bridge,
QC heatmap, process_runs) are tested in test_core.py.
"""

import os
import textwrap

import pandas as pd
import pytest
from pyteomics import mgf

from pipeline.proteomics_core import PSM_filter, nan_imputation
from pipeline.phos_processing import (
    _read_phosphositeplus_csv,
    _resolve_sup_corr_path,
    phos_filter,
    process_phos_dataset,
    sum_psms_phos,
)


def _phos_df():
    """
    Row 0 — pY, high abundance, high ptmRS  → survives pY filter
    Row 1 — pS only                         → dropped when exp_type has pY
    Row 2 — pY, low abundance               → dropped by abundance
    Row 3 — no Phos modification            → dropped by mod filter
    """
    return pd.DataFrame({
        "Abundance 126": [500.0, 500.0, 10.0, 500.0],
        "Abundance 127": [500.0, 500.0, 10.0, 500.0],
        "Modifications": ["Phospho", "Phospho", "Phospho", "TMT6plex only"],
        "PhosphoRS Best Site Probabilities": [
            "Y247(99.9)", "S155(99.9)", "Y100(99.9)", "Y200(99.9)",
        ],
        "Sequence": ["PEPTIDEY", "PEPTIDES", "PEPTIDEY", "PEPTIDEY"],
    })


# ═══════════════════════════════════════════════════════════════════════
# Stage 03 — Phos filter, phos only
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "exp_type,ptmrs_override,expected_sequences",
    [
        # pY_and_abundance: exp_type has 'pY' → drops non-Y AND low-abundance rows
        # → only row 0 (pY, high-abundance) survives
        ("BHA15_pY_DDA",  None, ["PEPTIDEY"]),
        # pST: no 'pY' marker → Y-specific filter skipped, but abundance + mod filters still apply
        # → rows 0 and 1 survive (row 2 too low abund, row 3 no Phospho mod)
        ("BHA15_pST_DDA", None, ["PEPTIDEY", "PEPTIDES"]),
        # ptmrs_strict: impossibly high localization threshold → nothing survives
        ("BHA15_pY_DDA",  999,  []),
    ],
    ids=["pY_and_abundance", "pST", "ptmrs_strict"],
)
def test_03_phos_filter(phos_filter_cfg, exp_type, ptmrs_override, expected_sequences):
    """Verifies both the count AND identity of surviving rows for each filter scenario."""
    cfg = {**phos_filter_cfg}
    if ptmrs_override is not None:
        cfg["phos_ptmrs_threshold"] = ptmrs_override
    filtered, _, end = phos_filter(_phos_df(), exp_type=exp_type, cfg=cfg)
    assert end == len(expected_sequences)
    assert sorted(filtered["Sequence"].tolist()) == sorted(expected_sequences)


# ═══════════════════════════════════════════════════════════════════════
# Sup-correction path resolution
#
# _resolve_sup_corr_path picks the phos-run's correction-factors CSV by
# trying, in order: metadata-provided path, then heuristic transforms
# (strip phos markers), then a walk of out_dir for */gprot/05_corr_factors.csv.
# ═══════════════════════════════════════════════════════════════════════

def test_resolve_sup_corr_direct_hit(tmp_path):
    """Relative path from metadata is joined with out_dir and returned when it exists."""
    (tmp_path / "corr.csv").write_text("")
    assert _resolve_sup_corr_path("corr.csv", str(tmp_path), "any_prefix") == str(tmp_path / "corr.csv")


def test_resolve_sup_corr_absolute(tmp_path):
    """Absolute path in metadata is used as-is (not joined with out_dir)."""
    abs_path = tmp_path / "abs.csv"
    abs_path.write_text("")
    assert _resolve_sup_corr_path(str(abs_path), str(tmp_path / "unrelated"), "any") == str(abs_path)


def test_resolve_sup_corr_not_found_raises(tmp_path):
    """Nothing on disk → FileNotFoundError with an actionable message."""
    with pytest.raises(FileNotFoundError, match="Could not locate sup correction"):
        _resolve_sup_corr_path("missing.csv", str(tmp_path), "anything")


@pytest.mark.parametrize(
    "candidates,prefix,expected_dir,forbidden_dir",
    [
        # Single candidate: walk finds it via marker-stripped prefix ("bha15_pst" → "bha15")
        (["bha15"],             "bha15_pst", "bha15",   None),
        # Multiple candidates: token-overlap scorer picks the one sharing a token with prefix
        (["unrelated", "xyz"],  "xyz_pst",   "xyz",     "unrelated"),
    ],
    ids=["single_hit", "multi_hit"],
)
def test_resolve_sup_corr_walk_fallback(tmp_path, candidates, prefix, expected_dir, forbidden_dir):
    """
    Walk fallback (metadata blank/NaN): scans out_dir for */gprot/05_corr_factors.csv.
    Marker-stripping normalizes the prefix; token-overlap picks best match when multiple hit.
    Asserting on lowercased path tolerates macOS case-insensitive FS canonicalization.
    """
    for base in candidates:
        (tmp_path / base / "gprot").mkdir(parents=True)
        (tmp_path / base / "gprot" / "05_corr_factors.csv").write_text("")

    resolved = _resolve_sup_corr_path("", str(tmp_path), prefix).lower()
    assert resolved.endswith(f"{expected_dir}/gprot/05_corr_factors.csv")
    if forbidden_dir is not None:
        assert forbidden_dir not in resolved


# ═══════════════════════════════════════════════════════════════════════
# Real-data integration (BHA15/16 subsets in tests/fixtures/)
#
# Only real-data coverage of sum_psms_phos — the biggest, most complex
# function in the pipeline. Synthetic fixture is impractical.
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.integration
def test_04_sum_psms_phos_real(real_phos, shipped_cfg, tmp_path):
    """
    Full stage 04 phos pipeline on real subset:
      raw PSMs → PSM_filter → nan_imputation → phos_filter → sum_psms_phos
    Verifies output is indexed by (Gene, Accessions, Site) and motif CSV is written.
    Falls back to tests/fixtures/phos_site_stub.csv if the shipped PSP file is
    absent (it is gitignored under data/).
    """
    from pathlib import Path as _Path
    psms  = pd.read_csv(real_phos["psms"],  sep="\t")
    libs  = pd.read_csv(real_phos["library"]).set_index("headers")["names"].to_dict()
    pepts = pd.read_csv(real_phos["pepts"], sep="\t")
    mods  = pd.read_csv(real_phos["mods"],  sep="\t")
    phos_site_path = shipped_cfg["phos_site_csv"]
    if not os.path.exists(phos_site_path):
        phos_site_path = str(_Path(__file__).parent / "fixtures" / "phos_site_stub.csv")
    phos_site = _read_phosphositeplus_csv(phos_site_path, shipped_cfg)

    filtered = PSM_filter(psms, libs, shipped_cfg)
    imputed = nan_imputation(filtered, mgf.read(real_phos["mgf"]), shipped_cfg)
    phos_filtered, _, end = phos_filter(imputed, exp_type="test_pST", cfg=shipped_cfg)

    if end == 0:
        pytest.skip("no phospho-passing PSMs in this subset")

    out = sum_psms_phos(
        phos_filtered, pepts, mods, phos_site,
        exp_type="test_pST", cfg=shipped_cfg, out_dir=str(tmp_path),
    )
    assert list(out.index.names) == ["Gene", "Accessions", "Site"]
    assert len(out) > 0
    assert os.path.exists(tmp_path / "test_pST_sum_motif.csv")


# ═══════════════════════════════════════════════════════════════════════
# Orchestration — process_phos_dataset on synthetic on-disk inputs
#
# Mirrors gprot's orchestration test. Requires: PSMs, MGF, library,
# PeptideGroups, ModificationSites, PhosphoSitePlus, and a pre-written
# gprot 05_corr_factors.csv (phos consumes gprot's output).
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def phos_run_env(tmp_path):
    """Minimal on-disk environment for process_phos_dataset. Returns (cfg, meta_df, out_dir)."""
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    lib_dir = tmp_path / "lib"; lib_dir.mkdir()
    out_dir = tmp_path / "out"; out_dir.mkdir()

    # ── MGF: 2 scans matching PSMs' Scan/Charge/mz keys ─────────────────
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

    # ── Library ─────────────────────────────────────────────────────────
    # NB: PSM_filter drops any column not in this map, so include every
    # column phos_filter / sum_psms_phos will later look up.
    pd.DataFrame({
        "headers": [
            "First Scan", "Charge", "mz in Da",
            "Annotated Sequence", "Sequence", "Modifications",
            "Master Protein Accessions", "Master Protein Descriptions",
            "PhosphoRS Best Site Probabilities",
            "Abundance 126", "Abundance 127",
        ],
        "names": [
            "First Scan", "Charge", "mz in Da",
            "Annotated Sequence", "Sequence", "Modifications",
            "Master Protein Accessions", "Master Protein Descriptions",
            "PhosphoRS Best Site Probabilities",
            "Sample_126", "Sample_127",
        ],
    }).to_csv(lib_dir / "library.csv", index=False)

    # ── PSMs: 2 rows, 1 gene, both phosphorylated ──────────────────────
    pd.DataFrame({
        "Search Engine Rank":          [1, 1],
        "Delta M in ppm":              [1.0, 1.0],
        "Expectation Value":           [0.01, 0.01],
        "Modifications":               ["Phospho [S1]; TMT6plex", "Phospho [Y2]; TMT6plex"],
        "Ions Score":                  [50, 50],
        "Master Protein Descriptions": ["GN=GENE_A desc", "GN=GENE_A desc"],
        "First Scan":                  [1, 2],
        "Charge":                      [2, 3],
        "mz in Da":                    [500.0, 600.0],
        "Annotated Sequence":          ["[K].SPEPTIDE.[R]", "[K].YPEPTIDE.[R]"],
        "Sequence":                    ["SPEPTIDE", "YPEPTIDE"],
        "Master Protein Accessions":   ["P1", "P1"],
        "PhosphoRS Best Site Probabilities": ["S1(99.9)", "Y2(99.9)"],
        "Abundance 126":               [500.0, 500.0],
        "Abundance 127":               [500.0, 500.0],
    }).to_csv(raw_dir / "psms.txt", sep="\t", index=False)

    # ── PeptideGroups: modifications-in-master-proteins format ─────────
    pd.DataFrame({
        "Annotated Sequence": ["[K].SPEPTIDE.[R]", "[K].YPEPTIDE.[R]"],
        "Modifications in Master Proteins": ["P1 [S1]", "P1 [Y2]"],
        "Positions in Master Proteins":     ["P1 [1-8]", "P1 [1-8]"],
    }).to_csv(raw_dir / "pepts.txt", sep="\t", index=False)

    # ── ModificationSites: one high-confidence site per PSM ────────────
    pd.DataFrame({
        "Confidence":          ["High", "High"],
        "Target Amino Acid":   ["S", "Y"],
        "Position in Peptide": [1, 2],
        "Peptide Sequence":    ["SPEPTIDE", "YPEPTIDE"],
        "Protein Accession":   ["P1", "P1"],
        "Position":            [1, 2],
        "Motif":               ["_SPEPTIDE_", "_YPEPTIDE_"],
    }).to_csv(raw_dir / "mods.txt", sep="\t", index=False)

    # ── PhosphoSitePlus: motif lookup ──────────────────────────────────
    pd.DataFrame({
        "Accession": ["P1", "P1"],
        "Site":      ["S1", "Y2"],
        "Motif":     ["_SPEPTIDE_", "_YPEPTIDE_"],
    }).to_csv(raw_dir / "phos_site.csv", index=False)

    # ── Pre-existing gprot correction factors (phos consumes this) ─────
    pd.DataFrame({"Abundance 126": [1.0], "Abundance 127": [1.0]}).to_csv(
        out_dir / "testphos_corr.csv", index=False
    )

    # ── Metadata: one experiment row ───────────────────────────────────
    meta_df = pd.DataFrame({
        "expType":  ["TESTPHOS"],
        "psms":     ["psms.txt"],
        "mgf":      ["spectra.mgf"],
        "library":  ["library.csv"],
        "pepts":    ["pepts.txt"],
        "mods":     ["mods.txt"],
        "sup_corr": ["testphos_corr.csv"],
    }).set_index("expType")

    cfg = {
        # PSM_filter thresholds — permissive
        "delta_ppm_range": (-5, 5),
        "expectation_max": 0.05,
        "require_mod_contains": "TMT",
        "ions_score_min": 0,
        "abundance_contains": "Abundance",
        "min_fraction_channels_present": 0.1,
        # phos filter — permissive
        "phos_min_abundance": 0,
        "phos_mod_filter": "Phospho",
        "phos_ptmrs_threshold": 0,
        "phos_enrichment_filter": "",
        "phos_filter_pY": False,
        # phos-site lookup columns (match the CSV we wrote)
        "phos_site_csv": str(raw_dir / "phos_site.csv"),
        "phos_site_accession_col": "Accession",
        "phos_site_site_col": "Site",
        "phos_site_motif_col": "Motif",
        # output
        "phos_output_prefix": "phos",
        # paths
        "raw_dir": str(raw_dir),
        "lib_dir": str(lib_dir),
        "out_dir": str(out_dir),
        "meta_cols": {
            "psms": "psms", "mgf": "mgf", "library": "library",
            "pepts": "pepts", "mods": "mods", "sup_corr": "sup_corr",
        },
    }
    return cfg, meta_df, out_dir


def test_phos_orchestration(phos_run_env):
    """End-to-end wiring: process_phos_dataset writes expected stage-labeled CSVs."""
    cfg, meta_df, out_dir = phos_run_env

    result = process_phos_dataset("TESTPHOS", meta_df, cfg)

    run_dir = out_dir / "TESTPHOS" / "phos"
    for stage_csv in [
        "01_psm_filtered.csv",
        "02_psm_imputed.csv",
        "03_phos_filtered.csv",
        "04_sum_psms_phos.csv",
        "06_corr_phos.csv",
    ]:
        assert (run_dir / stage_csv).exists(), f"missing {stage_csv}"

    # motif CSV written by sum_psms_phos
    assert (run_dir / "TESTPHOS_sum_motif.csv").exists()

    # Stats block has phos-specific shape
    assert result["experiment"] == "TESTPHOS"
    assert result["psms_raw"] == 2
    assert "phosphosites" in result
