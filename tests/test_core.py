"""
Tests for pipeline/proteomics_core.py — the shared functions that both
gprot and phos pipelines call:

  PSM_filter, nan_imputation, apply_sup_correction,
  bridgeCenter_data, qc_heatmap_post_bridge,
  process_runs, meanCenter_data, concat_dfs, infer_project_name, to_matrix.

Real-data smoke tests live at the bottom (parametrized × bha15, bha16).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyteomics import mgf

REPO_ROOT = Path(__file__).resolve().parent.parent

from pipeline.proteomics_core import (
    PSM_filter,
    apply_sup_correction,
    bridgeCenter_data,
    concat_dfs,
    infer_corr_paths,
    infer_project_name,
    meanCenter_data,
    nan_imputation,
    pick_post_bridge_csv,
    process_runs,
    qc_heatmap_post_bridge,
    run_post_bridge_outputs,
    to_matrix,
)


# ═══════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════

def test_mean_center():
    """Mean-centering: each row divided by its mean; z-scored rows have mean 0."""
    df = pd.DataFrame({"A": [2.0, 4.0], "B": [4.0, 8.0], "C": [6.0, 12.0]})
    mc, zs = meanCenter_data(df)
    assert mc.mean(axis=1).round(6).tolist() == [1.0, 1.0]
    assert np.allclose(zs.mean(axis=1), 0.0)


def test_concat_dfs():
    """Concatenates run DataFrames; columns sorted naturally (S1, S2, S3, S10); _n drops NaN rows."""
    d1 = pd.DataFrame({"S1": [1.0, 2.0], "S2": [3.0, np.nan]}, index=["A", "B"])
    d2 = pd.DataFrame({"S10": [5.0, 6.0], "S3": [7.0, 8.0]}, index=["A", "B"])
    u, n = concat_dfs({"run1": d1, "run2": d2})
    assert list(u.columns) == ["S1", "S2", "S3", "S10"]  # natural sort
    assert len(n) == 1                                    # _n drops rows with any NaN


@pytest.mark.parametrize("inputs,expected", [
    (["BHA15_DDA_sup", "BHA16_DDA_sup"],  "BHA"),      # common alpha prefix
    (["BHA15_DDA", "XYZ22_DDA"],          "MIXED"),    # no common prefix
    (["20260423_GST_EI_TMT_Set1"],        "GST"),      # date-prefixed name
    ([],                                  "PROJECT"),  # empty → default
])
def test_project_name(inputs, expected):
    """Infers project label from experiment names — common prefix, MIXED, date-prefix, or default."""
    assert infer_project_name(inputs) == expected


def test_to_matrix():
    """Strips ID columns (Gene, Accessions) into the index, leaving sample columns."""
    df = pd.DataFrame({
        "Gene":       ["A", "B"],
        "Accessions": ["P1", "P2"],
        "sample1":    [1.0, 2.0],
        "sample2":    [3.0, 4.0],
    })
    mat = to_matrix(df, ["Gene", "Accessions"])
    assert list(mat.columns) == ["sample1", "sample2"]
    assert mat.index.names == ["Gene", "Accessions"]


# ═══════════════════════════════════════════════════════════════════════
# Stage 01 — PSM filtering (writes 01_psm_filtered.csv)
# ═══════════════════════════════════════════════════════════════════════

def _psm_row(rank=1, ppm=1.0, exp=0.01, mod="TMT", ions=20, desc="GN=FOO desc"):
    """Build a single-row PSM DataFrame for filter tests."""
    return pd.DataFrame({
        "Search Engine Rank":          [rank],
        "Delta M in ppm":              [ppm],
        "Expectation Value":           [exp],
        "Modifications":               [mod],
        "Ions Score":                  [ions],
        "Master Protein Descriptions": [desc],
    })


@pytest.mark.parametrize(
    "kwargs,should_survive,expected_gene",
    [
        (dict(),                                    True,  "FOO"),   # all pass
        (dict(rank=2),                              False, None),    # rank != 1
        (dict(ppm=10.0),                            False, None),    # ppm out of range
        (dict(exp=0.5),                             False, None),    # expectation too high
        (dict(mod="no_label"),                      False, None),    # missing TMT modification
        (dict(ions=10),                             False, None),    # ions score below threshold
        (dict(desc="GN=TP53 tumor antigen"),        True,  "TP53"),  # gene extraction: TP53
        (dict(desc="GN=BRCA1 breast cancer"),       True,  "BRCA1"), # gene extraction: BRCA1
    ],
    ids=["all_pass", "rank", "ppm", "expectation", "mod", "ions_score", "gene_TP53", "gene_BRCA1"],
)
def test_01_psm_filter(kwargs, should_survive, expected_gene, psm_filter_cfg):
    """One row per case — each fails a different filter or exercises gene parsing."""
    df = _psm_row(**kwargs)
    libs = {"Master Protein Descriptions": "desc"}
    out = PSM_filter(df, libs, psm_filter_cfg)
    if should_survive:
        assert len(out) == 1
        assert out["Gene"].iloc[0] == expected_gene
    else:
        assert len(out) == 0


# ═══════════════════════════════════════════════════════════════════════
# Stage 02 — NaN imputation (writes 02_psm_imputed.csv)
# ═══════════════════════════════════════════════════════════════════════

def test_02_nan_impute_fills_from_mgf(mini_mgf, imputation_cfg):
    """NaN abundance channels are filled with the matching MGF spectrum's min intensity."""
    psms = pd.DataFrame({
        "Search Engine Rank": [1, 1],
        "First Scan":         [1, 2],
        "Charge":             [2, 3],
        "mz in Da":           [500.0, 600.0],
        "Abundance 126":      [500.0, np.nan],
        "Abundance 127":      [np.nan, 100.0],
        "Abundance 128":      [300.0, 200.0],
    })
    out = nan_imputation(psms, mgf.read(mini_mgf), imputation_cfg)
    row1 = out[out["First Scan"] == 1].iloc[0]
    row2 = out[out["First Scan"] == 2].iloc[0]
    assert row1["Abundance 127"] == 50.0    # imputed from scan 1 min intensity
    assert row2["Abundance 126"] == 25.0    # imputed from scan 2 min intensity


def test_02_nan_impute_drops_sparse_rows(mini_mgf):
    """Rows with fewer than min_fraction_channels_present non-NaN channels are dropped."""
    strict_cfg = {"abundance_contains": "Abundance", "min_fraction_channels_present": 0.67}
    psms = pd.DataFrame({
        "Search Engine Rank": [1, 1],
        "First Scan":         [1, 2],
        "Charge":             [2, 3],
        "mz in Da":           [500.0, 600.0],
        "Abundance 126":      [500.0, np.nan],
        "Abundance 127":      [500.0, np.nan],
        "Abundance 128":      [500.0, 200.0],  # row 2: only 1/3 non-NaN → dropped
    })
    out = nan_imputation(psms, mgf.read(mini_mgf), strict_cfg)
    assert (out["First Scan"] == 2).sum() == 0
    assert (out["First Scan"] == 1).sum() == 1  # row 1 (3/3) survives


# ═══════════════════════════════════════════════════════════════════════
# Stage 06 — Sup correction (writes 06_corr_prot.csv / 06_corr_pept.csv)
# ═══════════════════════════════════════════════════════════════════════

def test_06_apply_sup_correction():
    """Known channels divided by correction factors; unknown columns pass through unchanged."""
    prot = pd.DataFrame(
        {"Abundance 126": [300.0, 700.0], "Abundance 127": [600.0, 900.0], "Something Else": [42.0, 99.0]},
        index=pd.MultiIndex.from_tuples([("A", "P1"), ("B", "P2")], names=["Gene", "Accessions"]),
    )
    sup = pd.DataFrame({"Abundance 126": [0.5], "Abundance 127": [2.0]})
    out = apply_sup_correction(prot, sup)

    # Math: division applied to known channels
    assert out.loc[("A", "P1"), "Abundance 126"] == pytest.approx(600.0)  # 300/0.5
    assert out.loc[("A", "P1"), "Abundance 127"] == pytest.approx(300.0)  # 600/2.0
    # Passthrough: unknown col unchanged
    assert out.loc[("A", "P1"), "Something Else"] == 42.0
    assert out.loc[("B", "P2"), "Something Else"] == 99.0


def test_06_zero_factor_raises():
    """Correction factor of 0 would produce inf → should raise loudly instead of propagating."""
    prot = pd.DataFrame({"Abundance 126": [100.0]})
    sup = pd.DataFrame({"Abundance 126": [0.0]})
    with pytest.raises(ValueError, match="correction factor"):
        apply_sup_correction(prot, sup)


def test_06_nan_factor_raises():
    """Correction factor of NaN would produce NaN → should raise loudly instead of propagating."""
    prot = pd.DataFrame({"Abundance 126": [100.0]})
    sup = pd.DataFrame({"Abundance 126": [float("nan")]})
    with pytest.raises(ValueError, match="correction factor"):
        apply_sup_correction(prot, sup)


# ═══════════════════════════════════════════════════════════════════════
# Stage 07 — Pool/bridge normalization + QC heatmap (writes 07_post_pool_bridge.csv)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "cols,regex,expected",
    [
        # single bridge column: samples divided by bridge value
        (
            {"sample1": [10.0, 20.0], "sample2": [20.0, 40.0], "BRG1": [10.0, 20.0]},
            "BRG",
            {"sample1": [1.0, 1.0], "sample2": [2.0, 2.0]},
        ),
        # multiple bridge columns: divisor is row-wise mean of the bridges
        (
            {"sample1": [100.0, 200.0], "BRG_a": [10.0, 20.0], "BRG_b": [30.0, 40.0]},
            "BRG",
            {"sample1": [5.0, 6.666666666666667]},   # 100/mean(10,30)=5.0, 200/mean(20,40)=6.667
        ),
        # no matching bridge column: return copy unchanged (early-return code path)
        (
            {"sample1": [1.0, 2.0]},
            "POOL",
            {"sample1": [1.0, 2.0]},
        ),
    ],
    ids=["single_bridge", "multiple_bridges", "no_match"],
)
def test_07_bridge_center(cols, regex, expected):
    """
    Bridge normalization across three scenarios:
      single_bridge:    sample columns divided by the one bridge column, bridge dropped
      multiple_bridges: divisor is the row-wise MEAN of all bridge columns
      no_match:         no columns match the regex → return copy unchanged (early return)
    """
    df = pd.DataFrame(cols)
    out = bridgeCenter_data(df, brg_regex=regex)
    for col, vals in expected.items():
        assert out[col].tolist() == pytest.approx(vals)
    # No column matching the regex should remain (either dropped, or never present)
    assert not any(regex in c for c in out.columns)


def test_07_qc_heatmap_writes_png(tmp_path):
    """QC heatmap produces a valid, non-trivial PNG on disk."""
    from PIL import Image

    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.random((50, 6)), columns=[f"s{i}" for i in range(6)])
    out_png = tmp_path / "heatmap.png"
    qc_heatmap_post_bridge(df, out_png=str(out_png), top_n=30, zscore=True, title="test")

    assert out_png.exists()
    with Image.open(out_png) as img:
        img.verify()                            # decodes header + IDAT chunks
    with Image.open(out_png) as img:
        assert img.format == "PNG"
        assert img.size[0] > 100 and img.size[1] > 100  # not a placeholder


def test_07_qc_heatmap_empty_raises():
    """QC heatmap raises a clean ValueError on empty input instead of silently crashing."""
    with pytest.raises(ValueError):
        qc_heatmap_post_bridge(pd.DataFrame(), out_png="unused.png")


# ═══════════════════════════════════════════════════════════════════════
# Multi-run bridging (process_runs)
# ═══════════════════════════════════════════════════════════════════════

def test_process_runs():
    """
    Two runs, one shared protein, distinct bridge column names per run.
    brgA hand-computed: run1 [S1=200/BRG1=100, S2=300/BRG1=100] → [2.0, 3.0]
                        run2 [S3=220/BRG2=110, S4=330/BRG2=110] → [2.0, 3.0]
    """
    idx = pd.MultiIndex.from_tuples([("A", "P1")], names=["Gene", "Accessions"])
    run1 = pd.DataFrame({"BRG1": [100.0], "S1": [200.0], "S2": [300.0]}, index=idx)
    run2 = pd.DataFrame({"BRG2": [110.0], "S3": [220.0], "S4": [330.0]}, index=idx)

    _, concat = process_runs({"run1": run1, "run2": run2}, ["run1", "run2"], {"pool_regex": "BRG"})

    expected_keys = {"runCorr_DF_u", "runCorr_DF_n", "run_corr_mc_u", "run_corr_mc_n",
                     "run_corr_zs_u", "run_corr_zs_n", "raw_mc_DF_u", "raw_mc_DF_n",
                     "raw_zscore_DF_u", "raw_zscore_DF_n", "brgA_u", "brgA_n"}
    assert expected_keys.issubset(concat.keys())

    brgA = concat["brgA_n"]
    assert list(brgA.columns) == ["S1", "S2", "S3", "S4"]
    row = brgA.iloc[0]
    assert [row["S1"], row["S2"], row["S3"], row["S4"]] == [2.0, 3.0, 2.0, 3.0]


# ═══════════════════════════════════════════════════════════════════════
# Post-bridge writer + path helpers
# ═══════════════════════════════════════════════════════════════════════

def test_run_post_bridge_outputs(tmp_path):
    """Writes one CSV per concatDict entry, prefixed with the caller-supplied prefix."""
    meta = tmp_path / "meta.csv"
    meta.write_text("expType\nR1\nR2\n")

    idx = pd.MultiIndex.from_tuples([("A", "P1")], names=["Gene", "Accessions"])
    r1 = pd.DataFrame({"BRG1": [100.0], "S1": [200.0]}, index=idx)
    r2 = pd.DataFrame({"BRG2": [50.0], "S2": [100.0]}, index=idx)
    corrDFs = {"R1": r1, "R2": r2}

    out = tmp_path / "post"
    cfg = {
        "meta_prot_csv": str(meta),
        "meta_index": "expType",
        "post_bridge_dir": str(out),
        "pool_regex": "BRG",
    }
    result = run_post_bridge_outputs(corrDFs, cfg, prefix="testrun", pipeline="prot")

    # Returns concat dict from process_runs
    assert isinstance(result, dict)
    assert "brgA_n" in result
    # Wrote one CSV per key with the caller's prefix
    written = sorted(p.name for p in out.glob("testrun_*.csv"))
    assert "testrun_brgA_n.csv" in written
    assert "testrun_brgA_u.csv" in written


def test_post_bridge_empty_raises(tmp_path):
    """Empty metadata → ValueError instead of silently writing nothing."""
    meta = tmp_path / "meta.csv"
    meta.write_text("expType\n")
    cfg = {
        "meta_prot_csv": str(meta),
        "meta_index": "expType",
        "post_bridge_dir": str(tmp_path / "post"),
        "pool_regex": "BRG",
    }
    with pytest.raises(ValueError):
        run_post_bridge_outputs({}, cfg, prefix="x", pipeline="prot")


def test_infer_corr_paths(tmp_path):
    """Builds one 06_corr_prot.csv path per metadata row, under <out_dir>/<expType>/gprot/."""
    meta = tmp_path / "meta.csv"
    meta.write_text("expType,psms\nBHA15,x.txt\nBHA16,y.txt\n")
    paths = infer_corr_paths(str(meta), "expType", str(tmp_path))
    assert len(paths) == 2
    assert all(p.endswith("06_corr_prot.csv") for p in paths)
    assert "BHA15" in paths[0] and "BHA16" in paths[1]


def test_infer_corr_paths_empty_raises(tmp_path):
    """Empty metadata → ValueError, not a silent empty list."""
    meta = tmp_path / "meta.csv"
    meta.write_text("expType,psms\n")
    with pytest.raises(ValueError):
        infer_corr_paths(str(meta), "expType", str(tmp_path))


@pytest.mark.parametrize(
    "files_present,expected_suffix",
    [
        ([],                                          None),
        (["gProt_brgA_u.csv"],                        "gProt_brgA_u.csv"),
        (["gProt_brgA_n.csv", "gProt_brgA_u.csv"],    "gProt_brgA_n.csv"),
        (["gProt_brgA_n.csv"],                        "gProt_brgA_n.csv"),
    ],
    ids=["none", "only_u_falls_back", "both_prefers_n", "only_n"],
)
def test_pick_post_bridge_csv(tmp_path, files_present, expected_suffix):
    """Prefers brgA_n.csv, falls back to brgA_u.csv, returns None if neither exists."""
    d = tmp_path / "after-bridging"
    d.mkdir()
    for fname in files_present:
        (d / fname).write_text("")

    # out_dir is required even when post_bridge_dir is set — pick_post_bridge_csv's
    # .get() default expression evaluates cfg["out_dir"] unconditionally.
    cfg = {"out_dir": str(tmp_path), "post_bridge_dir": str(d), "post_bridge_prefix": "gProt"}
    result = pick_post_bridge_csv(cfg)

    if expected_suffix is None:
        assert result is None
    else:
        assert result.endswith(expected_suffix)


# ═══════════════════════════════════════════════════════════════════════
# run.py CLI entrypoint smoke tests
#
# Catches import-time regressions (moved modules, broken argparse wiring,
# config.yaml drift that breaks load_config at module scope) that stage-
# level tests can't see, because they import pipeline modules directly.
# ═══════════════════════════════════════════════════════════════════════

def _invoke_runpy(*args, timeout=30):
    """Invoke run.py from REPO_ROOT so config.yaml resolves relatively."""
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_runpy_help_exits_zero():
    """`python run.py --help` succeeds → imports + argparse + config load all wire up."""
    result = _invoke_runpy("--help")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "usage:" in result.stdout.lower()


def test_runpy_version_exits_zero():
    """`--version` prints a version string and exits 0."""
    result = _invoke_runpy("--version")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert result.stdout.strip() != ""


def test_runpy_help_advertises_both_pipeline_flags():
    """--gprot-only and --phos-only must appear in help text — they are the public API."""
    result = _invoke_runpy("--help")
    assert "--gprot-only" in result.stdout
    assert "--phos-only" in result.stdout


def test_runpy_unknown_flag_exits_nonzero():
    """Unknown flags are rejected by argparse with a non-zero exit."""
    result = _invoke_runpy("--definitely-not-a-real-flag")
    assert result.returncode != 0


# ═══════════════════════════════════════════════════════════════════════
# Real-data integration (BHA15/16 subsets in tests/fixtures/)
#
# Runs core functions against real Proteome Discoverer output.
# Parametrized × bha15, bha16.
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.integration
def test_01_psm_filter_real(real_gprot, shipped_cfg):
    """PSM_filter on real data: keeps >0 rows AND extracts real gene symbols."""
    psms = pd.read_csv(real_gprot["psms"], sep="\t")
    libs = pd.read_csv(real_gprot["library"]).set_index("headers")["names"].to_dict()
    filtered = PSM_filter(psms, libs, shipped_cfg)

    # Filter doesn't annihilate the input
    retention = len(filtered) / len(psms)
    assert 0.0 < retention <= 1.0, f"retention {retention:.2%}"

    # Gene extraction works on real Master Protein Descriptions
    if len(filtered) > 0:
        assert filtered["Gene"].notna().mean() > 0.3


@pytest.mark.integration
def test_02_nan_imputation_real(real_gprot, shipped_cfg):
    """nan_imputation on real MGF + real PSMs leaves no NaN in Abundance columns."""
    psms = pd.read_csv(real_gprot["psms"], sep="\t")
    out = nan_imputation(psms, mgf.read(real_gprot["mgf"]), shipped_cfg)
    assert len(out) > 0
    abund_cols = out.columns[out.columns.str.contains("Abundance")]
    assert not out[abund_cols].isna().any().any()
