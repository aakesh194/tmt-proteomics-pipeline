"""
Config tests — two distinct concerns:
  1. load_config() behavior on synthetic YAML (pure logic, no shipped file).
  2. Shipped config.yaml validation — every key present, correct type,
     reasonable range, referenced files exist on disk.
"""

import os
import textwrap
from pathlib import Path

import pytest
import yaml

from config.utils import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════
# 1. load_config() behavior
# ═══════════════════════════════════════════════════════════════════════

def _write_yaml(path, body):
    path.write_text(textwrap.dedent(body).lstrip())


def test_load_yaml(tmp_path):
    """load_config reads values from a YAML file exactly as written."""
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, """
        delta_ppm_max: 5
        out_dir: outputs
        expectation_max: 0.05
    """)
    cfg = load_config(yaml_path=str(cfg_path))
    assert cfg["delta_ppm_max"] == 5
    assert cfg["out_dir"] == "outputs"
    assert cfg["expectation_max"] == 0.05


def test_ppm_range_derived(tmp_path):
    """delta_ppm_range tuple is auto-computed from delta_ppm_max."""
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, "delta_ppm_max: 8\nout_dir: outputs\n")
    cfg = load_config(yaml_path=str(cfg_path))
    assert cfg["delta_ppm_range"] == (-8, 8)


def test_post_bridge_paths_derived(tmp_path):
    """post_bridge_dir and post_bridge_prefix are auto-derived from out_dir."""
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, "delta_ppm_max: 5\nout_dir: my_outputs\n")
    cfg = load_config(yaml_path=str(cfg_path))
    assert "my_outputs" in cfg["post_bridge_dir"]
    assert cfg["post_bridge_prefix"] == "gProt"


# ═══════════════════════════════════════════════════════════════════════
# 2. Shipped config.yaml — presence, types, ranges, file existence
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def shipped_cfg_raw():
    """Raw YAML dict from config.yaml — no load_config() derivation applied."""
    with open(REPO_ROOT / "config.yaml") as fh:
        return yaml.safe_load(fh)


STRING_KEYS = [
    "meta_prot_csv", "meta_phos_csv", "phos_site_csv",
    "phos_site_accession_col", "phos_site_site_col", "phos_site_motif_col",
    "meta_index", "raw_dir", "lib_dir", "out_dir",
    "require_mod_contains", "abundance_contains", "pool_regex",
    "phos_mod_filter", "phos_output_folder", "phos_output_file_prefix",
]

NUMERIC_KEYS = [
    "delta_ppm_max", "expectation_max", "ions_score_min",
    "gprot_top_quantile", "gprot_low_variance_quantile",
    "min_fraction_channels_present",
    "phos_min_abundance", "phos_ptmrs_threshold", "qc_top_n",
]

BOOL_KEYS = [
    "gprot_use_top_quantile", "gprot_use_low_variance",
    "gprot_post_bridge", "phos_post_bridge", "qc_zscore",
    "bridge_enabled",
]

META_COLS_SUBKEYS = ["psms", "mgf", "library", "pepts", "mods", "sup_corr"]


@pytest.mark.parametrize("key", STRING_KEYS)
def test_string_key(shipped_cfg_raw, key):
    """Every documented string setting is present and typed as str."""
    assert key in shipped_cfg_raw, f"missing key: {key}"
    assert isinstance(shipped_cfg_raw[key], str), f"{key} must be str"


@pytest.mark.parametrize("key", NUMERIC_KEYS)
def test_numeric_key(shipped_cfg_raw, key):
    """Every documented numeric setting is present and typed as int/float."""
    assert key in shipped_cfg_raw, f"missing key: {key}"
    assert isinstance(shipped_cfg_raw[key], (int, float)), f"{key} must be numeric"


@pytest.mark.parametrize("key", BOOL_KEYS)
def test_bool_key(shipped_cfg_raw, key):
    """Every documented true/false setting is present and typed as bool."""
    assert key in shipped_cfg_raw, f"missing key: {key}"
    assert isinstance(shipped_cfg_raw[key], bool), f"{key} must be bool"


def test_meta_cols_is_dict(shipped_cfg_raw):
    """meta_cols top-level value is a dict."""
    assert isinstance(shipped_cfg_raw["meta_cols"], dict)


@pytest.mark.parametrize("sub", META_COLS_SUBKEYS)
def test_meta_cols_subkey(shipped_cfg_raw, sub):
    """Each required meta_cols sub-key is present and mapped to a string."""
    assert sub in shipped_cfg_raw["meta_cols"]
    assert isinstance(shipped_cfg_raw["meta_cols"][sub], str)


# ── Value ranges ────────────────────────────────────────────────────────

def test_delta_ppm_max_positive(shipped_cfg_raw):
    """Mass-error window must be positive (negative would be nonsense)."""
    assert shipped_cfg_raw["delta_ppm_max"] > 0

def test_expectation_is_prob(shipped_cfg_raw):
    """expectation_max is a p-value → must be in (0, 1)."""
    assert 0 < shipped_cfg_raw["expectation_max"] < 1

def test_ions_score_non_negative(shipped_cfg_raw):
    """Ions score threshold cannot be negative."""
    assert shipped_cfg_raw["ions_score_min"] >= 0

def test_gprot_top_quantile_range(shipped_cfg_raw):
    """Top-abundance quantile is a fraction in (0, 1)."""
    assert 0 < shipped_cfg_raw["gprot_top_quantile"] < 1

def test_gprot_low_variance_quantile_range(shipped_cfg_raw):
    """Low-variance quantile is a fraction in (0, 1)."""
    assert 0 < shipped_cfg_raw["gprot_low_variance_quantile"] < 1

def test_min_fraction_channels_range(shipped_cfg_raw):
    """Minimum-channel fraction is in (0, 1]."""
    assert 0 < shipped_cfg_raw["min_fraction_channels_present"] <= 1

def test_phos_min_abundance_positive(shipped_cfg_raw):
    """Phospho abundance threshold cannot be zero or negative."""
    assert shipped_cfg_raw["phos_min_abundance"] > 0

def test_phos_ptmrs_threshold_range(shipped_cfg_raw):
    """PhosphoRS localization threshold is a percent probability in [0, 100]."""
    assert 0 <= shipped_cfg_raw["phos_ptmrs_threshold"] <= 100

def test_qc_top_n_positive(shipped_cfg_raw):
    """Heatmap top-N features must be positive."""
    assert shipped_cfg_raw["qc_top_n"] > 0

def test_phos_enrichment_filter_valid(shipped_cfg_raw):
    """Enrichment filter must be blank, 'pY', or 'pSQTQ' — no typos allowed."""
    assert shipped_cfg_raw.get("phos_enrichment_filter", "") in {"", "pY", "pSQTQ"}


# ── Referenced files actually exist ─────────────────────────────────────
# Marked @integration because these check the developer's local filesystem
# (raw_dir / lib_dir / metadata CSVs) — they fail on any machine that
# doesn't have the shipped data laid out under those paths.

@pytest.mark.integration
@pytest.mark.local_only
def test_meta_prot_csv_exists(shipped_cfg_raw):
    """gProt metadata CSV path in config points at a real file on disk."""
    assert os.path.exists(shipped_cfg_raw["meta_prot_csv"])

@pytest.mark.integration
@pytest.mark.local_only
def test_meta_phos_csv_exists(shipped_cfg_raw):
    """Phos metadata CSV path in config points at a real file on disk."""
    assert os.path.exists(shipped_cfg_raw["meta_phos_csv"])

@pytest.mark.integration
@pytest.mark.local_only
def test_raw_dir_exists(shipped_cfg_raw):
    """raw_dir path is a real directory (holds PSMs/MGF/PeptideGroups/etc.)."""
    assert os.path.isdir(shipped_cfg_raw["raw_dir"])

@pytest.mark.integration
@pytest.mark.local_only
def test_lib_dir_exists(shipped_cfg_raw):
    """lib_dir path is a real directory (holds library CSVs)."""
    assert os.path.isdir(shipped_cfg_raw["lib_dir"])
