"""Shared pytest fixtures for the proteomics pipeline test suite."""

import os
import textwrap
from pathlib import Path

import pandas as pd
import pytest


FIXTURES = Path(__file__).parent / "fixtures"

# When set (any truthy value), missing fixture files fail loudly instead of skipping.
# Intended for CI so silent skips don't mask a fixture-availability regression.
_REQUIRE_FIXTURES = os.environ.get("PROTEOMICS_REQUIRE_FIXTURES", "").lower() in {"1", "true", "yes"}


def _missing_fixture(msg):
    """Skip (dev) or fail (CI) with a consistent, actionable message."""
    full = (f"{msg}\n"
            "Set PROTEOMICS_REQUIRE_FIXTURES=0 to skip locally, "
            "or commit the missing fixture files under tests/fixtures/.")
    if _REQUIRE_FIXTURES:
        pytest.fail(full)
    pytest.skip(full)


def pytest_configure(config):
    """Register custom markers + silence upstream lib warnings we can't fix."""
    config.addinivalue_line(
        "markers",
        "integration: exercises real fixture files under tests/fixtures/ "
        "(slow, environment-dependent; skip with -m 'not integration')",
    )
    config.addinivalue_line(
        "markers",
        "local_only: requires the developer's local data/ directory "
        "(not committed to the repo). CI excludes these with -m 'not local_only'.",
    )
    # halo spinner lib uses deprecated Thread.setDaemon() on Python 3.10+
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning:halo")


# ── Config fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def psm_filter_cfg():
    return {
        "delta_ppm_range": (-5, 5),
        "expectation_max": 0.05,
        "require_mod_contains": "TMT",
        "ions_score_min": 15,
    }


@pytest.fixture
def phos_filter_cfg():
    return {
        "abundance_contains": "Abundance",
        "phos_min_abundance": 100,
        "phos_mod_filter": "Phos",
        "phos_enrichment_filter": "pY",
        "phos_ptmrs_threshold": 50,
    }


@pytest.fixture
def imputation_cfg():
    return {
        "abundance_contains": "Abundance",
        "min_fraction_channels_present": 0.33,
    }


# ── Synthetic data fixtures ─────────────────────────────────────────────

@pytest.fixture
def mini_mgf(tmp_path):
    """
    Minimal synthetic MGF: 2 spectra designed to merge with a PSM frame
    on (First Scan, Charge, mz in Da).
      Scan 1: charge=2, pepmass=500.0, intensity_min=50.0
      Scan 2: charge=3, pepmass=600.0, intensity_min=25.0
    """
    body = textwrap.dedent("""\
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
        """)
    path = tmp_path / "mini.mgf"
    path.write_text(body)
    return str(path)


@pytest.fixture
def gprot_psm_df():
    """4 PSMs across 2 genes — used by sum_peps and sup_Corrections tests."""
    return pd.DataFrame({
        "Gene":                        ["FOO", "FOO", "BAR", "BAR"],
        "Master Protein Accessions":   ["P1;P1b", "P1;P1b", "P2", "P2"],
        "Master Protein Descriptions": ["FOO_desc"] * 2 + ["BAR_desc"] * 2,
        "Sequence":                    ["PEPA", "PEPB", "PEPC", "PEPD"],
        "Abundance 126": [100.0, 200.0, 300.0, 400.0],
        "Abundance 127": [110.0, 210.0, 310.0, 410.0],
    })


# ── Real-data fixture paths (committed under tests/fixtures/) ───────────

def _paths(folder: str, stem: str, is_phos: bool):
    d = FIXTURES / folder
    files = {
        "psms":    d / f"{stem}_PSMs.txt",
        "mgf":     d / f"{stem}.mgf",
        "library": d / f"Library_{folder.upper()}.csv",
    }
    if is_phos:
        files["pepts"] = d / f"{stem}_PeptideGroups.txt"
        files["mods"]  = d / f"{stem}_ModificationSites.txt"
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        _missing_fixture(f"fixture files missing: {missing}")
    return {k: str(v) for k, v in files.items()}


@pytest.fixture(params=["bha15", "bha16"])
def real_gprot(request):
    return _paths(request.param, "gprot", is_phos=False)


@pytest.fixture(params=["bha15", "bha16"])
def real_phos(request):
    return _paths(request.param, "gphos", is_phos=True)


@pytest.fixture(scope="module")
def shipped_cfg():
    """Load the actual config.yaml used in production. Used by real-data tests."""
    from config.utils import load_config
    return load_config()
