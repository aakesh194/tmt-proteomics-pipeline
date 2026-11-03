"""Phosphoproteomics (phos) specific processing"""

import os
import re
import sys
import numpy as np
import pandas as pd

from pyteomics import mgf
from halo import Halo

from config.utils import load_config
from pipeline.proteomics_core import (
    PSM_filter,
    apply_sup_correction,
    nan_imputation,
    bridge_if_enabled,
    run_post_bridge_outputs,
    qc_heatmap_post_bridge,
    pick_post_bridge_csv,
    to_matrix,
    qc_warn_no_row_change,
    qc_warn_no_value_change,
    infer_project_name,
)


# GLOBAL DATA STORES
corrDict = {}
sumPSM = {}
corrDFs = {}
libsDict = {}

# Cache of parsed FASTA files keyed by absolute path \u2014 SwissProt is ~286 MB and
# we only want to parse it once even when multiple phos experiments run back-to-back.
_FASTA_CACHE = {}


def _reset_run_state():
    """Reset global data stores."""
    corrDict.clear()
    sumPSM.clear()
    corrDFs.clear()
    libsDict.clear()


_FASTA_HEADER_RE = re.compile(r"^>(?:sp|tr)\|([^|]+)\|")
_SITE_RE = re.compile(r"\s*([A-Za-z])\s*(\d+)\s*")


def _read_swissprot_fasta(path: str) -> dict:
    """Parse a UniProt SwissProt/TrEMBL FASTA into {accession: sequence}."""
    key = os.path.abspath(path)
    if key in _FASTA_CACHE:
        return _FASTA_CACHE[key]

    proteins = {}
    acc = None
    chunks = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith(">"):
                if acc is not None:
                    proteins[acc] = "".join(chunks).upper()
                m = _FASTA_HEADER_RE.match(line)
                acc = m.group(1) if m else line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if acc is not None:
            proteins[acc] = "".join(chunks).upper()

    _FASTA_CACHE[key] = proteins
    return proteins


def _motif_from_fasta(accession: str, site_str: str, peptide_seq: str,
                      fasta_dict: dict, flank: int = 7) -> str:
    """15-mer motif (flank=7) centered on `site_str` (e.g. 'S123'), lowercased
    at the phosphosite, underscore-padded past protein termini.

    Falls back to `peptide_seq` when the accession isn't in the FASTA or the
    site can't be resolved (unparseable, out of range, residue mismatch).
    """
    m = _SITE_RE.match(str(site_str) if site_str is not None else "")
    if not m:
        return peptide_seq if isinstance(peptide_seq, str) else ""
    residue = m.group(1).upper()
    pos = int(m.group(2))

    seq = fasta_dict.get(accession) if isinstance(accession, str) else None
    if not seq or pos < 1 or pos > len(seq):
        return peptide_seq if isinstance(peptide_seq, str) else ""
    if seq[pos - 1] != residue:
        return peptide_seq if isinstance(peptide_seq, str) else ""

    center = pos - 1
    parts = []
    for i in range(center - flank, center + flank + 1):
        if 0 <= i < len(seq):
            aa = seq[i]
            parts.append(aa.lower() if i == center else aa)
        else:
            parts.append("_")
    return "".join(parts)


def _resolve_sup_corr_path(sup_corr_value, out_dir: str, out_prefix: str) -> str:
    """
    Resolve the correction-factor CSV path for phos runs.

    Priority:
    1) Use the metadata-provided path under `out_dir` (or absolute path as-is).
    2) If missing, try `outputs/<gprot-exp>/gprot/05_corr_factors.csv` heuristics.
    3) If still missing, search `out_dir` for `*/gprot/05_corr_factors.csv` and pick a best match.
    """
    val = "" if sup_corr_value is None else str(sup_corr_value).strip()
    if val.lower() in {"", "nan", "none"}:
        val = ""

    candidates = []
    if val:
        candidates.append(val if os.path.isabs(val) else os.path.join(out_dir, val))

    # If metadata provides something like "<BASE>_sup_Corrections.csv", prefer "<BASE>/gprot/05_corr_factors.csv".
    if val.endswith("_sup_Corrections.csv"):
        base = os.path.basename(val).replace("_sup_Corrections.csv", "")
        candidates.append(os.path.join(out_dir, base, "gprot", "05_corr_factors.csv"))

    # Common expected location when gProt already ran for the same sample prefix.
    # Your folder structure is typically:
    #   outputs/<base>_dda/gprot/05_corr_factors.csv
    #   outputs/<base>_pst_dda/gphos/...
    prefix = str(out_prefix)
    prefix_l = prefix.lower()
    transforms = {prefix}

    # Strip common phospho markers to find the gProt base folder
    for marker in ("_pst", "_psty", "_py", "_phos", "_gphos"):
        if marker in prefix_l:
            transforms.add(prefix_l.replace(marker, ""))

    # Also try collapsing double-underscores caused by removals.
    # Sort so candidate order is deterministic across runs/platforms.
    normalized = set()
    for t in transforms:
        tt = str(t).replace("__", "_").strip("_")
        normalized.add(tt)
        normalized.add(tt.lower())
        normalized.add(tt.upper())
    transforms = sorted(normalized)

    for t in transforms:
        candidates.append(os.path.join(out_dir, t, "gprot", "05_corr_factors.csv"))

    # If metadata is just the filename, try finding it under a gprot folder for this experiment base.
    if val and os.path.basename(val) == "05_corr_factors.csv":
        for t in transforms:
            candidates.append(os.path.join(out_dir, t, "gprot", val))

    for p in candidates:
        if p and os.path.exists(p):
            return p

    # Last resort: search outputs for any gprot correction factors.
    matches = []
    if out_dir and os.path.isdir(out_dir):
        for root, dirs, files in os.walk(out_dir):
            # light pruning: don't descend into very large common folders
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
            if os.path.basename(root) == "gprot" and "05_corr_factors.csv" in files:
                matches.append(os.path.join(root, "05_corr_factors.csv"))

    if len(matches) == 1:
        return matches[0]

    if matches:
        # Prefer best token overlap with the current experiment name
        tokens = [t for t in str(out_prefix).split("_") if t]
        def score(path: str) -> int:
            p = path.lower()
            return sum(1 for t in tokens if t.lower() in p)
        matches.sort(key=score, reverse=True)
        return matches[0]

    tried = [p for p in candidates if p]
    raise FileNotFoundError(
        "Could not locate sup correction factors file.\n"
        f"Tried: {tried}\n"
        f"Also searched under out_dir='{out_dir}' for '*/gprot/05_corr_factors.csv' and found none.\n"
        "Fix by either (a) running gProt first, and/or (b) setting the `sup_corr` column in your phos metadata CSV "
        "to the correct path (relative to out_dir) of the gProt `05_corr_factors.csv`."
    )



# PHOS-SPECIFIC FUNCTIONS

def phos_filter(df, exp_type, cfg):
    """
    Filter phosphopeptides based on quality criteria.
    """
    #print(f"    Applying phospho-specific filters...")
    #print(f"      Starting PSMs: {len(df)}")
    start = len(df)
    
    # Drop unused columns
    phosfilter = df.drop([
        "Delta M in ppm", "Ions Score", "Delta mz in Da", "Master Protein Descriptions"
    ], axis=1, errors='ignore')
    
    # Filter by abundance
    if 'AbundAve' not in phosfilter.columns:
        abund = phosfilter.loc[:, phosfilter.columns.str.contains(cfg["abundance_contains"])]
        phosfilter = phosfilter.assign(AbundAve=abund.mean(axis=1))
    
    phosfilter = phosfilter[phosfilter['AbundAve'] >= cfg["phos_min_abundance"]]
    phosfilter = phosfilter.drop('AbundAve', axis=1)
    #print(f"      After abundance filter (>={cfg['phos_min_abundance']}): {len(phosfilter)}")
    
    # Keep only PTMs of interest (configurable)
    mod_filter = cfg.get("phos_mod_filter", "Phos")
    phosfilter = phosfilter[phosfilter['Modifications'].str.contains(mod_filter, na=False)]
    #print(f"      After phospho filter: {len(phosfilter)}")
    
    # Filter for pY if specified
    enrichment = cfg.get("phos_enrichment_filter")

    if enrichment == "pY" and "pY" in exp_type:
        phosfilter = phosfilter[
            phosfilter['PhosphoRS Best Site Probabilities']
            .str.contains(r'\bY\d', regex=True, na=False)
        ]
        #print(f"      After pY filter: {len(phosfilter)}")
    elif enrichment == "pSQTQ" and exp_type.endswith("pSQTQ"):
        phosfilter = phosfilter[phosfilter["Sequence"].str.contains(r"(SQ|TQ)", regex=True, na=False)]
    
    # PhosphoRS probability filter
    a = pd.DataFrame(
        phosfilter['PhosphoRS Best Site Probabilities']
        .str.findall(r"\b[-+]?(?:\d*\.\d+|\d+)")
        .astype('string').str.strip()
        .str.replace(r"\[|'|\]", "", regex=True)
        .str.split(',', expand=True)
    )
    a = a.replace(r'^\s*$', np.nan, regex=True).astype(float)
    a = a.assign(low=a.min(axis=1))
    phosfilter['ptmRSprobability'] = a['low']
    phosfilter = phosfilter[phosfilter['ptmRSprobability'] > cfg["phos_ptmrs_threshold"]]
    phosfilter = phosfilter.drop('ptmRSprobability', axis=1)
    #print(f"      After ptmRS filter (>{cfg['phos_ptmrs_threshold']}): {len(phosfilter)}")
    
    end = len(phosfilter)

    return phosfilter, start, end

def sum_psms_phos(df, pepts, fasta_dict, exp_type, cfg, out_dir=""):
    """
    Sum PSMs to phosphosite level and add site/motif annotations.

    1. Parses PhosphoRS localization
    2. Maps peptide sites to protein sites
    3. Generates motifs from the SwissProt FASTA (±FLANK around each site,
       lowercase at the phosphosite, underscore padding past termini);
       falls back to the peptide sequence when the accession or site can't
       be resolved in the FASTA
    4. Sums to final phosphosite level
    """
    #print(f"    Summarizing phosphosites with motif annotation...")
    
    # Parse PhosphoRS localization
    pRS = pd.DataFrame(
        df['PhosphoRS Best Site Probabilities']
        .str.findall(r"(\b[S,T,Y]\w+)")
        .astype('string').str.strip()
        .str.replace(r"\[|'|\]", "", regex=True)
    )
    
    # Concatenate with dataframe
    phosSum = pd.concat([
        df.drop([
            'PhosphoRS Best Site Probabilities', 'Modifications', 'Expectation Value',
            'First Scan', 'Charge', 'mz in Da', 'intensity_min', 'intensity_max',
            'countTMTchannels'
        ], axis=1, errors='ignore'),
        pRS
    ], axis=1)
    
    # Sum peptides by sequence and site
    phosSum = phosSum.groupby([
        'Annotated Sequence', 'Sequence', 'Gene', 'Master Protein Accessions', 'PhosphoRS Best Site Probabilities'
    ]).agg('sum')
    phosSum = phosSum.reset_index()
    phosSum['Annotated Sequence2'] = phosSum['Annotated Sequence'].astype('string').str.upper()
    
    # Get modification sites from peptide file
    pSeq = pepts[['Annotated Sequence', 'Modifications in Master Proteins', 'Positions in Master Proteins']]
    
    # Extract site assignments and positions
    pSeq2 = pd.DataFrame(
        pSeq['Modifications in Master Proteins']
        .str.findall(r"(?<=\[)([^]]+)(?=\])")
        .astype('string').str.strip()
        .str.replace(r"\[|'|\]", "", regex=True)
    )
    start = pd.DataFrame(
        pSeq['Positions in Master Proteins']
        .str.findall(r"(?<=\[)([^]]+)(?=\])")
        .astype('string').str.strip()
        .str.replace(r"\[|'|\]", "", regex=True)
    )
    start2 = start['Positions in Master Proteins'].str.split(',', expand=True)
    start2 = start2[0].str.split('-', expand=True)
    pSeq3 = pSeq.assign(Sites=pSeq2, peptStart=start2[0], peptEnd=start2[1])
    pSeq3['Annotated Sequence'] = pSeq3['Annotated Sequence'].astype('string')
    pSites = pSeq3.set_index('Annotated Sequence').to_dict()['Sites']
    pStart = pSeq3.set_index('Annotated Sequence').to_dict()['peptStart']
    pEnd = pSeq3.set_index('Annotated Sequence').to_dict()['peptEnd']
    phosSum['Sites'] = phosSum['Annotated Sequence2'].map(pSites)
    phosSum['peptStart'] = phosSum['Annotated Sequence2'].map(pStart)
    phosSum['peptEnd'] = phosSum['Annotated Sequence2'].map(pEnd)
    
    # Extract PhosphoRS localized sites (positions within peptide)
    pRSx = pd.DataFrame(
        phosSum['PhosphoRS Best Site Probabilities']
        .str.findall(r'(\d+)')
        .astype('string').str.strip()
        .str.replace(r"\[|'| |\]", "", regex=True)
    )
    phosSum['intSites'] = pRSx['PhosphoRS Best Site Probabilities'].str.split(',').astype('string').str.strip().str.replace(r"\[|'| |\]", "", regex=True)
    
    # Create unique identifier and dictionaries for site mapping
    phosSum['AA+ptmRS'] = phosSum['Sequence'] + '_' + phosSum['PhosphoRS Best Site Probabilities']
    pRSdict = phosSum.set_index('AA+ptmRS').to_dict()['intSites']
    peptStartdict = phosSum.set_index('AA+ptmRS').to_dict()['peptStart']
    
    # Convert intSites string to list
    for k, v in pRSdict.items():
        if isinstance(v, str):
            pRSdict[k] = v.split(',')
        else:
            pRSdict[k] = []
    
    # Calculate true protein-level sites
    trueSite = {}
    for i, j in peptStartdict.items():
        for k, v in pRSdict.items():
            if i == k and j is not None and not pd.isna(j):
                try:
                    trueSite[k] = '; '.join([str(int(float(j)) - 1 + int(vi)) for vi in v if vi])
                except (ValueError, TypeError):
                    trueSite[k] = ''
    
    phosSum['trueSite'] = phosSum['AA+ptmRS'].map(trueSite)
    
    # Extract accessions
    uniprotID = phosSum['Master Protein Accessions'].str.split(';', expand=True)
    phosSum['Accessions'] = uniprotID[0]
    phosSum = phosSum.drop(['intSites', 'Annotated Sequence', 'AA+ptmRS', 'Annotated Sequence2', 'Sites', 'Master Protein Accessions'], axis=1)
    phosSum = phosSum.groupby(['Sequence', 'Gene', 'Accessions', 'peptStart', 'peptEnd', 'trueSite', 'PhosphoRS Best Site Probabilities']).agg('sum')
    
    # Rename and prepare for motif mapping
    phosSum1 = phosSum.reset_index().rename(columns={'Sequence': 'Peptide Sequence', 'PhosphoRS Best Site Probabilities': 'ptmRS'})
    
    # Map true sites
    psiteA = phosSum1.to_dict()['trueSite']
    for k, v in psiteA.items():
        if isinstance(v, str):
            psiteA[k] = v.split('; ')
        else:
            psiteA[k] = []
    
	#\\d replaced from \d
    #phosSum1['ptmRS'] = phosSum1['ptmRS'].replace('\\d+', '', regex=True)
    phosSum1['ptmRS'] = phosSum1['ptmRS'].replace(r"\d+", "", regex=True)

    ptmRSdict = phosSum1.to_dict()['ptmRS']
    for k, v in ptmRSdict.items():
        if isinstance(v, str):
            ptmRSdict[k] = v.split(', ')
        else:
            ptmRSdict[k] = []
    
    # Create new site labels (e.g., S123, T456)
    newSite = []
    for i in range(len(psiteA)):
        try:
            if len(psiteA[i]) > 0 and len(ptmRSdict[i]) > 0:
                newSite.append(', '.join([ptmRSdict[i][j] + psiteA[i][j] for j in range(len(psiteA[i]))]))
            else:
                newSite.append('')
        except (IndexError, TypeError):
            newSite.append('')
    
    phosSum1 = phosSum1.drop('trueSite', axis=1)
    phosSum1['Site'] = newSite
    
    # Handle missed cleavages - normalize peptide sequences
    for i in phosSum1['Peptide Sequence']:
        if isinstance(i, str):
            if i.endswith('KK') or i.endswith('RK') or i.endswith('KR'):
                phosSum1['Peptide Sequence'] = phosSum1['Peptide Sequence'].replace(i, i[:-1])
    
    phosSum1['ID'] = phosSum1['Accessions'] + '_' + phosSum1['Peptide Sequence']
    phosSum1 = phosSum1.groupby(['Peptide Sequence', 'Gene', 'Accessions', 'Site', 'ptmRS', 'ID']).agg('sum', numeric_only=True).reset_index()

    # Generate motifs directly from the SwissProt FASTA, one per site in each row.
    flank = int(cfg.get("phos_motif_flank", 7))

    def build_motifs(row):
        sites = str(row['Site']).split(', ')
        return ', '.join(
            _motif_from_fasta(row['Accessions'], s, row['Peptide Sequence'], fasta_dict, flank)
            for s in sites
        )

    phosSum4 = phosSum1.copy()
    phosSum4['Motif'] = phosSum4.apply(build_motifs, axis=1)
    phosSum4 = phosSum4.drop(['Peptide Sequence', 'ptmRS', 'ID'], axis=1, errors='ignore')
    phosSum4 = phosSum4.groupby(['Gene', 'Accessions', 'Site', 'Motif']).agg('sum', numeric_only=True)

    motif_file = os.path.join(out_dir, f"{exp_type}_sum_motif.csv") if out_dir else f"{exp_type}_sum_motif.csv"
    phosSum4.to_csv(motif_file)

    phosSum5 = (
        phosSum4.reset_index()
        .drop('Motif', axis=1)
        .set_index(['Gene', 'Accessions', 'Site'])
        .groupby(['Gene', 'Accessions', 'Site'])
        .agg('sum', numeric_only=True)
    )

    return phosSum5



# DATASET PROCESSING
def process_phos_dataset(index, dataDF, cfg):
    """Process a single phos dataset."""
    #print(f"  Processing: {index}")
    
    raw_dir = cfg.get("raw_dir", "")
    out_dir = cfg.get("out_dir", "")
    lib_dir = cfg.get("lib_dir", "")
    out_label = cfg.get("phos_output_folder", cfg.get("phos_output_prefix", "phos"))
    file_prefix = cfg.get("phos_output_file_prefix", "")
    file_prefix_ = f"{file_prefix}_" if file_prefix else ""
    
    out_prefix = str(index)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    run_dir = os.path.join(out_dir, out_prefix, out_label)
    os.makedirs(run_dir, exist_ok=True)
    
    # Build file paths
    psms_path = os.path.join(raw_dir, dataDF.loc[index, cfg["meta_cols"]["psms"]])
    mgf_path = os.path.join(raw_dir, dataDF.loc[index, cfg["meta_cols"]["mgf"]])
    lib_path = os.path.join(lib_dir, dataDF.loc[index, cfg["meta_cols"]["library"]])
    pepts_path = os.path.join(raw_dir, dataDF.loc[index, cfg["meta_cols"]["pepts"]])
    sup_corr_value = dataDF.loc[index, cfg["meta_cols"]["sup_corr"]]

    fasta_path = cfg.get("swissprot_fasta")
    if not fasta_path:
        raise KeyError("config is missing 'swissprot_fasta' (path to SwissProt FASTA for motif generation)")
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(
            f"SwissProt FASTA not found at '{fasta_path}'. "
            "Set `swissprot_fasta` in config.yaml to a valid path."
        )

    with Halo(spinner="dots", color="cyan") as sp:
        # Load input files
        sp.text = f"  {out_prefix} — reading files"
        psms = pd.read_csv(psms_path, sep="\t")
        psm_start = len(psms)
        mgf_dict = mgf.read(mgf_path)
        libs = pd.read_csv(lib_path).set_index("headers").to_dict()["names"]
        pepts = pd.read_csv(pepts_path, sep="\t")
        sup_corr_path = _resolve_sup_corr_path(sup_corr_value, out_dir, out_prefix)
        corrSum = pd.read_csv(sup_corr_path)
        fasta_dict = _read_swissprot_fasta(fasta_path)
        
        libsDict[index] = libs
        
        # Filter and impute PSMs
        #print(f"    Filtering + imputation...")
        sp.text = f"  {out_prefix} — filtering..."
        psm_filtered = PSM_filter(psms, libs, cfg)
        psm_filtered.to_csv(os.path.join(run_dir, f"{file_prefix_}01_psm_filtered.csv"), index=False)
        PSMdf = nan_imputation(psm_filtered, mgf_dict, cfg)
        PSMdf.to_csv(os.path.join(run_dir, f"{file_prefix_}02_psm_imputed.csv"), index=False)
        qc_warn_no_row_change("PSM filtering+imputation", psm_start, len(PSMdf), context=out_prefix)
        #print(len(PSMdf))
        
        # Phospho-specific filtering
        PSMdf, psm_start, psm_end = phos_filter(PSMdf, index, cfg)
        PSMdf.to_csv(os.path.join(run_dir, f"{file_prefix_}03_phos_filtered.csv"), index=False)
        qc_warn_no_row_change("phospho filtering", psm_start, psm_end, context=out_prefix)
        
        # Summarize to phosphosite level with motifs
        sp.text = f"  {out_prefix} — motif mapping..."
        sumPSMdf = sum_psms_phos(PSMdf, pepts, fasta_dict, out_prefix, cfg, out_dir=run_dir)
        sumPSMdf.to_csv(os.path.join(run_dir, f"{file_prefix_}04_sum_psms_phos.csv"))
        qc_warn_no_row_change("phosphosite summarization", psm_end, len(sumPSMdf), context=out_prefix)
        sumPSM[index] = sumPSMdf
        
        # Apply supernatant corrections
        sp.text = f"  {out_prefix} — corrections"
        # Save a copy of the correction factors used for this run (named after metadata `sup_corr`)
        corr_copy_name = os.path.basename(str(sup_corr_path)) if sup_corr_path else "corr_factors.csv"
        corrSum.to_csv(os.path.join(run_dir, corr_copy_name), index=False)
        corrPhos = apply_sup_correction(sumPSMdf, corrSum)

        qc_warn_no_value_change("correction factor (phospho)", sumPSMdf, corrPhos, context=out_prefix, cols=corrSum.columns)
        
        # Rename columns using library mapping (exclude id columns)
        id_cols = {"Gene", "Accessions", "Site"}
        rename_map = {k: v for k, v in libs.items() if k in corrPhos.columns and k not in id_cols}
        if rename_map:
            corrPhos = corrPhos.rename(columns=rename_map)
        
        corrDFs[index] = corrPhos
        
        # Save output
        sp.text = f"  {out_prefix} — saving"
        corr_phos_path = os.path.join(run_dir, f"{file_prefix_}06_corr_phos.csv")
        corrPhos.to_csv(corr_phos_path)

        # print(f"      {index} — {psm_start} PSMs → {psm_end} after filters → {len(sumPSMdf)} sites")
        # print(f"Saved: {os.path.basename(corr_phos_path)}")
        #print(f"    ✓ Done: {index}")

        sp.succeed(f"{out_prefix} — {psm_start} PSMs → {psm_end} after filters → {len(sumPSMdf)} sites | saved: {corr_phos_path}")

        return {
            "experiment": out_prefix,
            "psms_raw": psm_start,
            "psms_after_filter": psm_end,
            "psm_retention_pct": round(psm_end / psm_start * 100, 1) if psm_start else 0,
            "phosphosites": len(sumPSMdf),
            "tmt_channels": len([c for c in corrPhos.columns if c not in {"Gene", "Accessions", "Site"}]),
        }


# PIPELINE EXECUTION
def run_phos_pipeline(cfg=None, exp_types=None):
    """Run the phos pipeline for all or selected experiment types."""
    if cfg is None:
        cfg = load_config()
    _reset_run_state()

    print(f"\n[phos]")
    out_label = cfg.get("phos_output_folder", cfg.get("phos_output_prefix", "phos"))
    file_prefix = cfg.get("phos_output_file_prefix", "")
    file_prefix_ = f"{file_prefix}_" if file_prefix else ""
    do_post_bridge = cfg.get("phos_post_bridge", cfg.get("phos_do_bridge", False))
    
    dataDF = pd.read_csv(cfg["meta_phos_csv"], sep=",").set_index(cfg["meta_index"])
    indices = dataDF.index if exp_types is None else exp_types
    
    #print(f"  Found {len(indices)} dataset(s) to process")
    
    stats = {}
    for index in indices:
        try:
            result = process_phos_dataset(index, dataDF, cfg)
            stats[index] = result
        except Exception:
            print(f"\n[error] phos processing failed for expType={index!r}", file=sys.stderr)
            raise

    out_dir = cfg["out_dir"]
    project = infer_project_name(list(indices))
    phos_prefix = f"{out_label}_{project}"

    for index in indices:
        try:
            out_prefix = str(index)
            run_dir = os.path.join(out_dir, out_prefix, out_label)
            phos_corr_path = os.path.join(run_dir, f"{file_prefix_}06_corr_phos.csv")

            with Halo(spinner="dots", color="cyan", text="generating combined QC heatmap...") as sp:
                phos_corr = pd.read_csv(phos_corr_path)

                # Convert to a feature x sample matrix.
                id_cols = ["Gene", "Accessions", "Site"]
                sample_cols = [c for c in phos_corr.columns if c not in id_cols]
                phos_mat = phos_corr.set_index(id_cols)[sample_cols]

                if cfg.get("bridge_enabled", True):
                    phos_brg = bridge_if_enabled(phos_mat, cfg)
                    phos_brg.to_csv(os.path.join(run_dir, f"{file_prefix_}07_post_pool_bridge.csv"))
                    qc_heatmap_post_bridge(
                        phos_brg.dropna(),
                        out_png=os.path.join(run_dir, "07_QC_heatmap_post_pool_bridge.png"),
                        top_n=cfg["qc_top_n"],
                        zscore=cfg["qc_zscore"],
                        title=f"{out_prefix} QC heatmap (post pool-bridge)",
                    )
                else:
                    qc_heatmap_post_bridge(
                        phos_mat.dropna(),
                        out_png=os.path.join(run_dir, "07_QC_heatmap_no_bridging.png"),
                        top_n=cfg["qc_top_n"],
                        zscore=cfg["qc_zscore"],
                        title=f"{out_prefix} QC heatmap (no bridging)",
                    )
                sp.succeed(f"{out_prefix} — QC outputs + heatmap ")
        except Exception:
            print(f"\n[error] phos QC failed for expType={index!r}", file=sys.stderr)
            raise

    # Combined post-bridge outputs (After bridging)
    if do_post_bridge:
        post_cfg = dict(cfg)
        post_cfg["post_bridge_dir"] = os.path.join(out_dir, "after-bridging", out_label)
        with Halo(spinner="dots", color="cyan", text="[phos] post-bridge outputs...") as sp:
            run_post_bridge_outputs(corrDFs, cfg=post_cfg, pipeline="phos", prefix=phos_prefix)
            sp.succeed(f"post-bridge outputs | saved: {post_cfg['post_bridge_dir']}")

        # Combined heatmap from post-bridge outputs (universal)
        post_cfg["post_bridge_prefix"] = phos_prefix
        post_csv = pick_post_bridge_csv(post_cfg)
        with Halo(spinner="dots", color="cyan", text="[phos] generating combined heatmap...") as sp:
            if post_csv:
                combined_df = pd.read_csv(post_csv)
                combined_mat = to_matrix(combined_df, ["Gene", "Accessions", "Site"])
                out_png = os.path.join(
                    post_cfg.get("post_bridge_dir", os.path.join(out_dir, "after-bridging", out_label)),
                    f"{phos_prefix}_QC_heatmap_bridged.png",
                )
                qc_heatmap_post_bridge(
                    combined_mat.dropna(),
                    out_png=out_png,
                    top_n=cfg["qc_top_n"],
                    zscore=cfg["qc_zscore"],
                    title=f"{os.path.splitext(os.path.basename(post_csv))[0]} QC heatmap (combined)",
                )
                sp.succeed(f"combined heatmap | saved: {out_png}")
            else:
                sp.warn("skipping combined heatmap — no post-bridge CSV found")
    
    print(f"\n[phos] done — all written to:", cfg["out_dir"])
    return stats



# MAIN

def main():
    """Main entry point for testing."""
    run_phos_pipeline()


if __name__ == "__main__":
    main()
