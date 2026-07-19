"""
phos_processing.py - Phosphoproteomics (phos) specific processing

Author: Aakesh Yoganathan
Lab: Tamir Lab, UNC Chapel Hill
"""

import os
import numpy as np
import pandas as pd

from pyteomics import mgf
from halo import Halo

from config.utils import load_config
from pipeline.proteomics_core import (
    PSM_filter,
    nan_imputation,
    bridgeCenter_data,
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


def _reset_run_state():
    """Reset global data stores."""
    corrDict.clear()
    sumPSM.clear()
    corrDFs.clear()
    libsDict.clear()

def _read_phosphositeplus_csv(path: str, cfg: dict) -> pd.DataFrame:
    """
    Read a PhosphoSitePlus motif CSV while tolerating occasional junk/metadata lines
    *before* the real header row (common in some PSP downloads/exports).
    """
    accession_col = cfg.get("phos_site_accession_col", "Accession")
    site_col = cfg.get("phos_site_site_col", "Site")
    motif_col = cfg.get("phos_site_motif_col", "Motif")
    required = {accession_col, site_col, motif_col}

    header_row = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i > 200:
                    break
                # Strip BOM and whitespace
                raw = line.lstrip("\ufeff").strip()
                if not raw:
                    continue
                cols = [c.strip().strip('"') for c in raw.split(",")]
                if required.issubset(set(cols)):
                    header_row = i
                    break
    except OSError:
        header_row = 0

    return pd.read_csv(path, sep=",", skiprows=header_row)

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

    # Also try collapsing double-underscores caused by removals
    normalized = set()
    for t in transforms:
        tt = str(t).replace("__", "_").strip("_")
        normalized.add(tt)
        normalized.add(tt.lower())
        normalized.add(tt.upper())
    transforms = normalized

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

def sum_psms_phos(df, pepts, mods, phos_site, exp_type, cfg, out_dir=""):
    """
    Sum PSMs to phosphosite level and add site/motif annotations.
    
    1. Parses PhosphoRS localization
    2. Maps peptide sites to protein sites
    3. Assigns motifs from modification files
    4. Assigns motifs from PhosphoSitePlus database
    5. Sums to final phosphosite level
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
    
    # Prepare modifications table for motif extraction
    mods1 = mods[mods['Confidence'] == 'High']
    if cfg.get("phos_filter_pY", True) and 'pY' in exp_type:
        mods1 = mods1[mods1['Target Amino Acid'] == 'Y']
    mods1 = mods1[['Target Amino Acid', 'Position in Peptide', 'Peptide Sequence', 'Protein Accession', 'Position', 'Motif']]
    mods1['Site in peptide'] = mods1['Target Amino Acid'].astype('string') + mods1['Position in Peptide'].astype('string')
    mods1['Site in Protein'] = mods1['Target Amino Acid'].astype('string') + mods1['Position'].astype('string')
    mods1['ID'] = mods1['Protein Accession'] + '_' + mods1['Peptide Sequence']
    mods2 = mods1[['ID', 'Site in Protein', 'Motif']]
    
    # Create dictionaries for motif mapping
    modsdict = {k: f.groupby('Site in Protein')['Motif'].apply(list).to_dict() for k, f in mods2.groupby('ID')}
    
    # Prepare PhosphoSitePlus dictionary
    # phos_site['Site'] = phos_site['Site']
    # phos_site['Motif'] = phos_site['Motif']
    # phos_site['Accession'] = phos_site['Accession']
    phosSiteDict = {k: f.groupby('Site')['Motif'].apply(list).to_dict() for k, f in phos_site.groupby('Accession')}
    
    phosSum2 = phosSum1.copy()
    phosSum2['Motif'] = ''

    # map motifs from mods file
    mods_lookup = (
        mods2.groupby(['ID', 'Site in Protein'])['Motif']
        .first()
    )

    def map_mods_motif(row):
        sites = str(row['Site']).split(', ')
        result = []
        for s in sites:
            try:
                result.append(mods_lookup.loc[(row['ID'], s)])
            except KeyError:
                result.append('X')
        return ', '.join(result)

    phosSum2['Motif'] = phosSum2.apply(map_mods_motif, axis=1)

    # map motifs from PhosphoSitePlus for anything still X
    phos_lookup = (
        phos_site.groupby(['Accession', 'Site'])['Motif']
        .first()
    )

    def map_phos_motif(row):
        if 'X' not in str(row['Motif']):
            return row['Motif']
        sites  = str(row['Site']).split(', ')
        motifs = str(row['Motif']).split(', ')
        result = []
        for s, m in zip(sites, motifs):
            if 'X' in m:
                try:
                    result.append(phos_lookup.loc[(row['Accessions'], s)])
                except KeyError:
                    result.append('X')
            else:
                result.append(m)
        return ', '.join(result)

    phosSum3 = phosSum2.copy()
    phosSum3['Motif'] = phosSum3.apply(map_phos_motif, axis=1)

    # fallback — use peptide sequence if still X
    def map_fallback(row):
        if 'X' not in str(row['Motif']):
            return row['Motif']
        motifs = str(row['Motif']).split(', ')
        return ', '.join([row['Peptide Sequence'] if 'X' in m else m for m in motifs])

    phosSum4 = phosSum3.copy()
    phosSum4['Motif'] = phosSum4.apply(map_fallback, axis=1)

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
    out_label = cfg.get("phos_output_prefix", "phos")
    
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
    mods_path = os.path.join(raw_dir, dataDF.loc[index, cfg["meta_cols"]["mods"]])
    sup_corr_value = dataDF.loc[index, cfg["meta_cols"]["sup_corr"]]


    phos_site_path = cfg.get("phos_site_csv", "Phosphosite Motifs_M.csv")


    with Halo(spinner="dots", color="cyan") as sp:
        # Load input files
        sp.text = f"  {out_prefix} — reading files"
        #print(f"    Reading PSMs: {os.path.basename(psms_path)}")
        psms = pd.read_csv(psms_path, sep="\t")
        psm_start = len(psms)
        #print(f"    Reading MGF: {os.path.basename(mgf_path)}")
        mgf_dict = mgf.read(mgf_path)
        #print(f"    Reading library: {os.path.basename(lib_path)}")
        libs = pd.read_csv(lib_path).set_index("headers").to_dict()["names"]
        #print(f"    Reading peptides: {os.path.basename(pepts_path)}")
        pepts = pd.read_csv(pepts_path, sep="\t")
        #print(f"    Reading mods: {os.path.basename(mods_path)}")
        mods = pd.read_csv(mods_path, sep='\t')
        sup_corr_path = _resolve_sup_corr_path(sup_corr_value, out_dir, out_prefix)
        corrSum = pd.read_csv(sup_corr_path)
        #print(f"    Reading PhosphoSitePlus: {os.path.basename(phos_site_path)}")
        phos_site = _read_phosphositeplus_csv(phos_site_path, cfg)
        
        libsDict[index] = libs
        
        # Filter and impute PSMs
        #print(f"    Filtering + imputation...")
        sp.text = f"  {out_prefix} — filtering..."
        psm_filtered = PSM_filter(psms, libs, cfg)
        psm_filtered.to_csv(os.path.join(run_dir, "01_psm_filtered.csv"), index=False)
        PSMdf = nan_imputation(psm_filtered, mgf_dict, cfg)
        PSMdf.to_csv(os.path.join(run_dir, "02_psm_imputed.csv"), index=False)
        qc_warn_no_row_change("PSM filtering+imputation", psm_start, len(PSMdf), context=out_prefix)
        #print(len(PSMdf))
        
        # Phospho-specific filtering
        PSMdf, psm_start, psm_end = phos_filter(PSMdf, index, cfg)
        PSMdf.to_csv(os.path.join(run_dir, "03_phos_filtered.csv"), index=False)
        qc_warn_no_row_change("phospho filtering", psm_start, psm_end, context=out_prefix)
        
        # Summarize to phosphosite level with motifs
        sp.text = f"  {out_prefix} — motif mapping..."
        sumPSMdf = sum_psms_phos(PSMdf, pepts, mods, phos_site, out_prefix, cfg, out_dir=run_dir)
        sumPSMdf.to_csv(os.path.join(run_dir, "04_sum_psms_phos.csv"))
        qc_warn_no_row_change("phosphosite summarization", psm_end, len(sumPSMdf), context=out_prefix)
        sumPSM[index] = sumPSMdf
        
        # Apply supernatant corrections
        sp.text = f"  {out_prefix} — corrections"
        # Save a copy of the correction factors used for this run (named after metadata `sup_corr`)
        corr_copy_name = os.path.basename(str(sup_corr_path)) if sup_corr_path else "corr_factors.csv"
        corrSum.to_csv(os.path.join(run_dir, corr_copy_name), index=False)
        corrPhos = sumPSMdf.copy()
        for col in corrPhos.columns:
            if col in corrSum.columns:
                corrPhos[col] = corrPhos[col] / corrSum[col][0]

        qc_warn_no_value_change("correction factor (phospho)", sumPSMdf, corrPhos, context=out_prefix, cols=corrSum.columns)
        
        # Rename columns using library mapping (exclude id columns)
        id_cols = {"Gene", "Accessions", "Site"}
        rename_map = {k: v for k, v in libs.items() if k in corrPhos.columns and k not in id_cols}
        if rename_map:
            corrPhos = corrPhos.rename(columns=rename_map)
        
        corrDFs[index] = corrPhos
        
        # Save output
        sp.text = f"  {out_prefix} — saving"
        corr_phos_path = os.path.join(run_dir, "06_corr_phos.csv")
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
    out_label = cfg.get("phos_output_prefix", "phos")
    do_post_bridge = cfg.get("phos_post_bridge", cfg.get("phos_do_bridge", False))
    
    dataDF = pd.read_csv(cfg["meta_phos_csv"], sep=",").set_index(cfg["meta_index"])
    indices = dataDF.index if exp_types is None else exp_types
    
    #print(f"  Found {len(indices)} dataset(s) to process")
    
    stats = {}
    for index in indices:
        result = process_phos_dataset(index, dataDF, cfg)
        stats[index] = result

    out_dir = cfg["out_dir"]
    project = infer_project_name(list(indices))
    phos_prefix = f"phos_{project}"

    for index in indices:
        out_prefix = str(index)
        run_dir = os.path.join(out_dir, out_prefix, out_label)
        phos_corr_path = os.path.join(run_dir, "06_corr_phos.csv")

        with Halo(spinner="dots", color="cyan", text="generating combined QC heatmap...") as sp:
            phos_corr = pd.read_csv(phos_corr_path)

            # Convert to a feature x sample matrix.
            id_cols = ["Gene", "Accessions", "Site"]
            sample_cols = [c for c in phos_corr.columns if c not in id_cols]
            phos_mat = phos_corr.set_index(id_cols)[sample_cols]

            # Pool-bridge using regex; drops pool channels after normalization.
            phos_brg = bridgeCenter_data(phos_mat, cfg["pool_regex"])

            phos_brg.to_csv(os.path.join(run_dir, "07_post_pool_bridge.csv"))

            # QC heatmap AFTER bridging
            qc_heatmap_post_bridge(
                phos_brg.dropna(),
                out_png=os.path.join(run_dir, "07_QC_heatmap_post_pool_bridge.png"),
                top_n=cfg["qc_top_n"],
                zscore=cfg["qc_zscore"],
                title=f"{out_prefix} QC heatmap (post pool-bridge)",
            )
            sp.succeed(f"{out_prefix} — QC outputs + heatmap ")

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
