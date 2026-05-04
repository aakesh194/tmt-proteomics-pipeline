"""
gprot_processing.py - Global proteomics (gProt) specific processing

Author: Aakesh Yoganathan
Lab: Tamir Lab, UNC Chapel Hill
"""

import os
import numpy as np
import pandas as pd

from pyteomics import mgf
from halo import Halo

from config.utils import load_config
from pipeline.proteomics_core import *

# cfg["out_dir"] = "outputs"
# cfg["pool_regex"] = r"BRG"
# cfg["qc_top_n"] = 300
# cfg["qc_zscore"] = True

# GLOBAL DATA STORES
corrDict = {}
corrPeps = {}
sumPSM = {}
corrDFs = {}
libsDict = {}


def _reset_run_state():
    """Reset global data stores."""
    corrDict.clear()
    corrPeps.clear()
    sumPSM.clear()
    corrDFs.clear()
    libsDict.clear()


def sup_Corrections(df, cfg):
    """Calculate supernatant correction factors for gProt."""
    
    # Calculate average abundance if not already present
    if 'AbundAve' not in df.columns:
        abund = df.loc[:, df.columns.str.contains(cfg["abundance_contains"])]
        df = df.assign(AbundAve=abund.mean(axis=1))
    
    #---#
    # Select top 25% most abundant PSMs
    # rawSup = df[df['AbundAve'] >= df['AbundAve'].quantile(0.75)]
    
    # # Sum PSMs by gene, protein, sequence
    # psmsSum = df.set_index(['Gene', 'Master Protein Accessions', 'Sequence'])
    # psmsSum = psmsSum.groupby(['Gene', 'Master Protein Accessions', 'Sequence']).agg('sum')
    # abund = psmsSum.loc[:, psmsSum.columns.str.contains(cfg["abundance_contains"])]
    # psmsSum2 = psmsSum.assign(AbundAve=abund.mean(axis=1))
    # psmsSum2 = psmsSum.reset_index()
    
    # # Calculate correction factors
    # Sup = psmsSum2.loc[:, psmsSum2.columns.str.contains(cfg["abundance_contains"])].div(psmsSum2['AbundAve'], axis=0)
    # Sup = Sup.mean(axis=0).to_frame().transpose()
    #---#
    
    use_top = cfg.get("gprot_use_top_quantile", False)
    q = cfg.get("gprot_top_quantile", 0.75)

    base_df = df
    if use_top:
        base_df = df[df["AbundAve"] >= df["AbundAve"].quantile(q)]

    psmsSum = base_df.set_index(["Gene", "Master Protein Accessions", "Sequence"])

    # Select top 25% most abundant PSMs
    #rawSup = df[df['AbundAve'] >= df['AbundAve'].quantile(0.75)]
    
    # Sum PSMs by gene, protein, sequence

    #psmsSum = df.set_index(['Gene', 'Master Protein Accessions', 'Sequence'])
    #psmsSum = rawSup.set_index(['Gene', 'Master Protein Accessions', 'Sequence'])
    psmsSum = psmsSum.groupby(['Gene', 'Master Protein Accessions', 'Sequence']).agg('sum')
    abund = psmsSum.loc[:, psmsSum.columns.str.contains(cfg["abundance_contains"])]
    psmsSum2 = psmsSum.assign(AbundAve=abund.mean(axis=1))
    psmsSum2 = psmsSum2.reset_index()
    
    # Calculate correction factors
    Sup = psmsSum2.loc[:, psmsSum2.columns.str.contains(cfg["abundance_contains"])].div(psmsSum2['AbundAve'], axis=0)
    Sup = Sup.mean(axis=0).to_frame().transpose()
    
    # Prepare peptide sum with labels
    #pepSum = psmsSum.reset_index()
    pepSum = psmsSum2.reset_index()

    uniprotID = pepSum['Master Protein Accessions'].str.split(';', expand=True)
    pepSum['Accessions'] = uniprotID[0]
    pepSum['Label'] = pepSum['Gene'] + '_' + pepSum['Accessions'] + '_' + pepSum['Sequence']
    pepSum = pepSum.set_index('Label')
    pepSum = pepSum.drop([
        'Gene', 'Accessions', 'Master Protein Descriptions', 'Master Protein Accessions', 'Sequence',
        'Expectation Value', 'First Scan', 'Charge', 'mz in Da', 'intensity_min', 'intensity_max',
        'countTMTchannels', 'Annotated Sequence', 'PhosphoRS Best Site Probabilities',
        'Modifications', 'Delta M in ppm', 'Delta mz in Da', 'Ions Score', 'AbundAve'
    ], axis=1, errors='ignore')
    
    return pepSum, Sup


def sum_peps(df):
    """Sum PSMs to protein level for gProt."""
    drop_cols = [
        'Expectation Value', 'First Scan', 'Charge', 'mz in Da', 'intensity_min', 'intensity_max',
        'countTMTchannels', 'Annotated Sequence', 'Sequence', 'PhosphoRS Best Site Probabilities',
        'Modifications', 'Delta M in ppm', 'Delta mz in Da', 'Ions Score', 'AbundAve'
    ]
    df = df.drop([c for c in drop_cols if c in df.columns], axis=1)
    
    uniprotID = df['Master Protein Accessions'].str.split(';', expand=True)
    df['Accessions'] = uniprotID[0]
    pepSum = df.set_index(['Gene', 'Accessions']).drop(['Master Protein Descriptions', 'Master Protein Accessions'], axis=1)
    
    # Sum peptides based on gene name and protein accession
    pepSum = pepSum.groupby(['Gene', 'Accessions']).agg('sum')
    
    return pepSum



# DATASET PROCESSING
def process_gprot_dataset(index, dataDF, cfg):
    """Process a single gProt dataset."""
    
    raw_dir = cfg.get("raw_dir", "")
    out_dir = cfg.get("out_dir", "")
    lib_dir = cfg.get("lib_dir", "")
    sup_corr_value = None
    try:
        sup_col = cfg.get("meta_cols", {}).get("sup_corr", "sup_corr")
        if sup_col in dataDF.columns:
            sup_corr_value = dataDF.loc[index, sup_col]
    except Exception:
        sup_corr_value = None
    
    out_prefix = str(index)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    run_dir = os.path.join(out_dir, out_prefix, "gprot")
    os.makedirs(run_dir, exist_ok=True)
    
    # Build file paths
    psms_path = os.path.join(raw_dir, dataDF.loc[index, cfg["meta_cols"]["psms"]])
    mgf_path = os.path.join(raw_dir, dataDF.loc[index, cfg["meta_cols"]["mgf"]])
    lib_path = os.path.join(lib_dir, dataDF.loc[index, cfg["meta_cols"]["library"]])
    
    
    with Halo(spinner="dots", color="cyan") as sp:
        # Load input files
        sp.text = f"  {out_prefix} — reading files"

        psms = pd.read_csv(psms_path, sep="\t")
        psm_start = len(psms)
        mgf_dict = mgf.read(mgf_path)
        lib_df = pd.read_csv(lib_path)
        if lib_df["headers"].duplicated().any():
            raise ValueError(f"Duplicate library headers in {lib_path}")
        libs = lib_df.set_index("headers")["names"].to_dict()

        # Filter and impute PSMs
        sp.text = f"  {out_prefix} — filtering..."
        psm_filtered = PSM_filter(psms, libs, cfg)
        psm_filtered.to_csv(os.path.join(run_dir, "01_psm_filtered.csv"), index=False)
        PSMdf = nan_imputation(psm_filtered, mgf_dict, cfg)
        PSMdf.to_csv(os.path.join(run_dir, "02_psm_imputed.csv"), index=False)
        qc_warn_no_row_change("PSM filtering+imputation", psm_start, len(PSMdf), context=out_prefix)

        # Calculate corrections
        sp.text = f"  {out_prefix} — corrections..."
        peps, corrSum = sup_Corrections(PSMdf, cfg)
        peps.to_csv(os.path.join(run_dir, "03_peps.csv"))
        
        corrDict[index] = corrSum
        sumPSMdf = sum_peps(PSMdf)
        sumPSMdf.to_csv(os.path.join(run_dir, "04_sum_peps.csv"))
        qc_warn_no_row_change("peptide summarization", len(PSMdf), len(sumPSMdf), context=out_prefix)
        sumPSM[index] = sumPSMdf

        # Save correction factors
        corr_sum_path = os.path.join(run_dir, "05_corr_factors.csv")
        corrSum.to_csv(corr_sum_path, index=False)

        # Optional convenience: also write correction factors to the filename referenced
        # by `sup_corr` in metadata (typically used by the phos pipeline).
        if sup_corr_value is not None:
            alias = str(sup_corr_value).strip()
            if alias and alias.lower() not in {"nan", "none"}:
                alias_path = alias if os.path.isabs(alias) else os.path.join(out_dir, alias)
                dir_to_create = os.path.dirname(alias_path) if os.path.dirname(alias_path) else out_dir
                os.makedirs(dir_to_create, exist_ok=True)
                corrSum.to_csv(alias_path, index=False)

        # Apply corrections
        corrProt = sumPSMdf.copy()
        corrPept = peps.copy()
        for col in corrProt.columns:
            if col in corrSum.columns:
                corrProt[col] = corrProt[col] / corrSum[col][0] 
        
        for col in corrPept.columns:
            if col in corrSum.columns:
                corrPept[col] = corrPept[col] / corrSum[col][0]

        qc_warn_no_value_change("correction factor (proteins)", sumPSMdf, corrProt, context=out_prefix, cols=corrSum.columns)
        qc_warn_no_value_change("correction factor (peptides)", peps, corrPept, context=out_prefix, cols=corrSum.columns)

        # Rename columns using library mapping (exclude id columns)
        id_cols = {"Gene", "Accessions"}
        rename_map = {k: v for k, v in libs.items() if k in corrProt.columns and k not in id_cols}
        if rename_map:
            corrProt = corrProt.rename(columns=rename_map)
        rename_map = {k: v for k, v in libs.items() if k in corrPept.columns and k not in id_cols}
        if rename_map:
            corrPept = corrPept.rename(columns=rename_map)
        
        # Store results
        corrDFs[index] = corrProt
        corrPeps[index] = corrPept
        
        # Save outputs
        sp.text = f"  {out_prefix} — saving..."
        corr_prot_path = os.path.join(run_dir, "06_corr_prot.csv")
        corr_pept_path = os.path.join(run_dir, "06_corr_pept.csv")
        corrProt.to_csv(corr_prot_path)
        corrPept.to_csv(corr_pept_path)

        sp.succeed(f"  {out_prefix} — proteins: {len(sumPSMdf)} | saved: {corr_prot_path}")

        return {
            "experiment": out_prefix,
            "psms_raw": psm_start,
            "psms_after_filter": len(PSMdf),
            "psm_retention_pct": round(len(PSMdf) / psm_start * 100, 1) if psm_start else 0,
            "proteins": len(sumPSMdf),
            "peptides": len(peps),
            "tmt_channels": len([c for c in corrProt.columns if c not in {"Gene", "Accessions"}]),
        }
    
    # tqdm.write(f"\n[gprot] {out_prefix}")
    # tqdm.write(f"        Proteins: {len(sumPSMdf)} | Saved: {corr_prot_path}")


# PIPELINE EXECUTION
def run_gprot_pipeline(cfg=None, exp_types=None):
    """Run the gProt pipeline for all or selected experiment types."""
    if cfg is None:
        cfg = load_config()

    _reset_run_state()
    do_post_bridge = cfg.get("gprot_post_bridge", cfg.get("gprot_do_bridge", False))
    print(f"\n[gprot]")
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    
    dataDF = pd.read_csv(cfg["meta_prot_csv"], sep=",").set_index(cfg["meta_index"])
    indices = dataDF.index if exp_types is None else exp_types
    
    #print(f"[gprot] Found {len(indices)} dataset(s) to process")
    
    stats = {}
    for index in indices:
        result = process_gprot_dataset(index, dataDF, cfg)
        stats[index] = result

    # Run QC per expType output.
    #corr_paths = infer_corr_paths(cfg["meta_prot_csv"], cfg["meta_index"], out_dir)
    for index in indices:
        out_prefix = str(index)
        run_dir = os.path.join(out_dir, out_prefix, "gprot")
        gprot_corr_path = os.path.join(run_dir, "06_corr_prot.csv")

        with Halo(spinner="dots", color="cyan", text="generating combined QC heatmap...") as sp:
            gprot_corr = pd.read_csv(gprot_corr_path)

            # Convert to a feature x sample matrix.
            id_cols = ["Gene", "Accessions"]
            sample_cols = [c for c in gprot_corr.columns if c not in id_cols]
            gprot_mat = gprot_corr.set_index(id_cols)[sample_cols]

            # Pool-bridge using regex; drops pool channels after normalization.
            gprot_brg = bridgeCenter_data(gprot_mat, cfg["pool_regex"])

            gprot_brg.to_csv(os.path.join(run_dir, "07_post_pool_bridge.csv"))

            # QC heatmap AFTER bridging
            qc_heatmap_post_bridge(
                gprot_brg.dropna(),
                out_png=os.path.join(run_dir, "07_QC_heatmap_post_pool_bridge.png"),
                top_n=cfg["qc_top_n"],
                zscore=cfg["qc_zscore"],
                title=f"{out_prefix} QC heatmap (post pool-bridge)",
            )
            #print(f"[run] Wrote QC outputs for: {base}", flush=True)
            sp.succeed(f"{out_prefix} — QC outputs + heatmap ")

    if do_post_bridge:
        # Combined post-bridge outputs (After bridging)
        post_cfg = dict(cfg)
        post_cfg["post_bridge_dir"] = os.path.join(out_dir, "after-bridging", "gprot")
        with Halo(spinner="dots", color="cyan", text="[gprot] post-bridge outputs...") as sp:
            run_post_bridge_outputs(corrDFs, cfg=post_cfg, pipeline="prot", prefix="gprot")
            sp.succeed(f"post-bridge outputs | saved: {post_cfg['post_bridge_dir']}")

        # Combined heatmap from post-bridge outputs (universal)
        post_csv = pick_post_bridge_csv(post_cfg)
        with Halo(spinner="dots", color="cyan", text="[gprot] generating combined heatmap...") as sp:
            if post_csv:
                combined_df = pd.read_csv(post_csv)
                combined_mat = to_matrix(combined_df, ["Gene", "Accessions"])
                out_png = os.path.join(
                    post_cfg.get("post_bridge_dir", os.path.join(out_dir, "after-bridging", "gprot")),
                    f"{cfg.get('post_bridge_prefix', 'gProt')}_QC_heatmap_bridged.png",
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

    # sp.succeed("Done. saved:", out_dir,)
    print(f"\n[gprot] done — all written to:", out_dir, flush=True)
    return stats


def main():
    cfg = load_config()
    run_gprot_pipeline(cfg=cfg)


if __name__ == "__main__":
    main()
