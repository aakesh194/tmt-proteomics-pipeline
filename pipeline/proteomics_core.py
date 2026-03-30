"""
proteomics_core.py - Shared functions for gProt and phos pipelines

Author: Aakesh Yoganathan
Lab: Tamir Lab, UNC Chapel Hill
"""

import os
import numpy as np
import pandas as pd
#from pyteomics import mgf
#import re
from natsort import natsorted
import matplotlib.pyplot as plt


# PSM FILTERING
def PSM_filter(df, libs, cfg):
    """
    Filter PSMs based on quality criteria.
    Used by both gProt and phos pipelines.
    """
    newdf = pd.DataFrame()
    
    # Filter: Search engine rank of 1
    PSMdf = df[df['Search Engine Rank'] == 1]
    
    # Filter: Delta M ppm range
    dmin, dmax = cfg["delta_ppm_range"]
    PSMdf = PSMdf[PSMdf["Delta M in ppm"].between(dmin, dmax)]
    
    # Filter: Expectation value
    PSMdf = PSMdf[PSMdf["Expectation Value"] < cfg["expectation_max"]]
    
    # Filter: Modifications (e.g., TMT labeled)
    PSMdf = PSMdf[PSMdf["Modifications"].str.contains(cfg["require_mod_contains"], na=False)]
    
    # Filter: Ions score
    PSMdf = PSMdf[PSMdf["Ions Score"] >= cfg["ions_score_min"]]
    
    # Select appropriate columns
    for key in libs:
        if key in PSMdf.columns:
            newdf = pd.concat([newdf, PSMdf[key]], axis=1)
    
    # Extract Gene name from Master Protein Descriptions
    if "Master Protein Descriptions" in newdf.columns:
        data1 = newdf["Master Protein Descriptions"].astype(str).str.split("GN=", n=1, expand=True)
        if data1.shape[1] > 1:
            data = data1[1].str.split(" ", n=1, expand=True)
            newdf["Gene"] = data[0]
        else:
            newdf["Gene"] = np.nan
    else:
        newdf["Gene"] = np.nan
    
    return newdf



# NAN IMPUTATION
def nan_imputation(df, mgf_data, cfg):
    """
    Impute missing values using minimum intensity from MGF files.
    Used by both gProt and phos pipelines.
    """
    # Convert MGF to DataFrame
    mgf_list = list(mgf_data)
    mgf_df = pd.DataFrame(mgf_list)

    # Expand the params column into separate DataFrame columns containing scan number, charge, monisotopic peptide m/z
    params_df = pd.json_normalize(mgf_df['params'])
    pepmass_df = (
        params_df['pepmass'].astype('string').str.strip()
        .str.replace(r"\(|'|\)", "", regex=True)
        .str.split(", ", n=1, expand=True)
    )
    params_df = pd.concat([params_df.drop('pepmass', axis=1), pepmass_df[0]], axis=1)
    params_df = params_df.rename(columns={0: 'pepmass'})
    params_df["charge"] = params_df["charge"].astype("string").str.strip().str.replace(r"\+", "", regex=True)
    
    # Concatenate the original DataFrame with the expanded params DataFrame only taking the scan number, charge, and peptide m/z
    mgf_df = pd.concat([mgf_df.drop('params', axis=1), params_df[['scans', 'charge', 'pepmass']]], axis=1)

    # Calculate the minimum and maximum intensity values for each scan
    intensity_min_max_df = mgf_df['intensity array'].apply(
        lambda x: pd.Series({'intensity_min': x.min(), 'intensity_max': x.max()})
    )
    mgf_df = pd.concat([mgf_df.drop(['m/z array', 'intensity array', 'charge array'], axis=1), intensity_min_max_df], axis=1)
    
    # Format mgf dataframe to match PSMs
    mgf_df = mgf_df.rename(columns={'scans': 'First Scan', 'charge': 'Charge', 'pepmass': 'mz in Da'})
    mgf_df['First Scan'] = mgf_df['First Scan'].astype('int64')
    mgf_df['Charge'] = mgf_df['Charge'].astype('int64')
    mgf_df['mz in Da'] = mgf_df['mz in Da'].astype('float64')

    # Merge with PSMs
    mdf = pd.merge(df, mgf_df[['First Scan', 'Charge', 'mz in Da', 'intensity_min', 'intensity_max']], 
                   on=['First Scan', 'Charge', 'mz in Da'])

    # Drop unused columns
    PSMfilter = mdf.drop(['Search Engine Rank'], axis=1, errors="ignore")

    # Count channels present
    abund_cols = PSMfilter.columns[PSMfilter.columns.str.contains(cfg["abundance_contains"])]
    PSMfilter = PSMfilter.assign(countTMTchannels=PSMfilter.loc[:, abund_cols].count(axis=1))

    # Filter by minimum channels
    min_needed = max(1, int(np.floor(len(abund_cols) * cfg["min_fraction_channels_present"])))
    PSMfilter = PSMfilter[PSMfilter['countTMTchannels'] >= min_needed]

    # Impute missing values with minimum intensity
    PSMfilter.loc[:, abund_cols] = (
        PSMfilter.loc[:, abund_cols]
        .apply(lambda x: x.fillna(PSMfilter['intensity_min']))
    )

    # Calculate average abundance across channels
    abund = PSMfilter.loc[:, abund_cols]
    PSMfilter = PSMfilter.assign(AbundAve=abund.mean(axis=1))

    return PSMfilter



# NORMALIZATION FUNCTIONS
def meanCenter_data(df, eT=None):
    """
    Mean-center and z-score normalize data.
    """
    ave = df.mean(axis=1)
    dfmc = df.div(ave, axis=0)
    stndev = df.std(axis=1).replace(0, np.nan)
    dfzs = df.sub(ave, axis=0).div(stndev, axis=0)
    return dfmc, dfzs


def bridgeCenter_data(df, brg_regex):
    """
    Normalize data to bridge channel(s).
    """
    brg_cols = df.columns[df.columns.str.contains(pat=brg_regex, regex=True)]

    if len(brg_cols) == 0:
        print(f"  ⚠️  No bridge columns found matching pattern: {brg_regex}")
        return df.copy()
    
    #print(f"  Found {len(brg_cols)} bridge column(s): {list(brg_cols)}")
    
    # Use mean of multiple bridge channels if 
    bridge = df[brg_cols].mean(axis=1)
    dfbrg = df.div(bridge, axis=0)
    dfbrg = dfbrg.drop(list(dfbrg.filter(regex=brg_regex)), axis=1)
    return dfbrg



# DATA CONCATENATION
def concat_dfs(data):
    """
    Concatenate multiple dataframes and sort columns naturally.
    """
    df_u = pd.concat(data, axis=1)
    df_u.columns = df_u.columns.droplevel(0)
    df_u = df_u[natsorted(df_u.columns)]
    df_n = df_u.dropna()
    return df_u, df_n



# MULTI-RUN PROCESSING
def process_runs(corrDFs, keys, cfg):
    """
    Process multiple runs with bridging normalization.
    """
    #print(f"Processing {len(keys)} runs for multi-run analysis...")
    
    run_corr = {}
    wMC = {}
    wMC_zscore = {}
    raw_mc = {}
    raw_zscore = {}
    brg_corr = {}
    
    # Concatenate all samples
    allBrgs = pd.concat([corrDFs[key] for key in keys], axis=1)
    allBrgs = allBrgs[natsorted(allBrgs.columns)]
    allBrgs = allBrgs.dropna()

    # Calculate bridge correction factors
    filtered_cols = allBrgs.filter(regex=cfg["pool_regex"])
    if filtered_cols.shape[1] == 0:
        print(f"-- No pool columns found for multi-run correction --")
        mcB_run = pd.DataFrame()
    else:
        mcB_run, _ = meanCenter_data(filtered_cols, "brg")
        mcB_run = mcB_run.mean(axis=0)
        mcB_run = mcB_run.to_frame().transpose()
    
    # Bridge normalization for each run
    for k in keys:
        brgA = bridgeCenter_data(corrDFs[k], cfg["pool_regex"])
        brg_corr[k + '_brgA'] = brgA
        brg_corr[k + '_brgS'] = bridgeCenter_data(brgA, cfg["bridge_regex"])

    # Apply run corrections
    for k in keys:
        raw_mc[k], raw_zscore[k] = meanCenter_data(corrDFs[k], None)
        if not mcB_run.empty:
            for col in corrDFs[k].columns:
                if col in mcB_run.columns:
                    df = corrDFs[k] / mcB_run[col][0]
                    df = df.drop(df.filter(regex=cfg["pool_regex"]).columns, axis=1)
                    run_corr[k] = df
                    wMC[k], wMC_zscore[k] = meanCenter_data(df, k)
    
    # Concatenate all results
    concatDict = {}
    concatDict['runCorr_DF_u'], concatDict['runCorr_DF_n'] = concat_dfs(run_corr)
    concatDict['run_corr_mc_u'], concatDict['run_corr_mc_n'] = concat_dfs(wMC)
    concatDict['run_corr_zs_u'], concatDict['run_corr_zs_n'] = concat_dfs(wMC_zscore)
    concatDict['raw_mc_DF_u'], concatDict['raw_mc_DF_n'] = concat_dfs(raw_mc)
    concatDict['raw_zscore_DF_u'], concatDict['raw_zscore_DF_n'] = concat_dfs(raw_zscore)
    concatDict['brgA_u'], concatDict['brgA_n'] = concat_dfs({key: value for key, value in brg_corr.items() if 'brgA' in key})
    concatDict['brgS_u'], concatDict['brgS_n'] = concat_dfs({key: value for key, value in brg_corr.items() if 'brgS' in key})
    
    return mcB_run, concatDict

# def run_post_bridge_outputs(corrDFs, cfg, keys=None, prefix=None, pipeline="prot", out_prefix_default="gProt"):
#     """Shared post-bridge runner for gProt and phos pipelines."""
#     meta_key = f"meta_{pipeline}_csv"

#     dataDF = pd.read_csv(cfg[meta_key], sep=",").set_index(cfg["meta_index"])
#     run_keys = list(dataDF.index) if keys is None else list(keys)

#     if not run_keys:
#         raise ValueError("run_post_bridge_outputs: no experiment types provided")

#     out_prefix = prefix or out_prefix_default
#     _, concatDict = process_runs(corrDFs, run_keys, prefix, cfg)

#     out_dir = cfg.get("post_bridge_dir") or ""
#     if out_dir:
#         os.makedirs(out_dir, exist_ok=True)

#     # Save all concatenated results
#     for name, df in concatDict.items():
#         out_name = f"{out_prefix}_{name}.csv"
#         out_path = os.path.join(out_dir, out_name) if out_dir else out_name
#         df.to_csv(out_path)
#         print(f"Saved: {out_name}")

#     return concatDict

def run_post_bridge_outputs(corrDFs, cfg, keys=None, prefix=None, pipeline="prot", out_prefix_default="gProt"):
    """Shared post-bridge runner for gProt and phos pipelines."""
    meta_key = f"meta_{pipeline}_csv"

    dataDF = pd.read_csv(cfg[meta_key], sep=",").set_index(cfg["meta_index"])
    run_keys = list(dataDF.index) if keys is None else list(keys)

    if not run_keys:
        raise ValueError("run_post_bridge_outputs: no experiment types provided")

    out_prefix = prefix or out_prefix_default
    _, concatDict = process_runs(corrDFs, run_keys, cfg)

    out_dir = cfg.get("post_bridge_dir") or ""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Save all concatenated results
    for name, df in concatDict.items():
        out_name = f"{out_prefix}_{name}.csv"
        out_path = os.path.join(out_dir, out_name) if out_dir else out_name
        df.to_csv(out_path)

    return concatDict



# QC HEATMAP
def qc_heatmap_post_bridge(df, out_png, top_n=300, zscore=True, title="QC heatmap (post pool-bridge)"):
    """
    Generate QC heatmap after pool bridging normalization.
    """
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("qc_heatmap_post_bridge: empty dataframe")

    # Select top variable rows
    var = df.var(axis=1, numeric_only=True)
    df2 = df.loc[var.sort_values(ascending=False).head(min(top_n, len(var))).index]

    mat = df2.to_numpy(dtype=float)

    if zscore:
        mu = np.nanmean(mat, axis=1, keepdims=True)
        sd = np.nanstd(mat, axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        mat = (mat - mu) / sd

    # Generate heatmap
    plt.figure(figsize=(max(10, df2.shape[1] * 0.35), 9))
    plt.imshow(mat, aspect="auto", interpolation="nearest", cmap='RdBu_r')
    plt.colorbar(label='Z-score' if zscore else 'Abundance')
    plt.xticks(range(df2.shape[1]), df2.columns, rotation=90, fontsize=7)
    
    plt.yticks([])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    
    #print(f"Saved: {out_png}")

##
    
def infer_corr_paths(meta_csv, meta_index, out_dir):
    data_df = pd.read_csv(meta_csv, sep=",").set_index(meta_index)
    if len(data_df.index) == 0:
        raise ValueError("meta_csv has no rows; cannot infer gProt_corr path.")
    paths = []
    for exp_type in data_df.index:
        paths.append(
            os.path.join(out_dir, f"{str(exp_type).replace('_sup','')}_gProt_corr.csv")
        )
    return paths

def to_matrix(df, id_cols):
    sample_cols = [c for c in df.columns if c not in id_cols]
    return df.set_index(id_cols)[sample_cols]


def pick_post_bridge_csv(cfg):
    post_dir = cfg.get("post_bridge_dir", os.path.join(cfg["out_dir"], "after-bridging"))
    prefix = cfg.get("post_bridge_prefix", "gProt")
    candidates = [
        f"{prefix}_brgA_n.csv",
        f"{prefix}_brgA_u.csv",
        f"{prefix}_brgS_n.csv",
        f"{prefix}_brgS_u.csv",
    ]
    for name in candidates:
        path = os.path.join(post_dir, name)
        if os.path.exists(path):
            return path
    return None



