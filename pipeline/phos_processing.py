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
    nan_imputation
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
    
    # Keep only phospho peptides
    phosfilter = phosfilter[phosfilter['Modifications'].str.contains("Phos", na=False)]
    #print(f"      After phospho filter: {len(phosfilter)}")
    
    # Filter for pY if specified
    if cfg.get("phos_filter_pY", True) and 'pY' in exp_type:
        phosfilter = phosfilter[phosfilter['Modifications'].str.contains("Y", na=False)]
        #print(f"      After pY filter: {len(phosfilter)}")
    
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


# def sum_psms_phos(df, pepts, mods, phos_site, exp_type, cfg, out_dir=""):
#     """
#     Sum PSMs to phosphosite level and add site/motif annotations.
    
#     1. Parses PhosphoRS localization
#     2. Maps peptide sites to protein sites
#     3. Assigns motifs from modification files
#     4. Assigns motifs from PhosphoSitePlus database
#     5. Sums to final phosphosite level
#     """
#     #print(f"    Summarizing phosphosites with motif annotation...")
    
#     # Parse PhosphoRS localization
#     pRS = pd.DataFrame(
#         df['PhosphoRS Best Site Probabilities']
#         .str.findall(r"(\b[S,T,Y]\w+)")
#         .astype('string').str.strip()
#         .str.replace(r"\[|'|\]", "", regex=True)
#     )
    
#     # Concatenate with dataframe
#     phosSum = pd.concat([
#         df.drop([
#             'PhosphoRS Best Site Probabilities', 'Modifications', 'Expectation Value',
#             'First Scan', 'Charge', 'mz in Da', 'intensity_min', 'intensity_max',
#             'countTMTchannels'
#         ], axis=1, errors='ignore'),
#         pRS
#     ], axis=1)
    
#     # Sum peptides by sequence and site
#     phosSum = phosSum.groupby([
#         'Annotated Sequence', 'Sequence', 'Gene', 'Master Protein Accessions', 'PhosphoRS Best Site Probabilities'
#     ]).agg('sum')
#     phosSum = phosSum.reset_index()
#     phosSum['Annotated Sequence2'] = phosSum['Annotated Sequence'].astype('string').str.upper()
    
#     # Get modification sites from peptide file
#     pSeq = pepts[['Annotated Sequence', 'Modifications in Master Proteins', 'Positions in Master Proteins']]
    
#     # Extract site assignments and positions
#     pSeq2 = pd.DataFrame(
#         pSeq['Modifications in Master Proteins']
#         .str.findall(r"(?<=\[)([^]]+)(?=\])")
#         .astype('string').str.strip()
#         .str.replace(r"\[|'|\]", "", regex=True)
#     )
#     start = pd.DataFrame(
#         pSeq['Positions in Master Proteins']
#         .str.findall(r"(?<=\[)([^]]+)(?=\])")
#         .astype('string').str.strip()
#         .str.replace(r"\[|'|\]", "", regex=True)
#     )
#     start2 = start['Positions in Master Proteins'].str.split(',', expand=True)
#     start2 = start2[0].str.split('-', expand=True)
#     pSeq3 = pSeq.assign(Sites=pSeq2, peptStart=start2[0], peptEnd=start2[1])
#     pSeq3['Annotated Sequence'] = pSeq3['Annotated Sequence'].astype('string')
#     pSites = pSeq3.set_index('Annotated Sequence').to_dict()['Sites']
#     pStart = pSeq3.set_index('Annotated Sequence').to_dict()['peptStart']
#     pEnd = pSeq3.set_index('Annotated Sequence').to_dict()['peptEnd']
#     phosSum['Sites'] = phosSum['Annotated Sequence2'].map(pSites)
#     phosSum['peptStart'] = phosSum['Annotated Sequence2'].map(pStart)
#     phosSum['peptEnd'] = phosSum['Annotated Sequence2'].map(pEnd)
    
#     # Extract PhosphoRS localized sites (positions within peptide)
#     pRSx = pd.DataFrame(
#         phosSum['PhosphoRS Best Site Probabilities']
#         .str.findall(r'(\d+)')
#         .astype('string').str.strip()
#         .str.replace(r"\[|'| |\]", "", regex=True)
#     )
#     phosSum['intSites'] = pRSx['PhosphoRS Best Site Probabilities'].str.split(',').astype('string').str.strip().str.replace(r"\[|'| |\]", "", regex=True)
    
#     # Create unique identifier and dictionaries for site mapping
#     phosSum['AA+ptmRS'] = phosSum['Sequence'] + '_' + phosSum['PhosphoRS Best Site Probabilities']
#     pRSdict = phosSum.set_index('AA+ptmRS').to_dict()['intSites']
#     peptStartdict = phosSum.set_index('AA+ptmRS').to_dict()['peptStart']
    
#     # Convert intSites string to list
#     for k, v in pRSdict.items():
#         if isinstance(v, str):
#             pRSdict[k] = v.split(',')
#         else:
#             pRSdict[k] = []
    
#     # Calculate true protein-level sites
#     trueSite = {}
#     for i, j in peptStartdict.items():
#         for k, v in pRSdict.items():
#             if i == k and j is not None and not pd.isna(j):
#                 try:
#                     trueSite[k] = '; '.join([str(int(float(j)) - 1 + int(vi)) for vi in v if vi])
#                 except (ValueError, TypeError):
#                     trueSite[k] = ''
    
#     phosSum['trueSite'] = phosSum['AA+ptmRS'].map(trueSite)
    
#     # Extract accessions
#     uniprotID = phosSum['Master Protein Accessions'].str.split(';', expand=True)
#     phosSum['Accessions'] = uniprotID[0]
#     phosSum = phosSum.drop(['intSites', 'Annotated Sequence', 'AA+ptmRS', 'Annotated Sequence2', 'Sites', 'Master Protein Accessions'], axis=1)
#     phosSum = phosSum.groupby(['Sequence', 'Gene', 'Accessions', 'peptStart', 'peptEnd', 'trueSite', 'PhosphoRS Best Site Probabilities']).agg('sum')
    
#     # Rename and prepare for motif mapping
#     phosSum1 = phosSum.reset_index().rename(columns={'Sequence': 'Peptide Sequence', 'PhosphoRS Best Site Probabilities': 'ptmRS'})
    
#     # Map true sites
#     psiteA = phosSum1.to_dict()['trueSite']
#     for k, v in psiteA.items():
#         if isinstance(v, str):
#             psiteA[k] = v.split('; ')
#         else:
#             psiteA[k] = []
    
# 	#\\d replaced from \d
#     #phosSum1['ptmRS'] = phosSum1['ptmRS'].replace('\\d+', '', regex=True)
#     phosSum1['ptmRS'] = phosSum1['ptmRS'].replace(r"\d+", "", regex=True)

#     ptmRSdict = phosSum1.to_dict()['ptmRS']
#     for k, v in ptmRSdict.items():
#         if isinstance(v, str):
#             ptmRSdict[k] = v.split(', ')
#         else:
#             ptmRSdict[k] = []
    
#     # Create new site labels (e.g., S123, T456)
#     newSite = []
#     for i in range(len(psiteA)):
#         try:
#             if len(psiteA[i]) > 0 and len(ptmRSdict[i]) > 0:
#                 newSite.append(', '.join([ptmRSdict[i][j] + psiteA[i][j] for j in range(len(psiteA[i]))]))
#             else:
#                 newSite.append('')
#         except (IndexError, TypeError):
#             newSite.append('')
    
#     phosSum1 = phosSum1.drop('trueSite', axis=1)
#     phosSum1['Site'] = newSite
    
#     # Handle missed cleavages - normalize peptide sequences
#     for i in phosSum1['Peptide Sequence']:
#         if isinstance(i, str):
#             if i.endswith('KK') or i.endswith('RK') or i.endswith('KR'):
#                 phosSum1['Peptide Sequence'] = phosSum1['Peptide Sequence'].replace(i, i[:-1])
    
#     phosSum1['ID'] = phosSum1['Accessions'] + '_' + phosSum1['Peptide Sequence']
#     phosSum1 = phosSum1.groupby(['Peptide Sequence', 'Gene', 'Accessions', 'Site', 'ptmRS', 'ID']).agg('sum', numeric_only=True).reset_index()
    
#     # Prepare modifications table for motif extraction
#     mods1 = mods[mods['Confidence'] == 'High']
#     if cfg.get("phos_filter_pY", True) and 'pY' in exp_type:
#         mods1 = mods1[mods1['Target Amino Acid'] == 'Y']
#     mods1 = mods1[['Target Amino Acid', 'Position in Peptide', 'Peptide Sequence', 'Protein Accession', 'Position', 'Motif']]
#     mods1['Site in peptide'] = mods1['Target Amino Acid'].astype('string') + mods1['Position in Peptide'].astype('string')
#     mods1['Site in Protein'] = mods1['Target Amino Acid'].astype('string') + mods1['Position'].astype('string')
#     mods1['ID'] = mods1['Protein Accession'] + '_' + mods1['Peptide Sequence']
#     mods2 = mods1[['ID', 'Site in Protein', 'Motif']]
    
#     # Create dictionaries for motif mapping
#     modsdict = {k: f.groupby('Site in Protein')['Motif'].apply(list).to_dict() for k, f in mods2.groupby('ID')}
    
#     # Prepare PhosphoSitePlus dictionary
#     # phos_site['Site'] = phos_site['Site']
#     # phos_site['Motif'] = phos_site['Motif']
#     # phos_site['Accession'] = phos_site['Accession']
#     phosSiteDict = {k: f.groupby('Site')['Motif'].apply(list).to_dict() for k, f in phos_site.groupby('Accession')}
    
#     phosSum2 = phosSum1.copy()
#     phosSum2['Motif'] = ''
    
#     # Assign motifs from mods file
#     #print(f"      Mapping motifs from modification file...")
#     for k, v in modsdict.items():
#         for i in range(len(phosSum2)):
#             if k == phosSum2['ID'].iloc[i] and not phosSum2['Motif'].iloc[i]:
#                 try:
#                     sites = str(phosSum2['Site'].iloc[i]).split(', ')
#                     m = []
#                     for j in sites:
#                         if j in v:
#                             m.extend(v[j])
#                         else:
#                             m.append('X')
#                     phosSum2.loc[i, 'Motif'] = ', '.join(m)
#                 except:
#                     pass
    
#     # Assign motifs from PhosphoSitePlus
#     #print(f"      Mapping motifs from PhosphoSitePlus...")
#     phosSum3 = phosSum2.fillna('X')
#     for f, g in phosSiteDict.items():
#         for i in range(len(phosSum3)):
#             if f == phosSum3['Accessions'].iloc[i] and 'X' in str(phosSum3['Motif'].iloc[i]):
#                 try:
#                     sites = str(phosSum3['Site'].iloc[i]).split(', ')
#                     motifs = str(phosSum3['Motif'].iloc[i]).split(', ')
#                     x = []
#                     for n, motif_part in zip(sites, motifs):
#                         if 'X' in motif_part and n in g:
#                             x.extend(g[n])
#                         else:
#                             x.append(motif_part)
#                     phosSum3.loc[i, 'Motif'] = ', '.join(x)
#                 except:
#                     pass
    
#     # Final fallback - use peptide sequence if still unknown
#     phosSum4 = phosSum3.copy()
#     for i in range(len(phosSum4)):
#         if 'X' in str(phosSum4['Motif'].iloc[i]):
#             try:
#                 motifs = str(phosSum4['Motif'].iloc[i]).split(', ')
#                 x = []
#                 for motif_part in motifs:
#                     if 'X' in motif_part:
#                         x.append(str(phosSum4['Peptide Sequence'].iloc[i]))
#                     else:
#                         x.append(motif_part)
#                 phosSum4.loc[i, 'Motif'] = ', '.join(x)
#             except:
#                 pass
    
#     phosSum4 = phosSum4.drop(['Peptide Sequence', 'ptmRS', 'ID'], axis=1, errors='ignore')
#     phosSum4 = phosSum4.groupby(['Gene', 'Accessions', 'Site', 'Motif']).agg('sum', numeric_only=True)
    
#     # # Save motif file
#     # motif_file = f"{exp_type}_sum_motif.csv"
#     # phosSum4.to_csv(motif_file)

#     motif_file = os.path.join(out_dir, f"{exp_type}_sum_motif.csv") if out_dir else f"{exp_type}_sum_motif.csv"
#     phosSum4.to_csv(motif_file)
#     #print(f"      Saved: {motif_file}")
    
#     # Final summary without motif (for main analysis)
#     phosSum5 = (
#         phosSum4.reset_index()
#         .drop('Motif', axis=1)
#         .set_index(['Gene', 'Accessions', 'Site'])
#         .groupby(['Gene', 'Accessions', 'Site'])
#         .agg('sum', numeric_only=True)
#     )
    
#     #print(f"      Final phosphosites: {len(phosSum5)}")
#     return phosSum5

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
    
    out_prefix = index.replace("_sup", "")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Build file paths
    psms_path = os.path.join(raw_dir, dataDF.loc[index, cfg["meta_cols"]["psms"]])
    mgf_path = os.path.join(raw_dir, dataDF.loc[index, cfg["meta_cols"]["mgf"]])
    lib_path = os.path.join(lib_dir, dataDF.loc[index, cfg["meta_cols"]["library"]])
    pepts_path = os.path.join(raw_dir, dataDF.loc[index, cfg["meta_cols"]["pepts"]])
    mods_path = os.path.join(raw_dir, dataDF.loc[index, cfg["meta_cols"]["mods"]])
    sup_corr_path = os.path.join(out_dir, dataDF.loc[index, cfg["meta_cols"]["sup_corr"]])


    phos_site_path = cfg.get("phos_site_csv", "Phosphosite Motifs_M.csv")


    with Halo(spinner="dots", color="cyan") as sp:
        # Load input files
        sp.text = f"  {out_prefix} — reading files"
        #print(f"    Reading PSMs: {os.path.basename(psms_path)}")
        psms = pd.read_csv(psms_path, sep="\t")
        #print(f"    Reading MGF: {os.path.basename(mgf_path)}")
        mgf_dict = mgf.read(mgf_path)
        #print(f"    Reading library: {os.path.basename(lib_path)}")
        libs = pd.read_csv(lib_path).set_index("headers").to_dict()["names"]
        #print(f"    Reading peptides: {os.path.basename(pepts_path)}")
        pepts = pd.read_csv(pepts_path, sep="\t")
        #print(f"    Reading mods: {os.path.basename(mods_path)}")
        mods = pd.read_csv(mods_path, sep='\t')
        #print(f"    Reading sup corrections: {sup_corr_path}")
        corrSum = pd.read_csv(sup_corr_path)
        #print(f"    Reading PhosphoSitePlus: {os.path.basename(phos_site_path)}")
        phos_site = pd.read_csv(phos_site_path, sep=',')
        
        libsDict[index] = libs
        
        # Filter and impute PSMs
        #print(f"    Filtering + imputation...")
        sp.text = f"  {out_prefix} — filtering..."
        PSMdf = nan_imputation(PSM_filter(psms, libs, cfg), mgf_dict, cfg)
        #print(len(PSMdf))
        
        # Phospho-specific filtering
        PSMdf, psm_start, psm_end = phos_filter(PSMdf, index, cfg)
        
        # Summarize to phosphosite level with motifs
        sp.text = f"  {out_prefix} — motif mapping..."
        sumPSMdf = sum_psms_phos(PSMdf, pepts, mods, phos_site, out_prefix, cfg, out_dir=out_dir)
        sumPSM[index] = sumPSMdf
        
        # Apply supernatant corrections
        sp.text = f"  {out_prefix} — corrections"
        corrPhos = sumPSMdf.copy()
        for col in corrPhos.columns:
            if col in corrSum.columns:
                corrPhos[col] = corrPhos[col] / corrSum[col][0]
        
        # Rename columns using library mapping
        for x in corrPhos.columns:
            if x in libs.keys():
                corrPhos = corrPhos.rename(columns=libs)
        
        corrDFs[index] = corrPhos
        
        # Save output
        sp.text = f"  {out_prefix} — saving"
        corr_phos_path = os.path.join(out_dir, f"{out_prefix}_phos_corr.csv")
        corrPhos.to_csv(corr_phos_path)
        # print(f"      {index} — {psm_start} PSMs → {psm_end} after filters → {len(sumPSMdf)} sites")
        # print(f"Saved: {os.path.basename(corr_phos_path)}")
        #print(f"    ✓ Done: {index}")

        sp.succeed(f"{out_prefix} — {psm_start} PSMs → {psm_end} after filters → {len(sumPSMdf)} sites | saved: {corr_phos_path}")


# PIPELINE EXECUTION
def run_phos_pipeline(cfg=None, exp_types=None):
    """Run the phos pipeline for all or selected experiment types."""
    if cfg is None:
        cfg = load_config()
    _reset_run_state()

    print(f"\n[phos]")
    
    dataDF = pd.read_csv(cfg["meta_phos_csv"], sep=",").set_index(cfg["meta_index"])
    indices = dataDF.index if exp_types is None else exp_types
    
    #print(f"  Found {len(indices)} dataset(s) to process")
    
    for index in indices:
        process_phos_dataset(index, dataDF, cfg)
    
    print(f"\n[phos] done — all written to:", cfg["out_dir"])


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point for testing."""
    run_phos_pipeline()


if __name__ == "__main__":
    main()
