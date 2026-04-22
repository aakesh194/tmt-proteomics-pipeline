# outputs

All pipeline outputs are organized by experiment and pipeline type.


## Per-experiment outputs

Each experiment gets its own folder named `{expType}/gprot/` or `{expType}/phos/`. Files are numbered in processing order.

### gprot (`{exp}/gprot/`)

| File | Description |
|------|-------------|
| `01_psm_filtered.csv` | PSMs after quality filtering |
| `02_psm_imputed.csv` | PSMs after missing value imputation |
| `03_peps.csv` | Peptide-level summed data |
| `04_sum_peps.csv` | Proteins summed from peptides |
| `05_corr_factors.csv` | Supernatant correction factors |
| `06_corr_pept.csv` | Corrected peptide abundances |
| `06_corr_prot.csv` | Corrected protein abundances |
| `07_post_pool_bridge.csv` | Pool-bridge normalized protein abundances |
| `07_QC_heatmap_post_pool_bridge.png` | QC heatmap — top variable proteins, z-scored. Biological groups should cluster together. |

### phos (`{exp}/phos/`)

| File | Description |
|------|-------------|
| `01_psm_filtered.csv` | PSMs after quality filtering |
| `02_psm_imputed.csv` | PSMs after missing value imputation |
| `03_phos_filtered.csv` | PSMs after phospho-specific filtering |
| `04_sum_psms_phos.csv` | Phosphosites summed from PSMs |
| `05_corr_factors.csv` | Supernatant correction factors (from gprot) |
| `06_corr_phos.csv` | Corrected phosphosite abundances |
| `07_post_pool_bridge.csv` | Pool-bridge normalized phosphosite abundances |
| `07_QC_heatmap_post_pool_bridge.png` | QC heatmap — top variable phosphosites, z-scored |
| `{exp}_sum_motif.csv` | Phosphosites with motif annotations |


## after-bridging/

Multi-run bridge-corrected combined outputs. Only generated when processing multiple runs. Split into `gprot/` and `phos/` subfolders.

| File | Description |
|------|-------------|
| `{pipeline}_runCorr_DF_u.csv` | Run-corrected, all sites |
| `{pipeline}_runCorr_DF_n.csv` | Run-corrected, no missing values |
| `{pipeline}_brgA_u.csv` | Bridge A corrected, all sites |
| `{pipeline}_brgA_n.csv` | Bridge A corrected, no missing values |
| `{pipeline}_run_corr_mc_*.csv` | Mean-centered run-corrected |
| `{pipeline}_run_corr_zs_*.csv` | Z-scored run-corrected |
| `{pipeline}_raw_mc_DF_*.csv` | Raw mean-centered |
| `{pipeline}_raw_zscore_DF_*.csv` | Raw z-scored |
| `gProt_QC_heatmap_bridged.png` | Combined gprot QC heatmap across all runs |
| `phos_QC_heatmap_bridged.png` | Combined phos QC heatmap across all runs |