# tmt-proteomics-pipeline
A command-line pipeline for processing raw DDA TMT mass spectrometry proteomics and phosphoproteomics data from Proteome Discoverer search results.

**Author:** Aakesh Yoganathan — Tamir Lab, UNC Chapel Hill


## Setup
**Clone the repository:**
```bash
git clone https://github.com/aakesh194/tmt-proteomics-pipeline.git
cd tmt-proteomics-pipeline
```

**Create and activate conda environment:**
```bash
conda create -n omics
conda activate omics
conda install pip
pip install -r requirements.txt
```

**Add your data files:**
- Raw files (PSMs, MGF, peptide groups, mods) → `data/raw-files/`
- Library CSVs → `data/library-files/`
- Metadata CSVs → `data/`
- PhosphoSitePlus motif file → `data/`

**Edit `config.yaml` to match your experiment — at minimum update the metadata file paths — then run:**
```bash
python run.py
```

## Usage
```bash
python run.py                  # full pipeline
python run.py --gprot-only     # global proteomics only
python run.py --phos-only      # phosphoproteomics only
python run.py --help           # for more help
```


## Outputs

All files are written to `outputs/`. After bridging outputs go to `outputs/after-bridging/`.

| File | Description |
|------|-------------|
| `{exp}_gProt_corr.csv` | Corrected protein abundances |
| `{exp}_pepts_corr.csv` | Corrected peptide abundances |
| `{exp}_sup_Corrections.csv` | Supernatant correction factors |
| `{exp}_gProt_corr_post_pool_bridge.csv` | Pool-bridge normalized protein abundances |
| `{exp}_QC_heatmap_post_pool_bridge.png` | QC heatmap after pool-bridge normalization — top 300 most variable proteins, z-scored. |
| `{exp}_phos_corr.csv` | Corrected phosphosite abundances |
| `{exp}_sum_motif.csv` | Phosphosites with motif annotations |
| `After bridging/` | Multi-run bridge-corrected combined outputs |
| `After bridging/gProt_QC_heatmap_bridged.png` | Combined QC heatmap across all runs after multi-run bridge correction |
---