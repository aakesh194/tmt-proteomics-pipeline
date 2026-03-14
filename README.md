# omics-pipeline

A command-line pipeline for processing raw DDA TMT mass spectrometry proteomics and phosphoproteomics data from Proteome Discoverer search results.

**Author:** Aakesh Yoganathan — Tamir Lab, UNC Chapel Hill

---

## Setup
```bash
git clone https://github.com/aakesh194/omics-pipeline.git
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Add your data files:
- Raw files (PSMs, MGF, peptide groups, mods) → `data/raw-files/`
- Library CSVs → `data/library-files/`
- Metadata CSVs → `data/`
- PhosphoSitePlus motif file → `data/`

Edit `config.yaml` to match your experiment — at minimum update the metadata file paths — then run:
```bash
python run.py
```

---

## Usage
```bash
python run.py                  # full pipeline
python run.py --gprot-only     # global proteomics only
python run.py --phos-only      # phosphoproteomics only
python run.py --help           # for more help
```

---

## Outputs

All files are written to `outputs/`. After bridging outputs go to `outputs/After bridging/`.

| File | Description |
|------|-------------|
| `{exp}_gProt_corr.csv` | Corrected protein abundances |
| `{exp}_pepts_corr.csv` | Corrected peptide abundances |
| `{exp}_sup_Corrections.csv` | Supernatant correction factors |
| `{exp}_phos_corr.csv` | Corrected phosphosite abundances |
| `{exp}_sum_motif.csv` | Phosphosites with motif annotations |
| `After bridging/` | Multi-run bridge-corrected combined outputs |