# tmt-proteomics-pipeline
A command-line pipeline for processing raw DDA TMT mass spectrometry proteomics and phosphoproteomics data from Proteome Discoverer search results.


## Setup
**Clone the repository:**
```bash
git clone https://github.com/tamirlab-unc/tmt-proteomics-pipeline
cd tmt-proteomics-pipeline
```

**Create and activate conda environment:**
```bash
conda create -n omics
conda activate omics
conda install pip
pip install -r requirements.txt
```
_If you don’t have conda installed, install [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (recommended) or use your own Python environment._

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

See `outputs/README.md` for the current folder structure and file descriptions.


<br>
<br>

---

Tamir Lab, UNC-Chapel Hill
