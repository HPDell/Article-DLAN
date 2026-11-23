# Data and Code for Article "Dimension-specific local attention for nonlinear regression in geographical and temporal spaces"

This repository stores all code and data for the article *Dimension-specific local attention for nonlinear regression in geographical and temporal spaces*. Four experiment are included (see [Datasets](#Datasets) below). 

Each experiment follows the same workflow:

1. Read data and train models: DLAN, GNNWR/GTNNWR, GCNNWR, GeoAggregator (**each in a separate Jupyter Notebook**)
2. Visualise results (**all case in a single Jupyter Notebook**)

Run all notebooks and Python entry points from this directory (`article/scripts/`). Paths are relative to the current working directory; the code does not search for the project or notebook directory.

## Directory structure and experiment mapping

```text
repository/
├── README.md
├── REPRODUCE_FINDINGS.md      # Entry points for each finding and current manuscript figure
├── requirements.txt
├── environment_snapshot.json
├── experiment_manifest.json  # Configurations, historical sources, parameters and file SHA256 hashes
├── data/
│   ├── basemap/              # Shared Wuhan and Shanghai boundary layers
│   ├── CE/                   # CE case, including CO2 data
│   ├── SIC/                  # SIC case, including origin-destination (OD) flow counts in Wuhan
│   ├── SHP/                  # SHP case, including secondhand housing price in Wuhan
│   ├── GVI/                  # GVI, including street-view images and related datasets on predicting green view index
│   └── historical/           # Separately labelled records from earlier protocols
├── evidence/                 # Saved metrics, predictions, provenance and figure inputs
├── models/                   # DLAN, GeoAggregator, GCNNWR and the published GNNWR adapter
├── CE.ipynb
├── SIC.ipynb
├── SHP.ipynb
├── GVI.ipynb
├── plot_results.ipynb        # Existing R plotting notebook; retained without changes
└── outputs/
    ├── reference/            # Copies of completed-run configurations, predictions, checkpoints, audits and histories
    └── <RUN_ID>/             # Replay/retraining results and evidence staged for publication
```

## Datasets

| Case | Dataset Name | Observations | Covariates | Coordinate dimensions | Training / validation / test |
| ---- | ------------ | -----------: | ---------: | --------------------: | ---------------------------- |
| CE   | CO2          |       13,000 |          8 |                     3 | 9,732 / 1,643 / 1,625        |
| SIC  | WuhanOD      |       13,807 |         35 |                     4 | 9,618 / 2,104 / 2,085        |
| SHP  | WuhanHP      |       19,599 |         17 |                     2 | 15,475 / 2,196 / 1,928       |
| GVI  | GVI          |        2,122 |        176 |                     2 | 1,696 / 213 / 213            |

See `data/<case>/README.md` for case-specific details. 

* CE predicts future years
* SHP holds out entire residential communities
* GVI uses a fixed spatial-group split without a buffer and regression on frozen features
* SIC is trained on $ln(\text{count}+ 1)$, with metrics calculated after recovering counts.

All coordinate dimensions are retained: CE uses GTNNWR and SIC uses four-dimensional GNNWR. 

**Note: The two-dimensional GCNNWR implementation is inapplicable to CE/SIC, and those sections explicitly record that it is not run.**

## Dependencies

Use Python 3.12 with **NumPy**, **pandas**, **PyTorch**, **scikit-lear**n, **JupyterLab**, **ipykernel**, **nbformat**, **nbclient** and the author-published `gnnwr==0.1.17`. Install PyTorch compatible with the CUDA toolkit and driver in a separate environment before installing the remaining dependencies.

```bash
cd article/scripts
python -m pip install -r requirements.txt
python -m ipykernel install --user --name article-dlan --display-name "Python (DLAN article)"
jupyter lab .
```

If the published package's dependency constraints conflict with the verified **PyTorch** environment, install the other dependencies first, then run `python -m pip install --no-deps gnnwr==0.1.17`. The adapter uses only the distribution's `networks.py`; it does not invoke its trainers, plotting or diagnostic modules. The network source is not duplicated in this package. See `models/PROVENANCE.json` for provenance and consistency checks.

The current R plotting notebook loads **tidyverse**, **patchwork**, **sf**, **tmap**, **ggpubr**, **ggcube** and **grid**, and also uses viridisLite/scales. **IRkernel** is required. 

Additional requrirements:

- **tmap** uses the version 4 API
- **sf** requires working GDAL/GEOS/PROJ installations.

The original statistical plotting environment is recorded in `evidence/R_sessionInfo.txt`; consult the notebook's setup cells for additional current dependencies. 

An existing R kernel can run `plot_results.ipynb` directly. 

## Workflow

### Train models

Run `CE.ipynb`, `SIC.ipynb`, `SHP.ipynb` and `GVI.ipynb` in that order, each with a fresh Python kernel, or execute the following commands from this directory:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=article-dlan CE.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=article-dlan SIC.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=article-dlan SHP.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=article-dlan GVI.ipynb
```

### Data visualisation (inline notebook display only)

Run `plot_results.ipynb` from the top with an R kernel to reproduce figures in the article.

## Step-by-step reproduction of reported findings

1. Install the Python and R dependencies in `README.md`.
2. Start fresh Python kernels and run `CE.ipynb`, `SIC.ipynb`, `SHP.ipynb`, then `GVI.ipynb` from top to bottom.
3. The final completed case creates the complete figure-input bundle in `outputs/replay/evidence/`. Each case saves its own fragments and completion manifest first.
4. Inspect the generated tables. To publish them into `evidence/`, set `ARTICLE_PUBLISH_EVIDENCE=1` and rerun the final result-summary cells only. Existing evidence is backed up under the run directory.
5. Open `plot_results.ipynb` with the R kernel and run the relevant figure cell after its setup/data cells. To retain its current saved outputs, make an execution copy or use nbconvert with a separate output notebook; set `ARTICLE_FIGURE_DIR` to a temporary directory to retain existing exported images.

### Figures

| Figure | Experiment / source                              | Plot notebook cell (1-based) |
| ------ | ------------------------------------------------ | ---------------------------- |
| 3      | `reported_metrics.csv`                           | 13                           |
| 4      | `reported_bandwidths.csv`                        | 20                           |
| 5      | `reported_predictions.csv`                       | 23                           |
| 6      | `reported_metrics.csv; reported_predictions.csv` | 36                           |
| 7      | `matched_hp_groups.csv`                          | 29                           |
| 8      | `reported_predictions.csv`                       | 44                           |

### Tables and other findings

| Finding                                 | Experiment                        | Generated output                                         |
| --------------------------------------- | --------------------------------- | -------------------------------------------------------- |
| Data dimensions and splits              | All four                          | `data/<case>/contract.json` and displayed data checks    |
| Main accuracy and seed variability      | All four                          | `reported_metrics.csv`, `reported_summary.csv`           |
| WuhanOD 3×3 width/candidate sensitivity | `SIC.ipynb` (DLAN factorial jobs) | `sic_factorial_metrics.csv`, `sic_factorial_summary.csv` |
| Training cost / epochs                  | All four                          | `reported_metrics.csv`, `reported_summary.csv`           |
| WuhanHP equal-community accuracy        | `SHP.ipynb`                       | `matched_hp_groups.csv`, `reported_metrics.csv`          |

