# DLAN article experiment reproduction package

## Directory structure and experiment mapping

```text
scripts/
├── README.md
├── REPRODUCE_FINDINGS.md      # Entry points for each finding and current manuscript figure
├── requirements.txt
├── environment_snapshot.json
├── experiment_manifest.json  # Configurations, historical sources, parameters and file SHA256 hashes
├── data/
│   ├── CE/                   # CO2 data and README
│   ├── SIC/                  # WuhanOD
│   ├── SHP/                  # WuhanHP
│   ├── GVI/                  # GVI, including month-augmented inputs
│   └── historical/           # Separately labelled records from earlier protocols
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

All paths are resolved relative to this directory. Models and data are loaded locally; the repository-root `models/`, `share/` and `experiments/` directories are no longer runtime dependencies. Original paths are retained as provenance. Each experiment follows **Read data → Model experiments (DLAN, GeoAggregator, GNNWR, GCNNWR) → Results summary**, with experiment code written directly in the notebook.

## Datasets

| Package name | Original case | Observations | Covariates | Coordinate dimensions | Training / validation / test |
| ------------ | ------------- | -----------: | ---------: | --------------------: | ---------------------------- |
| CE           | CO2           |       13,000 |          8 |                     3 | 9,732 / 1,643 / 1,625        |
| SIC          | WuhanOD       |       13,807 |         35 |                     4 | 9,618 / 2,104 / 2,085        |
| SHP          | WuhanHP       |       19,599 |         17 |                     2 | 15,475 / 2,196 / 1,928       |
| GVI          | GVI           |        2,122 |        176 |                     2 | 1,696 / 213 / 213            |

See `data/<case>/README.md` for case-specific details. CE predicts future years; SHP holds out entire residential communities; GVI uses a fixed spatial-group split without a buffer and regression on frozen features. SIC is trained on log1p(count), with metrics calculated after recovering counts. All coordinate dimensions are retained: CE uses GTNNWR and SIC uses four-dimensional GNNWR. The two-dimensional GCNNWR implementation is inapplicable to CE/SIC, and those sections explicitly record that it is not run.

## Dependencies

Use Python 3.12 with NumPy, pandas, PyTorch, scikit-learn, JupyterLab, ipykernel, nbformat, nbclient and the author-published `gnnwr==0.1.17`. Install PyTorch compatible with the CUDA toolkit and driver in a separate environment before installing the remaining dependencies.

```bash
cd article/scripts
python -m pip install -r requirements.txt
python -m ipykernel install --user --name article-dlan --display-name "Python (DLAN article)"
jupyter lab .
```

If the published package's dependency constraints conflict with the verified PyTorch environment, install the other dependencies first, then run `python -m pip install --no-deps gnnwr==0.1.17`. The adapter uses only the distribution's `networks.py`; it does not invoke its trainers, plotting or diagnostic modules. The network source is not duplicated in this package. See `models/PROVENANCE.json` for provenance and consistency checks.

The current R plotting notebook loads tidyverse (including ggplot2/dplyr/tidyr), patchwork, sf, tmap, ggpubr and grid, and also uses viridisLite/scales. IRkernel is required. tmap uses the version 4 API; sf requires working GDAL/GEOS/PROJ installations. The original statistical plotting environment is recorded in `../evidence/R_sessionInfo.txt`; consult the notebook's setup cells for additional current dependencies. An existing R kernel can run `plot_results.ipynb` directly. Reproducing the architecture artwork additionally requires Pillow, Draw.io Desktop CLI and Inkscape; see `REPRODUCE_FINDINGS.md`. Image encoders are not trained, and TensorFlow is not required.

## Default workflow: audit saved results and reconstruct evidence

Run `CE.ipynb`, `SIC.ipynb`, `SHP.ipynb` and `GVI.ipynb` in that order, each with a fresh Python kernel, or execute the following commands from this directory:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=article-dlan CE.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=article-dlan SIC.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=article-dlan SHP.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=article-dlan GVI.ipynb
```

The default `ARTICLE_MODE=replay` checks SHA256 hashes, row identifiers, splits, reference banks, checkpoints and existing checkpoint-reload audits. It then recomputes validation/test RMSE, MAE, R², tail errors, community errors and scale parameters from the saved artefacts. This audits existing results without retraining; recomputing metrics is distinct from reproducing model training.

The last case to complete merges fragments from all four cases into `outputs/replay/evidence/`. This includes all eight input CSV files currently read by `plot_results.ipynb`, together with the main table, supplementary comparisons and the SIC 3×3 sensitivity summary. Use separate RUN_ID values for replay and retraining. Fragments with different modes, source manifests or seed sets cannot be combined.

## Training from scratch

Set the following variables before starting a new kernel:

```bash
export ARTICLE_MODE=train
export ARTICLE_RUN_ID=retrain-v1
```

Then run the four notebooks in the same order. This mode trains the main comparison and the historical/supplementary configurations required by the summary figures. It is computationally intensive and defaults to ten seeds per configuration. The nine-cell SIC factorial experiment retains prediction head `(10,8)`; it is distinct from the earlier width-10 configuration with head `(36,3)`. Each fit saves its configuration, training history, predictions, checkpoint and reload audit. Completion metrics are written only after all audits pass.

Shared training settings are AdamW, learning rate 0.001, weight decay 0.0001, effective batch size 1,024, query microbatch size 64, at most 300 epochs, validation-loss early stopping with patience 30 and min_delta 1e-5, gradient clipping at 5, and ReduceLROnPlateau with factor 0.5 and patience 10. Parameter budgets and candidate counts are set separately for each configuration in the manifest. The main SIC comparison uses 7,069 parameters and DLAN K512; the other cases use 6,957 parameters.

A GPU is recommended for full training; the default audit can run on a CPU. Training software, randomisation, hardware and original timings are recorded. The original SIC K512/K2048 experiments used different execution environments, so their recorded times are not a comparison on identical hardware. New fits record their own timings rather than reusing historical values.

## Publishing results to the article evidence directory

The default `ARTICLE_PUBLISH_EVIDENCE=0` leaves `../evidence/` unchanged. After inspecting `outputs/<RUN_ID>/evidence/`, set `PUBLISH_EVIDENCE=True` in the final case's preparation cell and rerun only the publication cell at the end of Results summary. Alternatively, set `ARTICLE_PUBLISH_EVIDENCE=1` before starting the kernel and rerun. Existing evidence is backed up under the run directory before replacement.

Publication neither executes nor edits `plot_results.ipynb`, and its saved outputs remain intact. Explicitly execute the plotting notebook when figure updates are needed. See `REPRODUCE_FINDINGS.md` for the steps corresponding to each current manuscript figure and reported finding.

## Reproduction scope

The package contains frozen artefacts for 27 distinct configurations and 270 completed fits; the 14 main configurations account for 140 fits. Repeated references to K512, capacity or factorial experiments are deduplicated by source path, while main and historical comparisons remain separately summarised. The historical Gaussian-prior CSV is retained separately. It is neither a module ablation of the final DLAN nor retrained by these four notebooks.

Model organisation preserves the original numerical operations and state-dict keys, with local changes to filenames and imports. The published-library GNNWR adapter uses fixed training-reference landmarks under the parameter budget; it does not reproduce the author's default full-reference configuration.

The packaging step left `main.tex`, existing `article/evidence/`, images and `plot_results.ipynb` unchanged.

## Recorded execution issues in the existing plotting notebook

An execution copy run in a fresh R kernel with reconstructed evidence exposed the following pre-existing issues:

| Cell (1-based) | Issue                                                            | Effect                                                                                                                                                 |
| -------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 12             | `mutate()` refers to `accuracy_long$metric` before it is created | The main comparison plotting expression fails; the subsequent `ggsave()` may save an earlier plot, so file existence alone does not establish success. |
| 16             | `p` is used before it is defined                                 | The computation-time comparison export fails.                                                                                                          |
| 31             | `glimpse(od)` precedes the cell that creates `od`                | Execution fails in a fresh kernel; the variable is created in a later cell.                                                                            |

These are variable-reference/execution-order issues in the plotting code, rather than differences in the input CSV metrics. The packaging step preserved `plot_results.ipynb` and its saved results. `outputs/replay/plot_compatibility.json` records the execution-copy diagnostics. Successful exports of some figures do not establish that the entire notebook runs successfully.

Standard deviations in the SIC nine-cell sensitivity results are recomputed from the ten seeds in each cell. Some width-10/16 standard deviations in the current LaTeX table differ from these calculations. Reconstructed results are in `outputs/replay/evidence/sic_factorial_summary.csv`; the manuscript table was not changed during packaging.

The original `prepare_evidence.py`, `prepare_supplement_evidence.py` and `prepare_reported_results.py` remain as historical entry points for the earlier repository layout. The new reproduction workflow does not call them. `render_references.py` checks references and is outside the model experiment workflow.

## Data visualisation (inline notebook display only)

Run `data_visualisation.ipynb` from the top with an R kernel. CE uses ggcube for a space–time cube, SIC uses tmap for observed OD flows, and GVI uses tmap for Shanghai observation points; SHP is omitted. Dependencies are tidyverse (including ggplot2, dplyr, tidyr, purrr and stringr), ggcube (verified with 0.2.0), sf, tmap (verified with 4.3), viridisLite and IRkernel.

Data reading, transformation and spatial-object construction use tidyverse `%>%` pipelines: `crossing()` enumerates cases and arrays, `map2()` reads NPZ arrays and pairs coordinates with responses, `pivot_wider()`, `unnest()`, `filter()` and `transmute()` organise the observation table, and `pmap()` constructs OD line geometries. Low-level float64 decoding uses base R binary reading and preserves NumPy's C/Fortran storage order. Python, reticulate and intermediate CSV exports are unnecessary. This is in-memory pipeline processing; each case's arrays are still read into memory. SIC first looks for `data/OD-Wuhan/wuhan_subdis.gpkg` under the repository root, then falls back to the packaged contextual basemap. GVI's Shanghai boundary comes from geoBoundaries CHN ADM1 (2019, frozen release `9469f09`) and is embedded in the notebook for offline use. Both tmap maps use fixed response classes, a north arrow and a kilometre scale bar.

Results are displayed and retained only in notebook outputs. The notebook does not call ggsave, tmap_save or other figure/data export functions, and it leaves manuscript figures, experiment data, evidence and `plot_results.ipynb` unchanged.
