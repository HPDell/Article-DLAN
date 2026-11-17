# Step-by-step reproduction of reported findings

1. Install the Python and R dependencies in `README.md`.
2. Start fresh Python kernels and run `CE.ipynb`, `SIC.ipynb`, `SHP.ipynb`, then `GVI.ipynb` from top to bottom. The default mode re-audits the supplied saved predictions/checkpoints; it does not fit models.
3. The final completed case creates the complete figure-input bundle in `outputs/replay/evidence/`. Each case saves its own fragments and completion manifest first. The merge rejects missing cases, different seed sets and different bundle manifests.
4. Inspect the generated tables. To publish them into `article/evidence/`, set `ARTICLE_PUBLISH_EVIDENCE=1` and rerun the final result-summary cells only. Existing evidence is backed up under the run directory. This step is deliberately off by default.
5. Open `plot_results.ipynb` with the R kernel and run the relevant figure cell after its setup/data cells. To retain its current saved outputs, make an execution copy or use nbconvert with a separate output notebook; set `ARTICLE_FIGURE_DIR` to a temporary directory to retain existing exported images.
6. Full retraining is separate: set `ARTICLE_MODE=train`, select a new `ARTICLE_RUN_ID`, restart the kernels and run all four notebooks. Existing recorded fit times are audit data; new runs measure new times. No cross-machine bitwise identity is promised.

## Every figure currently referenced by main.tex

| Figure | Artwork | Experiment / source | Plot notebook cell (1-based) | Interpretation |
|---|---|---|---|---|
| 1 | `dlan_architecture.pdf` | `../figures/dlan_design/` builders and editable Draw.io files | Not a statistical plot cell | Current architecture diagram; do not substitute the historical R architecture cell. |
| 2 | `matched_relative_accuracy.pdf` | `all four case notebooks` → `reported_metrics.csv` | 13 | Ten-seed RMSE/MAE distributions |
| 3 | `matched_bandwidths.pdf` | `all four case notebooks` → `reported_bandwidths.csv` | 20 | Effective positive bandwidths from main DLAN checkpoints |
| 4 | `matched_co2_predictions.pdf` | `CE.ipynb` → `reported_predictions.csv` | 23 | Fixed seed 1, test year 2021 |
| 5 | `matched_od_tail.pdf` | `SIC.ipynb` → `reported_metrics.csv; reported_predictions.csv` | 36 | K512 main group, top-1% SSE share and seed-1 predictions |
| 6 | `matched_hp_groups.pdf` | `SHP.ipynb` → `matched_hp_groups.csv` | 29 | Ten-seed held-out-community errors and group sizes |
| 7 | `matched_gvi_response.pdf` | `GVI.ipynb` → `reported_predictions.csv` | 44 | Four planar models plus recovered OLS buffer; descriptive response quintiles |

## Tables and other findings

| Finding | Experiment | Generated output |
|---|---|---|
| Data dimensions and splits | All four, Read data section | `data/<case>/contract.json` and displayed data checks |
| Main accuracy and seed variability | All four, main jobs only | `reported_metrics.csv`, `reported_summary.csv` |
| WuhanOD 3×3 width/candidate sensitivity | `SIC.ipynb`, DLAN factorial jobs | `sic_factorial_metrics.csv`, `sic_factorial_summary.csv` |
| Training cost / epochs | All four; saved or newly measured timing | `reported_metrics.csv`, `reported_summary.csv` |
| WuhanHP equal-community accuracy | `SHP.ipynb` | `matched_hp_groups.csv`, `reported_metrics.csv` |
| GVI global linear reference | `GVI.ipynb`, GCNNWR fitted OLS buffers | `gvi_ols_buffer.csv`, OLS rows in `reported_predictions.csv` |

## Retained plotting cells outside the current manuscript

`plot_results.ipynb` currently exports thirteen named statistical figures. The main manuscript currently cites only the artwork listed above. The older seed plot, cost plot, capacity comparison, width/support plot, month-augmented GVI plot, analytical retrieval plot and historical Gaussian diagnostic remain reproducible from the staging bundle without being reclassified as current manuscript figures.

The 9-cell SIC factorial summary is computed from all 90 saved test runs. Width-10 factorial uses head `(10,8)` and 5,541 parameters; it is distinct from archived width-10/head `(36,3)` with 6,957 parameters. Standard deviations and timing are recomputed from their own configurations, not copied between widths. This package does not edit the manually maintained LaTeX sensitivity table.

Current architecture reproduction follows `../figures/dlan_design/README.md`: run its Python builders, export with Draw.io Desktop, and prepare print artwork with Inkscape. These are drawing tools, not model dependencies. The current plotting notebook does not generate the architecture artwork; use the editable Draw.io sources instead.

Historical Gaussian ablation is a frozen separate-protocol record in `data/historical/`; the new notebooks do not claim to retrain it. No final-model FiLM or response-path ablation is invented by this packaging.

## Existing notebook execution caveat

The untouched plotting notebook currently fails in a fresh kernel at cells 12 (`accuracy_long`), 16 (`p`) and 31 (`od`). These diagnostics were reproduced using an in-memory copy and temporary export paths; see `README.md`. Evidence compatibility was verified numerically, but the complete current plotting notebook is not marked execution-clean. Fixing it is a separate step; this package has preserved its source and outputs.
