# DLAN manuscript for IJGIS

The manuscript is `main.tex`; all prose, tables and glossary definitions are in that file.
References use the user-supplied `zotero.bib` and `tfcad.bst`.
British English, the Interact class and sentence/clause-level source lines are retained.

Build in this directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=_out main.tex
```

All compiler outputs go to `_out/`.
The desktop editor's Tectonic preview may be unavailable when its bundle cannot be downloaded; the delivered PDF is built with local pdfLaTeX.

## Completed main experiment, 8 November 2026

The main results use only `dlan_matched4_10seeds_v1`: 100 completed fits, ten seeds for each supported configuration and exactly 6,957 trainable parameters per regressor.

- CO2: DLAN and GeoAggregator, three-dimensional coordinates, future-year test.
- WuhanOD: DLAN and GeoAggregator, four endpoint coordinates, held-out unordered endpoint pairs.
- WuhanHP: DLAN, GeoAggregator and GCNNWR, two-dimensional coordinates, community-disjoint test.
- GVI: the same three models, 176 fixed features and two-dimensional coordinates, inherited 1-km spatial groups.

DLAN has lower mean test RMSE on CO2, WuhanHP and GVI; GeoAggregator is better on WuhanOD.
DLAN-P is explained as a possible isotropic restriction in the methodology; the main experiment retains coordinate-specific DLAN even for planar cases.
GNNWR is excluded from this experiment because TensorFlow is unavailable; GCNNWR supports only the planar cases.
Historical baseline scores are not substituted for these excluded cells.
The separately prepared GVI three-coordinate experiment has no formal completed result and is not included.
Final-model module-isolating ablations remain future work; the historical three-seed joint-attention Gaussian ablation is explicitly separate.

## Reproduce the analysis

```sh
python scripts/prepare_evidence.py
Rscript scripts/plot_results.R
python scripts/render_references.py
```

The evidence script verifies saved row IDs, responses, metrics, configuration/data/source hashes and recorded prediction-reload audits.
It checks 200 held-out prediction arrays and reads 100 checkpoints without training.
The GVI OLS diagnostic is recovered from already fitted GCNNWR coefficient buffers; it does not fit an additional model.
All current primary evidence tables use the `matched_` filename prefix, plus `gvi_ols_buffer.csv`.
Earlier evidence files remain historical records and are not pooled with the main comparison.
The plotting script generates eleven figures, including the architecture, from current evidence and the explicitly labelled historical Gaussian ablation.
`render_references.py` only checks keys; it never reinserts inline bibliography entries.
No source datasets are copied into the article directory.
