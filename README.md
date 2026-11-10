# DLAN manuscript for IJGIS

`main.tex` contains the manuscript prose, tables and glossaries definitions.
References use the user-supplied `zotero.bib`; the class, author metadata, citation macros and double spacing are preserved.
New and revised sentences and longer clauses start on separate source lines.

Build from `article/`:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=_out main.tex
```

All compiler products are written to `_out/`.
The native editor is open; its preview compiler cannot download the uncached Tectonic bundle, so the delivered PDF uses local pdfLaTeX.

## Evidence incorporated on 10 November 2026

- Main comparison: 140 fits, fourteen configurations, ten seeds each, 6,957 trainable parameters. Original matched-case results are extended by the pinned upstream PyTorch GNNWR/GTNNWR baselines. GTNNWR is used for CO2; GNNWR retains all four coordinates on WuhanOD. GCNNWR remains limited to the planar cases.
- Separate WuhanOD capacity comparison: DLAN, GeoAggregator and GNNWR, ten seeds each and 7,069 parameters each.
- DLAN width 16: ten seeds, 10,773 parameters, compared with the width-12 baseline without claiming equal capacity.
- Candidate support: ten seeds each at K512 and K2048, compared with the reused K1024 baseline. These new runs were trained on another machine; the partial K256 record is excluded.
- GVI 3D: DLAN and GeoAggregator, ten seeds each at 6,957 parameters. Original spatial partitions/features are unchanged; widths and DLAN reference budget change. This is not future-month holdout.
- Historical HPC WuhanOD: ten complete fits, reported separately because coordinate scaling, microbatch size and execution environment differ.
- The coordinate-scale/microbatch cross-diagnostic was incomplete at the frozen snapshot and is excluded from final attribution.

The archived equal-budget outcome remains dataset-specific. The WuhanOD capacity comparison additionally distinguishes mean error from median and paired win count; the lower DLAN mean does not imply superiority in most seeds.
Final-model FiLM/response-path ablations remain unrun, while completed configuration comparisons and the historical Gaussian ablation are individually labelled.

## WuhanOD reported-result correction

The reported WuhanOD configuration is attention width 12, head `(10,8)`, `candidate_count=512`, 7,069 parameters and ten seeds.
Its exact mean RMSE is **565.2762909669982**, sample SD 154.3068999460001, MAE 50.28194197849897 and mean R2 0.9051198068162931.
K2048 has mean RMSE 587.5989767259074; it is not the source of 565.2763.
The historical 6,957-parameter / width-10 score 919.9232 and the width-12 K1024 score 608.7311 remain labelled diagnostics.
The reported WuhanOD comparators use the complete 7,069-parameter baseline group.
Other cases retain their 6,957-parameter configurations; the paper no longer claims one global parameter budget.
K512 ran on the second machine, so its fit time is not a same-hardware speed comparison with the baselines.

`prepare_reported_results.py` selects these explicit source records, rechecks all twenty K512 prediction arrays and verifies the saved projection shapes.
The source manifest is `evidence/reported_configuration_manifest.json`.
Primary tables and R plots read `reported_metrics.csv`, `reported_summary.csv`, `reported_predictions.csv` and `reported_bandwidths.csv`.
Some exported figure filenames retain their historical `matched_` prefix for compatibility; the source tables, titles and captions specify the reported configurations.
Historical `matched_` evidence remains unchanged by this selection step.

## Reproduce the manuscript evidence and figures

```sh
python scripts/prepare_evidence.py
python scripts/prepare_supplement_evidence.py
python scripts/prepare_reported_results.py
Rscript scripts/plot_results.R
python scripts/render_references.py
```

These scripts read saved artifacts and do not train networks.
The main and supplementary audit checks 440 held-out prediction arrays from 220 unique completed fits; 20 additional arrays come from the separate historical HPC experiment.
The GVI OLS diagnostic is recovered from an existing fitted buffer, not refitted.
Source hashes, completion records, metrics and the ongoing-diagnostic status snapshot are saved under `evidence/`.
The R script generates fourteen figures; nine tables remain inline in `main.tex`.
The reference validator supports both direct citations and the user's glossary-driven citation macros.
Current validation finds 53 cited keys and zero missing keys.

Historical evidence files remain for provenance and are not pooled across budgets, machines or information conditions.
No source dataset is redistributed by the article build.
