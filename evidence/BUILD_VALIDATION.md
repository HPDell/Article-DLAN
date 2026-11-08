# Validation of the updated DLAN manuscript

Date: 8 November 2026 (Asia/Shanghai).

## Experimental evidence

- The main comparison uses `dlan_matched4_10seeds_v1` only: 100 completed fits, ten seeds per supported configuration, and exactly 6,957 trainable parameters each.
- `python article/scripts/prepare_evidence.py` passed. It recalculated 200 validation/test prediction arrays, verified IDs/targets/configuration digests, checked fixed data/source hashes, read all 100 checkpoints, and confirmed the recorded reload audits. No training or new neural inference was run.
- The GVI OLS diagnostic uses the existing training-fitted coefficient buffers, with no new fit. It is explicitly separate from the parameter-matched neural systems.
- WuhanHP map geometry is taken from the row-aligned original projected coordinates; the frozen model inputs inherit an earlier coordinate normalisation. Targets were checked for equality before mapping.
- Final-model module-isolating ablations remain unrun. The manuscript clearly separates the completed historical Gaussian ablation from the new main experiment.
- GVI 3-D formal training remains outside the reported completed comparison.

## Document and figures

- `Rscript article/scripts/plot_results.R` passed: eleven R-generated vector figures, including the architecture and labelled historical Gaussian comparison.
- `python article/scripts/render_references.py`: 61 cited keys found in the supplied `zotero.bib`; zero missing. Empty citation syntax was removed, and the XGBoost discussion uses the matching Chen and Guestrin record. The unrelated Lundberg spatial-service citation is no longer used as a feature-attribution source.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=_out main.tex` passed: 44-page PDF with eleven figures and six inline tables. The final log has no undefined citations/references, TeX errors or overfull boxes.
- Sentence boundaries and long-clause source breaks were checked without changing the user's class/format settings.
- Poppler rendered all 44 pages for a contact-sheet inspection; representative quantitative figures were also inspected directly. QA outputs are in `_out/matched_qa/`.
- The built-in editor's compilation was attempted but failed because its Tectonic bundle could not be downloaded. The delivered PDF was successfully compiled using local pdfLaTeX.

The old partial-result snapshots are preserved as historical evidence, not combined with the new main experiment. Author details, funding and conflict declarations remain for author completion.
