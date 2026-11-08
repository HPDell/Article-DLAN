# Validation of the working draft

Date: 7 November 2026 (Asia/Shanghai).

- `python article/scripts/prepare_evidence.py`: passed; 76 saved prediction sets checked, no fitting or checkpoint inference run.
- `Rscript article/scripts/plot_results.R`: passed; nine figures exported to vector PDF and PNG. R environment recorded in `R_sessionInfo.txt`.
- `python article/scripts/render_references.py`: embedded 62 distinct cited references, each with a retained original metadata-source URL. Ambiguous title-search matches were discarded or replaced by verified publisher/arXiv records.
- `make -C article`: passed through latexmk/pdfLaTeX; 24 pages. The final log contains no undefined references/citations, LaTeX errors or overfull boxes.
- Rendered all 24 PDF pages with Poppler; inspected the contact sheet and representative figure panels. QA renders are in `_out/qa/`.
- One manuscript `.tex` file; nine external figures; six inline tables; glossaries manages the abbreviation list without an external indexing dependency.
- The built-in LaTeX editor was opened. Its compiler could not obtain an uncached Tectonic bundle from relay.fullyjustified.net. The actual delivered PDF was successfully built and verified using the supplied Makefile and local TeX installation.

This validation confirms document construction and the stated saved-output audit. It does not complete the missing GVI dataset, final ten-seed adaptive WuhanOD run, unavailable higher-dimensional baselines, final-model module ablations or fixed-hardware scaling experiments. These items are identified in the manuscript and README.
