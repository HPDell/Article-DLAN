# DLAN manuscript for IJGIS

`main.tex` contains all manuscript text, tables and glossary definitions, with each sentence starting on a new line.
References are read from the user-supplied `zotero.bib` using the publisher's `tfcad.bst` author–date style.
The unmodified Interact class is `interact.cls`.
British English is used; author metadata remain to be supplied.

Build from this directory with `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=_out main.tex`.
All compiler output goes to `_out/`.
`Rscript scripts/plot_results.R` regenerates the R figures from frozen evidence tables.
`python scripts/prepare_evidence.py` checks saved predictions without training.
`python scripts/render_references.py` validates the citation keys against `zotero.bib`; it no longer inserts inline bibliography entries.

## Citation migration, 8 November 2026

28 of 62 cited works matched the supplied export by DOI or normalised exact title.
The 34 unmatched original keys remain in the manuscript, so their citations are unresolved until those records are supplied.
See [the complete missing-reference list and key mapping](evidence/missing_references.md).
The Zotero export has not been modified.

## Evidence snapshot, 7 November 2026

- CO2: complete 10-seed comparison, 60 saved validation/test prediction sets recalculated.
- WuhanHP: complete planar configuration and three neural baselines, 12 test prediction sets recalculated. The planar configuration is called DLAN-P; it uses the historical joint block and fixed Euclidean retrieval. It is explicitly distinguished from the final coordinate-specific product retrieval.
- WuhanOD: final bandwidth-adaptive retrieval has four completed seeds used in this frozen draft. The completed ten-seed marginal-union experiment is historical evidence and is separately labelled. Its scores are not substituted for the final DLAN model.
- GVI: no dataset or outputs found in the supplied checkout. Data description, experiment and result completion items are visible in the manuscript.
- Final DLAN component ablations and fixed-hardware scaling experiments are specified but unrun. Historical joint-attention Gaussian ablations are clearly identified.
- GNNWR CO2 and higher-dimensional GCNNWR comparisons have no supported measured result here.

The old article directory was cleared before drafting, as requested. No training or long experiment was started for manuscript preparation. This is a substantive working draft, not a submission-ready paper with a completed four-dataset experiment matrix.

## Reproduction requirements

Python with numpy, pandas, torch, requests and BeautifulSoup; R with ggplot2 and patchwork; a TeX installation with latexmk, pdfLaTeX, natbib and glossaries. `make` needs no network, Python or R when the supplied figures already exist. Glossaries uses the no-index workflow, so latexmk resolves the abbreviation list without an external makeglossaries invocation.
