# DLAN manuscript for IJGIS

`main.tex` contains all manuscript text, tables, glossary definitions and references. Only figures are included externally. The publisher's unmodified Interact class is in `template/`; provenance is recorded in `template/SOURCE.txt`. British English and author–date citations are used. Author metadata remain to be supplied.

Build from this directory with `make`; all compiler output goes to `_out/`. `make figures` regenerates the R figures from the frozen evidence tables. `make audit` rechecks the repository's saved predictions and regenerates those tables; it does not train models. `make references` embeds the verified reference metadata into the one manuscript file. Verified reference metadata are frozen in `evidence/references_crossref.json`; the default build needs no reference-network requests.

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
