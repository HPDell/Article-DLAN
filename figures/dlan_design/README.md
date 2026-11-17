# DLAN architecture figures

The current figure set is reviewed against the implemented `gtdr` model and the manuscript's **34 pc (143.933 mm)** text width. Notation: coordinates `u`, coordinate index `l`, attention-block index `r`, and FiLM-layer index `s`; block/layer counts are `R`/`S`. Panel (b) now explicitly shows dimension-wise score branches, their fusion before one joint softmax, and the shared-weight fan-out to independent value pools. Full review: `PUBLICATION_REVIEW.md`; measured results: `validation.json`.

| Figure | Editable SVG / Draw.io | Font-independent print SVG | Intended width | Smallest text |
|---|---|---|---:|---:|
| Main architecture | `dlan_architecture.svg` / `.drawio` | `dlan_architecture_print.svg` | 143.933 mm | 8.23 pt |
| Full attention mechanism | `dlan_attention_detail.svg` / `.drawio` | `dlan_attention_detail_print.svg` | 143.933 mm | 12.40 pt |
| FiLM hidden-layer mechanism | `dlan_film_detail.svg` / `.drawio` | `dlan_film_detail_print.svg` | 104 mm | 8.30 pt |

The main figure's minimum stroke is about 0.52 pt. Both detail figures fit within the current manuscript's 650 pt text height. The 8 pt threshold is an internal legibility target; the publisher's general minimum line width is 0.3 pt. Use the specified widths; narrowing the figures invalidates those measurements.

The editable SVGs were exported with official draw.io Desktop **32.6.10** and contain native text plus the embedded full Draw.io model. Open either `.svg` or `.drawio` in Draw.io to edit. The `*_print.svg` copies convert lettering to vector paths and specify physical dimensions, avoiding font substitution during transfer. All SVGs are vector-only and contain no `foreignObject` or raster images. Print copies are produced with Inkscape from the official Draw.io exports.

Other files:

- `dlan_architecture_144mm.png`: preview rendered at the intended width and 150 dpi.
- `dlan_architecture_grayscale.png`: grayscale check at that same size.
- `dlan_mechanism.*`: earlier three-panel mechanism reference, retained for encoder/FiLM details. The current explicit dimension-wise attention drawing is in `dlan_architecture` and `dlan_attention_detail`.
- `dlan_retrieval.*`: candidate retrieval, self-exclusion and scale gradient paths.
- `DLAN_structure_zh.md`: full Chinese structure description, equations and tensor shapes.
- `figure_captions.md`: separate English captions for the main and detail figures.
- `source_manifest.json`: implementation/manuscript hashes used for the fidelity review.
- `revisions/v1/`, `revisions/v2/` and `revisions/v3/`: preserved earlier layouts for comparison.

## Regenerate

`dimension_attention_panel.py` builds the same explicit attention mechanism for both current figures, preventing drift between the overview and detail. Requires Python with Pillow, draw.io Desktop CLI and Inkscape. No model fitting is involved.

```bash
export FONTCONFIG_FILE="$PWD/article/figures/dlan_design/fontconfig.conf"
python article/figures/dlan_design/build_dlan_drawio.py
python article/figures/dlan_design/build_publication_drawio.py
python article/figures/dlan_design/build_detail_panels.py
DRAWIO_BIN=drawio bash article/figures/dlan_design/export_drawio.sh
python article/figures/dlan_design/prepare_print_artwork.py
python article/figures/dlan_design/audit_artwork.py
```

If system font rules override standard fonts, use the supplied task-local `fontconfig.conf` (Linux) to resolve Arial to DejaVu Sans, as used in this delivery. The builders use DejaVu Sans metrics for line breaks. The print copies preserve rendered letter shapes, while their Draw.io sources remain editable. Inkscape's warnings about unused CSS background/theme properties do not affect the checked geometry or glyph paths.

Existing manuscript source, model code and the manuscript's previous figure references are preserved; this directory contains the independent replacement artwork.
