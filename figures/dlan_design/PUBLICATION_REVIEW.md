# Publication artwork review — 2026-11-12

## Scope and acceptance profile

The deliverable is an accurate, editable DLAN architecture figure set with a compact main figure and legible mechanism details. It is designed for the current `interact` manuscript, whose `article/interact.cls:30` sets `textwidth=34pc` (408 pt; 143.933 mm) and `textheight=650pt`. The main figure and attention detail use that full width; the compact FiLM detail uses 104 mm.

This review establishes figure accuracy and production readiness under the stated layout profile. Journal acceptance and any later production-specific requirements remain editorial decisions. The 8 pt minimum text criterion is this project's legibility target, not a claimed IJGIS rule. Taylor & Francis's general artwork guide specifies consistent standard lettering, publication-size preparation, separate files and lines of at least 0.3 pt:
https://authorservices.taylorandfrancis.com/publishing-your-research/making-your-submission/submit-electronic-artwork/

## Structural fidelity audit

| Requirement | Authoritative implementation and visual evidence | Result |
|---|---|---|
| Training-reference task and no query response input | `models/DeepGWRTransformer.py:423–469`; main panel (a), caption and Chinese explanation | Pass |
| Positive scales, joint multidimensional retrieval, union over blocks/heads | `models/gtdr_reference_trainer.py:24–77,100–103`; retrieval diagram and main Top-K block | Pass |
| Retrieval uses detached scales; attention retains scale gradients | Trainer line 100 and `_forward_gtdr` lines 225–227; dedicated retrieval diagram | Pass |
| Response enters reference values, not reference keys or initial query | Model lines 445–457; main reference K/V box and both mechanism figures | Pass |
| Independent coordinate/head projections, averaged content, summed geometry | Model lines 214–230; main panel (b), attention detail and caption | Pass |
| One masked neighbour softmax per head; one shared set of weights for all coordinate values | Model lines 228–236; main panel (b) explicitly includes masking and attention dropout | Pass |
| Query-state residual updates; fixed reference tokens across blocks | Model lines 237–238 and 459–469; main panel (a), captions and detailed text | Pass |
| FiLM bounded scale, additive shift, GELU, dropout and final affine readout | `FiLMLocalPredictor`, lines 311–342; main panel (c), FiLM detail | Pass |
| FiLM generators zero-initialised | Model lines 329–330; FiLM detail and captions | Pass |

The diagram's content mean C, directional bias G and Gaussian term P are scalar score components. α̃ denotes weights after attention dropout. The main figure's ×R and ×S frames denote independently parameterised repeated blocks/layers; they do not imply shared weights. The same final condition hᵢ is reused by all FiLM hidden layers.

## Visual and technical audit

- Removed the oversized title, implementation configuration footer and dense explanatory prose from the main figure. Panel labels, semantic colour and aligned computation blocks carry the hierarchy.
- Kept the direct xᵢ prediction route, both query residual bypasses, response-enriched values, joint softmax and FiLM operators visible.
- Inspected the rendered main figure, a 150 dpi preview at 143.933 mm and its grayscale version. Text is legible, operator signs remain distinct, and the meaning does not depend on colour. Corrected the residual branch route to avoid passing through the query token.
- Extracted attention as a separate full-width detail and compressed FiLM's vertical spacing. Both fit within the manuscript's 650 pt page height at their specified widths, with at least 35 mm reserved for captions. Inspected both rendered details and rerouted β to enter the addition operator horizontally.
- `audit_artwork.py` verifies native SVG text sizes at final width, minimum stroke width, label fit in the main figure, matching editable cells in the SVG metadata, valid edge endpoints, absence of raster images / foreignObject, font-independent print SVGs, page fit of the detail figures, and unchanged model/manuscript hashes.
- `validation.json` records measurements from the actual exported SVG, rather than builder intentions. The main figure minimum is 8.23 pt text and approximately 0.52 pt stroke.
- Editing SVGs retain native text and embedded Draw.io models. `*_print.svg` converts lettering to vector paths to remove font substitution during transfer; separate `.drawio` sources retain full editability. The print artwork has explicit physical dimensions.
- Original source/code and manuscript text remain unchanged. The first diagram is retained in `revisions/v1`; the complete mechanism overview is retained as `dlan_mechanism` for large-screen inspection. Use the separate detail figures for page-sized publication.

## Deliverable choice

Use `dlan_architecture_print.svg` at the manuscript full width. Use `dlan_attention_detail_print.svg` at full width and `dlan_film_detail_print.svg` at 104 mm when mechanism details are needed. Use `figure_captions.md` for the separate captions. Do not shrink the full-width main figure to one narrow column: the verified minimum font size applies to the stated 34 pc width.


## 2026-11-12 revision: visible dimension-specific scoring and joint attention

The previous diagram encoded dimension specificity mainly as subscripts inside a single projection box. That was mathematically consistent but visually understated. The new main panel (b) expands representative coordinate branches `l=1,2,…,D` and gives attention most of the horizontal page area. FiLM moves to a horizontal panel below the overview and attention.

Each coordinate branch visibly enters a common fusion node through its own score card. The fusion node explicitly averages content scores and sums the two geographical contributions. Exactly one joint softmax lies above the fusion. Three purple arrows distribute its output to the coordinate pools, whilst separate green value-projection branches supply the other pooling inputs. The context branches then converge at concatenation/output projection. The detail figure shows the same topology with the score formulas expanded inside each coordinate card. Its `l=D` branch uses generic subscript l with l fixed to D.

`audit_joint_attention` in `audit_artwork.py` checks the actual exported diagram cell topology: each independent Q/K projector feeds its coordinate score, each score feeds the common fusion, fusion feeds the unique softmax, that softmax feeds every pool, each pool also receives exactly its own value projection, and all pools feed concatenation. The role metadata survives the official Draw.io export, and the embedded editable graph is checked against the source. This graph audit supports the intended structure; the correctness of the score decomposition is grounded separately in `GeoAttentionBlock._forward_gtdr`, lines 214–238.

The full-width main figure still exceeds the 8 pt internal legibility target and 0.3 pt publisher line criterion. The larger standalone attention figure uses about 12.40 pt minimum lettering and fits the page with its caption reserve. Current numerical measurements are in `validation.json`. Captions now define the displayed terms and explicitly distinguish per-dimension scores from a jointly normalised attention distribution. The model and manuscript source hashes remain unchanged.


## Notation update: coordinates u, dimension l, attention block r

All current editable diagrams and exported SVGs use u for coordinates, l for the coordinate index and r for the attention-block index. FiLM hidden layers use s to avoid reusing r; their layer counts are R and S, respectively. The coordinate encoder is Eᵤ and positive scales are bᵣₗₕ. Raw coordinates are distinguished as u(raw) in the accompanying explanation, and the residual intermediate state is written as h-bar rather than u. Archived revisions retain their historical notation. This change affects notation only; the model/manuscript hashes and attention topology checks remain unchanged.
