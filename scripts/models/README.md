# Model definitions and adapters

| File | Role |
|---|---|
| `DLAN.py` | Original Fourier encoding, dimension-specific Q/K/V, joint product-kernel attention, FiLM predictor and base scaling/trainer definitions; public alias `DLAN` preserves original state-dict keys. |
| `training.py` | Original exact bandwidth-metric candidate retrieval, AdamW/MSE loop, gradient accumulation, early stopping and ReduceLROnPlateau. |
| `GeoAggregator.py` | Repository-reviewed GeoAggregator implementation from P18; original local rotary-width/registration corrections retained. This is not labelled an unmodified official release. |
| `geometry.py` | Original multidimensional rotary adapter, retaining all CO2 time and WuhanOD endpoint coordinates. |
| `GCNNWR.py` | Repository GCNNWR construction: distance image, convolutional multiplier network and frozen training OLS. |
| `GNNWR.py` | Adapter only. Loads SWNN/STPNN from the installed `gnnwr==0.1.17` distribution; implements training-only landmark selection, frozen OLS and the controlled training loop. No copy of the standard library's network source is bundled. |
| `adapters.py` | Existing train-reference adapters, grid statistics and matched configurations. The old 6,957-parameter WuhanOD preset is historical, not the main result. |
| `experiment_adapter.py` | Constructs each explicit specification in `experiment_manifest.json`, including the 7,069-parameter WuhanOD main group and 3D GVI. |
| `PROVENANCE.json` | Original source hashes, packaged source hashes and published GNNWR provenance. |

GNNWR/GTNNWR retain the author's network definitions. The published 0.1.17 network module was parsed and its AST verified identical to the original module pinned at commit `2711bdee9821bfe242992d2eb885a78be1cf2628`. Its network module is loaded directly from the installed distribution to avoid importing unrelated package UI/plotting modules.

The small parameter budget uses training-only fixed landmarks, rather than the upstream full-reference default. CE uses GTNNWR (128 references); SIC uses four-coordinate GNNWR (119 references); SHP/GVI use 134 references. GCNNWR is not extended to CE/SIC by dropping dimensions.

DLAN and GeoAggregator use labelled training-reference responses; GNNWR/GTNNWR use reference coordinates plus fitted OLS coefficients; GCNNWR uses fitted grid statistics and OLS. Equal parameter counts do not imply identical information access.
