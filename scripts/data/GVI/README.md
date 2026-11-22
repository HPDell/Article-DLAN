# GVI — GVI

There are 2,122 street-view green view index observations with 176 frozen features: 128 Sentinel features, 16 SVF features, and 32 canopy-height and valid-fraction features. The experiments train regression models only; image encoders remain frozen.

## Ready-to-use data

- `ordinary.npz`: Frozen model inputs, responses, row identifiers and splits; 2,122 rows, 176 covariates and 2 coordinate dimensions.
- `contract.json`: Original experiment contract, retaining provenance, SHA256 hashes, splits, preprocessing and scoring definitions. Historical source paths provide provenance and are not runtime dependencies of the new notebooks.
- `map_coordinates.npz`: Row-aligned coordinates for residual maps only, stored in the `coordinates` field.

## Splits and scales

A fixed split by 1 km spatial groups is used without a buffer. Cross-split imagery-footprint overlaps from the original exploratory split are retained. The split does not hold out a future period.

Training / validation / test observations: 1,696 / 213 / 213.
Coordinate order: (projected_x, projected_y); the supplementary version adds elapsed capture month.
Response processing: `actual` is the GVI proportion in [0,1].

Any additional covariate, target and coordinate transformations fitted by the trainer use training rows only. Validation and test responses never enter the reference bank, and training queries exclude their own rows.

## Array fields

| Field | Shape | Description |
|---|---|---|
| `x` | `(2122, 176)` | Covariate matrix |
| `y` | `(2122,)` | Training target before the trainer applies target standardisation |
| `actual` | `(2122,)` | Response on the scale used to calculate final metrics |
| `u` | `(2122, 2)` | Frozen coordinates supplied to the trainer; the trainer fits its own coordinate transformation using training rows |
| `groups` | `(2122,)` | Original group identifiers |
| `row_id` | `(2122,)` | Stable row identifiers for alignment across models |
| `train_ids` | `(1696,)` | Positional indices of training rows |
| `validation_ids` | `(213,)` | Positional indices of validation rows |
| `test_ids` | `(213,)` | Positional indices of test rows |
| `u_raw` | `(2122, 2)` | Source coordinates for provenance and diagnostics; replacing `u` invalidates compatibility with existing checkpoints |
| `coordinate_center` | `(2,)` | Coordinate centre recorded during data preparation |
| `coordinate_scale` | `(2,)` | Coordinate scale recorded during data preparation |

## Running the experiment

Run `../../GVI.ipynb`. The default `replay` mode audits saved predictions; `ARTICLE_MODE=train` retrains models from the frozen inputs.
Each configuration uses seeds 1–10. Training leaves this data directory unchanged.

## Month-augmented inputs

`month_augmented.npz` and `month_augmented_contract.json` provide the three-coordinate version. Observations, covariates, responses and spatial splits match the planar version. DLAN uses width/head configuration `(6,16,33)` with 768 candidates; GeoAggregator uses `(24,14,31)`. This compares configurations with equal parameter counts, rather than a single-factor ablation that changes only temporal information.
