# SIC — WuhanOD

Positive travel flows between Wuhan subdistricts for 20–26 July 2020. All 35 covariates and all four origin–destination coordinate dimensions are retained.

## Ready-to-use data

- `ordinary.npz`: Frozen model inputs, responses, row identifiers and splits; 13,807 rows, 35 covariates and 4 coordinate dimensions.
- `contract.json`: Original experiment contract, retaining provenance, SHA256 hashes, splits, preprocessing and scoring definitions. Historical source paths provide provenance and are not runtime dependencies of the new notebooks.
- `map_coordinates.npz`: Row-aligned coordinates for residual maps only, stored in the `coordinates` field.

## Splits and scales

A fixed observation-level training/validation/test split is used. It evaluates neither temporal extrapolation nor the exclusion of entire origin or destination groups.

Training / validation / test observations: 9,618 / 2,104 / 2,085.
Coordinate order: (origin_x, origin_y, destination_x, destination_y).
Response processing: `y` is log1p(count), and `actual` contains original counts. Apply expm1 to predictions and clip them to non-negative values before calculating metrics on the count scale.

Any additional covariate, target and coordinate transformations fitted by the trainer use training rows only. Validation and test responses never enter the reference bank, and training queries exclude their own rows.

## Array fields

| Field | Shape | Description |
|---|---|---|
| `x` | `(13807, 35)` | Covariate matrix |
| `u` | `(13807, 4)` | Frozen coordinates supplied to the trainer; the trainer fits its own coordinate transformation using training rows |
| `y` | `(13807,)` | Training target before the trainer applies target standardisation |
| `actual` | `(13807,)` | Response on the scale used to calculate final metrics |
| `groups` | `(13807,)` | Original group identifiers |
| `row_id` | `(13807,)` | Stable row identifiers for alignment across models |
| `train_ids` | `(9618,)` | Positional indices of training rows |
| `validation_ids` | `(2104,)` | Positional indices of validation rows |
| `test_ids` | `(2085,)` | Positional indices of test rows |
| `u_raw` | `(13807, 4)` | Source coordinates for provenance and diagnostics; replacing `u` invalidates compatibility with existing checkpoints |
| `coordinate_center` | `(4,)` | Coordinate centre recorded during data preparation |
| `coordinate_scale` | `(4,)` | Coordinate scale recorded during data preparation |

## Running the experiment

Run `../../SIC.ipynb`. The default `replay` mode audits saved predictions; `ARTICLE_MODE=train` retrains models from the frozen inputs.
Each configuration uses seeds 1–10. Training leaves this data directory unchanged.
