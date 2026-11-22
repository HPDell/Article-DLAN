# CE — CO2

Regional panel for 2014–2021, sourced from the MGTWR data bundle at `data/MGTWR_2SCALL/data/co2data_13k.csv`. The source provides neither variable names nor physical units, so x1–x8 and the stored response scale are retained without assuming tonnage or an additional exponential transformation.

## Ready-to-use data

- `ordinary.npz`: Frozen model inputs, responses, row identifiers and splits; 13,000 rows, 8 covariates and 3 coordinate dimensions.
- `contract.json`: Original experiment contract, retaining provenance, SHA256 hashes, splits, preprocessing and scoring definitions. Historical source paths provide provenance and are not runtime dependencies of the new notebooks.
- `map_coordinates.npz`: Row-aligned coordinates for residual maps only, stored in the `coordinates` field.

## Splits and scales

Training covers 2014–2019, validation uses 2020, and testing uses 2021. Regions may recur across years, so this evaluates future-year prediction rather than holding out unseen regions.

Training / validation / test observations: 9,732 / 1,643 / 1,625.
Coordinate order: (u, v, year).
Response processing: the stored CO₂ response is standardised within the trainer and restored to its stored scale for evaluation.

Any additional covariate, target and coordinate transformations fitted by the trainer use training rows only. Validation and test responses never enter the reference bank, and training queries exclude their own rows.

## Array fields

| Field | Shape | Description |
|---|---|---|
| `x` | `(13000, 8)` | Covariate matrix |
| `u` | `(13000, 3)` | Frozen coordinates supplied to the trainer; the trainer fits its own coordinate transformation using training rows |
| `u_raw` | `(13000, 3)` | Source coordinates for provenance and diagnostics; replacing `u` invalidates compatibility with existing checkpoints |
| `y` | `(13000,)` | Training target before the trainer applies target standardisation |
| `actual` | `(13000,)` | Response on the scale used to calculate final metrics |
| `groups` | `(13000,)` | Original group identifiers |
| `row_id` | `(13000,)` | Stable row identifiers for alignment across models |
| `train_ids` | `(9732,)` | Positional indices of training rows |
| `validation_ids` | `(1643,)` | Positional indices of validation rows |
| `test_ids` | `(1625,)` | Positional indices of test rows |
| `coordinate_center` | `(3,)` | Coordinate centre recorded during data preparation |
| `coordinate_scale` | `(3,)` | Coordinate scale recorded during data preparation |

## Running the experiment

Run `../../CE.ipynb`. The default `replay` mode audits saved predictions; `ARTICLE_MODE=train` retrains models from the frozen inputs.
Each configuration uses seeds 1–10. Training leaves this data directory unchanged.
