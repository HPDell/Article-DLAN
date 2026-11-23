# SHP — WuhanHP

Wuhan housing data contain dwellings nested within residential communities and 17 covariates. Dwellings in the same community share a location. Map coordinates are stored separately so that the existing coordinate scaling in the model inputs is not mistaken for distances in metres.

## Ready-to-use data

- `ordinary.npz`: Frozen model inputs, responses, row identifiers and splits; 19,599 rows, 17 covariates and 2 coordinate dimensions.
- `contract.json`: Original experiment contract, retaining provenance, SHA256 hashes, splits, preprocessing and scoring definitions. Historical source paths provide provenance and are not runtime dependencies of the new notebooks.
- `map_coordinates.npz`: Row-aligned coordinates for residual maps only, stored in the `coordinates` field.

## Splits and scales

Entire communities are assigned to the training, validation or test set; communities do not overlap between sets.

Training / validation / test observations: 15,475 / 2,196 / 1,928.
Coordinate order: (x, y).
Response processing: use the response scale stored in the files without applying an additional exponential back-transformation.

Any additional covariate, target and coordinate transformations fitted by the trainer use training rows only. Validation and test responses never enter the reference bank, and training queries exclude their own rows.

## Array fields

| Field | Shape | Description |
|---|---|---|
| `x` | `(19599, 17)` | Covariate matrix |
| `u` | `(19599, 2)` | Frozen coordinates supplied to the trainer; the trainer fits its own coordinate transformation using training rows |
| `y` | `(19599,)` | Training target before the trainer applies target standardisation |
| `actual` | `(19599,)` | Response on the scale used to calculate final metrics |
| `groups` | `(19599,)` | Original group identifiers |
| `row_id` | `(19599,)` | Stable row identifiers for alignment across models |
| `train_ids` | `(15475,)` | Positional indices of training rows |
| `validation_ids` | `(2196,)` | Positional indices of validation rows |
| `test_ids` | `(1928,)` | Positional indices of test rows |
| `u_raw` | `(19599, 2)` | Source coordinates for provenance and diagnostics; replacing `u` invalidates compatibility with existing checkpoints |
| `coordinate_center` | `(2,)` | Coordinate centre recorded during data preparation |
| `coordinate_scale` | `(2,)` | Coordinate scale recorded during data preparation |

## Basemap

The Wuhan subdistrict basemap is shared with SIC and stored once at `../basemap/wuhan_subdis.gpkg` (`data/basemap/wuhan_subdis.gpkg` from the scripts working directory). It is used by `plot_results.ipynb` and is excluded from model training.

## Running the experiment

Run `../../SHP.ipynb`. The default `replay` mode audits saved predictions; `ARTICLE_MODE=train` retrains models from the frozen inputs.
Each configuration uses seeds 1–10. Training leaves this data directory unchanged.
