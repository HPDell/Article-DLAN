# Ready-to-use experiment data

| Notebook | Directory | Original case |
|---|---|---|
| `CE.ipynb` | `CE/` | CO2 |
| `SIC.ipynb` | `SIC/` | WuhanOD |
| `SHP.ipynb` | `SHP/` | WuhanHP |
| `GVI.ipynb` | `GVI/` | GVI |

Inputs are copied, byte-for-byte, from the fixed experiment bundles. Each case documents its arrays, units, transformations and split. No symlinks or project-root dataset paths are required. `historical/` contains a separately labelled frozen diagnostic used only by a retained plotting cell.

## Shared basemaps

`basemap/` contains the Wuhan layer shared by SIC and SHP and the Shanghai layer used by GVI. See `basemap/README.md` for file names and coordinate systems. These files provide map context and are excluded from model training.
