# Shared basemaps

| File | Used by | Stored coordinate reference system |
|---|---|---|
| `wuhan_subdis.gpkg` | SIC observed-flow map and SHP community-residual map | EPSG:4326 (WGS 84) |
| `shanghai_county.geojson` | GVI observation map | EPSG:4490 (China Geodetic Coordinate System 2000) |

The Wuhan GeoPackage contains the `wuhan_subdis` layer with 196 polygon features. The Shanghai GeoJSON contains 17 polygon features. The files were relocated from the existing packaged case folders without changing their contents. The identical Wuhan copies previously stored under SIC and SHP have been consolidated into one file.

From the `article/scripts/` working directory, notebooks read `data/basemap/<filename>`. The SIC and GVI data maps repair geometries and transform them to EPSG:32650 and EPSG:32651, respectively. The SHP residual map overlays its EPSG:4547 observation points on the Wuhan layer using tmap's coordinate transformation.

These layers provide cartographic context only. They are not used for model training, split construction, feature generation or observation filtering. SHA256 hashes are recorded in `SHA256.json`.
