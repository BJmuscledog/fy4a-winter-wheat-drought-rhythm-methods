# RSE Method Code Package

This package curates the key method scripts used in our *Remote Sensing of Environment* study on resolving daytime photosynthetic rhythms of winter wheat under drought stress from FY-4A observations.

## Scope

The package focuses on the reproducible **method pipeline** rather than all figure drafts, notebooks, and intermediate outputs in the full project. Large local datasets and generated figures are intentionally excluded.

## Pipeline covered

1. **Data harmonization and atmospheric correction**
2. **BRDF normalization to NBAR-like reflectance**
3. **Phenological alignment using MODIS EVI + DTW/TWDTW-style matching**
4. **Daytime rhythm metric extraction from hourly NIRv curves**
5. **Correlation-based diagnosis of environmental controls**
6. **Pixel-level XGBoost + SHAP attribution**

## Folder structure

- `src/` — packaged Python scripts with original research logic preserved
- `docs/METHODS_MAP.md` — mapping from manuscript sections to scripts
- `requirements.txt` — Python dependencies to recreate the analysis environment

## Important note before reuse

These scripts are preserved close to the original research workflow. Most of them still contain **project-specific absolute paths** (for example `F:\...` or `H:\...`). Before running the scripts elsewhere, please search for path variables such as:

- `BASE_DIR`
- `OUT_DIR`
- `BRDF_DIR_2022`, `BRDF_DIR_2023`
- `WHEAT_MASK_TIF`, `HHH_SHP_PATH`
- `csv_2022`, `csv_2023`
- `OUTPUT_BASE_DIR`

and replace them with paths valid on your machine.

## Suggested usage order

1. Atmospheric correction: `Atmosphere_correction.py` + `fy6s_atc_func.py`
2. BRDF normalization: `brdf_fy_v2.py` + `get_kvol_geo_func.py`
3. Phenological alignment: `region_dtw_align.py` / `combined_drought_dtw.py`
4. Rhythm metrics: `plot_metrics.py` + `generate_metrics_cache.py`
5. Correlation diagnostics: `plot_env_factor_correlations.py` + `plot_env_corr_matrix_4panels.py`
6. Attribution: `XGBoost_shap_GEE-Adapted.py`

## Packaged scripts

- `Atmosphere_correction.py`
- `fy6s_atc_func.py`
- `brdf_fy_v2.py`
- `get_kvol_geo_func.py`
- `read_data_func.py`
- `write_tif.py`
- `region_dtw_align.py`
- `combined_drought_dtw.py`
- `plot_metrics.py`
- `generate_metrics_cache.py`
- `plot_env_factor_correlations.py`
- `plot_env_corr_matrix_4panels.py`
- `XGBoost_shap_GEE-Adapted.py`
- `pixel_shap_utils.py`

## Reproducibility note

This package is intended as a shareable research snapshot of the core method code. It does **not** yet convert the project into a fully parameterized Python package or command-line tool; instead, it preserves the original scripts used to generate the manuscript analyses.
