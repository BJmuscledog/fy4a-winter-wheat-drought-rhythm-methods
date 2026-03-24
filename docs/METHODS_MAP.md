# Manuscript-to-code map

## Section 3.1 — Data harmonization and BRDF normalization

- `Atmosphere_correction.py` — atmospheric correction driver
- `fy6s_atc_func.py` — 6S-based atmospheric correction routines
- `brdf_fy_v2.py` — FY-4A BRDF normalization workflow
- `get_kvol_geo_func.py` — Ross–Li kernel calculations
- `read_data_func.py` — raster reading helpers
- `write_tif.py` — GeoTIFF writing helpers

## Section 3.2 — Phenological alignment

- `region_dtw_align.py` — MODIS EVI alignment by DTW on a common seasonal axis
- `combined_drought_dtw.py` — combined drought context + DTW alignment figure script

## Section 3.3 — Daytime rhythm metric extraction

- `plot_metrics.py` — pixel-level hourly NIRv processing and metric extraction
- `generate_metrics_cache.py` — cache-generation driver for metric tables

Core metrics implemented in `plot_metrics.py` include:
- `t_peak`
- `MDI`
- `A_NIRv`
- `Skew`
- `Recovery_rate`
- `NIRv_integral`
- `centroid`
- `centroid_shift`

## Section 3.4 — Multi-scale attribution of environmental drivers

### Correlation-based diagnosis
- `plot_env_factor_correlations.py` — scenario-wise Spearman correlation tables
- `plot_env_corr_matrix_4panels.py` — four-panel correlation matrix visualization

### Machine-learning attribution
- `XGBoost_shap_GEE-Adapted.py` — pixel-level feature table construction, model training, and SHAP analysis
- `pixel_shap_utils.py` — lightweight SHAP utility helpers

## Notes

- The packaged scripts retain their original research-script style and path settings.
- Figure-only polishing scripts, notebooks, and intermediate caches are intentionally excluded from this package.
