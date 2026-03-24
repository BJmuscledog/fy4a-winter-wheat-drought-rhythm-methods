# FY-4A winter wheat drought-rhythm methods

This repository packages the key method code used in our *Remote Sensing of Environment* study on resolving daytime photosynthetic rhythms of winter wheat under drought stress from FY-4A observations.

The goal is to provide a transparent, manuscript-aligned **companion-code repository** for the core analyses, while keeping the scientific choices explicit and reproducible.

## Availability

Code for the workflow is **openly available** in this repository.

Raw FY-4A, VIIRS, TROPOSIF, and meteorological datasets are **not redistributed here** because access is governed by the original data providers and their respective access policies.

This repository therefore provides the **code, configuration, and documentation required to reproduce the workflow**, assuming the user has independent access to the original source datasets described in the manuscript.

## What this repository covers

The packaged workflow spans:

1. **Data harmonization and atmospheric correction**
2. **BRDF normalization to NBAR-like reflectance**
3. **Phenological alignment using MODIS EVI + DTW-style matching**
4. **Daytime rhythm metric extraction from hourly NIRv curves**
5. **Correlation-based diagnosis of environmental controls**
6. **Pixel-level XGBoost + SHAP attribution**

Large local datasets, figure drafts, notebooks, and heavyweight caches are intentionally excluded.

## Repository structure

```text
.
├── configs/
│   └── environment_variables.example.txt
├── docs/
│   ├── METHODS_MAP.md
│   └── REPRODUCIBILITY.md
├── scripts/
│   └── smoke_check.py
├── src/
│   ├── project_config.py
│   ├── Atmosphere_correction.py
│   ├── brdf_fy_v2.py
│   ├── combined_drought_dtw.py
│   ├── fy6s_atc_func.py
│   ├── generate_metrics_cache.py
│   ├── get_kvol_geo_func.py
│   ├── pixel_shap_utils.py
│   ├── plot_env_corr_matrix_4panels.py
│   ├── plot_env_factor_correlations.py
│   ├── plot_metrics.py
│   ├── read_data_func.py
│   ├── region_dtw_align.py
│   ├── write_tif.py
│   └── XGBoost_shap_GEE-Adapted.py
├── CITATION.cff
├── environment.yml
├── LICENSE
└── requirements.txt
```

## What changed relative to the original project snapshot

This public version improves the original script bundle in a few important ways:

- centralizes shared parameters in `src/project_config.py`
- makes drought/non-drought grouping schemes explicit
- documents where analysis windows differ across methods
- provides environment-variable overrides for machine-specific paths
- adds reproducibility notes, citation metadata, and a lightweight smoke check

## Quick start

### 1. Create an environment

Conda:

```bash
conda env create -f environment.yml
conda activate fy4a-wheat-drought-rhythm
```

or pip:

```bash
pip install -r requirements.txt
```

### 2. Check configuration

The repository still depends on local datasets, but you can verify the configuration layer without running the full workflow:

```bash
python scripts/smoke_check.py
```

If your local data paths differ from the original project, set environment variables listed in `configs/environment_variables.example.txt`.

### 3. Suggested workflow order

1. Atmospheric correction: `Atmosphere_correction.py` + `fy6s_atc_func.py`
2. BRDF normalization: `brdf_fy_v2.py` + `get_kvol_geo_func.py`
3. Phenological alignment: `region_dtw_align.py` / `combined_drought_dtw.py`
4. Rhythm metrics: `plot_metrics.py` + `generate_metrics_cache.py`
5. Correlation diagnostics: `plot_env_factor_correlations.py` + `plot_env_corr_matrix_4panels.py`
6. Attribution: `XGBoost_shap_GEE-Adapted.py`

## Important reproducibility note

This repository is intended as a transparent **companion-code archive**, not yet as a fully packaged end-user software library. The scripts preserve the original research workflow, and some of them still operate in a script-first style. Please read:

- `docs/METHODS_MAP.md`
- `docs/REPRODUCIBILITY.md`

before reusing the code for a new study.
