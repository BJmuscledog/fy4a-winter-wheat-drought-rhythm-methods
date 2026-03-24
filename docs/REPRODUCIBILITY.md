# Reproducibility and design notes

This repository packages the key method code used in the FY-4A winter wheat drought-rhythm study. The original project was developed as a research workflow rather than as a software library, so a few scientific choices are important to document explicitly.

## 1. Shared configuration

All cross-script configuration that materially affects reproducibility now lives in `src/project_config.py`, including:

- data and output paths
- drought/non-drought class grouping schemes
- analysis date windows
- smoothing settings for DTW figures
- BRDF kernel parameters
- visualization-only clipping choices

Machine-specific path changes can be applied with environment variables listed in `configs/environment_variables.example.txt`.

## 2. Drought-class grouping schemes

Two grouping schemes are kept explicit instead of leaving the choice hidden in different scripts:

- `regime_default`: drought = `{2, 3, 4}`, wet = `{0, 1}`
  - used for regime-level and machine-learning analyses
  - favours broader sample support and dryland/non-dryland consistency
- `strict_contrast`: drought = `{3, 4}`, wet = `{0}`
  - used for stricter effect-size-style contrasts in metric-focused analysis

The selected default for each module is documented in `project_config.py`.

## 3. Date windows

Different steps of the workflow legitimately use different windows:

- metric extraction uses a two-day hourly window after phenological alignment
- correlation and XGBoost analyses use the target comparison day associated with each year

These windows are now centralized rather than redefined ad hoc inside each script.

## 4. What is exploratory vs. default

The repository keeps some exploratory options, but they are **not** the default:

- `AUTO_SELECT_TIME_SCALE = False` in the XGBoost/SHAP workflow
- `USE_ALL_ENV_FACTORS = True` by default
- debug-only subsampling is only triggered when `DEBUG_MODE = True`

This is intentional: the public repository should default to the more transparent and less optimistic configuration.

## 5. Visualization-only operations

Some scripts use percentile clipping or visual scaling to improve figure readability. These operations are intended for visualization only and should not be interpreted as part of the raw metric computation.

In particular, `plot_metrics.py` uses shared percentile bounds from `project_config.py` when clipping figure ranges.

## 6. Current limitations

The repository is much cleaner and more internally consistent than the original project snapshot, but it is still not a fully packaged command-line application. In particular:

- large local datasets are not included
- most methods remain script-based rather than library-style APIs
- several scripts still depend on GDAL/GeoPandas system installations

For that reason, this repository should be treated as a transparent companion-code archive for the manuscript, not as a polished end-user software product.
