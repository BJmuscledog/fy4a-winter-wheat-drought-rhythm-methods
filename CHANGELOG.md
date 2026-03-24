# Changelog

## v1.0.0 - 2026-03-24

Initial public release of the FY-4A winter wheat drought-rhythm methods repository.

### Added

- centralized shared configuration in `src/project_config.py`
- environment setup files (`requirements.txt`, `environment.yml`)
- reproducibility notes in `docs/REPRODUCIBILITY.md`
- smoke-test script in `scripts/smoke_check.py`
- citation metadata in `CITATION.cff`
- license and public-facing repository documentation

### Changed

- unified cross-script drought/non-drought grouping definitions
- centralized analysis date windows and key numerical settings
- updated metric, DTW, correlation, BRDF, and XGBoost/SHAP scripts to read shared configuration
- improved repository structure for manuscript-aligned reuse and inspection

### Notes

- large source datasets are not redistributed in this repository
- the repository is intended as companion code for the manuscript workflow
