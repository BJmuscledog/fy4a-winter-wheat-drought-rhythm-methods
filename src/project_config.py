"""Shared configuration for the FY-4A winter wheat drought-rhythm repository.

This module centralizes the small set of parameters that were previously
hard-coded across several research scripts. The goal is not to over-engineer
the original workflow, but to make the repository more transparent,
internally consistent, and easier to adapt on a different machine.
"""

from __future__ import annotations

import os
from copy import deepcopy
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: str) -> Path:
    """Return a filesystem path, optionally overridden by an environment variable."""
    return Path(os.getenv(name, default)).expanduser()


def _env_int(name: str, default: int) -> int:
    """Parse an integer environment variable with a safe default."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


BASE_DATA_DIR = _env_path("FY4_BASE_DATA_DIR", r"F:\风云数据\Fy_p2_data")
ANCILLARY_DATA_DIR = _env_path("FY4_ANCILLARY_DATA_DIR", r"F:\G_disk\FY4\data\ancillary_data")
FY4_OUTPUT_ROOT = _env_path("FY4_OUTPUT_ROOT", r"F:\FY4")

PAPER_FIG_DIR = _env_path(
    "FY4_PAPER_FIG_DIR",
    str(FY4_OUTPUT_ROOT / "outputs_paper_figs"),
)
METRIC_OUTPUT_DIR = _env_path(
    "FY4_METRIC_OUTPUT_DIR",
    str(PAPER_FIG_DIR / "box_grid_Analysis"),
)
XGB_OUTPUT_DIR = _env_path(
    "FY4_XGB_OUTPUT_DIR",
    str(FY4_OUTPUT_ROOT / "outputs_final" / "pixel_level_xgb_shap_gee"),
)

WHEAT_MASK_TIF = _env_path(
    "FY4_WHEAT_MASK_TIF",
    r"F:\G_disk\FY4\Winter_wheat_map\Winter_wheat_map.tif",
)
HHH_SHP_PATH = _env_path(
    "FY4_HHH_SHP_PATH",
    r"F:\风云数据\Fy_p1_data\Shp\Huanghuaihai\Huanghuaihai.shp",
)

DTW_CSV_2022 = _env_path(
    "FY4_DTW_CSV_2022",
    r"F:\G_disk\FY4\data\ancillary_data\dtw\HHH_WinterWheat_MODIS_2022.csv",
)
DTW_CSV_2023 = _env_path(
    "FY4_DTW_CSV_2023",
    r"F:\G_disk\FY4\data\ancillary_data\dtw\HHH_WinterWheat_MODIS_2023.csv",
)

DROUGHT_TIMESERIES_CSV = _env_path(
    "FY4_DROUGHT_TIMESERIES_CSV",
    r"F:\G_disk\FY4\data\Drought_GEE\HHH_Wheat_SPEI_SM_Drought_2022_2023_Optimizedn.csv",
)

DROUGHT_MASK_PATHS = {
    "2022": _env_path(
        "FY4_DROUGHT_MASK_2022",
        r"F:\G_disk\FY4\data\Drought_GEE\drive-download-20251227T184021Z-1-001\SMPct_DroughtClass_20220423.tif",
    ),
    "2023": _env_path(
        "FY4_DROUGHT_MASK_2023",
        r"F:\G_disk\FY4\data\Drought_GEE\drive-download-20251227T184021Z-1-001\SMPct_DroughtClass_20230425.tif",
    ),
}


CLASS_SCHEMES = {
    "regime_default": {
        "drought": {2, 3, 4},
        "wet": {0, 1},
        "description": (
            "Broad dryland/non-dryland grouping used for regime-level and "
            "machine-learning analyses to maintain sample support."
        ),
    },
    "strict_contrast": {
        "drought": {3, 4},
        "wet": {0},
        "description": (
            "Stricter contrast used for effect-size-style metric comparisons "
            "where a sharper drought-vs-wet separation is desired."
        ),
    },
}

METRIC_CLASS_SCHEME = os.getenv("FY4_METRIC_CLASS_SCHEME", "strict_contrast")
CORRELATION_CLASS_SCHEME = os.getenv("FY4_CORRELATION_CLASS_SCHEME", "regime_default")
ML_CLASS_SCHEME = os.getenv("FY4_ML_CLASS_SCHEME", "regime_default")


def get_class_scheme(name: str) -> dict:
    """Return a defensive copy of a named drought-class grouping scheme."""
    if name not in CLASS_SCHEMES:
        raise KeyError(f"Unknown class scheme: {name}")
    return deepcopy(CLASS_SCHEMES[name])


METRIC_TARGET_DATES = {
    "2022": [date(2022, 4, 22), date(2022, 4, 23)],
    "2023": [date(2023, 4, 24), date(2023, 4, 25)],
}
METRIC_HOURS = list(range(9, 17))
METRIC_SAMPLE_PIXELS = _env_int("FY4_METRIC_SAMPLE_PIXELS", 5000)

CORRELATION_TARGET_DATES = {
    "2022_Drought": {date(2022, 4, 23)},
    "2023_Wet": {date(2023, 4, 25)},
}

XGB_YEARLY_CONFIGS = [
    {
        "year_tag": "2022_Drought",
        "brdf_dir": BASE_DATA_DIR / "2022_0423_week" / "Brdf_hhh_v2",
        "target_dates": [date(2022, 4, 23)],
        "era5_env_raster": _env_path(
            "FY4_ERA5_ENV_RASTER_2022",
            r"F:\G_disk\FY4\data\ancillary_data\ERA5_hour\Env_daily_1d3d_dry_20220423.tif",
        ),
        "era5_hourly_raster": _env_path(
            "FY4_ERA5_HOURLY_RASTER_2022",
            r"F:\G_disk\FY4\data\ancillary_data\ERA5_hour\ERA5_hourly_dry_20220423_hourly_stack.tif",
        ),
        "drought_tif": DROUGHT_MASK_PATHS["2022"],
        "gee_env_tif": _env_path(
            "FY4_GEE_ENV_TIF_2022",
            r"F:\G_disk\FY4\data\ancillary_data\GEE\Env_factors_20220423.tif",
        ),
        "par_k": 4.6,
    },
    {
        "year_tag": "2023_Wet",
        "brdf_dir": BASE_DATA_DIR / "2023_0423_week" / "Brdf_hhh_v2",
        "target_dates": [date(2023, 4, 25)],
        "era5_env_raster": _env_path(
            "FY4_ERA5_ENV_RASTER_2023",
            r"F:\G_disk\FY4\data\ancillary_data\ERA5_hour\Env_daily_1d3d_wet_20230425.tif",
        ),
        "era5_hourly_raster": _env_path(
            "FY4_ERA5_HOURLY_RASTER_2023",
            r"F:\G_disk\FY4\data\ancillary_data\ERA5_hour\ERA5_hourly_wet_20230425_hourly_stack.tif",
        ),
        "drought_tif": DROUGHT_MASK_PATHS["2023"],
        "gee_env_tif": _env_path(
            "FY4_GEE_ENV_TIF_2023",
            r"F:\G_disk\FY4\data\ancillary_data\GEE\Env_factors_20230425.tif",
        ),
        "par_k": 4.5,
    },
]

DTW_SMOOTHING = {"window": 7, "poly": 3}
COMBINED_DTW_SMOOTHING = {"window": 5, "poly": 2}
BRDF_KERNEL_PARAMETERS = {"a": 1, "b": 1}
BRDF_CORRECTION_FACTOR_CLIP = (0.1, 10.0)
METRIC_VISUAL_CLIP_PERCENTILES = (1, 99)
XGB_DEBUG_SUBSAMPLE_N = _env_int("FY4_XGB_DEBUG_SUBSAMPLE_N", 1000)


def build_yearly_class_mapping(scheme_name: str) -> dict:
    """Return the year-tag -> class-set mapping expected by XGBoost scripts."""
    scheme = get_class_scheme(scheme_name)
    return {
        "2022_Drought": {"drought": set(scheme["drought"]), "wet": set(scheme["wet"])},
        "2023_Wet": {"drought": set(scheme["drought"]), "wet": set(scheme["wet"])},
    }


def configuration_summary() -> dict:
    """Small helper used by smoke-check scripts and documentation."""
    return {
        "metric_class_scheme": METRIC_CLASS_SCHEME,
        "correlation_class_scheme": CORRELATION_CLASS_SCHEME,
        "ml_class_scheme": ML_CLASS_SCHEME,
        "metric_dates": {k: [str(vv) for vv in values] for k, values in METRIC_TARGET_DATES.items()},
        "correlation_dates": {k: [str(vv) for vv in values] for k, values in CORRELATION_TARGET_DATES.items()},
        "xgb_years": [cfg["year_tag"] for cfg in XGB_YEARLY_CONFIGS],
    }
