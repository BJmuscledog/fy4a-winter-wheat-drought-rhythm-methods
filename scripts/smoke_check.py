"""Repository smoke-check utility.

This script performs a lightweight import/configuration sanity check without
requiring the large local datasets to be present. It reports shared parameter
choices and indicates which expected local paths are missing on the current
machine.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from project_config import (  # noqa: E402
    ANCILLARY_DATA_DIR,
    BASE_DATA_DIR,
    DROUGHT_MASK_PATHS,
    DROUGHT_TIMESERIES_CSV,
    DTW_CSV_2022,
    DTW_CSV_2023,
    HHH_SHP_PATH,
    METRIC_OUTPUT_DIR,
    PAPER_FIG_DIR,
    WHEAT_MASK_TIF,
    XGB_OUTPUT_DIR,
    configuration_summary,
)


def main() -> int:
    print("FY-4A winter wheat drought-rhythm repository smoke check")
    print("=" * 68)
    print("Shared configuration summary:")
    for key, value in configuration_summary().items():
        print(f"  - {key}: {value}")

    print("\nPath availability on this machine:")
    path_checks = {
        "BASE_DATA_DIR": BASE_DATA_DIR,
        "ANCILLARY_DATA_DIR": ANCILLARY_DATA_DIR,
        "PAPER_FIG_DIR": PAPER_FIG_DIR,
        "METRIC_OUTPUT_DIR": METRIC_OUTPUT_DIR,
        "XGB_OUTPUT_DIR": XGB_OUTPUT_DIR,
        "WHEAT_MASK_TIF": WHEAT_MASK_TIF,
        "HHH_SHP_PATH": HHH_SHP_PATH,
        "DTW_CSV_2022": DTW_CSV_2022,
        "DTW_CSV_2023": DTW_CSV_2023,
        "DROUGHT_TIMESERIES_CSV": DROUGHT_TIMESERIES_CSV,
        "DROUGHT_MASK_2022": DROUGHT_MASK_PATHS["2022"],
        "DROUGHT_MASK_2023": DROUGHT_MASK_PATHS["2023"],
    }
    missing = []
    for label, path in path_checks.items():
        exists = Path(path).exists()
        print(f"  - {label}: {'OK' if exists else 'MISSING'} -> {path}")
        if not exists:
            missing.append(label)

    print("\nResult:")
    if missing:
        print("  Smoke check completed with missing local data paths.")
        print("  This is expected on a clean machine; update paths through")
        print("  environment variables documented in configs/environment_variables.example.txt.")
        return 0

    print("  Smoke check completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
