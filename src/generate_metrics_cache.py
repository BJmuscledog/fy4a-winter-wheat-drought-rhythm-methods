# -*- coding: utf-8 -*-
"""Generate cached metric tables used by downstream rhythm-analysis figures.

This script now resolves imports relative to the packaged repository rather
than the original monolithic project tree, which makes the public repository
self-consistent.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

import plot_metrics as pm


def build_cache(force: bool = False) -> None:
    print("=" * 70)
    print("Generating metric cache files (metrics_pix_22.csv / metrics_pix_23.csv)")
    print("=" * 70)

    out_dir = Path(pm.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_22 = out_dir / "metrics_pix_22.csv"
    cache_23 = out_dir / "metrics_pix_23.csv"

    if cache_22.exists() and cache_23.exists() and not force:
        df22 = pd.read_csv(cache_22)
        df23 = pd.read_csv(cache_23)
        print("\n[OK] Cache files already exist:")
        print(f"  - {cache_22} ({len(df22)} rows)")
        print(f"  - {cache_23} ({len(df23)} rows)")
        print("\nUse --force to regenerate them.")
        return

    print("\n[INFO] Building caches from source rasters...")

    sample_candidates = glob.glob(str(Path(pm.BRDF_DIR_2022) / "*.tif")) + glob.glob(
        str(Path(pm.BRDF_DIR_2023) / "*.tif")
    )
    if not sample_candidates:
        raise FileNotFoundError("No .tif files found in BRDF source directories.")

    sample_fp = sample_candidates[0]
    print(f"[INFO] Reference raster: {Path(sample_fp).name}")

    gt, proj, nx, ny = pm.get_raster_info(sample_fp)
    print(f"[INFO] Raster size: {nx} x {ny}")

    print("[INFO] Resampling winter wheat mask...")
    mask_arr = pm.resample_mask_to_ref(pm.WHEAT_MASK_TIF, sample_fp)
    print(f"[INFO] Valid mask pixels: {int(np.sum(mask_arr))}")

    print("\n[INFO] Processing 2022 data...")
    print(f"  Dates: {pm.DATES_2022}")
    print(f"  Hours: {pm.HOURS}")
    df_pixels_22 = pm.build_daily_hourly_pixel_nirv(pm.BRDF_DIR_2022, pm.DATES_2022, pm.HOURS, (ny, nx))

    print("\n[INFO] Processing 2023 data...")
    print(f"  Dates: {pm.DATES_2023}")
    print(f"  Hours: {pm.HOURS}")
    df_pixels_23 = pm.build_daily_hourly_pixel_nirv(pm.BRDF_DIR_2023, pm.DATES_2023, pm.HOURS, (ny, nx))

    print("\n[INFO] Loading study-area boundary...")
    gdf_hhh = gpd.read_file(pm.HHH_SHP_PATH)
    gdf_hhh_4326 = gdf_hhh.to_crs(epsg=4326)

    print("\n[INFO] Applying masks and boundary filters (2022)...")
    processed_pix_22 = pm.build_processed_pixel_values_region(
        df_pixels_22, mask_arr, gt, proj, nx, ny, gdf_hhh_4326, pm.HOURS
    )

    print("\n[INFO] Applying masks and boundary filters (2023)...")
    processed_pix_23 = pm.build_processed_pixel_values_region(
        df_pixels_23, mask_arr, gt, proj, nx, ny, gdf_hhh_4326, pm.HOURS
    )

    print("\n[INFO] Computing 2022 rhythm metrics...")
    metrics_pix_22 = pm.metrics_from_processed_pixels(processed_pix_22, pm.HOURS)
    print(f"  Done: {len(metrics_pix_22)} pixels")

    print("\n[INFO] Computing 2023 rhythm metrics...")
    metrics_pix_23 = pm.metrics_from_processed_pixels(processed_pix_23, pm.HOURS)
    print(f"  Done: {len(metrics_pix_23)} pixels")

    print("\n[INFO] Writing cache files...")
    metrics_pix_22.to_csv(cache_22, index=False)
    metrics_pix_23.to_csv(cache_23, index=False)

    print("\n" + "=" * 70)
    print("[DONE] Metric caches generated successfully.")
    print("=" * 70)
    print(f"  - {cache_22}")
    print(f"  - {cache_23}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cached FY-4A rhythm metric tables.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate cache files even if they already exist.",
    )
    args = parser.parse_args()
    build_cache(force=args.force)


if __name__ == "__main__":
    main()
