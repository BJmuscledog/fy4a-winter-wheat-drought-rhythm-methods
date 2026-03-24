# -*- coding: utf-8 -*-
"""
plot_nirv_geo_facet.py

Produce a classic multi-panel comparison of NIRv diurnal cycles,
with each subplot corresponding to a geographic grid cell.
- HHH_SHP_PATH is drawn as a global background in light gray.
- Subplots are overlaid on this background, with transparent backgrounds.
- Subplots strictly align with geographic grid lines.
- Each subplot: NIRv diurnal cycle (2022 vs 2023)
- NEW PIXEL-STATISTICS LOGIC:
    1. For each grid cell, for each pixel, for each hour: collect NIRv values from all available dates (e.g., 3 days).
    2. If >= 2 days have values for that pixel-hour: average them.
    3. If 1 day has value for that pixel-hour: use that value.
    4. If 0 days have value: that pixel-hour has no value.
    5. THEN, for each grid cell, for each hour: calculate median, Q25, Q75 from all *processed* pixel-hour values.
- Empty grid cells: marked with a large 'X' (diagonal cross) in light gray.
- IF 2022 HAS COMPLETE 9-16H DATA, THE GRID CELL IS CONSIDERED VALID FOR PLOTTING 2022.
2023 data will be plotted if available, otherwise skipped.
- IF 2022 LACKS COMPLETE 9-16H DATA, THE GRID CELL IS MARKED AS EMPTY ('X').
- All subplot internal axes/grids/titles removed.
- Global geographic grid (lat/lon) labels on the main figure axes.
- Y-axis latitude labels rotated 90 degrees.
- No legend.

Author: Assistant (adapted to user's folder layout and new requirements)
"""
import os
import glob
import re
from datetime import datetime, timedelta, date
import warnings
import traceback
import sys 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from osgeo import gdal, osr
import geopandas as gpd
from shapely.geometry import box
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle 
from shapely.geometry import Point
from project_config import (
    BASE_DATA_DIR,
    DROUGHT_MASK_PATHS,
    HHH_SHP_PATH as SHARED_HHH_SHP_PATH,
    METRIC_CLASS_SCHEME,
    METRIC_HOURS,
    METRIC_OUTPUT_DIR,
    METRIC_SAMPLE_PIXELS,
    METRIC_TARGET_DATES,
    METRIC_VISUAL_CLIP_PERCENTILES,
    WHEAT_MASK_TIF as SHARED_WHEAT_MASK_TIF,
    get_class_scheme,
)
warnings.filterwarnings("ignore")
# 添加在文件顶部的 import 区
from scipy.stats import skew, ttest_rel, wilcoxon
import math
# --- New for drought mask sampling ---
import numpy as np
# Projection constants
TARGET_CRS = 'EPSG:32650'  # UTM Zone 50N for HHH plain
OUTPUT_RESOLUTION = 1000.0  # Output resolution (m), ~1km
# --- GDAL_DATA Environment Variable Fix ---
if 'GDAL_DATA' not in os.environ:
    conda_env_path = os.path.dirname(os.path.dirname(sys.executable)) 
    gdal_data_path_win = os.path.join(conda_env_path, 'Library', 'share', 'gdal') 
    gdal_data_path_linux = os.path.join(conda_env_path, 'share', 'gdal')
    
    if os.path.exists(gdal_data_path_win):
        os.environ['GDAL_DATA'] = gdal_data_path_win
        print(f"Set GDAL_DATA to: {gdal_data_path_win}")
    elif os.path.exists(gdal_data_path_linux):
        os.environ['GDAL_DATA'] = gdal_data_path_linux
        print(f"Set GDAL_DATA to: {gdal_data_path_linux}")
    else:
        print(f"GDAL_DATA path not found. GDAL warnings may persist.")
# --- End GDAL_DATA Fix ---

# ---------------- USER PARAMETERS ----------------
# ... (existing parameters) ...

# --- NEW: Specific Grid Definitions for Individual Plots ---
# Format: (min_lon, max_lon, min_lat, max_lat, name)
SPECIFIC_GRIDS = [
    (113.4, 116.5, 33.3, 36.1, "Specific_Grid_A_113.4E_33.0N"),
    (119.6, 122.7, 37.0, 38.9, "Specific_Grid_B_119.6E_37.0N")
]
# Drought mask (SMPct) paths
DROUGHT_MASKS = {
    "2022": os.fspath(DROUGHT_MASK_PATHS["2022"]),
    "2023": os.fspath(DROUGHT_MASK_PATHS["2023"]),
}
# Classes to keep
DROUGHT_CLASSES = {3, 4}  # 中度+重度
NONDROUGHT_CLASSES = {0}  # 无干旱
# ---------------- USER PARAMETERS ----------------
BASE_DIR = r'F:\风云数据\Fy_p2_data'
BRDF_DIR_2022 = os.path.join(BASE_DIR, r'2022_0423_week\Brdf_hhh')
BRDF_DIR_2023 = os.path.join(BASE_DIR, r'2023_0423_week\Brdf_hhh')

WHEAT_MASK_TIF = r'F:\G_disk\FY4\Winter_wheat_map\Winter_wheat_map.tif'
HHH_SHP_PATH = r'F:\风云数据\Fy_p1_data\Shp\Huanghuaihai\Huanghuaihai.shp' # For defining overall study area extent and background
OUT_DIR = r'F:\FY4\outputs_paper_figs\box_grid_Analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# Shared repository configuration overrides. We keep these assignments here so
# the original script structure stays recognizable, but the effective values
# now come from project_config.py.
CLASS_SCHEME = get_class_scheme(METRIC_CLASS_SCHEME)
DROUGHT_CLASSES = set(CLASS_SCHEME["drought"])
NONDROUGHT_CLASSES = set(CLASS_SCHEME["wet"])
BASE_DIR = os.fspath(BASE_DATA_DIR)
BRDF_DIR_2022 = os.fspath(BASE_DATA_DIR / '2022_0423_week' / 'Brdf_hhh')
BRDF_DIR_2023 = os.fspath(BASE_DATA_DIR / '2023_0423_week' / 'Brdf_hhh')
WHEAT_MASK_TIF = os.fspath(SHARED_WHEAT_MASK_TIF)
HHH_SHP_PATH = os.fspath(SHARED_HHH_SHP_PATH)
OUT_DIR = os.fspath(METRIC_OUTPUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# Cache (for heavy intermediate computations)
USE_CACHE = True
CACHE_DIR = os.path.join(OUT_DIR, "cache_plot_metrics")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_METRICS_22 = os.path.join(CACHE_DIR, "metrics_pix_22n2.csv")
CACHE_METRICS_23 = os.path.join(CACHE_DIR, "metrics_pix_23n2.csv")
CACHE_22_DRY = os.path.join(CACHE_DIR, "metrics_22_dry.csv")
CACHE_22_WET = os.path.join(CACHE_DIR, "metrics_22_wet.csv")
CACHE_23_DRY = os.path.join(CACHE_DIR, "metrics_23_dry.csv")
CACHE_23_WET = os.path.join(CACHE_DIR, "metrics_23_wet.csv")

# 缓存像元级节律指标表（供 Fig_Metrics_Spiral_Maps.ipynb / Fix_Classification_and_Boundary.py 复用）
CACHE_22 = os.path.join(OUT_DIR, 'metrics_pix_22n2.csv')
CACHE_23 = os.path.join(OUT_DIR, 'metrics_pix_23n2.csv')

# Analysis time window (BJT hours)
DATES_2022 = [date(2022,4,d) for d in range(22,24)] # Example: 21, 22 April (2 days)
DATES_2023 = [date(2023,4,d) for d in range(24,26)] # Example: 25, 26 April (2 days)
HOURS = list(range(9,17)) # 9..16 BJT

# Sampling for speed: if mask contains many pixels, sample this many pixels for distribution (None = use all)
SAMPLE_PIXELS = 5000

DATES_2022 = list(METRIC_TARGET_DATES["2022"])
DATES_2023 = list(METRIC_TARGET_DATES["2023"])
HOURS = list(METRIC_HOURS)
SAMPLE_PIXELS = METRIC_SAMPLE_PIXELS

# ============== 统一科研绘图格式配置 ==============
FONT_FAMILY = "Times New Roman"
FONT_SIZE = 9
KEY_FONT_WEIGHT = "bold"
FIG_WIDTH_INCH = 5.75  # ~14.6 cm text width with 3.17 cm margins on A4

sns.set(style="white", rc={'axes.edgecolor': 'black'})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [FONT_FAMILY]
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['axes.labelsize'] = FONT_SIZE
plt.rcParams['axes.titlesize'] = FONT_SIZE
plt.rcParams['axes.labelweight'] = KEY_FONT_WEIGHT
plt.rcParams['axes.titleweight'] = KEY_FONT_WEIGHT
plt.rcParams['xtick.labelsize'] = FONT_SIZE
plt.rcParams['ytick.labelsize'] = FONT_SIZE
plt.rcParams['legend.fontsize'] = FONT_SIZE
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['patch.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.25
plt.rcParams['ytick.major.width'] = 0.25
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300  # High resolution for publication
plt.rcParams['figure.constrained_layout.use'] = False
# Default single-figure width fits ~14.6 cm text block (A4 with 3.17 cm margins)
plt.rcParams['figure.figsize'] = (FIG_WIDTH_INCH, 4.0)

# !!! MODIFIED: Color assignments for 2022 (drought) and 2023 (non-drought)
# Ukraine flag colors: Blue (#0057B7) and Yellow (#FFD700)
COLOR_2022_DROUGHT = "#FFD700" # Yellow for 2022 (drought year) - Ukraine flag yellow
COLOR_2023_NON_DROUGHT = "#0057B7" # Blue for 2023 (non-drought year) - Ukraine flag blue

# --- NEW: Geographic Grid Parameters ---
NUM_ROWS = 6 
NUM_COLS = 4 
SUBPLOT_PANEL_SIZE_INCH = 1.2 
FIG_SCALE_FACTOR = 1.0 

# ---------------- Helpers: IO & parsing ----------------
def read_band_safe(fp, band=1):
    ds = gdal.Open(fp)
    if ds is None:
        print(f"[WARN] Failed to open {fp}")
        return None
    if ds.RasterCount < band:
        print(f"[WARN] Band {band} not found in {fp}")
        return None
    return ds.GetRasterBand(band).ReadAsArray().astype(float)

def resample_mask_to_ref(mask_fp, ref_fp):
    """Resample mask to reference raster grid using GDAL Warp in memory (nearest neighbor)."""
    ref_ds = gdal.Open(ref_fp)
    if ref_ds is None:
        raise FileNotFoundError(f"Reference raster not found: {ref_fp}")
    
    ref_gt = ref_ds.GetGeoTransform()
    ref_proj = ref_ds.GetProjection()
    
    warped = gdal.Warp('', mask_fp, format='MEM',
                        dstSRS=ref_proj,
                        outputBounds=(ref_gt[0], ref_gt[3] + ref_ds.RasterYSize * ref_gt[5],
                                      ref_gt[0] + ref_ds.RasterXSize * ref_gt[1], ref_gt[3]),
                        width=ref_ds.RasterXSize, height=ref_ds.RasterYSize,
                        resampleAlg=gdal.GRA_NearestNeighbour)
    
    if warped is None:
        raise RuntimeError(f"GDAL Warp failed for mask resampling. Check {mask_fp} and {ref_fp}")
    arr = warped.GetRasterBand(1).ReadAsArray()
    return arr.astype(bool)

def get_raster_info(ref_fp):
    """Get georeferencing info from a raster file."""
    ds = gdal.Open(ref_fp)
    if ds is None:
        raise FileNotFoundError(f"Raster not found: {ref_fp}")
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    nx = ds.RasterXSize
    ny = ds.RasterYSize
    ds = None 
    print("[DEBUG] Reference raster GeoTransform (gt):", gt)  # 新增：输出gt数组
    print("[DEBUG] Reference raster Projection WKT (first 200 chars):", proj[:200])  # 新增：输出proj_wkt
    print("[DEBUG] Reference raster dimensions: {}x{}".format(nx, ny))  # 新增：输出尺寸
    return gt, proj, nx, ny

def parse_timestamp_from_filename(name, target_dates):
    """
    Try to parse timestamp from filename (flexible): prefer YYYYMMDDHHMMSS,
    fallback to MMDDHHMMSS plus infer year from target_dates[0].year.
    Returns a datetime (UTC) or None.
    """
    m = re.search(r'(\d{8})(\d{6})', name)
    if m:
        try:
            dt_utc = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            return dt_utc
        except Exception:
            pass
    m2 = re.search(r'(\d{4})(\d{6})', name)
    if m2:
        year = str(target_dates[0].year)
        try:
            dt_utc = datetime.strptime(year + m2.group(1) + m2.group(2), "%Y%m%d%H%M%S")
            return dt_utc
        except Exception:
            pass
    return None

# ---------------- NEW: Build daily-hourly NIRv pixel values ----------------
def build_daily_hourly_pixel_nirv(brdf_dir, target_dates, hours, ref_shape):
    """
    Collect all NIRv values for each pixel for each hour and each day.
    Returns a DataFrame with ['dt_local', 'hour', 'pixel_y', 'pixel_x', 'value'].
    ref_shape is the (height, width) of the reference raster.
    """
    files = sorted(glob.glob(os.path.join(brdf_dir, "*.tif")))
    if len(files) == 0:
        print(f"[WARN] No tif files in {brdf_dir}")
        return pd.DataFrame(columns=['dt_local', 'hour', 'pixel_y', 'pixel_x', 'value'])

    records = []
    map_h, map_w = ref_shape # Use shape from reference raster
    
    for fp in files:
        name = os.path.basename(fp)
        dt_utc = parse_timestamp_from_filename(name, target_dates)
        if dt_utc is None:
            continue
        dt_local = dt_utc + timedelta(hours=8) # convert UTC -> BJT
        
        if dt_local.date() in target_dates and dt_local.hour in hours:
            r_arr = read_band_safe(fp, 2) # Red band
            n_arr = read_band_safe(fp, 3) # NIR band
            
            if r_arr is None or n_arr is None:
                continue

            # Crop to reference shape
            r_arr_crop = r_arr[:map_h, :map_w]
            n_arr_crop = n_arr[:map_h, :map_w]

            with np.errstate(divide='ignore', invalid='ignore'):
                nirv_arr = n_arr_crop * ((n_arr_crop - r_arr_crop) / (n_arr_crop + r_arr_crop + 1e-9))
            
            # Debug: Check raw NIRv statistics for 2023 data
            if dt_local.year == 2023 and len(records) < 3:  # Only print for first few files
                valid_nirv_debug = nirv_arr[np.isfinite(nirv_arr)]
                if len(valid_nirv_debug) > 0:
                    print(f"[DEBUG] {dt_local.strftime('%Y-%m-%d %H:%M')} NIRv stats: "
                          f"n={len(valid_nirv_debug)}, min={np.nanmin(valid_nirv_debug):.4f}, "
                          f"max={np.nanmax(valid_nirv_debug):.4f}, mean={np.nanmean(valid_nirv_debug):.4f}, "
                          f"median={np.nanmedian(valid_nirv_debug):.4f}")
                    print(f"[DEBUG] Raw NIR band stats: min={np.nanmin(n_arr_crop):.4f}, "
                          f"max={np.nanmax(n_arr_crop):.4f}, mean={np.nanmean(n_arr_crop):.4f}")
                    print(f"[DEBUG] Raw Red band stats: min={np.nanmin(r_arr_crop):.4f}, "
                          f"max={np.nanmax(r_arr_crop):.4f}, mean={np.nanmean(r_arr_crop):.4f}")
            
            # Find all valid (non-NaN) NIRv values in this scene
            valid_y, valid_x = np.where(np.isfinite(nirv_arr))
            valid_values = nirv_arr[valid_y, valid_x]
            
            if len(valid_values) > 0:
                df_temp = pd.DataFrame({
                    'dt_local': dt_local.date(), # Store date for daily aggregation
                    'hour': dt_local.hour,
                    'pixel_y': valid_y,
                    'pixel_x': valid_x,
                    'value': valid_values
                })
                records.append(df_temp)
                print(f"[COLLECT] {dt_local.strftime('%Y-%m-%d %H:%M')}: {len(valid_values)} pixels collected.")
    
    if len(records) == 0:
        return pd.DataFrame(columns=['dt_local', 'hour', 'pixel_y', 'pixel_x', 'value'])
    
    df_all_pixels = pd.concat(records, ignore_index=True)
    return df_all_pixels

# ---------------- NEW: Extract and process pixel values then aggregate to grid summaries ----------------
def extract_and_process_pixels_by_grid(df_all_pixels, mask_arr, gt, proj_wkt, nx, ny,
                                       y_bins, x_bins, sample_pixels=None, gdf_extent=None, hours_to_check=None):  # 参数: lat/lon_bins -> y/x_bins
    """
    新的核心处理函数：... (描述不变，但用projected grid)
    返回 DataFrame: ['hour','median','q25','q75','count','y_bin_idx','x_bin_idx']
    """
    if df_all_pixels.empty:
        return pd.DataFrame(columns=['hour','median','q25','q75','count','y_bin_idx','x_bin_idx'])

    # Apply wheat mask
    df_all_pixels_masked = df_all_pixels[mask_arr[df_all_pixels['pixel_y'], df_all_pixels['pixel_x']]].copy()
    
    if df_all_pixels_masked.empty:
        print("[WARN] No pixels remain after applying wheat mask.")
        return pd.DataFrame(columns=['hour','median','q25','q75','count','y_bin_idx','x_bin_idx'])
    
    # Prepare CRS (src and tgt same, projected)
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(proj_wkt)  # Projected proj_wkt
    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromEPSG(32650)  # TARGET_CRS
    coord_trans = osr.CoordinateTransformation(src_srs, tgt_srs)  # Likely identity

    # Build unary_union (projected gdf)
    if gdf_extent is not None:
        gdf_union = gdf_extent.unary_union
    else:
        gdf_union = None

    # Calculate x/y (meters) for masked pixels
    x_maps = np.empty(len(df_all_pixels_masked), dtype=float)  # Eastings
    y_maps = np.empty(len(df_all_pixels_masked), dtype=float)  # Northings

    rows_arr = df_all_pixels_masked['pixel_y'].values
    cols_arr = df_all_pixels_masked['pixel_x'].values
    
    print(f"[INFO] Calculating projected x/y for {len(rows_arr)} pixels...")
    transformed_points = []
    for i in range(len(rows_arr)):
        row = int(rows_arr[i])
        col = int(cols_arr[i])
        x_map = gt[0] + col * gt[1] + row * gt[2]  # Projected gt: Eastings (m)
        y_map = gt[3] + col * gt[4] + row * gt[5]  # Northings (m)
        try:
            xt, yt, _ = coord_trans.TransformPoint(x_map, y_map)  # If identity, xt=x_map
            transformed_points.append((float(xt), float(yt)))
        except Exception:
            transformed_points.append((np.nan, np.nan))
    
    x_maps = np.array([p[0] for p in transformed_points])
    y_maps = np.array([p[1] for p in transformed_points])

    # Sample debug (5 pixels)
    if len(rows_arr) > 0:
        import random
        sample_indices = random.sample(range(len(rows_arr)), min(5, len(rows_arr)))
        for i in sample_indices:
            row = int(rows_arr[i])
            col = int(cols_arr[i])
            x_map = gt[0] + col * gt[1] + row * gt[2]
            y_map = gt[3] + col * gt[4] + row * gt[5]
            try:
                xt, yt, _ = coord_trans.TransformPoint(x_map, y_map)
                print(f"[DEBUG] Sample pixel (row={row}, col={col}): x_map={x_map:.0f}, y_map={y_map:.0f}, x={xt:.0f}, y={yt:.0f}")
            except Exception as e:
                print(f"[DEBUG] Sample pixel transform error: {e}")

    # Filter by projected shapefile extent
    if gdf_union is not None:
        mask_in_shape = np.array([gdf_union.contains(Point(xt, yt)) if (np.isfinite(xt) and np.isfinite(yt)) else False
                                  for xt, yt in zip(x_maps, y_maps)], dtype=bool)
        df_all_pixels_masked = df_all_pixels_masked[mask_in_shape].copy()
        x_maps = x_maps[mask_in_shape]
        y_maps = y_maps[mask_in_shape]
        print(f"[INFO] After projected shapefile mask: {len(df_all_pixels_masked)} pixels remain.")
        print(f"[DEBUG] Before strict geo mask: {len(df_all_pixels_masked)} pixels")
    
    if df_all_pixels_masked.empty:
        print("[WARN] No pixels remain after geographic filtering.")
        return pd.DataFrame(columns=['hour','median','q25','q75','count','y_bin_idx','x_bin_idx'])
    
    # Assign bin indices
    df_all_pixels_masked['y_bin_idx'] = np.digitize(y_maps, y_bins) - 1  # y_bins ascending (min_y south to max_y north)
    df_all_pixels_masked['x_bin_idx'] = np.digitize(x_maps, x_bins) - 1

    # Strict bounding box filter
    grid_min_x, grid_max_x = x_bins.min(), x_bins.max()
    grid_min_y, grid_max_y = y_bins.min(), y_bins.max()
    epsilon = 1e3  # ~1m for meter scale
    
    valid_geo_mask = (x_maps >= grid_min_x - epsilon) & (x_maps <= grid_max_x + epsilon) & \
                     (y_maps >= grid_min_y - epsilon) & (y_maps <= grid_max_y + epsilon)
    
    df_all_pixels_masked = df_all_pixels_masked[valid_geo_mask].copy()
    if df_all_pixels_masked.empty:
        print("[INFO] After strict projected bounding box filter, no pixels remain.")
        return pd.DataFrame(columns=['hour','median','q25','q75','count','y_bin_idx','x_bin_idx'])
    
    df_all_pixels_masked['y_bin_idx'] = np.clip(df_all_pixels_masked['y_bin_idx'], 0, NUM_ROWS - 1)
    df_all_pixels_masked['x_bin_idx'] = np.clip(df_all_pixels_masked['x_bin_idx'], 0, NUM_COLS - 1)

    print(f"[INFO] {len(df_all_pixels_masked)} pixels assigned to projected grid cells. Now performing daily averaging...")

    # --- NEW LOGIC: Daily averaging ...
    def custom_daily_average(series):
        valid_counts = series.count()
        if valid_counts >= 2:
            return series.mean()
        elif valid_counts == 1:
            return series.iloc[0]
        else:
            return np.nan

    processed_pixel_values = df_all_pixels_masked.groupby(
        ['y_bin_idx', 'x_bin_idx', 'pixel_y', 'pixel_x', 'hour']  # y/x
    )['value'].apply(custom_daily_average).reset_index(name='processed_value')

    print(f"[INFO] Daily averaging complete. {len(processed_pixel_values)} processed pixel-hour values generated.")

    # Aggregate
    final_summary = processed_pixel_values.groupby(
        ['y_bin_idx', 'x_bin_idx', 'hour']  # y/x
    )['processed_value'].agg(
        median=lambda x: np.nanmedian(x),
        q25=lambda x: np.nanpercentile(x, 25),
        q75=lambda x: np.nanpercentile(x, 75),
        count='count'
    ).reset_index()

    final_summary['q75_minus_q25'] = final_summary['q75'] - final_summary['q25']
    
    # Reindex
    all_combinations_idx = pd.MultiIndex.from_product([np.arange(NUM_ROWS),
                                                      np.arange(NUM_COLS),
                                                      hours_to_check if hours_to_check is not None else sorted(df_all_pixels_masked['hour'].unique())],
                                                     names=['y_bin_idx', 'x_bin_idx', 'hour'])
    final_summary = final_summary.set_index(['y_bin_idx', 'x_bin_idx', 'hour']).reindex(all_combinations_idx)
    return final_summary.reset_index()


# ---------------- NEW: Plotting Specific Grid Trends ----------------
def plot_specific_grid_trends(summary_data_22, summary_data_23, specific_grids, out_dir, lon_bins, lat_bins, num_rows, num_cols):
    """
    绘制指定地理格网的NIRv日内变化趋势单图，包含坐标轴、图例等。
    现在明确接收 lon_bins 和 lat_bins, 以及 num_rows, num_cols 作为参数。
    """
    print("[INFO] Plotting specific grid trends...")
    
    # 确定全局Y轴范围 (与多面板图保持一致，或单独计算)
    all_q25 = pd.concat([summary_data_22['q25'], summary_data_23['q25']]).dropna()
    all_q75 = pd.concat([summary_data_22['q75'], summary_data_23['q75']]).dropna()
    
    global_min_y = np.nanmin(all_q25) if not all_q25.empty else 0.1
    global_max_y = np.nanmax(all_q75) if not all_q75.empty else 0.6
    if np.isnan(global_min_y) or np.isnan(global_max_y) or global_min_y == global_max_y:
        global_min_y, global_max_y = 0.1, 0.6
    margin = (global_max_y - global_min_y) * 0.15
    global_ylim = (max(0, global_min_y - margin), min(1.0, global_max_y + margin))

    for min_lon, max_lon, min_lat, max_lat, name in specific_grids:
        print(f"  Processing specific grid: {name}")

        # 计算每个大格网的中心点
        large_grid_centers_lon = (lon_bins[:-1] + lon_bins[1:]) / 2
        large_grid_centers_lat = (lat_bins[:-1] + lat_bins[1:]) / 2

        # 找到所有落在目标范围内的 '大' 格网的索引
        valid_large_grid_lon_indices = np.where(
            (large_grid_centers_lon >= min_lon) & (large_grid_centers_lon <= max_lon)
        )[0]
        
        # 原始 lat_bins 是升序的，lat_bin_idx=0 对应的是最高的纬度（也就是最大的 lat_bins[idx+1]）
        # large_grid_centers_lat[idx] 对应的是 lat_bins[idx] 到 lat_bins[idx+1]
        # 如果 lat_bin_idx 0 是最北的行，那么它对应的是 `large_grid_centers_lat` 中最高的索引
        # 所以 `lat_bin_idx` 与 `large_grid_centers_lat` 的索引是反向的
        original_lat_center_indices = np.where(
            (large_grid_centers_lat >= min_lat) & (large_grid_centers_lat <= max_lat)
        )[0]

        # 将原始 lat_center_indices 转换为 'r_idx' (行索引, 0是最高纬度)
        valid_large_grid_lat_indices = [num_rows - 1 - idx for idx in original_lat_center_indices]
        valid_large_grid_lat_indices = sorted(valid_large_grid_lat_indices) # 确保r_idx是顺序的


        if len(valid_large_grid_lon_indices) == 0 or len(valid_large_grid_lat_indices) == 0:
            print(f"  [WARN] No predefined grid cells found within {name}'s boundaries. Skipping this grid.")
            continue

        specific_grid_data_22_list = []
        specific_grid_data_23_list = []

        for r_idx in valid_large_grid_lat_indices:
            for c_idx in valid_large_grid_lon_indices:
                cell_data_22 = summary_data_22[(summary_data_22['lat_bin_idx'] == r_idx) & 
                                               (summary_data_22['lon_bin_idx'] == c_idx)]
                cell_data_23 = summary_data_23[(summary_data_23['lat_bin_idx'] == r_idx) & 
                                               (summary_data_23['lon_bin_idx'] == c_idx)]
                specific_grid_data_22_list.append(cell_data_22)
                specific_grid_data_23_list.append(cell_data_23)
        
        df_22_combined = pd.concat(specific_grid_data_22_list).dropna(subset=['median'])
        df_23_combined = pd.concat(specific_grid_data_23_list).dropna(subset=['median'])

        if df_22_combined.empty and df_23_combined.empty:
            print(f"  [WARN] No data found for specific grid: {name}. Skipping plotting.")
            continue

        aggregated_22 = df_22_combined.groupby('hour').agg(
            median=('median', np.nanmean), 
            q25=('q25', np.nanmean),
            q75=('q75', np.nanmean),
            count=('count', 'sum') 
        ).reset_index()

        aggregated_23 = df_23_combined.groupby('hour').agg(
            median=('median', np.nanmean),
            q25=('q25', np.nanmean),
            q75=('q75', np.nanmean),
            count=('count', 'sum')
        ).reset_index()
        
        # Reindex to ensure all hours are present
        aggregated_22 = aggregated_22.set_index('hour').reindex(HOURS).reset_index()
        aggregated_23 = aggregated_23.set_index('hour').reindex(HOURS).reset_index()

        fig, ax = plt.subplots(figsize=(6, 4)) # 单图尺寸
        
        # Plot 2022 data with transparent lines
        if not aggregated_22['median'].isna().all():
            ax.plot(aggregated_22['hour'], aggregated_22['median'], '-', color=COLOR_2022_DROUGHT, lw=2, marker='o', markersize=5, label='2022 Median', alpha=0.7)
            ax.fill_between(aggregated_22['hour'], aggregated_22['q25'], aggregated_22['q75'], color=COLOR_2022_DROUGHT, alpha=0.25, label='2022 IQR')
        
        # Plot 2023 data with transparent lines
        if not aggregated_23['median'].isna().all():
            ax.plot(aggregated_23['hour'], aggregated_23['median'], '-', color=COLOR_2023_NON_DROUGHT, lw=2, marker='o', markersize=5, label='2023 Median', alpha=0.7)
            ax.fill_between(aggregated_23['hour'], aggregated_23['q25'], aggregated_23['q75'], color=COLOR_2023_NON_DROUGHT, alpha=0.25, label='2023 IQR')
        
        ax.set_ylim(global_ylim)
        ax.set_xlim(min(HOURS)-0.5, max(HOURS)+0.5)
        ax.set_xticks(HOURS[::2]) # 每两个小时一个刻度
        ax.set_xlabel('T_peak (hour of day)', fontsize=FONT_SIZE, fontweight=KEY_FONT_WEIGHT)
        ax.set_ylabel('NIRv', fontsize=FONT_SIZE, fontweight=KEY_FONT_WEIGHT)
        ax.set_title(f'NIRv Diurnal Cycle for {name}', fontsize=FONT_SIZE, fontweight=KEY_FONT_WEIGHT)
        ax.legend(loc='upper left', fontsize=FONT_SIZE)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 添加经纬度标注
        ax.text(0.02, 0.10, f"Lon: {min_lon:.1f}-{max_lon:.1f}°E\nLat: {min_lat:.1f}-{max_lat:.1f}°N",
                transform=ax.transAxes, fontsize=FONT_SIZE, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', lw=0.5, alpha=0.7))

        outfp = os.path.join(out_dir, f"Fig_NIRv_Diurnal_{name}.png")
        fig.savefig(outfp, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[SAVED] Specific grid figure: {outfp}")

print("Specific grid plotting function defined.")
# ---------------- Plotting (no changes from previous version) ----------------
def plot_nirv_geo_faceted(summary_data_22, summary_data_23, gdf_extent, y_bins, x_bins, outfp):  # 参数: lat/lon_bins -> y/x_bins
    """
    Draw a multi-panel (geo-faceted) plot... (Projected version)
    """
    # Calculate figure size (不变)
    grid_width_inches = NUM_COLS * SUBPLOT_PANEL_SIZE_INCH
    grid_height_inches = NUM_ROWS * SUBPLOT_PANEL_SIZE_INCH
    left_margin_inches = 0.7 
    right_margin_inches = 0.3 
    bottom_margin_inches = 0.7 
    top_margin_inches = 0.7 
    fig_width = grid_width_inches + left_margin_inches + right_margin_inches
    fig_height = grid_height_inches + bottom_margin_inches + top_margin_inches
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.patch.set_alpha(0.0)  # Transparent figure
    
    # Define main axes
    map_ax_left = left_margin_inches / fig_width
    map_ax_bottom = bottom_margin_inches / fig_height
    map_ax_width = grid_width_inches / fig_width
    map_ax_height = grid_height_inches / fig_height
    ax_map = fig.add_axes([map_ax_left, map_ax_bottom, map_ax_width, map_ax_height])
    ax_map.patch.set_alpha(0.0)  # Transparent axis
    ax_map.set_aspect('auto') 

    # Set projected extent
    ax_map.set_xlim(x_bins.min(), x_bins.max())  # Eastings
    ax_map.set_ylim(y_bins.min(), y_bins.max())  # Northings
    
    # Plot projected SHP background with visible boundaries (RSE style)
    # Make boundaries more visible to show administrative boundaries clearly
    gdf_extent.plot(ax=ax_map, color='white', edgecolor='#000000', linewidth=1.0, alpha=1.0, zorder=1)

    # Add gridlines
    ax_map.grid(True, linestyle='-', alpha=0.9, color='gray', zorder=2, linewidth=0.7) 
    ax_map.set_xticks(x_bins)
    ax_map.set_yticks(y_bins)
    
    # Set ticks/labels (m units)
    ax_map.tick_params(axis='both', which='major', length=4, labelsize=FONT_SIZE)
    ax_map.set_xticklabels([f'{x:.0f}m' for x in x_bins], rotation=45, ha='right')  # Eastings
    ax_map.set_yticklabels([f'{y:.0f}m' for y in y_bins], rotation=90, va='center')  # Northings
    
    ax_map.set_xlabel('Eastings (m)', fontsize=FONT_SIZE, fontweight=KEY_FONT_WEIGHT, labelpad=10)
    ax_map.set_ylabel('Northings (m)', fontsize=FONT_SIZE, fontweight=KEY_FONT_WEIGHT, labelpad=10)
    
    # Remove spines
    for spine in ax_map.spines.values():
        spine.set_visible(False)

    ax_map.set_title("NIRv Diurnal Comparison Across Projected Grid Cells", 
                     fontsize=FONT_SIZE-1, fontweight=KEY_FONT_WEIGHT, loc='center', y=1.02) 

    # Global Y limits (不变)
    all_q25 = pd.concat([summary_data_22['q25'], summary_data_23['q25']]).dropna()
    all_q75 = pd.concat([summary_data_22['q75'], summary_data_23['q75']]).dropna()
    global_min_y = np.nanmin(all_q25) if not all_q25.empty else 0.1
    global_max_y = np.nanmax(all_q75) if not all_q75.empty else 0.6
    if np.isnan(global_min_y) or np.isnan(global_max_y) or global_min_y == global_max_y:
        global_min_y, global_max_y = 0.1, 0.6 
    margin = (global_max_y - global_min_y) * 0.15
    global_ylim = (max(0, global_min_y - margin), min(1.0, global_max_y + margin)) 

    # Overlay subplots
    for r_idx in range(NUM_ROWS):
        for c_idx in range(NUM_COLS):
            cell_x_start = x_bins[c_idx]  # Eastings
            cell_x_end = x_bins[c_idx+1]
            
            # r_idx=0 top (highest y/Northings)
            cell_y_top = y_bins[NUM_ROWS - r_idx]  # y_bins ascending: min_y (south) to max_y (north)
            cell_y_bottom = y_bins[NUM_ROWS - 1 - r_idx]
            
            # Inset axes
            ax = inset_axes(
                ax_map, 
                width="100%", height="100%", 
                bbox_to_anchor=(cell_x_start, cell_y_bottom, cell_x_end - cell_x_start, cell_y_top - cell_y_bottom),
                bbox_transform=ax_map.transData, 
                loc='lower left', borderpad=0
            )
            
            ax.patch.set_alpha(0.0) 
            
            # Get data (y/x idx)
            cell_data_22 = summary_data_22[(summary_data_22['y_bin_idx'] == r_idx) & 
                                           (summary_data_22['x_bin_idx'] == c_idx)]
            cell_data_23 = summary_data_23[(summary_data_23['y_bin_idx'] == r_idx) & 
                                           (summary_data_23['x_bin_idx'] == c_idx)]
            
            # Check complete data (不变)
            summary_22_reindexed = cell_data_22.set_index('hour').reindex(HOURS)
            is_complete_22 = (summary_22_reindexed['median'].notna().sum() == len(HOURS)) and \
                             ((summary_22_reindexed['count'] > 0).all())
            
            if not is_complete_22:
                ax.plot([0, 1], [0, 1], color='lightgray', linestyle='-', lw=2, transform=ax.transAxes)
                ax.plot([0, 1], [1, 0], color='lightgray', linestyle='-', lw=2, transform=ax.transAxes)
            else:
                med_22 = summary_22_reindexed['median'].values
                q25_22 = summary_22_reindexed['q25'].values
                q75_22 = summary_22_reindexed['q75'].values
                # Transparent lines to show underlying shapefile boundaries
                ax.plot(HOURS, med_22, '-', color=COLOR_2022_DROUGHT, lw=1.5, marker='o', markersize=3, 
                       markeredgecolor='none', alpha=0.6, zorder=3)
                ax.fill_between(HOURS, q25_22, q75_22, color=COLOR_2022_DROUGHT, alpha=0.2, linewidth=0, zorder=2)
                
                summary_23_reindexed = cell_data_23.set_index('hour').reindex(HOURS)
                is_complete_23 = (summary_23_reindexed['median'].notna().sum() == len(HOURS)) and \
                                 ((summary_23_reindexed['count'] > 0).all())
                if is_complete_23:
                    med_23 = summary_23_reindexed['median'].values
                    q25_23 = summary_23_reindexed['q25'].values
                    q75_23 = summary_23_reindexed['q75'].values
                    # Transparent lines to show underlying shapefile boundaries
                    ax.plot(HOURS, med_23, '-', color=COLOR_2023_NON_DROUGHT, lw=1.5, marker='o', markersize=3, 
                           markeredgecolor='none', alpha=0.6, zorder=3)
                    ax.fill_between(HOURS, q25_23, q75_23, color=COLOR_2023_NON_DROUGHT, alpha=0.2, linewidth=0, zorder=2)
            
            # Styling (不变)
            ax.set_ylim(global_ylim)
            ax.set_xlim(min(HOURS)-0.5, max(HOURS)+0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.5)

    fig.savefig(outfp, dpi=300, bbox_inches='tight', transparent=True)  # Add transparent=True for PNG
    plt.close(fig)
    print(f"[SAVED] Figure: {outfp}")


# ------------------------- 区域级别（整片区域）像元指标计算与绘图 -------------------------

def build_processed_pixel_values_region(df_all_pixels, mask_arr, gt, proj_wkt, nx, ny, gdf_extent, hours_to_check):
    """
    ... 返回 ['pixel_y','pixel_x','hour','processed_value','lon','lat']
    """
    if df_all_pixels.empty:
        return pd.DataFrame(columns=['pixel_y','pixel_x','hour','processed_value','lon','lat'])

    # Filter mask range
    valid_mask_idx = (df_all_pixels['pixel_y'] >= 0) & (df_all_pixels['pixel_y'] < mask_arr.shape[0]) & \
                     (df_all_pixels['pixel_x'] >= 0) & (df_all_pixels['pixel_x'] < mask_arr.shape[1])
    df = df_all_pixels[valid_mask_idx].copy()
    print(f"[DEBUG] After mask range filter: {len(df)} pixels")
    if df.empty:
        return pd.DataFrame(columns=['pixel_y','pixel_x','hour','processed_value','lon','lat'])

    # Apply wheat mask
    unique_pixels = df[['pixel_y','pixel_x']].drop_duplicates()
    mask_vals = []
    for _, row in unique_pixels.iterrows():
        py, px = int(row['pixel_y']), int(row['pixel_x'])
        try:
            mask_vals.append(bool(mask_arr[py, px]))
        except Exception:
            mask_vals.append(False)
    unique_pixels['mask'] = mask_vals
    keep_pixels = unique_pixels[unique_pixels['mask']][['pixel_y','pixel_x']]

    if keep_pixels.empty:
        print("[WARN] No pixels inside wheat mask for region-level processing.")
        return pd.DataFrame(columns=['pixel_y','pixel_x','hour','processed_value','lon','lat'])

    # Merge masked
    df = df.merge(keep_pixels, on=['pixel_y','pixel_x'], how='inner')
    print(f"[DEBUG] After wheat mask merge: {len(df)} pixels")

    # Compute lon/lat (geographic coordinates)
    # Since the raster is already in WGS84, we can directly compute lon/lat from GeoTransform
    # GeoTransform: (ulx, xres, xskew, uly, yskew, yres)
    # For geographic: xres > 0 (east), yres < 0 (north to south)
    py_arr = keep_pixels['pixel_y'].values
    px_arr = keep_pixels['pixel_x'].values
    
    # Calculate lon/lat directly from GeoTransform (WGS84)
    # lon = ulx + px * xres + py * xskew
    # lat = uly + px * yskew + py * yres
    lon_arr = gt[0] + px_arr * gt[1] + py_arr * gt[2]
    lat_arr = gt[3] + px_arr * gt[4] + py_arr * gt[5]
    
    keep_pixels = keep_pixels.copy()
    keep_pixels['lon'] = lon_arr
    keep_pixels['lat'] = lat_arr
    
    print(f"[DEBUG] Coordinate ranges: lon=[{lon_arr.min():.4f}, {lon_arr.max():.4f}], lat=[{lat_arr.min():.4f}, {lat_arr.max():.4f}]")
    print(f"[DEBUG] HHH extent bounds: lon=[{gdf_extent.total_bounds[0]:.4f}, {gdf_extent.total_bounds[2]:.4f}], lat=[{gdf_extent.total_bounds[1]:.4f}, {gdf_extent.total_bounds[3]:.4f}]")

    # Filter by HHH extent using geopandas (more reliable than Point.contains)
    if gdf_extent is not None:
        # Create GeoDataFrame from pixels
        pixel_gdf = gpd.GeoDataFrame(
            keep_pixels,
            geometry=gpd.points_from_xy(keep_pixels['lon'], keep_pixels['lat']),
            crs='EPSG:4326'
        )
        
        # Ensure gdf_extent is in the same CRS
        if gdf_extent.crs != pixel_gdf.crs:
            gdf_extent = gdf_extent.to_crs(pixel_gdf.crs)
        
        # Use spatial join or within check (more reliable than contains)
        # Option 1: Use within (point within polygon)
        pixel_gdf['within'] = pixel_gdf.geometry.within(gdf_extent.unary_union)
        keep_pixels_filtered = pixel_gdf[pixel_gdf['within']][['pixel_y', 'pixel_x', 'lon', 'lat']].copy()
        
        print(f"[DEBUG] Before extent filter: {len(keep_pixels)} unique pixels")
        print(f"[DEBUG] After HHH extent filter: {len(keep_pixels_filtered)} pixels remain")
        
        if len(keep_pixels_filtered) == 0:
            print("[WARN] After HHH extent filtering, no pixels remain.")
            print(f"[DEBUG] Sample pixel coordinates: first 5 pixels")
            for i in range(min(5, len(keep_pixels))):
                row = keep_pixels.iloc[i]
                print(f"  Pixel ({row['pixel_y']}, {row['pixel_x']}): lon={row['lon']:.4f}, lat={row['lat']:.4f}")
            # Try a bounding box check as fallback
            bbox = gdf_extent.total_bounds
            bbox_mask = (
                (keep_pixels['lon'] >= bbox[0]) & (keep_pixels['lon'] <= bbox[2]) &
                (keep_pixels['lat'] >= bbox[1]) & (keep_pixels['lat'] <= bbox[3])
            )
            keep_pixels_bbox = keep_pixels[bbox_mask].copy()
            print(f"[DEBUG] Bounding box filter: {len(keep_pixels_bbox)} pixels (fallback)")
            if len(keep_pixels_bbox) == 0:
                return pd.DataFrame(columns=['pixel_y','pixel_x','hour','processed_value','lon','lat'])
            else:
                keep_pixels_filtered = keep_pixels_bbox
                print("[INFO] Using bounding box filter instead of geometry contains check")
        
        # Reduce df to only include pixels within extent
        df = df.merge(keep_pixels_filtered[['pixel_y','pixel_x']], on=['pixel_y','pixel_x'], how='inner')
        print(f"[DEBUG] After extent filter: {len(df)} pixels, py range {df['pixel_y'].min()}-{df['pixel_y'].max()}")
        
        # Create pix_to_ll dictionary from filtered pixels
        pix_to_ll = {}
        for _, row in keep_pixels_filtered.iterrows():
            pix_to_ll[(int(row['pixel_y']), int(row['pixel_x']))] = (float(row['lon']), float(row['lat']))
    else:
        # No extent filtering, create pix_to_ll from all keep_pixels
        pix_to_ll = {}
        for _, row in keep_pixels.iterrows():
            pix_to_ll[(int(row['pixel_y']), int(row['pixel_x']))] = (float(row['lon']), float(row['lat']))

    # Daily average
    def custom_daily_average(series):
        valid_counts = series.count()
        if valid_counts >= 2:
            return series.mean()
        elif valid_counts == 1:
            return series.iloc[0]
        else:
            return np.nan

    grouped = df.groupby(['pixel_y','pixel_x','hour'])['value'].apply(custom_daily_average).reset_index(name='processed_value')
    # Attach lon/lat
    lons = []; lats = []
    for _, row in grouped.iterrows():
        key = (int(row['pixel_y']), int(row['pixel_x']))
        ll = pix_to_ll.get(key, (np.nan, np.nan))
        lons.append(ll[0]); lats.append(ll[1])
    grouped['lon'] = lons
    grouped['lat'] = lats

    # Keep hours
    grouped = grouped[grouped['hour'].isin(hours_to_check)].copy()
    if grouped.empty:
        print("[WARN] No processed pixel-hour results for given hours.")
        return pd.DataFrame(columns=['pixel_y','pixel_x','hour','processed_value','lon','lat'])
    
    print(f"[DEBUG] Final grouped rows: {len(grouped)}, py range {grouped['pixel_y'].min()}-{grouped['pixel_y'].max()}")
    return grouped

# -------------- 复用的指标计算函数（与之前一致） ----------------
def compute_metrics_from_hourly(hours_list, med_vals):
    hrs = list(hours_list)
    vals = np.array(med_vals, dtype=float)
    if np.all(np.isnan(vals)):
        return {k: np.nan for k in ['MDI','t_peak','A_NIRv','Skew','Recovery_rate','NIRv_integral','centroid','centroid_shift']}
    # indices
    def idxs(subhrs):
        return [hrs.index(h) for h in subhrs if h in hrs]
    morning_idx = idxs([9,10,11])
    noon_idx = idxs([12,13,14])
    afternoon_idx = idxs([15,16])
    max_val = np.nanmax(vals)
    min_val = np.nanmin(vals)
    A_NIRv = max_val - min_val
    try:
        t_peak = int(hrs[int(np.nanargmax(vals))])
    except Exception:
        t_peak = np.nan
    mean_morning = np.nan
    min_noon = np.nan
    if len(morning_idx) > 0:
        mean_morning = np.nanmean(vals[morning_idx])
    if len(noon_idx) > 0:
        min_noon = np.nanmin(vals[noon_idx])
    MDI = np.nan
    if not np.isnan(mean_morning) and mean_morning != 0 and not np.isnan(min_noon):
        MDI = 1.0 - (min_noon / mean_morning)
    skewness = np.nan
    try:
        if np.sum(~np.isnan(vals)) >= 3:
            skewness = float(skew(vals[~np.isnan(vals)]))
    except Exception:
        skewness = np.nan
    recovery_rate = np.nan
    if (len(noon_idx) > 0) and (len(afternoon_idx) > 0):
        mean_afternoon = np.nanmean(vals[afternoon_idx])
        time_diff = (np.nanmean([h for h in afternoon_idx]) - np.nanmean([h for h in noon_idx])) if (len(noon_idx)>0 and len(afternoon_idx)>0) else 1.0
        if not np.isnan(mean_afternoon) and not np.isnan(min_noon) and time_diff != 0 and abs(min_noon) > 1e-9:
            recovery_rate = (mean_afternoon - min_noon) / time_diff / (abs(min_noon))
    # Calculate integral (AUC) - handle NaN values properly
    integral = np.nan
    try:
        # Filter out NaN values for integration
        valid_mask = ~np.isnan(vals)
        n_valid = np.sum(valid_mask)
        if n_valid >= 2:  # Need at least 2 points for integration
            valid_vals = vals[valid_mask]
            valid_hrs = np.array(hrs)[valid_mask]
            # Ensure hours are sorted for integration
            sort_idx = np.argsort(valid_hrs)
            valid_vals_sorted = valid_vals[sort_idx]
            valid_hrs_sorted = valid_hrs[sort_idx]
            # Use trapezoidal rule for integration
            integral = float(np.trapz(valid_vals_sorted, valid_hrs_sorted))
            # Check if result is valid and reasonable
            if not np.isfinite(integral):
                integral = np.nan
            # Additional check: integral should be positive for NIRv values
            # (NIRv is typically positive, so integral over time should be positive)
            # But we don't filter negative values here, just check for validity
    except Exception as e:
        # Debug: print error if needed
        # print(f"[DEBUG] Integral calculation error: {e}")
        integral = np.nan
    centroid = np.nan
    centroid_shift = np.nan
    if np.nansum(vals) > 0:
        centroid = float(np.nansum(np.array(hrs) * vals) / np.nansum(vals))
        centroid_shift = centroid - float(np.mean(hrs))
    return {
        'MDI': MDI,
        't_peak': t_peak,
        'A_NIRv': A_NIRv,
        'Skew': skewness,
        'Recovery_rate': recovery_rate,
        'NIRv_integral': integral,
        'centroid': centroid,
        'centroid_shift': centroid_shift
    }


def metrics_from_processed_pixels(processed_pixel_values_df, hours_list):
    if processed_pixel_values_df.empty:
        return pd.DataFrame(columns=['pixel_y','pixel_x','lon','lat'] + ['MDI','t_peak','A_NIRv','Skew','Recovery_rate','NIRv_integral','centroid','centroid_shift'])
    # pivot
    pivot = processed_pixel_values_df.pivot_table(index=['pixel_y','pixel_x','lon','lat'], columns='hour', values='processed_value', aggfunc='first')
    pivot = pivot.reindex(columns=hours_list)
    records = []
    integral_count = 0
    integral_nan_count = 0
    for idx, row in pivot.iterrows():
        py, px, lon, lat = idx
        med_vals = row.values.tolist()
        mets = compute_metrics_from_hourly(hours_list, med_vals)
        mets.update({'pixel_y': int(py), 'pixel_x': int(px), 'lon': float(lon) if np.isfinite(lon) else np.nan, 'lat': float(lat) if np.isfinite(lat) else np.nan})
        records.append(mets)
        # Debug NIRv_integral
        if np.isfinite(mets.get('NIRv_integral', np.nan)):
            integral_count += 1
        else:
            integral_nan_count += 1
    
    if len(records) == 0:
        return pd.DataFrame(columns=['pixel_y','pixel_x','lon','lat'] + ['MDI','t_peak','A_NIRv','Skew','Recovery_rate','NIRv_integral','centroid','centroid_shift'])
    
    result_df = pd.DataFrame(records)
    print(f"[DEBUG] NIRv_integral stats: valid={integral_count}, NaN={integral_nan_count}, total={len(records)}")
    if integral_count > 0:
        valid_integrals = result_df['NIRv_integral'].dropna()
        print(f"[DEBUG] NIRv_integral range: [{valid_integrals.min():.6f}, {valid_integrals.max():.6f}], mean={valid_integrals.mean():.6f}")
    return result_df


# ---------------- Drought mask filtering (pixel-level metrics) ----------------
def sample_mask_values(mask_fp, lon, lat):
    """Sample mask values by lon/lat arrays from a raster."""
    ds = gdal.Open(mask_fp)
    if ds is None:
        raise FileNotFoundError(f"Mask not found: {mask_fp}")
    gt = ds.GetGeoTransform()
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    x = ((lon - gt[0]) / gt[1]).astype(int)
    y = ((lat - gt[3]) / gt[5]).astype(int)
    h, w = arr.shape
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = np.full(len(lon), np.nan)
    out[valid] = arr[y[valid], x[valid]]
    return out


def filter_metrics_by_mask(metrics_df, mask_fp, keep_classes, debug_metric=None):
    """Filter pixel-level metrics by drought mask classes."""
    if metrics_df.empty:
        return metrics_df
    if 'lon' not in metrics_df.columns or 'lat' not in metrics_df.columns:
        print("[WARN] Metrics DF missing lon/lat; cannot apply drought mask filter.")
        return metrics_df
    lon = pd.to_numeric(metrics_df['lon'], errors='coerce').to_numpy()
    lat = pd.to_numeric(metrics_df['lat'], errors='coerce').to_numpy()
    mask_vals = sample_mask_values(mask_fp, lon, lat)
    df = metrics_df.copy()
    df['drought_class'] = mask_vals
    
    # 调试信息：检查掩膜值分布
    unique_classes, counts = np.unique(mask_vals[~np.isnan(mask_vals)], return_counts=True)
    print(f"[DEBUG] Mask value distribution: {dict(zip(unique_classes, counts))}")
    print(f"[DEBUG] Before filtering: n={len(df)}, keep_classes={keep_classes}")
    
    df = df[df['drought_class'].isin(keep_classes)]
    
    # 调试信息：检查过滤后的数据
    print(f"[DEBUG] After filtering: n={len(df)}")
    if debug_metric and debug_metric in df.columns:
        vals = df[debug_metric].astype(float).values
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            print(f"[DEBUG] {debug_metric} after mask: n={len(vals)}, median={np.nanmedian(vals):.4f}, "
                  f"mean={np.nanmean(vals):.4f}, min={np.nanmin(vals):.4f}, max={np.nanmax(vals):.4f}")
    
    df = df.drop(columns=['drought_class'])
    return df


def paired_stat_and_ci(arr_a, arr_b, n_boot=2000, random_seed=42):
    a = np.array(arr_a, dtype=float)
    b = np.array(arr_b, dtype=float)
    valid = (~np.isnan(a)) & (~np.isnan(b))
    a = a[valid]; b = b[valid]
    n = len(a)
    if n == 0:
        return {'n_pairs': 0, 'mean_a': np.nan, 'mean_b': np.nan, 'mean_diff': np.nan, 't_p': np.nan, 'wilcoxon_p': np.nan, 'ci_low': np.nan, 'ci_high': np.nan}
    mean_a = float(np.nanmean(a)); mean_b = float(np.nanmean(b))
    mean_diff = float(np.nanmean(a - b))
    try:
        t_stat, t_p = ttest_rel(a, b)
    except Exception:
        t_p = np.nan
    wilc_p = np.nan
    try:
        if n >= 3:
            wilc_stat, wilc_p = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
    except Exception:
        wilc_p = np.nan
    rng = np.random.RandomState(random_seed)
    diffs = a - b
    bs_means = []
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        bs_means.append(np.mean(diffs[idx]))
    ci_low, ci_high = np.percentile(bs_means, [2.5, 97.5])
    return {'n_pairs': n, 'mean_a': mean_a, 'mean_b': mean_b, 'mean_diff': mean_diff, 't_p': float(t_p) if not np.isnan(t_p) else np.nan, 'wilcoxon_p': float(wilc_p) if not np.isnan(wilc_p) else np.nan, 'ci_low': float(ci_low), 'ci_high': float(ci_high)}


def plot_region_metrics_boxplots(metrics_pix_22, metrics_pix_23, out_dir, metrics_to_plot=None,
                                 figsize=None, dpi=300, show_mean=True):
    """
    Publication-style boxplots for region-level per-pixel metrics (2022 vs 2023),
    with distinct colors for the two years (uses COLOR_2022_DROUGHT and COLOR_2023_NON_DROUGHT).
    Replace the previous version with this one.
    """
    import matplotlib.patches as mpatches
    import matplotlib as mpl
    mpl.rcParams['font.family'] = plt.rcParams.get('font.family', 'Arial')
    mpl.rcParams['font.size'] = 9

    if figsize is None:
        figsize = (FIG_WIDTH_INCH, 5.0)
    if metrics_to_plot is None:
        metrics_to_plot = ['MDI','t_peak','A_NIRv','Skew','Recovery_rate','NIRv_integral','centroid_shift']

    palette = {'2022': COLOR_2022_DROUGHT, '2023': COLOR_2023_NON_DROUGHT}

    # merge on pixel coords (inner) -> paired pixels
    merged = pd.merge(metrics_pix_22, metrics_pix_23, on=['pixel_y','pixel_x'], suffixes=('_22','_23'), how='inner')
    if merged.empty:
        print("[WARN] No paired pixels between years for region-level metrics.")
    results = []

    # p -> stars
    def p_to_stars(p):
        if np.isnan(p):
            return 'ns'
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        return 'ns'

    title_map = {
        'MDI': 'Midday depression index (MDI)',
        't_peak': 'Hour of maximum NIRv (T_peak)',
        'A_NIRv': 'Amplitude of NIRv (A_NIRv)',
        'Skew': 'Skewness of diurnal NIRv (Sk_NIRv)',
        'Recovery_rate': 'Recovery rate of NIRv (R_rate)',
        'NIRv_integral': 'NIRv integral over 9–16h (NIRv_AUC)',
        'centroid_shift': 'Centroid shift from mean hour (C_shift)'
    }
    
    ylabel_map = {
        'MDI': 'MDI',
        't_peak': 'T_peak (hour of day)',
        'A_NIRv': 'A_NIRv',
        'Skew': 'Skewness',
        'Recovery_rate': 'R_rate',
        'NIRv_integral': 'NIRv_AUC',
        'centroid_shift': 'C_shift (hours)'
    }

    for metric in metrics_to_plot:
        col22 = metric + '_22'
        col23 = metric + '_23'
        if col22 not in merged.columns or col23 not in merged.columns:
            print(f"[WARN] Missing metric {metric}, skip.")
            continue
        
        arr22 = merged[col22].values.astype(float)
        arr23 = merged[col23].values.astype(float)
        
        # Debug: check data availability for NIRv_integral
        if metric == 'NIRv_integral':
            valid_22_before = np.sum(~np.isnan(arr22))
            valid_23_before = np.sum(~np.isnan(arr23))
            valid_both_before = np.sum((~np.isnan(arr22)) & (~np.isnan(arr23)))
            print(f"[DEBUG] {metric} BEFORE filtering: 2022 valid={valid_22_before}/{len(arr22)}, 2023 valid={valid_23_before}/{len(arr23)}, paired valid={valid_both_before}")
            if valid_22_before > 0:
                print(f"[DEBUG] {metric} 2022 range: [{np.nanmin(arr22):.6f}, {np.nanmax(arr22):.6f}], mean={np.nanmean(arr22):.6f}")
            if valid_23_before > 0:
                print(f"[DEBUG] {metric} 2023 range: [{np.nanmin(arr23):.6f}, {np.nanmax(arr23):.6f}], mean={np.nanmean(arr23):.6f}")

        # 新增：硬阈值滤除特定指标离群值
        if metric == 'Recovery_rate':
            # 移除2023 >100 或 < -10 (paired点)
            outlier_mask = np.abs(arr23) > 100  # 或具体: (arr23 > 100) | (arr23 < -10)
            arr22 = np.where(outlier_mask, np.nan, arr22)
            arr23 = np.where(outlier_mask, np.nan, arr23)
            print(f"[INFO] Filtered {np.sum(outlier_mask)} outliers for {metric} (Recovery_rate 2023 extremes)")
        elif metric == 'centroid_shift':
            # 移除 >5 (绝对值)
            outlier_mask = (np.abs(arr22) > 5) | (np.abs(arr23) > 5)  # 加括号，用逻辑 | (已隐式)
            arr22 = np.where(outlier_mask, np.nan, arr22)
            arr23 = np.where(outlier_mask, np.nan, arr23)
            print(f"[INFO] Filtered {np.sum(outlier_mask)} outliers for {metric} (>5 abs)")

        # 重新valid_mask
        valid_mask = (~np.isnan(arr22)) & (~np.isnan(arr23))
        arr22 = arr22[valid_mask]
        arr23 = arr23[valid_mask]

        # 新增：质量控制 - 滤除指定指标的离群值 (IQR方法)
        outlier_metrics = ['A_NIRv', 'MDI', 'centroid_shift', 'Recovery_rate']
        if metric in outlier_metrics and len(arr22) >= 2:
            # 计算paired IQR阈值 (只在paired有效数据上)
            valid_mask = (~np.isnan(arr22)) & (~np.isnan(arr23))
            arr22_valid = arr22[valid_mask]
            arr23_valid = arr23[valid_mask]
            
            # IQR阈值 (对每个年份分别计算，保守滤除)
            q1_22, q3_22 = np.nanpercentile(arr22_valid, [25, 75])
            iqr_22 = q3_22 - q1_22
            lower_22, upper_22 = q1_22 - 1.5 * iqr_22, q3_22 + 1.5 * iqr_22
            
            q1_23, q3_23 = np.nanpercentile(arr23_valid, [25, 75])
            iqr_23 = q3_23 - q1_23
            lower_23, upper_23 = q1_23 - 1.5 * iqr_23, q3_23 + 1.5 * iqr_23
            
            # 滤除：移除两个年份中任一为离群的paired点 (保守，保持paired)
            outlier_mask = ((arr22 < lower_22) | (arr22 > upper_22) | (arr23 < lower_23) | (arr23 > upper_23)) & valid_mask
            arr22 = np.where(outlier_mask, np.nan, arr22)
            arr23 = np.where(outlier_mask, np.nan, arr23)
            
            # 新增：空数组检查 (避免percentile unpack错误)
            valid_mask = (~np.isnan(arr22)) & (~np.isnan(arr23))
            arr22_valid = arr22[valid_mask]
            arr23_valid = arr23[valid_mask]
            if len(arr22_valid) < 2:  # 太少数据，跳过滤除
                print(f"[WARN] Too few valid pairs for {metric} (n={len(arr22_valid)}), skipping outlier filter.")
            else:
                print(f"[INFO] Filtered {np.sum(outlier_mask)} outliers for {metric} (paired n={len(arr22_valid)} after filter)")
        
        # Re-compute final valid mask after all filtering
        valid_mask_final = (~np.isnan(arr22)) & (~np.isnan(arr23))
        if len(arr22) > 0 and len(arr23) > 0:
            arr22_final = arr22[valid_mask_final]
            arr23_final = arr23[valid_mask_final]
        else:
            arr22_final = np.array([], dtype=float)
            arr23_final = np.array([], dtype=float)
        
        # Debug: check final data for NIRv_integral
        if metric == 'NIRv_integral':
            print(f"[DEBUG] {metric} AFTER all filtering: paired valid={len(arr22_final)}/{len(merged)}")
            if len(arr22_final) > 0:
                print(f"[DEBUG] {metric} final range: 2022=[{np.nanmin(arr22_final):.6f}, {np.nanmax(arr22_final):.6f}], 2023=[{np.nanmin(arr23_final):.6f}, {np.nanmax(arr23_final):.6f}]")
        
        # Use final filtered arrays
        if len(arr22_final) == 0:
            print(f"[WARN] No valid paired data for {metric} after filtering. Skipping plot.")
            continue
        
        arr22 = arr22_final
        arr23 = arr23_final
        stat = paired_stat_and_ci(arr22, arr23)  # 基于最终滤除后数据

        # build plotting dataframe (paired)
        df_plot = pd.DataFrame({
            'value': np.concatenate([arr22, arr23]),
            'year': ['2022'] * len(arr22) + ['2023'] * len(arr23),
            'pair_id': list(range(len(arr22))) + list(range(len(arr23)))
        })
        df_plot = df_plot[np.isfinite(df_plot['value'])]

        # Calculate y-axis limits based on data statistics BEFORE plotting
        # This ensures the boxplot fits well without being compressed or cut off
        # Special handling for t_peak to make it more spacious
        if len(arr22) > 0 and len(arr23) > 0:
            bp_data_22 = arr22[~np.isnan(arr22)]
            bp_data_23 = arr23[~np.isnan(arr23)]
            
            if len(bp_data_22) > 0 and len(bp_data_23) > 0:
                # Calculate percentiles for proper y-axis range
                q1_22, q3_22 = np.percentile(bp_data_22, [25, 75])
                q1_23, q3_23 = np.percentile(bp_data_23, [25, 75])
                iqr_22 = q3_22 - q1_22
                iqr_23 = q3_23 - q1_23
                
                # Use 1.5*IQR rule for whiskers, but ensure we include all data within reasonable bounds
                lower_whisker = min(q1_22 - 1.5*iqr_22, q1_23 - 1.5*iqr_23, np.nanmin(bp_data_22), np.nanmin(bp_data_23))
                upper_whisker = max(q3_22 + 1.5*iqr_22, q3_23 + 1.5*iqr_23, np.nanmax(bp_data_22), np.nanmax(bp_data_23))
                
                # Adjust margin based on metric type
                # For t_peak (hour values), use smaller margin for tighter fit
                # ========== T_PEAK Y-LIMIT MODIFICATION ==========
        if metric == 't_peak':
            y_min_plot = 5  # 固定 Y 轴最小值
            y_max_plot = 20 # 固定 Y 轴最大值
            data_range = y_max_plot - y_min_plot
        else:
            # --- 保留所有其他指标的原始动态范围逻辑 ---
            if len(arr22) > 0 and len(arr23) > 0:
                bp_data_22 = arr22[~np.isnan(arr22)]
                bp_data_23 = arr23[~np.isnan(arr23)]
                
                if len(bp_data_22) > 0 and len(bp_data_23) > 0:
                    # Calculate percentiles for proper y-axis range
                    q1_22, q3_22 = np.percentile(bp_data_22, [25, 75])
                    q1_23, q3_23 = np.percentile(bp_data_23, [25, 75])
                    iqr_22 = q3_22 - q1_22
                    iqr_23 = q3_23 - q1_23
                    
                    lower_whisker = min(q1_22 - 1.5*iqr_22, q1_23 - 1.5*iqr_23, np.nanmin(bp_data_22), np.nanmin(bp_data_23))
                    upper_whisker = max(q3_22 + 1.5*iqr_22, q3_23 + 1.5*iqr_23, np.nanmax(bp_data_22), np.nanmax(bp_data_23))
                    
                    # (原 t_peak 的特殊逻辑已移除，因为它现在在上面的 if 块中处理)
                    data_range = upper_whisker - lower_whisker
                    if data_range > 0:
                        margin = data_range * 0.03
                        y_min_plot = lower_whisker - margin
                        y_max_plot = upper_whisker + margin
                    else:
                        y_min_plot = lower_whisker - 0.1
                        y_max_plot = upper_whisker + 0.1
                else:
                    # Fallback to data min/max
                    y_min_plot = min(np.nanmin(arr22), np.nanmin(arr23))
                    y_max_plot = max(np.nanmax(arr22), np.nanmax(arr23))
                    data_range = y_max_plot - y_min_plot
                    margin = data_range * 0.05 if data_range > 0 else 0.1
                    y_min_plot = y_min_plot - margin
                    y_max_plot += margin
            else:
                y_min_plot, y_max_plot = 0.0, 1.0
                data_range = 1.0
        # ========== Y-LIMIT MODIFICATION END ==========
        sns.set_style("white")
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # basic boxplot (seaborn will fill boxes with palette if provided)
        order = ['2022','2023']
        sns.boxplot(x='year', y='value', data=df_plot, order=order, ax=ax,
                    showcaps=True, showfliers=False, saturation=0.75,
                    palette=[palette['2022'], palette['2023']],
                    boxprops={'zorder':2, 'alpha':0.9},
                    medianprops={'color':'black', 'linewidth':2.0},
                    whiskerprops={'color':'black','linewidth':0.9},
                    capprops={'color':'black','linewidth':0.9})
        
        # Set y-axis limits immediately after plotting to ensure proper fit
        ax.set_ylim(bottom=y_min_plot, top=y_max_plot)

        # Post-process artists to ensure boxed edge = black & facecolor kept
        for i, artist in enumerate(ax.artists):
            face = [palette['2022'], palette['2023']][i]
            artist.set_edgecolor('black')
            artist.set_facecolor(face)
            artist.set_alpha(0.9)
            artist.set_linewidth(0.8)

        # overlay jittered points per year with matching colors (slightly transparent)
        for y_idx, year in enumerate(order):
            sub = df_plot[df_plot['year'] == year]
            sns.stripplot(x='year', y='value', data=sub, order=[year], ax=ax,
                          size=2.6, jitter=0.22, color=palette[year], alpha=0.45, dodge=True)

        # mark means with white diamond and black edge
        if show_mean:
            means = [np.nanmean(arr22), np.nanmean(arr23)]
            for i, m in enumerate(means):
                ax.scatter(i, m, marker='D', s=46, facecolors='white', edgecolors='black', linewidths=0.8, zorder=10)

        # add a small legend (patches) - RSE style
        p1 = mpatches.Patch(facecolor=palette['2022'], edgecolor='black', label='Drought year')
        p2 = mpatches.Patch(facecolor=palette['2023'], edgecolor='black', label='Non-drought year')
        ax.legend(handles=[p1,p2], loc='upper right', fontsize=FONT_SIZE, frameon=False, framealpha=0.9)

# aesthetics - RSE journal style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE, width=0.8)
        ax.set_xlabel('')
        
        # ========== 在这里添加修改 ==========
        # 将 "2022" 和 "2023" 替换为 RSE 标签
        ax.set_xticklabels(['Drought year', 'Non-drought year'], fontsize=FONT_SIZE, fontweight=KEY_FONT_WEIGHT)
        # ========== 修改结束 ==========
        
        # Set Y-axis label (simplified)
        ylabel = ylabel_map.get(metric, metric)
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE, fontweight=KEY_FONT_WEIGHT)
        
        
        # significance annotation - RSE journal style (规范化格式)
        x1, x2 = 0, 1
        y_range = y_max_plot - y_min_plot
        
        # Adjust y_loc based on metric type (t_peak needs more space)
        if metric == 't_peak':
            y_loc = y_max_plot + 0.12 * y_range  # More space for t_peak
        else:
            y_loc = y_max_plot + 0.10 * y_range  # Standard space
        
        # Draw bracket line (horizontal line with vertical ends)
        bracket_height = 0.015 * y_range
        ax.plot([x1, x1, x2, x2], [y_loc - bracket_height, y_loc, y_loc, y_loc - bracket_height], 
               lw=1.2, color='black', clip_on=False)
        
        star = p_to_stars(stat['t_p'])
        
        # Format p-value (TeX style with italic p)
        if stat['t_p'] < 0.001:
            p_text = r"$p$ < 0.001"
        elif np.isnan(stat['t_p']):
            p_text = r"$p$ = NA"
        else:
            p_text = r"$p$ = {:.3f}".format(stat['t_p'])
        
        # Format CI (保持统一格式，对齐)
        ci_text = "95% CI [{:.3g}, {:.3g}]".format(stat['ci_low'], stat['ci_high'])
        
        # Significance stars (bold, centered above bracket)
        if star != 'ns':
            ax.text((x1 + x2) * 0.5, y_loc + 0.02*y_range, star, 
                   ha='center', va='bottom', fontsize=FONT_SIZE, fontweight=KEY_FONT_WEIGHT, color='black')
        
        # p-value and CI (centered, below bracket, TeX rendered)
        annotation_text = p_text + "\n" + ci_text
        ax.text((x1 + x2) * 0.5, y_loc - 0.035*y_range, annotation_text, 
               ha='center', va='top', fontsize=FONT_SIZE, color='black')

        # Expand y-limit to include significance annotation
        # Adjust limits based on metric type
        if metric == 't_peak':
            top_limit = y_loc + 0.16*y_range  # More space for t_peak annotation
        else:
            top_limit = y_loc + 0.14*y_range
        ax.set_ylim(bottom=y_min_plot, top=top_limit)
        
        # Add sample sizes below X-axis labels (using axes coordinates for proper positioning)
        n_22 = len(arr22)
        n_23 = len(arr23)
# NEW (y 坐标从 -0.12 改为 -0.08):
        ax.text(0, -0.08, f'n = {n_22:,}', transform=ax.transAxes,
                ha='center', va='top', fontsize=FONT_SIZE, color='black')

        
        # NEW (y 坐标从 -0.12 改为 -0.08):
        ax.text(1, -0.08, f'n = {n_23:,}', transform=ax.transAxes,
                ha='center', va='top', fontsize=FONT_SIZE, color='black')

        fig.tight_layout()
        outfp_png = os.path.join(out_dir, f"Region_Box_{metric}.png")
        fig.savefig(outfp_png, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
        # optionally save vector PDF (uncomment if you want)
        # outfp_pdf = os.path.join(out_dir, f"Region_Box_{metric}.pdf")
        # fig.savefig(outfp_pdf, dpi=300, bbox_inches='tight', pad_inches=0.02, format='pdf')
        plt.close(fig)
        print(f"[SAVED] Publication-style region boxplot: {outfp_png}")

        results.append((metric, stat))

    # print summary
    print("\n--- Region-level metrics paired comparison summary (publication style) ---")
    print("metric, n_pairs, mean_2022, mean_2023, mean_diff, t_p, wilcoxon_p, ci_low, ci_high")
    for metric, stat in results:
        print(f"{metric}, {stat['n_pairs']}, {stat['mean_a']:.6g}, {stat['mean_b']:.6g}, {stat['mean_diff']:.6g}, "
              f"{stat['t_p']:.3e}, {stat['wilcoxon_p']:.3e}, {stat['ci_low']:.6g}, {stat['ci_high']:.6g}")

    return results


def plot_violin_box_metrics_2x3(metrics_pix_22, metrics_pix_23, out_dir,
                                figsize=None, dpi=300):
    """
    绘制 2×3 的 violin + boxplot 组合图，用于六个指标的干旱年 vs 非干旱年对比。
    顺序：A_NIRv, centroid_shift, MDI | Recovery_rate, Skew, t_peak
    """
    import matplotlib as mpl
    import matplotlib.patches as mpatches
    from matplotlib.ticker import LinearLocator, FuncFormatter
    from scipy.stats import mannwhitneyu

    if metrics_pix_22.empty or metrics_pix_23.empty:
        print("[WARN] Empty metrics for one of the years; skip violin+box 2x3.")
        return

    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['axes.edgecolor'] = 'black'
    mpl.rcParams['axes.linewidth'] = 1.2
    mpl.rcParams['xtick.major.width'] = 1.2
    mpl.rcParams['ytick.major.width'] = 1.2

    # 颜色：乌克兰配色（亮黄、亮蓝）
    color_dry = "#FFD700"
    color_wet = "#1E90FF"
    palette = {'Dry': color_dry, 'Wet': color_wet}

    metrics_order = ['A_NIRv', 'centroid_shift', 'MDI', 'Recovery_rate', 'Skew', 't_peak']
    ylabel_map = {
        'A_NIRv': 'A_NIRv',
        'centroid_shift': 'C_shift (hours)',
        'MDI': 'MDI',
        'Recovery_rate': 'R_rate',
        'Skew': 'Skewness',
        't_peak': 'T_peak (hours)'
    }
    panel_letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    merged = pd.merge(metrics_pix_22, metrics_pix_23, on=['pixel_y', 'pixel_x'], suffixes=('_22', '_23'), how='inner')
    if merged.empty:
        print("[WARN] No paired pixels between years for violin+box 2x3.")
        return

    if figsize is None:
        figsize = (FIG_WIDTH_INCH, 7.5)
    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
    axes = axes.flatten()

    # 小数位规则
    def tick_formatter_for(metric: str):
        fmt = "%.2f"
        return FuncFormatter(lambda x, pos: fmt % x)

    # 小提琴宽度与带宽调整
    width_default = 0.7
    width_thin = 0.55  # c/d/e 更瘦
    bw_default = 0.35
    bw_tpeak = 0.6

    def iqr_limits(a: np.ndarray, b: np.ndarray) -> tuple:
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if len(a) == 0 or len(b) == 0:
            vals = np.concatenate([a, b]) if len(a)+len(b) else np.array([0.0])
            lo, hi = np.nanmin(vals), np.nanmax(vals)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = -1.0, 1.0
            margin = (hi - lo) * 0.06
            return lo - margin, hi + margin
        q1a, q3a = np.percentile(a, [25, 75]); iqra = q3a - q1a
        q1b, q3b = np.percentile(b, [25, 75]); iqrb = q3b - q1b
        lo = min(q1a - 1.5*iqra, q1b - 1.5*iqrb, np.nanmin(a), np.nanmin(b))
        hi = max(q3a + 1.5*iqra, q3b + 1.5*iqrb, np.nanmax(a), np.nanmax(b))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = (np.nanmin(np.concatenate([a, b])), np.nanmax(np.concatenate([a, b])))
        margin = (hi - lo) * 0.06 if hi > lo else 0.1
        return lo - margin, hi + margin

    def p_to_stars(p: float) -> str:
        if not np.isfinite(p):
            return "ns"
        if p < 1e-3:
            return "***"
        if p < 1e-2:
            return "**"
        if p < 5e-2:
            return "*"
        return "ns"

    def stars_vertical(star: str) -> str:
        if star == "***":
            return "*\n*\n*"
        if star == "**":
            return "*\n*"
        if star == "*":
            return "*"
        return ""

    def cliffs_delta_from_u(n1: int, n2: int, u_stat: float) -> float:
        # Cliff's delta via U statistic: d = 2U/(n1*n2) - 1
        if n1 <= 0 or n2 <= 0 or not np.isfinite(u_stat):
            return np.nan
        return (2.0 * float(u_stat)) / float(n1 * n2) - 1.0

    for idx, metric in enumerate(metrics_order):
        ax = axes[idx]
        col22 = metric + '_22'
        col23 = metric + '_23'
        if col22 not in merged.columns or col23 not in merged.columns:
            ax.set_visible(False)
            continue

        arr22 = merged[col22].astype(float).values
        arr23 = merged[col23].astype(float).values

        # 与现有 boxplot 清理策略一致的关键规则
        if metric == 'Recovery_rate':
            mask = np.abs(arr23) > 100
            arr22 = np.where(mask, np.nan, arr22)
            arr23 = np.where(mask, np.nan, arr23)
        elif metric == 'centroid_shift':
            mask = (np.abs(arr22) > 5) | (np.abs(arr23) > 5)
            arr22 = np.where(mask, np.nan, arr22)
            arr23 = np.where(mask, np.nan, arr23)
        # 进一步温和去除极端值：对于 R_rate 与 MDI 使用 1-99 分位裁剪
        if metric in ('Recovery_rate', 'MDI'):
            vals = np.concatenate([arr22[np.isfinite(arr22)], arr23[np.isfinite(arr23)]])
            if vals.size >= 10:
                p1, p99 = np.percentile(vals, [5, 95])
                arr22 = np.where((arr22 < p1) | (arr22 > p99), np.nan, arr22)
                arr23 = np.where((arr23 < p1) | (arr23 > p99), np.nan, arr23)

        valid = np.isfinite(arr22) & np.isfinite(arr23)
        arr22 = arr22[valid]; arr23 = arr23[valid]
        if len(arr22) == 0 or len(arr23) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=FONT_SIZE, color='gray', transform=ax.transAxes)
            ax.set_axis_off()
            continue

        # y 轴范围
        if metric == 't_peak':
            y_min, y_max = 5, 20
        else:
            y_min, y_max = iqr_limits(arr22, arr23)

        df_plot = pd.DataFrame({
            'group': np.array(['Dry'] * len(arr22) + ['Wet'] * len(arr23)),
            'value': np.concatenate([arr22, arr23])
        })

        # violin（横向，透明度 0.6，展示 KDE）
        # 选择宽度与带宽
        v_width = width_default
        if metric in ('MDI', 'Recovery_rate', 'Skew'):
            v_width = width_thin
        v_bw = bw_tpeak if metric == 't_peak' else bw_default

        v = sns.violinplot(
            y='group', x='value', data=df_plot, ax=ax,
            order=['Dry', 'Wet'], palette=[palette['Dry'], palette['Wet']],
            inner=None, cut=0, linewidth=1.2, saturation=1.0, scale='width',
            width=v_width, bw=v_bw
        )
        # 设置透明度
        for coll in ax.collections:
            try:
                coll.set_alpha(0.6)
            except Exception:
                pass

        # 叠加 box（横向），红色中位线、黑色边框，保留帽线
        sns.boxplot(
            y='group', x='value', data=df_plot, ax=ax,
            order=['Dry', 'Wet'], width=0.25, showfliers=False,
            boxprops={'facecolor':'none', 'edgecolor':'black', 'linewidth':1.6, 'zorder':3},
            medianprops={'color':'red', 'linewidth':2.0},
            whiskerprops={'color':'black','linewidth':1.3},
            capprops={'color':'black','linewidth':1.3}
        )

        # Q25/Q75 短实线（与箱体留一点空隙）
        try:
            q1_22, q3_22 = np.percentile(arr22, [25, 75])
            q1_23, q3_23 = np.percentile(arr23, [25, 75])
            tick_half = 0.12  # 垂向长度的一半
            # Dry at y=0
            ax.plot([q1_22, q1_22], [0 - tick_half, 0 + tick_half], color='black', lw=1.4, zorder=4)
            ax.plot([q3_22, q3_22], [0 - tick_half, 0 + tick_half], color='black', lw=1.4, zorder=4)
            # Wet at y=1
            ax.plot([q1_23, q1_23], [1 - tick_half, 1 + tick_half], color='black', lw=1.4, zorder=4)
            ax.plot([q3_23, q3_23], [1 - tick_half, 1 + tick_half], color='black', lw=1.4, zorder=4)
        except Exception:
            pass

        # x=0 参考线（横向）
        if np.isfinite(y_min) and np.isfinite(y_max):
            ax.axvline(0, color='black', linestyle='--', linewidth=1.1, zorder=1, alpha=0.6)

        # 轴范围与网格
        ax.set_xlim(y_min, y_max) if metric == 't_peak' else ax.set_xlim(y_min, y_max)
        # 垂直方向三条灰线（定义 3 行）：-0.5, 0.5, 1.5
        for yline in [-0.5, 0.5, 1.5]:
            ax.axhline(yline, color='#e0e0e0', linestyle='-', linewidth=1.0, zorder=0)
        # X 方向 4 列（5 条竖线，含边界刻度）
        ax.xaxis.set_major_locator(LinearLocator(5))
        ax.xaxis.set_major_formatter(tick_formatter_for(metric))
        ax.grid(True, axis='x', color='#e0e0e0', linestyle='-', linewidth=0.9, alpha=0.9)
        ax.set_facecolor('white')
        ax.tick_params(axis='both', labelsize=FONT_SIZE, width=1.2, length=5)
        ax.set_ylabel('')  # Y 显示分组标签即可
        ax.set_xlabel(ylabel_map.get(metric, metric), fontsize=FONT_SIZE, fontweight=KEY_FONT_WEIGHT)
        # 仅在首列子图（a, d）保留 Dry/Wet 标注，其余列移除以压缩横向间距
        if idx in (0, 3):  # 第一列
            ax.set_yticklabels(['Dry', 'Wet'], fontsize=FONT_SIZE, fontweight=KEY_FONT_WEIGHT)
        else:
            ax.set_yticklabels([])
            ax.set_yticks([])
        for tick in ax.get_xticklabels():
            tick.set_fontweight('bold')
        for tick in ax.get_yticklabels():
            tick.set_fontweight('bold')
        # 全边框保留
        for side in ['top','right','bottom','left']:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.3)

        # 显著性
        try:
            res = mannwhitneyu(arr22, arr23, alternative='two-sided')
            u_stat, p = float(res.statistic), float(res.pvalue)
        except Exception:
            u_stat, p = np.nan, np.nan
        star = p_to_stars(p)
        # 右侧中部：靠近 x 轴最大值
        xlo, xhi = ax.get_xlim()
        x_range = xhi - xlo
        x_star = xhi - 0.02 * x_range
        yticks = ax.get_yticks()
        y_star = np.mean(yticks) if len(yticks) > 0 else 0.5
        ax.text(x_star, y_star, stars_vertical(star), ha='right', va='center',
                fontsize=FONT_SIZE, fontweight=KEY_FONT_WEIGHT, color='black', clip_on=False)

        # Cliff's delta（打印，不画图）
        d = cliffs_delta_from_u(len(arr22), len(arr23), u_stat)
        try:
            print(f"[EFFECT] {metric}: Cliff's d = {d:.3f} (n_dry={len(arr22)}, n_wet={len(arr23)})")
        except Exception:
            print(f"[EFFECT] {metric}: Cliff's d = {d}")

        # 子图角标
        ax.text(0.02, 0.98, panel_letters[idx], transform=ax.transAxes,
                ha='left', va='top', fontsize=FONT_SIZE, fontweight=KEY_FONT_WEIGHT,
                fontfamily=FONT_FAMILY)

    # 整图图例
    handles = [
        mpatches.Patch(facecolor=color_dry, edgecolor='black', linewidth=1.2, label='Dry (drought year)'),
        mpatches.Patch(facecolor=color_wet, edgecolor='black', linewidth=1.2, label='Wet (non-drought year)'),
    ]
    legend = fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False,
                        bbox_to_anchor=(0.5, 1.04), prop={'size':12})
    for text in legend.get_texts():
        text.set_fontweight('bold')

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_png = os.path.join(out_dir, "Fig_Metrics_ViolinBox_2x3.png")
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    # 可选矢量
    try:
        out_svg = os.path.join(out_dir, "Fig_Metrics_ViolinBox_2x3.svg")
        fig.savefig(out_svg, dpi=dpi, bbox_inches='tight', pad_inches=0.05, format='svg')
    except Exception:
        pass
    try:
        out_pdf = os.path.join(out_dir, "Fig_Metrics_ViolinBox_2x3.pdf")
        fig.savefig(out_pdf, dpi=dpi, bbox_inches='tight', pad_inches=0.05, format='pdf')
    except Exception:
        pass
    plt.close(fig)
    print(f"[SAVED] 2x3 violin+box composite: {out_png}")


# === New: Single-metric vertical violin with four groups (22_dry,22_wet,23_dry,23_wet) ===
def plot_violin_four_groups_grid(metrics_groups, metrics_to_plot, out_dir,
                                 fig_width_cm=17.0, dpi=300):
    """
    Draw a single 3x2 grid of vertical violin+box plots (four groups each),
    plus a 4th row legend matching DY/NDY x DL/NDL labels.
    """
    import matplotlib as mpl
    import matplotlib.patches as mpatches
    from scipy.stats import mannwhitneyu
    mpl.rcParams['axes.edgecolor'] = 'black'
    mpl.rcParams['axes.linewidth'] = 1.0

    palette = {
        # 2022：更鲜亮的深浅黄
        '2022_dry': "#f5a623",   # 深黄橙
        '2022_wet': "#ffe68a",   # 浅亮黄
        # 2023：更鲜亮的深浅蓝
        '2023_dry': "#2b83ba",   # 深蓝
        '2023_wet': "#7fb6ff",   # 浅亮蓝
    }

    order = ['2022_dry', '2022_wet', '2023_dry', '2023_wet']
    label_map = {
        '2022_dry': 'Dryland 2022',
        '2022_wet': 'Nondryland 2022',
        '2023_dry': 'Dryland 2023',
        '2023_wet': 'Nondryland 2023',
    }

    def iqr_limits(arrs):
        vals = np.concatenate(arrs)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            return (np.nanmin(vals) if len(vals) else -1.0,
                    np.nanmax(vals) if len(vals) else 1.0)
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        lo = min(lo, np.nanmin(vals))
        hi = max(hi, np.nanmax(vals))
        if lo == hi:
            lo -= 0.1
            hi += 0.1
        return lo, hi

    def p_to_stars(p):
        """Convert p-value to significance stars."""
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    panel_letters = ['a', 'b', 'c', 'd', 'e', 'f']

    # Figure layout: 3 rows x 2 cols, plus a legend row
    fig_width_in = fig_width_cm / 2.54
    fig_height_in = fig_width_in * 1.75
    fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.18], hspace=0.25, wspace=0.25)

    for idx, metric in enumerate(metrics_to_plot):
        rows = []
        for key in order:
            df = metrics_groups.get(key, pd.DataFrame())
            if metric in df.columns:
                vals = df[metric].astype(float).values
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    rows.append(pd.DataFrame({'group': key, 'value': vals}))
        if len(rows) == 0:
            print(f"[WARN] No data for metric {metric} across all groups; skip.")
            continue
        df_plot = pd.concat(rows, ignore_index=True)
        if df_plot.empty:
            print(f"[WARN] Empty plot dataframe for metric {metric}; skip.")
            continue

        # 对于 centroid_shift，过滤掉小于 -1.3 的值
        if metric == 'centroid_shift':
            n_before = len(df_plot)
            df_plot = df_plot[df_plot['value'] >= -1.3].copy()
            n_after = len(df_plot)
            n_filtered = n_before - n_after
            if n_filtered > 0:
                print(f"[INFO] {metric}: Filtered out {n_filtered} values < -1.3 (before: {n_before}, after: {n_after})")
            if df_plot.empty:
                print(f"[WARN] No data remaining for {metric} after filtering; skip.")
                continue

        # 百分位裁剪（仅用于可视化，不影响统计检验）
        # 所有指标都使用 1-99 百分位
        vals_all = df_plot['value'].to_numpy()
        vals_all = vals_all[np.isfinite(vals_all)]
        if len(vals_all) > 5:
            p_low, p_high = np.percentile(vals_all, METRIC_VISUAL_CLIP_PERCENTILES)
            # 调试信息：检查裁剪前的数据分布
            print(f"[INFO] {metric}: Before clipping - n_total={len(vals_all)}, "
                  f"median={np.nanmedian(vals_all):.4f}, mean={np.nanmean(vals_all):.4f}, "
                  f"min={np.nanmin(vals_all):.4f}, max={np.nanmax(vals_all):.4f}, "
                  f"percentiles: p1={p_low:.4f}, p99={p_high:.4f}")
            df_plot['value_plot'] = df_plot['value'].clip(p_low, p_high)
            print(f"[INFO] {metric}: Clipped to [{p_low:.4f}, {p_high:.4f}] for visualization")
            # 打印每个组的样本量和统计信息（裁剪前）
            for g in order:
                vals_g = df_plot.loc[df_plot['group'] == g, 'value'].values
                vals_g = vals_g[np.isfinite(vals_g)]
                if len(vals_g) > 0:
                    print(f"[INFO] {metric} {g}: n={len(vals_g)}, median={np.nanmedian(vals_g):.4f}, "
                          f"mean={np.nanmean(vals_g):.4f}, min={np.nanmin(vals_g):.4f}, max={np.nanmax(vals_g):.4f}, "
                          f"q25={np.nanpercentile(vals_g, 25):.4f}, q75={np.nanpercentile(vals_g, 75):.4f}")
        else:
            df_plot['value_plot'] = df_plot['value']

        # 分组统计
        grouped_vals = [df_plot[df_plot['group'] == g]['value_plot'].values for g in order if not df_plot[df_plot['group'] == g].empty]
        med_per_group = []
        # 用于显著性检验：使用原始数据（value列，不裁剪）
        # 统计检验基于原始数据，百分位裁剪仅用于可视化
        group_data_raw = {}
        for g in order:
            vals_plot = df_plot.loc[df_plot['group'] == g, 'value_plot'].values  # 用于可视化（裁剪后）
            vals_for_test = df_plot.loc[df_plot['group'] == g, 'value'].values  # 用于统计检验（原始数据，不裁剪）
            group_data_raw[g] = vals_for_test
            med_per_group.append(np.nanmedian(vals_plot) if len(vals_plot) else np.nan)
        # 导出绘图用数据（每个指标一份）
        csv_path = os.path.join(out_dir, f"Violin4_{metric}_data.csv")
        df_export = df_plot[['group', 'value', 'value_plot']].copy()
        df_export.to_csv(csv_path, index=False)
        print(f"[CSV] Saved violin data for {metric} -> {csv_path}")

        # 预先计算 y 轴范围，供柱形底部使用
        if metric == 't_peak':
            y_min_base, y_max_base = 5, 20
        else:
            y_min_base, y_max_base = iqr_limits(grouped_vals)
            y_range = y_max_base - y_min_base
            margin = y_range * 0.06 if y_range > 0 else 0.1
            y_min_base -= margin
            y_max_base += margin

        # 压缩布局：为显著性标记留出空间（控制在子图内部）
        data_range = y_max_base - y_min_base
        y_max_plot = y_max_base + 0.22 * data_range

        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        # 1) 浅色柱形：从 0 到中位数
        bar_colors = [palette[g] for g in order]
        bar_heights = [max(0, m - y_min_base) if np.isfinite(m) else 0 for m in med_per_group]
        ax.bar(range(len(order)), bar_heights, bottom=y_min_base,
               color=bar_colors, alpha=0.25, width=0.55, zorder=1)

        # 2) 小提琴（无须线，只显示 KDE）
        sns.violinplot(
            data=df_plot, x='group', y='value_plot', order=order,
            palette=[palette[g] for g in order], inner=None, cut=0,
            saturation=1.0, linewidth=1.0, width=0.7, alpha=0.7, ax=ax
        )

        # 3) 中位数红线 + 四分位黑线（无散点）
        for xi, g in enumerate(order):
            vals = df_plot.loc[df_plot['group'] == g, 'value_plot'].values
            if len(vals) == 0:
                continue
            med = np.nanmedian(vals)
            q1, q3 = np.nanpercentile(vals, [25, 75])
            ax.hlines(med, xi - 0.22, xi + 0.22, colors='red', linewidth=2.0, zorder=4)
            ax.hlines(q1, xi - 0.18, xi + 0.18, colors='black', linewidth=1.2, zorder=4)
            ax.hlines(q3, xi - 0.18, xi + 0.18, colors='black', linewidth=1.2, zorder=4)

        # 4) 显著性检验：使用Mann-Whitney U + Bonferroni校正（快速版本，与2×3小提琴图一致）
        # 使用原始数据（不裁剪）进行统计检验
        significance_pairs = []
        
        def cliffs_delta_from_u(n1, n2, u_stat):
            """快速计算Cliff's delta：从U统计量直接计算（与2×3小提琴图一致）"""
            if n1 <= 0 or n2 <= 0 or not np.isfinite(u_stat):
                return np.nan
            # Cliff's delta via U statistic: d = 2U/(n1*n2) - 1
            return (2.0 * float(u_stat)) / float(n1 * n2) - 1.0
        
        # 直接进行两两比较，使用Bonferroni校正（不进行Kruskal-Wallis预检验以加速）
        n_comparisons = len(order) * (len(order) - 1) // 2  # 6对比较
        
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                g1, g2 = order[i], order[j]
                vals1_raw = group_data_raw.get(g1, np.array([]))
                vals2_raw = group_data_raw.get(g2, np.array([]))
                
                # 过滤NaN值
                vals1_raw = vals1_raw[np.isfinite(vals1_raw)]
                vals2_raw = vals2_raw[np.isfinite(vals2_raw)]
                
                if len(vals1_raw) >= 3 and len(vals2_raw) >= 3:
                    try:
                        # Mann-Whitney U检验（快速）
                        res = mannwhitneyu(vals1_raw, vals2_raw, alternative='two-sided')
                        u_stat, p_raw = float(res.statistic), float(res.pvalue)
                        
                        # Bonferroni校正
                        p_corrected = min(p_raw * n_comparisons, 1.0)
                        
                        # 快速计算效应量（从U统计量）
                        effect_size = cliffs_delta_from_u(len(vals1_raw), len(vals2_raw), u_stat)
                        abs_effect = abs(effect_size) if np.isfinite(effect_size) else 0
                        
                        # 使用校正后的p值
                        star = p_to_stars(p_corrected)
                        
                        # 对于大样本量，需要同时满足统计显著性和效应量阈值
                        min_effect_size = 0.1
                        is_large_sample = len(vals1_raw) > 1000 or len(vals2_raw) > 1000
                        has_meaningful_effect = abs_effect >= min_effect_size if is_large_sample else True
                        
                        if star and star != "ns" and has_meaningful_effect:
                            # 使用裁剪后的数据计算y位置
                            vals1_plot = df_plot.loc[df_plot['group'] == g1, 'value_plot'].values
                            vals2_plot = df_plot.loc[df_plot['group'] == g2, 'value_plot'].values
                            significance_pairs.append({
                                'i': i, 'j': j,
                                'g1': g1, 'g2': g2,
                                'p': p_corrected, 'p_raw': p_raw, 'star': star,
                                'effect_size': effect_size,
                                'n1': len(vals1_raw), 'n2': len(vals2_raw),
                                'y1': np.nanmax(vals1_plot) if len(vals1_plot) > 0 else y_max_base,
                                'y2': np.nanmax(vals2_plot) if len(vals2_plot) > 0 else y_max_base
                            })
                            print(f"[SIG] {metric}: {g1} vs {g2}: p_raw={p_raw:.4e}, p_corrected={p_corrected:.4e}, "
                                  f"star={star}, effect_size={effect_size:.4f}, n1={len(vals1_raw)}, n2={len(vals2_raw)}")
                        elif star and star != "ns" and not has_meaningful_effect:
                            print(f"[SIG] {metric}: {g1} vs {g2}: p_corrected={p_corrected:.4e} (significant) but "
                                  f"effect_size={effect_size:.4f} < {min_effect_size} (too small, not shown)")
                    except Exception as e:
                        print(f"[WARN] Mann-Whitney U test failed for {metric} {g1} vs {g2}: {e}")
                        pass

        # 绘制显著性标记
        if significance_pairs:
            # 按y值排序，从高到低排列标记
            significance_pairs.sort(key=lambda x: max(x['y1'], x['y2']), reverse=True)
            
            # 计算标记位置（在数据区域上方，20%空间内）
            y_marker_base = y_max_base + (-0.04) * data_range
            marker_spacing = 0.065 * data_range
            star_offset = 0.04 * data_range
            
            # 限制最多显示6对比较
            max_pairs = min(5, len(significance_pairs))
            
            for idx_pair, pair in enumerate(significance_pairs[:max_pairs]):
                i, j = pair['i'], pair['j']
                star = pair['star']
                if star == "ns":
                    continue
                y_marker = y_marker_base + idx_pair * marker_spacing
                
                # 确保标记在y轴范围内
                if y_marker > y_max_plot - 0.01 * data_range:
                    continue
                
                # 绘制连接线
                ax.plot([i, j], [y_marker, y_marker], 
                        color='black', linewidth=1.4, zorder=5, clip_on=True)
                # 绘制星号
                ax.text((i + j) / 2, y_marker + star_offset, star,
                        ha='center', va='top', fontsize=FONT_SIZE + 1, fontweight='bold',
                        color='black', zorder=6, clip_on=True)

        # y-limits（包含显著性标记区域）
        ax.set_ylim(y_min_base, y_max_plot)

        ax.set_xticklabels([])  # 不显示横坐标标签
        ax.tick_params(axis='y', labelsize=FONT_SIZE, pad=1)
        from matplotlib.ticker import FormatStrFormatter
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        ax.grid(True, axis='y', linestyle='--', alpha=0.35, linewidth=0.9)

        ax.set_xlabel('')
        ylabel_map = {'centroid_shift': 'C_shift', 't_peak': 'T_peak', 'Recovery_rate': 'R_rate'}
        ax.set_ylabel(ylabel_map.get(metric, metric), fontsize=FONT_SIZE, fontweight=KEY_FONT_WEIGHT)

        ax.text(0.02, 0.94, panel_letters[idx], transform=ax.transAxes,
                ha='left', va='top', fontsize=FONT_SIZE, fontweight=KEY_FONT_WEIGHT)

    # Legend row (4th row)
    ax_leg = fig.add_subplot(gs[3, :])
    ax_leg.axis('off')
    handles = [
        mpatches.Patch(facecolor=palette['2022_dry'], edgecolor='black', linewidth=1.0, label='DY-DL'),
        mpatches.Patch(facecolor=palette['2023_dry'], edgecolor='black', linewidth=1.0, label='NDY-DL'),
        mpatches.Patch(facecolor=palette['2022_wet'], edgecolor='black', linewidth=1.0, label='DY-NDL'),
        mpatches.Patch(facecolor=palette['2023_wet'], edgecolor='black', linewidth=1.0, label='NDY-NDL'),
    ]
    ax_leg.legend(
        handles=handles,
        loc='center',
        ncol=2,
        frameon=False,
        prop={'family': FONT_FAMILY, 'size': FONT_SIZE, 'weight': KEY_FONT_WEIGHT},
        columnspacing=2.0,
        handlelength=2.4,
    )

    out_png = os.path.join(out_dir, "Fig_Violin4_3x2.png")
    out_pdf = os.path.join(out_dir, "Fig_Violin4_3x2.pdf")
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(out_pdf, dpi=dpi, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print(f"[SAVED] Vertical violin 3x2 grid: {out_png}")


def plot_median_bars_all_metrics(metrics_groups, metrics_to_plot, out_dir,
                                  dpi=300):
    """
    Plot median bar chart for each metric separately.
    Each metric has four separate bars (2022_dry, 2022_wet, 2023_dry, 2023_wet).
    Six bar charts side by side should have the same width as five violin plots.
    """
    import matplotlib as mpl
    mpl.rcParams['axes.edgecolor'] = 'black'
    mpl.rcParams['axes.linewidth'] = 1.0

    palette = {
        # 2022：更鲜亮的深浅黄
        '2022_dry': "#f5a623",   # 深黄橙
        '2022_wet': "#ffe68a",   # 浅亮黄
        # 2023：更鲜亮的深浅蓝
        '2023_dry': "#2b83ba",   # 深蓝
        '2023_wet': "#7fb6ff",   # 浅亮蓝
    }

    order = ['2022_dry', '2022_wet', '2023_dry', '2023_wet']
    label_map = {
        '2022_dry': 'Dryland 2022',
        '2022_wet': 'Nondryland 2022',
        '2023_dry': 'Dryland 2023',
        '2023_wet': 'Nondryland 2023',
    }
    
    ylabel_map = {
        'A_NIRv': 'A_NIRv',
        'centroid_shift': 'C_shift',
        'MDI': 'MDI',
        'Recovery_rate': 'R_rate',
        'Skew': 'Skewness',
        't_peak': 'T_peak'
    }
    
    # 计算单个图的宽度：5张小提琴图总宽度 / 6张柱形图
    # 小提琴图单个宽度是4.6英寸，5张总宽度是23英寸
    # 6张柱形图总宽度也应该是23英寸，所以每张约3.83英寸
    # 但为了更紧凑，我们使用与小提琴图相同的单个宽度，然后调整
    violin_single_width = 4.6  # 小提琴图单个宽度
    violin_total_width = violin_single_width * 5  # 5张小提琴图总宽度
    bar_single_width = violin_total_width / 6  # 6张柱形图每张宽度
    bar_height = 5.0  # 与小提琴图高度一致
    median_rows = []  # 收集所有指标的中位数，便于输出 CSV
    
    # 为每个指标绘制单独的柱形图
    for metric in metrics_to_plot:
        # 收集该指标的中位数
        medians = {}
        for g in order:
            df = metrics_groups.get(g, pd.DataFrame())
            if metric in df.columns:
                vals = df[metric].astype(float).values
                vals = vals[np.isfinite(vals)]
                medians[g] = np.nanmedian(vals) if len(vals) > 0 else np.nan
                print(f"[DEBUG] {metric} {g}: n={len(vals)}, median={medians[g]:.4f}")
                median_rows.append({
                    "metric": metric,
                    "group": g,
                    "median": medians[g],
                    "n": len(vals)
                })
            else:
                medians[g] = np.nan
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(bar_single_width, bar_height), dpi=dpi)
        
        # X轴位置（四个柱子）
        x_positions = np.arange(len(order))
        
        # 柱形宽度（四个柱子并排，不重叠）
        bar_width = 0.6
        
        # 绘制四个柱形
        for idx, g in enumerate(order):
            median_val = medians[g]
            if np.isfinite(median_val):
                ax.bar(x_positions[idx], median_val, width=bar_width, 
                      color=palette[g], alpha=0.8, edgecolor='black', 
                      linewidth=1.0, zorder=2)
        
        # 设置X轴标签
        ax.set_xticks(x_positions)
        ax.set_xticklabels([label_map[g] for g in order], 
                           fontsize=11, fontweight=KEY_FONT_WEIGHT, rotation=45, ha='right')
        
        # Y轴标签和格式
        ax.set_ylabel(ylabel_map.get(metric, metric), fontsize=12, fontweight='bold')
        from matplotlib.ticker import FormatStrFormatter
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis='y', labelsize=11)
        
        # 网格和样式
        ax.grid(True, axis='y', linestyle='--', alpha=0.35, linewidth=0.9)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        ax.tick_params(axis='both', labelsize=11, width=1.0)
        
        # 调整布局
        fig.tight_layout()
        out_png = os.path.join(out_dir, f"Median_Bar_{metric}.png")
        fig.savefig(out_png, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f"[SAVED] Median bar for {metric}: {out_png}")

    # 导出所有指标的中位数汇总 CSV
    if median_rows:
        median_df = pd.DataFrame(median_rows)
        csv_path = os.path.join(out_dir, "Median_Bars_AllMetrics_data.csv")
        median_df.to_csv(csv_path, index=False)
        print(f"[CSV] Saved median bar data -> {csv_path}")


def plot_grid_specific_boxplots(metrics_pix_22, metrics_pix_23, gdf_extent, gt, proj, nx, ny, 
                                 out_dir, lon_bins, lat_bins, num_rows, num_cols,
                                 metrics_to_plot=['t_peak', 'centroid_shift'],
                                 selected_grids=None):
    """
    Plot boxplots for specific grid cells for t_peak and centroid_shift metrics.
    This helps visualize if outliers in 2022 are reduced when looking at specific locations.
    
    Parameters:
    -----------
    metrics_pix_22 : DataFrame with pixel-level metrics for 2022
    metrics_pix_23 : DataFrame with pixel-level metrics for 2023
    gdf_extent : GeoDataFrame of study area extent
    gt : GeoTransform tuple
    proj : Projection WKT string
    nx, ny : Raster dimensions
    out_dir : Output directory
    lon_bins : Longitude bin edges
    lat_bins : Latitude bin edges
    num_rows, num_cols : Grid dimensions
    metrics_to_plot : List of metrics to plot (default: ['t_peak', 'centroid_shift'])
    selected_grids : List of (row_idx, col_idx) tuples for specific grids, or None to select automatically
    """
    import matplotlib.patches as mpatches
    import matplotlib as mpl
    
    print("[INFO] Plotting grid-specific boxplots for selected metrics...")
    print(f"[DEBUG] Input data: metrics_pix_22 shape={metrics_pix_22.shape}, metrics_pix_23 shape={metrics_pix_23.shape}")
    print(f"[DEBUG] Grid bins: lon_bins range=[{lon_bins.min():.4f}, {lon_bins.max():.4f}], lat_bins range=[{lat_bins.min():.4f}, {lat_bins.max():.4f}]")
    
    # Convert pixel coordinates to geographic coordinates if needed
    if 'lon' not in metrics_pix_22.columns or 'lat' not in metrics_pix_22.columns:
        print("[INFO] Computing lon/lat from pixel coordinates...")
        # Compute lon/lat from pixel coordinates
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(proj)
        tgt_srs = osr.SpatialReference()
        tgt_srs.ImportFromEPSG(4326)
        coord_trans = osr.CoordinateTransformation(src_srs, tgt_srs)
        
        lons_22 = []; lats_22 = []
        for _, row in metrics_pix_22.iterrows():
            py, px = int(row['pixel_y']), int(row['pixel_x'])
            x_map = gt[0] + px * gt[1] + py * gt[2]
            y_map = gt[3] + px * gt[4] + py * gt[5]
            try:
                lon, lat, _ = coord_trans.TransformPoint(x_map, y_map)
                lons_22.append(float(lon)); lats_22.append(float(lat))
            except Exception as e:
                print(f"[WARN] Transform failed for pixel ({py},{px}): {e}")
                lons_22.append(np.nan); lats_22.append(np.nan)
        metrics_pix_22 = metrics_pix_22.copy()
        metrics_pix_22['lon'] = lons_22
        metrics_pix_22['lat'] = lats_22
        
        lons_23 = []; lats_23 = []
        for _, row in metrics_pix_23.iterrows():
            py, px = int(row['pixel_y']), int(row['pixel_x'])
            x_map = gt[0] + px * gt[1] + py * gt[2]
            y_map = gt[3] + px * gt[4] + py * gt[5]
            try:
                lon, lat, _ = coord_trans.TransformPoint(x_map, y_map)
                lons_23.append(float(lon)); lats_23.append(float(lat))
            except Exception as e:
                print(f"[WARN] Transform failed for pixel ({py},{px}): {e}")
                lons_23.append(np.nan); lats_23.append(np.nan)
        metrics_pix_23 = metrics_pix_23.copy()
        metrics_pix_23['lon'] = lons_23
        metrics_pix_23['lat'] = lats_23
    else:
        print("[INFO] Using existing lon/lat columns from metrics data")
        metrics_pix_22 = metrics_pix_22.copy()
        metrics_pix_23 = metrics_pix_23.copy()
    
    # Filter out NaN values in lon/lat before assigning grid indices
    valid_22 = metrics_pix_22['lon'].notna() & metrics_pix_22['lat'].notna()
    valid_23 = metrics_pix_23['lon'].notna() & metrics_pix_23['lat'].notna()
    print(f"[DEBUG] Valid coordinates: 2022={valid_22.sum()}/{len(valid_22)}, 2023={valid_23.sum()}/{len(valid_23)}")
    
    # Check if coordinates are within grid bounds
    in_bounds_22 = valid_22 & (metrics_pix_22['lon'] >= lon_bins.min()) & (metrics_pix_22['lon'] <= lon_bins.max()) & \
                   (metrics_pix_22['lat'] >= lat_bins.min()) & (metrics_pix_22['lat'] <= lat_bins.max())
    in_bounds_23 = valid_23 & (metrics_pix_23['lon'] >= lon_bins.min()) & (metrics_pix_23['lon'] <= lon_bins.max()) & \
                   (metrics_pix_23['lat'] >= lat_bins.min()) & (metrics_pix_23['lat'] <= lat_bins.max())
    print(f"[DEBUG] In grid bounds: 2022={in_bounds_22.sum()}/{len(in_bounds_22)}, 2023={in_bounds_23.sum()}/{len(in_bounds_23)}")
    
    if in_bounds_22.sum() == 0 and in_bounds_23.sum() == 0:
        print("[ERROR] No data points within grid bounds! Check coordinate ranges.")
        print(f"[DEBUG] 2022 lon range: [{metrics_pix_22['lon'].min():.4f}, {metrics_pix_22['lon'].max():.4f}]")
        print(f"[DEBUG] 2022 lat range: [{metrics_pix_22['lat'].min():.4f}, {metrics_pix_22['lat'].max():.4f}]")
        print(f"[DEBUG] 2023 lon range: [{metrics_pix_23['lon'].min():.4f}, {metrics_pix_23['lon'].max():.4f}]")
        print(f"[DEBUG] 2023 lat range: [{metrics_pix_23['lat'].min():.4f}, {metrics_pix_23['lat'].max():.4f}]")
        return
    
    # Assign grid cell indices based on lon/lat (only for valid points)
    metrics_pix_22['lat_bin_idx'] = -1  # Initialize with invalid index
    metrics_pix_22['lon_bin_idx'] = -1
    metrics_pix_23['lat_bin_idx'] = -1
    metrics_pix_23['lon_bin_idx'] = -1
    
    # Use digitize only for valid points within bounds
    # Note: lat_bins and lon_bins are sorted ascending
    # np.digitize(x, bins) returns the index of the bin that x belongs to
    # For bins = [a, b, c, d] (4 boundaries = 3 bins):
    #   x < a: returns 0
    #   a <= x < b: returns 1
    #   b <= x < c: returns 2  
    #   c <= x < d: returns 3
    #   x >= d: returns len(bins) = 4
    # So we need to subtract 1 and handle the rightmost edge case
    if in_bounds_22.sum() > 0:
        lat_vals_22 = metrics_pix_22.loc[in_bounds_22, 'lat'].values
        lon_vals_22 = metrics_pix_22.loc[in_bounds_22, 'lon'].values
        
        # Use digitize with all bins (including edges)
        lat_idx_22 = np.digitize(lat_vals_22, lat_bins) - 1
        lon_idx_22 = np.digitize(lon_vals_22, lon_bins) - 1
        
        # Handle rightmost edge: values exactly at max should be in last bin
        lat_idx_22[lat_idx_22 >= num_rows] = num_rows - 1
        lon_idx_22[lon_idx_22 >= num_cols] = num_cols - 1
        
        # Handle leftmost edge: values exactly at min should be in first bin
        lat_idx_22[lat_idx_22 < 0] = 0
        lon_idx_22[lon_idx_22 < 0] = 0
        
        # Final clip to ensure valid range
        lat_idx_22 = np.clip(lat_idx_22, 0, num_rows - 1)
        lon_idx_22 = np.clip(lon_idx_22, 0, num_cols - 1)
        
        metrics_pix_22.loc[in_bounds_22, 'lat_bin_idx'] = lat_idx_22
        metrics_pix_22.loc[in_bounds_22, 'lon_bin_idx'] = lon_idx_22
    
    if in_bounds_23.sum() > 0:
        lat_vals_23 = metrics_pix_23.loc[in_bounds_23, 'lat'].values
        lon_vals_23 = metrics_pix_23.loc[in_bounds_23, 'lon'].values
        
        lat_idx_23 = np.digitize(lat_vals_23, lat_bins) - 1
        lon_idx_23 = np.digitize(lon_vals_23, lon_bins) - 1
        
        lat_idx_23[lat_idx_23 >= num_rows] = num_rows - 1
        lon_idx_23[lon_idx_23 >= num_cols] = num_cols - 1
        lat_idx_23[lat_idx_23 < 0] = 0
        lon_idx_23[lon_idx_23 < 0] = 0
        
        lat_idx_23 = np.clip(lat_idx_23, 0, num_rows - 1)
        lon_idx_23 = np.clip(lon_idx_23, 0, num_cols - 1)
        
        metrics_pix_23.loc[in_bounds_23, 'lat_bin_idx'] = lat_idx_23
        metrics_pix_23.loc[in_bounds_23, 'lon_bin_idx'] = lon_idx_23
    
    # Reverse lat_bin_idx (0 = top/north, num_rows-1 = bottom/south)
    # Only for valid indices
    valid_idx_22 = metrics_pix_22['lat_bin_idx'] >= 0
    valid_idx_23 = metrics_pix_23['lat_bin_idx'] >= 0
    
    metrics_pix_22['row_idx'] = -1
    metrics_pix_22['col_idx'] = -1
    metrics_pix_23['row_idx'] = -1
    metrics_pix_23['col_idx'] = -1
    
    if valid_idx_22.sum() > 0:
        metrics_pix_22.loc[valid_idx_22, 'row_idx'] = num_rows - 1 - metrics_pix_22.loc[valid_idx_22, 'lat_bin_idx']
        metrics_pix_22.loc[valid_idx_22, 'col_idx'] = metrics_pix_22.loc[valid_idx_22, 'lon_bin_idx']
    
    if valid_idx_23.sum() > 0:
        metrics_pix_23.loc[valid_idx_23, 'row_idx'] = num_rows - 1 - metrics_pix_23.loc[valid_idx_23, 'lat_bin_idx']
        metrics_pix_23.loc[valid_idx_23, 'col_idx'] = metrics_pix_23.loc[valid_idx_23, 'lon_bin_idx']
    
    print(f"[DEBUG] Assigned grid indices: 2022={valid_idx_22.sum()} pixels, 2023={valid_idx_23.sum()} pixels")
    if valid_idx_22.sum() > 0:
        print(f"[DEBUG] 2022 row_idx range: [{metrics_pix_22.loc[valid_idx_22, 'row_idx'].min()}, {metrics_pix_22.loc[valid_idx_22, 'row_idx'].max()}]")
        print(f"[DEBUG] 2022 col_idx range: [{metrics_pix_22.loc[valid_idx_22, 'col_idx'].min()}, {metrics_pix_22.loc[valid_idx_22, 'col_idx'].max()}]")
    if valid_idx_23.sum() > 0:
        print(f"[DEBUG] 2023 row_idx range: [{metrics_pix_23.loc[valid_idx_23, 'row_idx'].min()}, {metrics_pix_23.loc[valid_idx_23, 'row_idx'].max()}]")
        print(f"[DEBUG] 2023 col_idx range: [{metrics_pix_23.loc[valid_idx_23, 'col_idx'].min()}, {metrics_pix_23.loc[valid_idx_23, 'col_idx'].max()}]")
    
    # Select grids if not provided (select grids that actually have data)
    if selected_grids is None:
        # Find grids that have data
        valid_grids_22 = set(zip(metrics_pix_22.loc[valid_idx_22, 'row_idx'], 
                                 metrics_pix_22.loc[valid_idx_22, 'col_idx']))
        valid_grids_23 = set(zip(metrics_pix_23.loc[valid_idx_23, 'row_idx'], 
                                 metrics_pix_23.loc[valid_idx_23, 'col_idx']))
        all_valid_grids = valid_grids_22.union(valid_grids_23)
        print(f"[DEBUG] Found {len(all_valid_grids)} grids with data")
        
        if len(all_valid_grids) == 0:
            print("[ERROR] No grids have data! Cannot plot.")
            return
        
        # Select representative grids (prefer grids with data)
        candidate_grids = [
            (0, 0),  # Top-left
            (0, num_cols-1),  # Top-right
            (num_rows-1, 0),  # Bottom-left
            (num_rows-1, num_cols-1),  # Bottom-right
            (num_rows//2, num_cols//2),  # Center
        ]
        if num_rows > 3 and num_cols > 3:
            candidate_grids.append((num_rows//2, 0))  # Middle-left
            candidate_grids.append((num_rows//2, num_cols-1))  # Middle-right
        
        # Filter to only include grids with data
        selected_grids = [g for g in candidate_grids if g in all_valid_grids]
        
        # If we don't have enough grids, add more from valid grids
        if len(selected_grids) < 3:
            remaining_grids = sorted(list(all_valid_grids - set(selected_grids)))
            selected_grids.extend(remaining_grids[:6 - len(selected_grids)])
        
        print(f"[DEBUG] Selected {len(selected_grids)} grids for plotting: {selected_grids}")
    
    palette = {'2022': COLOR_2022_DROUGHT, '2023': COLOR_2023_NON_DROUGHT}
    
    for metric in metrics_to_plot:
        if metric not in metrics_pix_22.columns or metric not in metrics_pix_23.columns:
            print(f"[WARN] Metric {metric} not found, skipping...")
            continue
        
        # Create figure with subplots for each selected grid
        n_grids = len(selected_grids)
        n_cols_plot = min(3, n_grids)
        n_rows_plot = (n_grids + n_cols_plot - 1) // n_cols_plot
        
        fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(n_cols_plot*3.5, n_rows_plot*4), dpi=300)
        # Handle axes array properly
        if n_grids == 1:
            axes = [axes]
        elif n_rows_plot == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
            axes = axes.flatten() if hasattr(axes, 'flatten') else list(axes)
        else:
            axes = axes.flatten()
        
        title_map = {
            't_peak': 'Hour of maximum NIRv (T_peak)',
            'centroid_shift': 'Centroid shift from mean hour (C_shift)'
        }
        
        for grid_idx, (row_idx, col_idx) in enumerate(selected_grids):
            if grid_idx >= len(axes):
                break
            ax = axes[grid_idx]
            
            # Get data for this grid cell (filter by valid indices first)
            mask_22 = (metrics_pix_22['row_idx'] == row_idx) & (metrics_pix_22['col_idx'] == col_idx) & valid_idx_22
            mask_23 = (metrics_pix_23['row_idx'] == row_idx) & (metrics_pix_23['col_idx'] == col_idx) & valid_idx_23
            
            grid_data_22 = metrics_pix_22[mask_22][metric].dropna()
            grid_data_23 = metrics_pix_23[mask_23][metric].dropna()
            
            print(f"[DEBUG] Grid ({row_idx},{col_idx}): 2022 n={len(grid_data_22)}, 2023 n={len(grid_data_23)}")
            
            if len(grid_data_22) == 0 and len(grid_data_23) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=FONT_SIZE, color='gray', transform=ax.transAxes)
                lat_center = (lat_bins[num_rows - 1 - row_idx] + lat_bins[num_rows - row_idx]) / 2 if row_idx < num_rows else np.nan
                lon_center = (lon_bins[col_idx] + lon_bins[col_idx + 1]) / 2 if col_idx < num_cols else np.nan
                ax.set_title(f'Grid ({row_idx},{col_idx})\n({lon_center:.2f}°E, {lat_center:.2f}°N)', 
                            fontsize=FONT_SIZE, fontweight=KEY_FONT_WEIGHT)
                continue
            
            # Prepare data for plotting
            plot_data = []
            if len(grid_data_22) > 0:
                plot_data.append(pd.DataFrame({'value': grid_data_22.values, 'year': '2022'}))
            if len(grid_data_23) > 0:
                plot_data.append(pd.DataFrame({'value': grid_data_23.values, 'year': '2023'}))
            
            if len(plot_data) == 0:
                continue
            
            df_plot = pd.concat(plot_data, ignore_index=True)
            
            # Calculate y-axis limits (with special handling for t_peak)
            if len(grid_data_22) > 0 and len(grid_data_23) > 0:
                q1_22, q3_22 = np.percentile(grid_data_22, [25, 75])
                q1_23, q3_23 = np.percentile(grid_data_23, [25, 75])
                iqr_22 = q3_22 - q1_22
                iqr_23 = q3_23 - q1_23
                lower_whisker = min(q1_22 - 1.5*iqr_22, q1_23 - 1.5*iqr_23, 
                                   np.nanmin(grid_data_22), np.nanmin(grid_data_23))
                upper_whisker = max(q3_22 + 1.5*iqr_22, q3_23 + 1.5*iqr_23, 
                                   np.nanmax(grid_data_22), np.nanmax(grid_data_23))
                data_range = upper_whisker - lower_whisker
                if metric == 't_peak':
                    # Round to 0.5 increments for t_peak
                    y_min_plot = np.floor(lower_whisker * 2) / 2
                    y_max_plot = np.ceil(upper_whisker * 2) / 2
                    margin = min(data_range * 0.02, 0.5)
                else:
                    margin = data_range * 0.03 if data_range > 0 else 0.1
                    y_min_plot = lower_whisker - margin
                    y_max_plot = upper_whisker + margin
                if metric == 't_peak':
                    y_min_plot = max(0, y_min_plot - margin)
                    y_max_plot += margin
            elif len(grid_data_22) > 0:
                q1_22, q3_22 = np.percentile(grid_data_22, [25, 75])
                iqr_22 = q3_22 - q1_22
                lower_whisker = q1_22 - 1.5*iqr_22
                upper_whisker = q3_22 + 1.5*iqr_22
                data_range = upper_whisker - lower_whisker
                if metric == 't_peak':
                    y_min_plot = np.floor(lower_whisker * 2) / 2
                    y_max_plot = np.ceil(upper_whisker * 2) / 2
                    margin = min(data_range * 0.02, 0.5)
                    y_min_plot = max(0, y_min_plot - margin)
                    y_max_plot += margin
                else:
                    margin = data_range * 0.03 if data_range > 0 else 0.1
                    y_min_plot = lower_whisker - margin
                    y_max_plot = upper_whisker + margin
            else:
                q1_23, q3_23 = np.percentile(grid_data_23, [25, 75])
                iqr_23 = q3_23 - q1_23
                lower_whisker = q1_23 - 1.5*iqr_23
                upper_whisker = q3_23 + 1.5*iqr_23
                data_range = upper_whisker - lower_whisker
                if metric == 't_peak':
                    y_min_plot = np.floor(lower_whisker * 2) / 2
                    y_max_plot = np.ceil(upper_whisker * 2) / 2
                    margin = min(data_range * 0.02, 0.5)
                    y_min_plot = max(0, y_min_plot - margin)
                    y_max_plot += margin
                else:
                    margin = data_range * 0.03 if data_range > 0 else 0.1
                    y_min_plot = lower_whisker - margin
                    y_max_plot = upper_whisker + margin
            
            # Plot boxplot
            order = []
            if len(grid_data_22) > 0:
                order.append('2022')
            if len(grid_data_23) > 0:
                order.append('2023')
            
            sns.boxplot(x='year', y='value', data=df_plot, order=order, ax=ax,
                       showcaps=True, showfliers=True, saturation=0.75,  # Show outliers
                       palette=[palette[y] for y in order],
                       boxprops={'zorder':2, 'alpha':0.9},
                       medianprops={'color':'black', 'linewidth':2.0},
                       whiskerprops={'color':'black','linewidth':0.9},
                       capprops={'color':'black','linewidth':0.9},
                       flierprops={'marker':'o', 'markersize':3, 'alpha':0.5, 'markeredgecolor':'gray'})
            
            # Post-process artists
            for i, artist in enumerate(ax.artists):
                if i < len(order):
                    face = palette[order[i]]
                    artist.set_edgecolor('black')
                    artist.set_facecolor(face)
                    artist.set_alpha(0.9)
                    artist.set_linewidth(0.8)
            
            # Set y-axis limits
            ax.set_ylim(bottom=y_min_plot, top=y_max_plot)
            
            # Styling - RSE journal style
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.8)
            ax.spines['left'].set_linewidth(0.8)
            ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE, width=0.8)
            ax.set_xlabel('')
            
            # Set Y-axis label (simplified)
            if metric == 't_peak':
                ylabel = 'T_peak (hours)'
            elif metric == 'centroid_shift':
                ylabel = 'C_shift (hours)'
            else:
                ylabel = metric
            ax.set_ylabel(ylabel, fontsize=FONT_SIZE, fontweight=KEY_FONT_WEIGHT)
            
            # Grid location info
            lat_center = (lat_bins[num_rows - 1 - row_idx] + lat_bins[num_rows - row_idx]) / 2
            lon_center = (lon_bins[col_idx] + lon_bins[col_idx + 1]) / 2
            ax.set_title(f'Grid ({row_idx},{col_idx})\n({lon_center:.2f}°E, {lat_center:.2f}°N)', 
                        fontsize=FONT_SIZE, fontweight=KEY_FONT_WEIGHT)
            
            # Add sample sizes below X-axis labels (after ylim is set)
            n_22 = len(grid_data_22)
            n_23 = len(grid_data_23)
            # Expand bottom limit to make room for sample size labels
            current_bottom = ax.get_ylim()[0]
            current_top = ax.get_ylim()[1]
            expanded_bottom = current_bottom - 0.08 * (current_top - current_bottom)
            ax.set_ylim(bottom=expanded_bottom, top=current_top)
            
            # Position sample sizes in the margin space
            y_pos_n = expanded_bottom + 0.02 * (current_top - expanded_bottom)
            if len(order) == 2:
                ax.text(0, y_pos_n, f'n = {n_22:,}', 
                       ha='center', va='bottom', fontsize=FONT_SIZE, color='black')
                ax.text(1, y_pos_n, f'n = {n_23:,}', 
                       ha='center', va='bottom', fontsize=FONT_SIZE, color='black')
            elif len(order) == 1:
                year_idx = 0 if order[0] == '2022' else 1
                n_val = n_22 if order[0] == '2022' else n_23
                ax.text(year_idx, y_pos_n, f'n = {n_val:,}', 
                       ha='center', va='bottom', fontsize=FONT_SIZE, color='black')
        
        # Hide unused subplots
        for idx in range(n_grids, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(title_map.get(metric, metric) + ' — Grid-specific Boxplots', 
                    fontsize=FONT_SIZE, fontweight=KEY_FONT_WEIGHT, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        
        outfp = os.path.join(out_dir, f"Grid_Specific_Box_{metric}_2022.png")
        fig.savefig(outfp, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"[SAVED] Grid-specific boxplot: {outfp}")
    
    print("[INFO] Grid-specific boxplot plotting completed.")





# ---------------- Main ----------------
def main():
    try:
        # 1) Find a sample tif for reference (for mask resampling and geo-info)
        sample_fp = None
        sample_candidates = glob.glob(os.path.join(BRDF_DIR_2022, '*.tif')) + glob.glob(os.path.join(BRDF_DIR_2023, '*.tif'))
        if not sample_candidates:
            raise FileNotFoundError("No .tif files found in both BRDF directories.")
        sample_fp = sample_candidates[0]
        print(f"[INFO] Using reference raster: {os.path.basename(sample_fp)}")

        # # Reproject reference tif to TARGET_CRS
        # projected_ref_fp = os.path.join(OUT_DIR, 'projected_ref.tif')
        # ref_ds = gdal.Open(sample_fp)
        # ref_gt = ref_ds.GetGeoTransform()
        # ref_proj = ref_ds.GetProjection()
        # ref_srs = osr.SpatialReference(ref_proj) if ref_proj else osr.SpatialReference()
        # ref_srs.ImportFromEPSG(4326)  # Source SRS

        # tgt_srs = osr.SpatialReference()
        # tgt_srs.ImportFromEPSG(32650)  # Target UTM

        # # Clamp source bounds to SHP to avoid invalid lat
        # shp_min_lon, shp_min_lat, shp_max_lon, shp_max_lat = gdf_hhh_4326.total_bounds
        # tif_min_lat = ref_gt[3] + ny * ref_gt[5]  # Tif south lat ~30.27
        # safe_min_lat = max(shp_min_lat, tif_min_lat)  # Clamp to SHP 31.38
        # safe_bounds_src = (shp_min_lon, safe_min_lat, shp_max_lon, shp_max_lat)
        # print(f"[DEBUG] Clamped src bounds (avoid invalid lat): {safe_bounds_src}")

        # # Transform 4 corners to target SRS
        # transform = osr.CoordinateTransformation(ref_srs, tgt_srs)
        # bl = transform.TransformPoint(safe_bounds_src[0], safe_bounds_src[1])  # BL
        # br = transform.TransformPoint(safe_bounds_src[2], safe_bounds_src[1])  # BR
        # tl = transform.TransformPoint(safe_bounds_src[0], safe_bounds_src[3])  # TL
        # tr = transform.TransformPoint(safe_bounds_src[2], safe_bounds_src[3])  # TR
        # output_bounds_tgt = (min(bl[0], br[0], tl[0], tr[0]), min(bl[1], br[1], tl[1], tr[1]),
        #                     max(bl[0], br[0], tl[0], tr[0]), max(bl[1], br[1], tl[1], tr[1]))
        # print(f"[DEBUG] Target outputBounds: {output_bounds_tgt}")

        # # Warp ref tif
        # warped = gdal.Warp(projected_ref_fp, ref_ds, format='GTiff',
        #                 dstSRS=TARGET_CRS, outputBounds=output_bounds_tgt,
        #                 xRes=OUTPUT_RESOLUTION, yRes=OUTPUT_RESOLUTION,
        #                 resampleAlg=gdal.GRA_NearestNeighbour,
        #                 te_srs='EPSG:4326')  # Explicit src SRS for bounds
        # ref_ds = None
        # if warped is None:
        #     raise RuntimeError("Warp failed - check clamped bounds")
        # warped = None

        # # Update reference
        # sample_fp = projected_ref_fp
        # gt, proj, nx, ny = get_raster_info(sample_fp)
        # print(f"[INFO] Projected reference to {TARGET_CRS}: {nx}x{ny} pixels (clamped)")

        # # Project wheat mask with same bounds
        # mask_ds = gdal.Open(WHEAT_MASK_TIF)
        # projected_mask_fp = os.path.join(OUT_DIR, 'projected_wheat_mask.tif')
        # gdal.Warp(projected_mask_fp, mask_ds, format='GTiff', dstSRS=TARGET_CRS,
        #         outputBounds=output_bounds_tgt,  # Same safe
        #         xRes=OUTPUT_RESOLUTION, yRes=OUTPUT_RESOLUTION, resampleAlg=gdal.GRA_NearestNeighbour,
        #         te_srs='EPSG:4326')
        # mask_ds = None

        # # Resample mask
        # print("[INFO] Resampling wheat mask to projected reference grid...")
        # mask_arr = resample_mask_to_ref(projected_mask_fp, sample_fp)
        # print(f"[INFO] Mask resampled. Valid pixels (mask sum) = {np.sum(mask_arr)}")

        # 直接用原 sample_fp
        gt, proj, nx, ny = get_raster_info(sample_fp)
        print(f"[INFO] Using original WGS84 reference: {nx}x{ny} pixels")
        mask_arr = resample_mask_to_ref(WHEAT_MASK_TIF, sample_fp)  # 原 mask

        # 3) Define geographic grid boundaries
        print("[INFO] Defining geographic grid boundaries...")
        # Get overall extent from HHH shapefile
        gdf_hhh = gpd.read_file(HHH_SHP_PATH)
        gdf_hhh_4326 = gdf_hhh.to_crs(epsg=4326)
        min_lon, min_lat, max_lon, max_lat = gdf_hhh_4326.total_bounds
        print("[DEBUG] HHH shapefile bounds (WGS84): min_lon={:.4f}, min_lat={:.4f}, max_lon={:.4f}, max_lat={:.4f}".format(min_lon, min_lat, max_lon, max_lat))
        # # Projected bounds (m)
        # min_x, min_y, max_x, max_y = gdf_hhh_proj.total_bounds
        # print(f"[DEBUG] Projected HHH bounds ({TARGET_CRS}): min_x={min_x:.0f}, min_y={min_y:.0f}, max_x={max_x:.0f}, max_y={max_y:.0f}")

        # Geographic bins
        lat_bins = np.linspace(min_lat, max_lat, NUM_ROWS + 1)
        lon_bins = np.linspace(min_lon, max_lon, NUM_COLS + 1)
        print(f"[DEBUG] lat_bins: {lat_bins}")
        print(f"[DEBUG] lon_bins: {lon_bins}")

        # Optional: Original geographic bins for debug
        lat_bins = np.linspace(min_lat, max_lat, NUM_ROWS + 1)
        lon_bins = np.linspace(min_lon, max_lon, NUM_COLS + 1)
        print(f"[DEBUG] lat_bins (original): {lat_bins}")
        print(f"[DEBUG] lon_bins (original): {lon_bins}")

        use_cached_metrics = (
            USE_CACHE
            and os.path.exists(CACHE_METRICS_22)
            and os.path.exists(CACHE_METRICS_23)
        )

        if use_cached_metrics:
            print("[INFO] Loading cached per-pixel metrics...")
            metrics_pix_22 = pd.read_csv(CACHE_METRICS_22)
            metrics_pix_23 = pd.read_csv(CACHE_METRICS_23)
        else:
            # 4) NEW: Build daily-hourly NIRv pixel values for each year
            print("[INFO] Building daily-hourly pixel NIRv values for 2022...")
            df_pixels_22 = build_daily_hourly_pixel_nirv(BRDF_DIR_2022, DATES_2022, HOURS, (ny, nx))
            if not df_pixels_22.empty:
                nirv_22_raw = df_pixels_22['value'].astype(float).values
                nirv_22_raw = nirv_22_raw[np.isfinite(nirv_22_raw)]
                print(f"[DEBUG] 2022 Raw NIRv (all collected): n={len(nirv_22_raw)}, "
                      f"min={np.nanmin(nirv_22_raw):.4f}, max={np.nanmax(nirv_22_raw):.4f}, "
                      f"mean={np.nanmean(nirv_22_raw):.4f}, median={np.nanmedian(nirv_22_raw):.4f}")
            
            print("[INFO] Building daily-hourly pixel NIRv values for 2023...")
            df_pixels_23 = build_daily_hourly_pixel_nirv(BRDF_DIR_2023, DATES_2023, HOURS, (ny, nx))
            if not df_pixels_23.empty:
                nirv_23_raw = df_pixels_23['value'].astype(float).values
                nirv_23_raw = nirv_23_raw[np.isfinite(nirv_23_raw)]
                print(f"[DEBUG] 2023 Raw NIRv (all collected): n={len(nirv_23_raw)}, "
                      f"min={np.nanmin(nirv_23_raw):.4f}, max={np.nanmax(nirv_23_raw):.4f}, "
                      f"mean={np.nanmean(nirv_23_raw):.4f}, median={np.nanmedian(nirv_23_raw):.4f}")

            if (df_pixels_22.empty) and (df_pixels_23.empty):
                raise RuntimeError("No pixel values collected for either year. Aborting.")
            
            # DEBUG 输出，确认投影与 shapefile CRS
            print("[DEBUG] raster proj WKT (first 200 chars):", str(proj)[:200])
            try:
                print("[DEBUG] shapefile CRS:", gdf_hhh_4326.crs)
            except Exception:
                print("[DEBUG] gdf_hhh_4326 not defined or has no .crs")

            # ------------------ Region-level per-pixel metrics (NEW) ------------------
            print("[INFO] Building processed per-pixel hourly values for entire HHH region (2022)...")
            processed_pix_22 = build_processed_pixel_values_region(df_pixels_22, mask_arr, gt, proj, nx, ny, gdf_hhh_4326, HOURS)  # gdf_hhh_proj -> gdf_hhh_4326
            print(f"[INFO] processed_pix_22 rows: {len(processed_pix_22)}")

            print("[INFO] Building processed per-pixel hourly values for entire HHH region (2023)...")
            processed_pix_23 = build_processed_pixel_values_region(df_pixels_23, mask_arr, gt, proj, nx, ny, gdf_hhh_4326, HOURS)  # 同上
            print(f"[INFO] processed_pix_23 rows: {len(processed_pix_23)}")

            # compute per-pixel metrics
            print("[INFO] Calculating per-pixel diurnal metrics for region (2022)...")
            metrics_pix_22 = metrics_from_processed_pixels(processed_pix_22, HOURS)
            print("[INFO] Calculating per-pixel diurnal metrics for region (2023)...")
            metrics_pix_23 = metrics_from_processed_pixels(processed_pix_23, HOURS)

            if USE_CACHE:
                metrics_pix_22.to_csv(CACHE_METRICS_22, index=False)
                metrics_pix_23.to_csv(CACHE_METRICS_23, index=False)
                print(f"[CACHE] Saved metrics to {CACHE_DIR}")

        print("[DEBUG] sample metrics 2022:\n", metrics_pix_22.head())
        print("[DEBUG] sample metrics 2023:\n", metrics_pix_23.head())

        # ================= Drought mask splits (dryland vs wetland) =================
        print("[INFO] Applying drought/non-drought masks to pixel metrics...")
        print(f"[DEBUG] Total metrics 2022: n={len(metrics_pix_22)}")
        print(f"[DEBUG] Total metrics 2023: n={len(metrics_pix_23)}")
        if 'A_NIRv' in metrics_pix_22.columns:
            a_nirv_22_all = metrics_pix_22['A_NIRv'].astype(float).values
            a_nirv_22_all = a_nirv_22_all[np.isfinite(a_nirv_22_all)]
            print(f"[DEBUG] A_NIRv 2022 (all, before mask): n={len(a_nirv_22_all)}, "
                  f"median={np.nanmedian(a_nirv_22_all):.4f}, mean={np.nanmean(a_nirv_22_all):.4f}, "
                  f"min={np.nanmin(a_nirv_22_all):.4f}, max={np.nanmax(a_nirv_22_all):.4f}")
        if 'A_NIRv' in metrics_pix_23.columns:
            a_nirv_23_all = metrics_pix_23['A_NIRv'].astype(float).values
            a_nirv_23_all = a_nirv_23_all[np.isfinite(a_nirv_23_all)]
            print(f"[DEBUG] A_NIRv 2023 (all, before mask): n={len(a_nirv_23_all)}, "
                  f"median={np.nanmedian(a_nirv_23_all):.4f}, mean={np.nanmean(a_nirv_23_all):.4f}, "
                  f"min={np.nanmin(a_nirv_23_all):.4f}, max={np.nanmax(a_nirv_23_all):.4f}")
        
        use_cached_splits = (
            USE_CACHE
            and os.path.exists(CACHE_22_DRY)
            and os.path.exists(CACHE_22_WET)
            and os.path.exists(CACHE_23_DRY)
            and os.path.exists(CACHE_23_WET)
        )
        if use_cached_splits:
            print("[INFO] Loading cached drought splits...")
            metrics_22_dry = pd.read_csv(CACHE_22_DRY)
            metrics_22_wet = pd.read_csv(CACHE_22_WET)
            metrics_23_dry = pd.read_csv(CACHE_23_DRY)
            metrics_23_wet = pd.read_csv(CACHE_23_WET)
        else:
            metrics_22_dry = filter_metrics_by_mask(metrics_pix_22, DROUGHT_MASKS["2022"], DROUGHT_CLASSES, debug_metric='A_NIRv')
            metrics_22_wet = filter_metrics_by_mask(metrics_pix_22, DROUGHT_MASKS["2022"], NONDROUGHT_CLASSES, debug_metric='A_NIRv')
            metrics_23_dry = filter_metrics_by_mask(metrics_pix_23, DROUGHT_MASKS["2023"], DROUGHT_CLASSES, debug_metric='A_NIRv')
            metrics_23_wet = filter_metrics_by_mask(metrics_pix_23, DROUGHT_MASKS["2023"], NONDROUGHT_CLASSES, debug_metric='A_NIRv')
            if USE_CACHE:
                metrics_22_dry.to_csv(CACHE_22_DRY, index=False)
                metrics_22_wet.to_csv(CACHE_22_WET, index=False)
                metrics_23_dry.to_csv(CACHE_23_DRY, index=False)
                metrics_23_wet.to_csv(CACHE_23_WET, index=False)
                print(f"[CACHE] Saved drought splits to {CACHE_DIR}")

        # ================= 为 Fix_Classification_and_Boundary / Fig_Metrics_Spiral_Maps 导出缓存 =================
        # 按需求：缓存 CSV 代表
        #   - 2022：干旱年所有像元（不区分旱地/非旱地）
        #   - 2023：非干旱年所有像元（不区分旱地/非旱地）
        # 使用全部可用像元，不进行抽样或样本平衡
        cache_22_df = metrics_pix_22.copy()
        cache_23_df = metrics_pix_23.copy()

        n22 = len(cache_22_df)
        n23 = len(cache_23_df)
        print(f"[INFO] Total pixels for cache CSV: 2022_all={n22}, 2023_all={n23} (no sampling applied)")

        # 写出缓存 CSV（结构与原 metrics_df 一致，包含 pixel_y, pixel_x, lon, lat 及所有节律指标列）
        # 使用全部像元，不进行抽样
        if not cache_22_df.empty:
            cache_22_df.to_csv(CACHE_22, index=False)
            print(f"[SAVE] Cached metrics for 2022 (all pixels, no sampling): {CACHE_22} (rows={len(cache_22_df)}, cols={len(cache_22_df.columns)})")
        else:
            print("[WARN] cache_22_df is empty; skip writing metrics_pix_22n.csv")

        if not cache_23_df.empty:
            cache_23_df.to_csv(CACHE_23, index=False)
            print(f"[SAVE] Cached metrics for 2023 (all pixels, no sampling): {CACHE_23} (rows={len(cache_23_df)}, cols={len(cache_23_df.columns)})")
        else:
            print("[WARN] cache_23_df is empty; skip writing metrics_pix_23n.csv")

        metrics_groups = {
            '2022_dry': metrics_22_dry,
            '2022_wet': metrics_22_wet,
            '2023_dry': metrics_23_dry,
            '2023_wet': metrics_23_wet,
        }
        metrics_to_plot = ['A_NIRv', 'centroid_shift', 'MDI', 'Recovery_rate', 'Skew', 't_peak']
        print("[INFO] Plotting vertical violins 3x2 grid (four groups)...")
        plot_violin_four_groups_grid(metrics_groups, metrics_to_plot, OUT_DIR, fig_width_cm=17.0, dpi=300)

        # # 单指标出版风格箱线图（暂时停用）
        # print("[INFO] Plotting region-level boxplots and running paired tests...")
        # region_results = plot_region_metrics_boxplots(metrics_pix_22, metrics_pix_23, OUT_DIR)
        # 
        # # Plot grid-specific boxplots for t_peak and centroid_shift（暂时停用）
        # print("[INFO] Plotting grid-specific boxplots for t_peak and centroid_shift...")
        # plot_grid_specific_boxplots(metrics_pix_22, metrics_pix_23, gdf_hhh_4326, gt, proj, nx, ny, 
        #                             OUT_DIR, lon_bins, lat_bins, NUM_ROWS, NUM_COLS,
        #                             metrics_to_plot=['t_peak', 'centroid_shift'],
        #                             selected_grids=None)
        # ------------------------------------------------------------------------

        # # 新增：导出 t_peak 2022-2023 差值影像 (GeoTIFF)
        # print("[INFO] Exporting t_peak difference raster (2022 - 2023)...")
        # if not metrics_pix_22.empty and not metrics_pix_23.empty:
        #     # Paired diff
        #     merged_tpeak = pd.merge(metrics_pix_22[['pixel_y', 'pixel_x', 't_peak']], 
        #                             metrics_pix_23[['pixel_y', 'pixel_x', 't_peak']], 
        #                             on=['pixel_y', 'pixel_x'], suffixes=('_22', '_23'))
        #     if not merged_tpeak.empty:
        #         print(f"[DEBUG] Paired t_peak pixels: n={len(merged_tpeak)}, py range {merged_tpeak['pixel_y'].min()}-{merged_tpeak['pixel_y'].max()}, px range {merged_tpeak['pixel_x'].min()}-{merged_tpeak['pixel_x'].max()}")  # 新增调试
        #         diff_arr = merged_tpeak['t_peak_22'] - merged_tpeak['t_peak_23']
                
        #         # Raster array (init NoData)
        #         diff_raster = np.full((ny, nx), -9999.0, dtype=np.float32)
                
        #         # Fill (放宽: if 0 <= py < ny and 0 <= px < nx)
        #         filled_count = 0
        #         for _, row in merged_tpeak.iterrows():
        #             py, px = int(row['pixel_y']), int(row['pixel_x'])
        #             if 0 <= py < ny and 0 <= px < nx:  # 已有限制，但调试print py/px
        #                 diff_raster[py, px] = float(diff_arr.iloc[filled_count])  # 用iloc确保顺序
        #                 filled_count += 1
                
        #         print(f"[DEBUG] Filled {filled_count} pixels in raster (total cells {ny*nx})")
                
        #         # Write GeoTIFF
        #         driver = gdal.GetDriverByName('GTiff')
        #         out_ds = driver.Create(os.path.join(OUT_DIR, 't_peak_diff_2022_2023.tif'), nx, ny, 1, gdal.GDT_Float32)
        #         out_ds.SetGeoTransform(gt)
        #         out_ds.SetProjection(proj)
        #         out_band = out_ds.GetRasterBand(1)
        #         out_band.WriteArray(diff_raster)
        #         out_band.SetNoDataValue(-9999.0)
        #         out_band.FlushCache()
        #         out_ds = None
        #         print(f"[SAVED] t_peak diff raster: {os.path.join(OUT_DIR, 't_peak_diff_2022_2023.tif')} (n={len(merged_tpeak)} paired)")
        #     else:
        #         print("[WARN] No paired pixels for t_peak diff export.")
        # else:
        #     print("[WARN] No metrics data for t_peak diff export.")
        # # ------------------------------------------------------------------------

        # # centroid_shift diff
        # print("[INFO] Exporting centroid_shift difference raster (2022 - 2023)...")
        # if not metrics_pix_22.empty and not metrics_pix_23.empty:
        #     merged_cent = pd.merge(metrics_pix_22[['pixel_y', 'pixel_x', 'centroid_shift']], 
        #                         metrics_pix_23[['pixel_y', 'pixel_x', 'centroid_shift']], 
        #                         on=['pixel_y', 'pixel_x'], suffixes=('_22', '_23'))
        #     if not merged_cent.empty:
        #         print(f"[DEBUG] Paired centroid_shift pixels: n={len(merged_cent)}, py range {merged_cent['pixel_y'].min()}-{merged_cent['pixel_y'].max()}")
        #         diff_arr = merged_cent['centroid_shift_22'] - merged_cent['centroid_shift_23']
                
        #         diff_raster = np.full((ny, nx), -9999.0, dtype=np.float32)
        #         filled_count = 0
        #         for _, row in merged_cent.iterrows():
        #             py, px = int(row['pixel_y']), int(row['pixel_x'])
        #             if 0 <= py < ny and 0 <= px < nx:
        #                 diff_raster[py, px] = float(diff_arr.iloc[filled_count])
        #                 filled_count += 1
                
        #         print(f"[DEBUG] Filled {filled_count} pixels for centroid_shift")
                
        #         driver = gdal.GetDriverByName('GTiff')
        #         out_ds = driver.Create(os.path.join(OUT_DIR, 'centroid_shift_diff_2022_2023.tif'), nx, ny, 1, gdal.GDT_Float32)
        #         out_ds.SetGeoTransform(gt)
        #         out_ds.SetProjection(proj)
        #         out_band = out_ds.GetRasterBand(1)
        #         out_band.WriteArray(diff_raster)
        #         out_band.SetNoDataValue(-9999.0)
        #         out_band.FlushCache()
        #         out_ds = None
        #         print(f"[SAVED] centroid_shift diff raster: {os.path.join(OUT_DIR, 'centroid_shift_diff_2022_2023.tif')} (n={len(merged_cent)} paired)")
        #     else:
        #         print("[WARN] No paired pixels for centroid_shift diff export.")

        # # MDI diff
        # print("[INFO] Exporting MDI difference raster (2022 - 2023)...")
        # if not metrics_pix_22.empty and not metrics_pix_23.empty:
        #     merged_mdi = pd.merge(metrics_pix_22[['pixel_y', 'pixel_x', 'MDI']], 
        #                         metrics_pix_23[['pixel_y', 'pixel_x', 'MDI']], 
        #                         on=['pixel_y', 'pixel_x'], suffixes=('_22', '_23'))
        #     if not merged_mdi.empty:
        #         print(f"[DEBUG] Paired MDI pixels: n={len(merged_mdi)}, py range {merged_mdi['pixel_y'].min()}-{merged_mdi['pixel_y'].max()}")
        #         diff_arr = merged_mdi['MDI_22'] - merged_mdi['MDI_23']
                
        #         diff_raster = np.full((ny, nx), -9999.0, dtype=np.float32)
        #         filled_count = 0
        #         for _, row in merged_mdi.iterrows():
        #             py, px = int(row['pixel_y']), int(row['pixel_x'])
        #             if 0 <= py < ny and 0 <= px < nx:
        #                 diff_raster[py, px] = float(diff_arr.iloc[filled_count])
        #                 filled_count += 1
                
        #         print(f"[DEBUG] Filled {filled_count} pixels for MDI")
                
        #         driver = gdal.GetDriverByName('GTiff')
        #         out_ds = driver.Create(os.path.join(OUT_DIR, 'MDI_diff_2022_2023.tif'), nx, ny, 1, gdal.GDT_Float32)
        #         out_ds.SetGeoTransform(gt)
        #         out_ds.SetProjection(proj)
        #         out_band = out_ds.GetRasterBand(1)
        #         out_band.WriteArray(diff_raster)
        #         out_band.SetNoDataValue(-9999.0)
        #         out_band.FlushCache()
        #         out_ds = None
        #         print(f"[SAVED] MDI diff raster: {os.path.join(OUT_DIR, 'MDI_diff_2022_2023.tif')} (n={len(merged_mdi)} paired)")
        #     else:
        #         print("[WARN] No paired pixels for MDI diff export.")



        # # 5) NEW: Process pixel values (daily average) and aggregate to grid summaries
        # print("[INFO] Processing pixel values and aggregating to grid summaries for 2022...")
        # summary_grid_22 = extract_and_process_pixels_by_grid(
        #     df_pixels_22, mask_arr, gt, proj, nx, ny, y_bins, x_bins,  # lat/lon -> y/x
        #     sample_pixels=SAMPLE_PIXELS, gdf_extent=gdf_hhh_proj, hours_to_check=HOURS  # gdf_hhh_4326 -> gdf_hhh_proj
        # )
        # print("[INFO] Processing pixel values and aggregating to grid summaries for 2023...")
        # summary_grid_23 = extract_and_process_pixels_by_grid(
        #     df_pixels_23, mask_arr, gt, proj, nx, ny, y_bins, x_bins,  # 同上
        #     sample_pixels=SAMPLE_PIXELS, gdf_extent=gdf_hhh_proj, hours_to_check=HOURS  # 同上
        # )

        # # 6) Plot geo-faceted figure
        # outfp_facet = os.path.join(OUT_DIR, "Fig_NIRv_Diurnal_GeoFaceted_Projected.png")  # 可选改名
        # plot_nirv_geo_faceted(summary_grid_22, summary_grid_23, gdf_hhh_proj, y_bins, x_bins, outfp_facet)  # gdf_hhh_4326 -> gdf_hhh_proj, lat/lon -> y/x

        # # 7) NEW: Plot specific grid trends
        # print("\n--- Plotting Specific Grid Trends ---")
        # if not summary_grid_22.empty or not summary_grid_23.empty:
        #     plot_specific_grid_trends(summary_grid_22, summary_grid_23, SPECIFIC_GRIDS, OUT_DIR, lon_bins, lat_bins, NUM_ROWS, NUM_COLS)
        # else:
        #     print("[WARN] No grid summary data available to plot specific trends.")

        # print("[DONE] All finished.")

    except Exception as e:
        print("[FATAL] Exception:", e)
        traceback.print_exc()

        print("[DONE] All finished.")

if __name__ == "__main__":
    main()
