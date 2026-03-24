# -*- coding: utf-8 -*-
"""
像元级环境因子集成 + XGBoost + SHAP 可解释性分析
- 版本：V11.0-MultiTarget (Fixed SHAP/XGBoost Compatibility)
- 核心改进：
  1. 修复 XGBoost base_score 格式导致的 SHAP 报错问题
  2. 直接使用 GEE 导出的多波段 ENV_FACTORS TIF
  3. 支持多目标变量
"""

import os
import re
import glob
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import json
import tempfile
from tqdm import tqdm
import warnings
import traceback
from scipy.stats import skew, spearmanr
warnings.filterwarnings('ignore')
import xgboost as xgb
# 可视化/ML
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from xgboost import XGBRegressor
import shap
from project_config import (
    ANCILLARY_DATA_DIR,
    ML_CLASS_SCHEME,
    WHEAT_MASK_TIF as SHARED_WHEAT_MASK_TIF,
    XGB_DEBUG_SUBSAMPLE_N,
    XGB_OUTPUT_DIR,
    XGB_YEARLY_CONFIGS as SHARED_XGB_YEARLY_CONFIGS,
    build_yearly_class_mapping,
)

# GDAL
from osgeo import gdal, osr
gdal.UseExceptions()

# ==================== 调试与路径配置 ====================
DEBUG_MODE = False  # True 时会做限量处理

# 根目录
BASE_DATA_DIR   = r'F:\风云数据\Fy_p2_data'
ANC_BASE_DIR    = r'F:\G_disk\FY4\data\ancillary_data'
OUTPUT_BASE_DIR = r'F:\FY4\outputs_final\pixel_level_xgb_shap_gee'

# Shared repository configuration overrides.
ANC_BASE_DIR = os.fspath(ANCILLARY_DATA_DIR)
OUTPUT_BASE_DIR = os.fspath(XGB_OUTPUT_DIR)

# 🆕 ERA5 环境因子数据路径（与 plot_env_factor_correlations.py 一致）
# ERA5 日尺度数据
ERA5_ENV_RASTER_2022 = r"F:\G_disk\FY4\data\ancillary_data\ERA5_hour\Env_daily_1d3d_dry_20220423.tif"
ERA5_ENV_RASTER_2023 = r"F:\G_disk\FY4\data\ancillary_data\ERA5_hour\Env_daily_1d3d_wet_20230425.tif"
# ERA5 小时级数据
ERA5_HOURLY_RASTER_2022 = r"F:\G_disk\FY4\data\ancillary_data\ERA5_hour\ERA5_hourly_dry_20220423_hourly_stack.tif"
ERA5_HOURLY_RASTER_2023 = r"F:\G_disk\FY4\data\ancillary_data\ERA5_hour\ERA5_hourly_wet_20230425_hourly_stack.tif"

# ERA5 栅格 band 名（1-based 映射）
ERA5_BANDS = [
    "temp_1d", "dew_1d", "u10_1d", "v10_1d", "tp_1d", "ssrd_1d", "swdown_1d", "par_1d", "wind10_1d", "vpd_1d",
    "temp_3d", "dew_3d", "u10_3d", "v10_3d", "tp_3d", "ssrd_3d", "swdown_3d", "par_3d", "wind10_3d", "vpd_3d",
    "precip_sum_7d",
]
# 1-based 索引映射
ERA5_BAND_TO_INDEX = {name: idx + 1 for idx, name in enumerate(ERA5_BANDS)}

# 静态掩膜
WHEAT_MASK_TIF = r'F:\G_disk\FY4\Winter_wheat_map\Winter_wheat_map.tif'
WHEAT_MASK_TIF = os.fspath(SHARED_WHEAT_MASK_TIF)

# 干旱分级图路径
DROUGHT_MASKS = {
    "2022_Drought": r"F:\G_disk\FY4\data\Drought_GEE\drive-download-20251227T184021Z-1-001\SMPct_DroughtClass_20220423.tif",
    "2023_Wet": r"F:\G_disk\FY4\data\Drought_GEE\drive-download-20251227T184021Z-1-001\SMPct_DroughtClass_20230425.tif",
}

# 干旱等级定义：干旱年旱地={2,3,4}，干旱年湿润地={0,1}；非干旱年干旱地={2,3,4}，非干旱年湿润地={0,1}
DROUGHT_CLASSES = {
    "2022_Drought": {"drought": {2, 3, 4}, "wet": {0, 1}},  # 干旱年：旱地={2,3,4}，湿润地={0,1}
    "2023_Wet": {"drought": {2, 3, 4}, "wet": {0, 1}},      # 非干旱年：干旱地={2,3,4}，湿润地={0,1}
}

# ==================== 🔥 目标变量选择 ====================
# Shared class-scheme override for public repository consistency.
DROUGHT_CLASSES = build_yearly_class_mapping(ML_CLASS_SCHEME)

AVAILABLE_TARGETS = {
    'nirv': 'target_nirv',
    't_peak': 'target_t_peak',
    'mdi': 'target_mdi',
    'c_shift': 'centroid_shift'
}

TARGET_MODE = 'diurnal_core'

if TARGET_MODE == 'all':
    TRAINING_TARGETS = list(AVAILABLE_TARGETS.values())
elif TARGET_MODE in AVAILABLE_TARGETS:
    TRAINING_TARGETS = [AVAILABLE_TARGETS[TARGET_MODE]]
elif TARGET_MODE == 'diurnal_core':
    TRAINING_TARGETS = [AVAILABLE_TARGETS[k] for k in ('c_shift', 't_peak', 'mdi')]
else:
    raise ValueError("Invalid TARGET_MODE")

# ==================== 环境因子选择（基于相关性分析优化）====================
# 指定的环境因子及时间尺度（列名优先级从前到后尝试）
# 根据相关性分析，优化为：
# - 气孔调节：VPD, LST（相关性最强）
# - 含水量调节：SM, P (precip)（使用7d降水）
# - 碳调节：PAR, SW↓ (swdown), AOD（补充AOD）
# - 其他：保留1-2个最强的（elevation, soc）
ENV_CHOICES = [
    # 气孔调节（Stomatal Regulation）
    # 基于相关性总表选择最佳时间尺度
    ("VPD", ["vpd_hmean_9_16"]),      # 相关性：0.735（最高）
    ("LST", ["lst_3d"]),               # 相关性：0.672（最高）
    ("Temp", ["temp_hmean_9_16"]),    # 相关性：0.745（最高）
    # 含水量调节（Water Content Regulation）
    ("SM", ["sm_mean_last_7d"]),      # 相关性：0.335（唯一选项）
    ("PPT", ["precip_sum_7d"]),       # 相关性：0.611（最高），使用7d降水
    # 碳调节（Carbon Regulation）
    ("PAR", ["par_hmean_9_16"]),      # 相关性：0.657（最高，移除SW↓相关选项）
    ("AOD", ["aod_mean_last_7d"]),    # 相关性：0.344（唯一选项）
    # 其他（Other）- 保留1-2个最强的
    ("Elev", ["elevation"]),          # 相关性：0.261（唯一选项）
    ("SOC", ["soc"]),                 # 相关性：0.519（唯一选项）
]

# ==================== 特征选择策略 ====================
# True: 使用所有可用环境因子（排除 LAI/NDVI）
USE_ALL_ENV_FACTORS = True
# True: 对每个目标变量在候选时间尺度中自动选择最重要的尺度
AUTO_SELECT_TIME_SCALE = False
# 预筛选模型的树数量（用于快速估计重要性）
PRESELECT_N_ESTIMATORS = 200

# 候选时间尺度集合（用于自动选择）
TIME_SCALE_GROUPS = {
    "VPD": ["vpd_hmean_9_16", "vpd_3d", "vpd_1d", "vpd_mean_last_3d"],
    "LST": ["lst_3d", "lst_1d", "lst_mean_last_3d"],
    "Temp": ["temp_hmean_9_16", "temp_3d", "temp_1d"],
    "SM": ["sm_mean_last_7d", "sm_7d", "sm_mean_7d", "sm_mean_last_30d"],
    "PPT": ["precip_sum_7d", "tp_3d", "tp_1d", "tp_7d", "precip_sum_last_30d"],
    "PAR": ["par_hmean_9_16", "par_1d", "par_inst_umol_s", "par_cum_to_obs_mol", "par_cum_day_mol"],
    "AOD": ["aod_mean_last_7d", "aod_7d", "aod_mean_7d"],
    "Elev": ["elevation", "elev", "dem", "altitude"],
    "SOC": ["soc", "soil_organic_carbon", "organic_carbon", "soil_carbon"],
}

# 从ENV_CHOICES中提取所有可能的环境因子列名（精确到时间尺度）
# 基于相关性总表选择的最佳时间尺度
FINAL_FEATURE_SET_BASE = [
    # 从ENV_CHOICES中提取的所有选项
    # 气孔调节
    'vpd_hmean_9_16',
    'lst_3d',
    'temp_hmean_9_16',
    # 含水量调节
    'sm_mean_last_7d',
    'precip_sum_7d',
    # 碳调节
    'par_hmean_9_16',  # PAR最佳时间尺度
    'aod_mean_last_7d',
    # 其他（保留1-2个最强的）
    'elevation',
    'soc',
]

# 兼容旧的环境因子名称（如果数据中有的话）
FINAL_FEATURE_SET = FINAL_FEATURE_SET_BASE + [
    # 兼容旧名称（保留一些备用选项以防数据中不存在最佳选项）
    'vpd_mean_last_3d', 'vpd_3d', 'vpd_1d',  # VPD备用
    'lst_mean_last_3d', 'lst_1d',  # LST备用
    'temp_1d', 'temp_3d',  # Temp备用
    'par_inst_umol_s', 'par_cum_to_obs_mol', 'par_cum_day_mol', 'par_1d',  # PAR备用
    'tp_7d', 'tp_3d', 'tp_1d', 'precip_sum_last_30d',  # PPT备用
]

# GEE 环境因子 TIF 路径（包含 LST, SM, AOD, Elev, SOC 等静态和动态因子）
# 如果这些文件不存在，将从 ERA5 数据中缺失这些因子
GEE_ENV_TIF_2022 = r"F:\G_disk\FY4\data\ancillary_data\GEE\Env_factors_20220423.tif"
GEE_ENV_TIF_2023 = r"F:\G_disk\FY4\data\ancillary_data\GEE\Env_factors_20230425.tif"

# 额外环境因子（AOD/SMAP/DEM/SOIL）导出目录
EXPORT_ENV_BASE_DIR = r"F:\G_disk\FY4\data\ancillary_data\export_tif"

# 年度批次配置
YEARLY_CONFIGS = [
    {
        "year_tag": "2022_Drought",
        "brdf_dir": os.path.join(BASE_DATA_DIR, r'2022_0423_week\Brdf_hhh_v2'),
        "target_dates": [date(2022, 4, 23)],
        "era5_env_raster": ERA5_ENV_RASTER_2022,
        "era5_hourly_raster": ERA5_HOURLY_RASTER_2022,
        "drought_tif": DROUGHT_MASKS["2022_Drought"],
        "gee_env_tif": GEE_ENV_TIF_2022,  # 可选：GEE 环境因子 TIF
        "par_k": 4.6,
    },
    {
        "year_tag": "2023_Wet",
        "brdf_dir": os.path.join(BASE_DATA_DIR, r'2023_0423_week\Brdf_hhh_v2'),
        "target_dates": [date(2023, 4, 25)],
        "era5_env_raster": ERA5_ENV_RASTER_2023,
        "era5_hourly_raster": ERA5_HOURLY_RASTER_2023,
        "drought_tif": DROUGHT_MASKS["2023_Wet"],
        "gee_env_tif": GEE_ENV_TIF_2023,  # 可选：GEE 环境因子 TIF
        "par_k": 4.5,
    },
]

# Shared yearly-config override so public and manuscript-facing scripts read the
# same date windows and input locations by default.
YEARLY_CONFIGS = [
    {
        "year_tag": cfg["year_tag"],
        "brdf_dir": os.fspath(cfg["brdf_dir"]),
        "target_dates": cfg["target_dates"],
        "era5_env_raster": os.fspath(cfg["era5_env_raster"]),
        "era5_hourly_raster": os.fspath(cfg["era5_hourly_raster"]),
        "drought_tif": os.fspath(cfg["drought_tif"]),
        "gee_env_tif": os.fspath(cfg["gee_env_tif"]),
        "par_k": cfg["par_k"],
    }
    for cfg in SHARED_XGB_YEARLY_CONFIGS
]

# 其它参数
SAMPLE_PIXELS = 0          # 0 表示不抽样，保持全量高分辨率
RED_BAND = 2
NIR_BAND = 3
DIURNAL_HOURS = [9, 10, 11, 12, 13, 14, 15, 16] 
VALIDATION_RATIO = 0              # 0 表示不做holdout验证，仅使用CV
MIN_VALIDATION_SAMPLES = 0        # 无holdout验证
EARLY_STOPPING_ROUNDS = 30        # 仍保留参数以防需要

# ==================== 工具方法 ====================
def parse_fy4_timestamp_from_name(fp, year):
    name = os.path.basename(fp)
    match = re.search(r'(\d{10})\.tif$', name)
    if match and year:
        try:
            time_str = str(year) + match.group(1)
            dt_utc = datetime.strptime(time_str, "%Y%m%d%H%M%S")
            return dt_utc + timedelta(hours=8)
        except ValueError:
            pass
    m3 = re.search(r'(\d{8})(?:[_-]?(\d{2}))?', name)
    if m3:
        dstr, hh = m3.group(1), m3.group(2)
        try:
            d = datetime.strptime(dstr, "%Y%m%d")
            return d + timedelta(hours=int(hh)) if hh else d
        except:
            return None
    return None

def resample_to_ref(src_fp, ref_fp, band_index=1):
    ref_ds = gdal.Open(ref_fp)
    if ref_ds is None:
        raise RuntimeError("Could not open ref: " + ref_fp)
    gt, proj = ref_ds.GetGeoTransform(), ref_ds.GetProjection()
    cols, rows = ref_ds.RasterXSize, ref_ds.RasterYSize
    ref_ds = None
    
    mem = gdal.Warp('', src_fp, format='MEM', width=cols, height=rows, 
                    dstSRS=proj, resampleAlg='near')
    if mem is None:
        raise RuntimeError(f"Failed to warp {src_fp}")
    arr = mem.GetRasterBand(band_index).ReadAsArray().astype(float)
    mem = None
    return arr

def get_geo_coords_from_ref(ref_fp):
    ds = gdal.Open(ref_fp)
    gt = ds.GetGeoTransform()
    cols, rows = ds.RasterXSize, ds.RasterYSize
    ds = None
    xs = np.arange(cols) * gt[1] + gt[0] + gt[1]/2.0
    ys = np.arange(rows) * gt[5] + gt[3] + gt[5]/2.0
    lon, lat = np.meshgrid(xs, ys)
    return lon, lat

def compute_regression_metrics(y_true, y_pred):
    """兼容旧版 sklearn：RMSE/MAE/R2 计算，不使用 squared 参数。"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2

def get_all_env_features(df, targets):
    """使用所有环境因子列（排除 LAI 和非环境字段）。"""
    cols = [c.lower() for c in df.columns]
    exclude = {
        'date', 'lon', 'lat', 'drought_class',
        'target_nirv', 'target_t_peak', 'target_mdi', 'centroid_shift',
        'a_nirv', 'skew', 'recovery_rate', 'nirv_integral', 'centroid',
    }
    exclude.update([t.lower() for t in targets])
    env_cols = []
    for c in cols:
        if c in exclude:
            continue
        if 'lai' in c or 'ndvi' in c:
            continue
        env_cols.append(c)
    # 去重并保持顺序
    return list(dict.fromkeys(env_cols))

def select_best_scales_by_shap(df, target, scale_groups):
    """
    对每个因子组（如VPD/LST/Temp等）在候选时间尺度中选择 SHAP 重要性最高的尺度。
    返回：selected_features(list), group_choice(dict)
    """
    d = df.dropna(subset=[target]).copy()
    if d.empty:
        return [], {}

    # 收集候选特征
    candidates = []
    group_to_features = {}
    cols = set([c.lower() for c in d.columns])
    for group_name, options in scale_groups.items():
        available = [opt for opt in options if opt.lower() in cols]
        if available:
            group_to_features[group_name] = available
            candidates.extend(available)

    if not candidates:
        return [], {}

    X = d[candidates].copy()
    X.fillna(X.median(numeric_only=True), inplace=True)
    y = d[target].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    model = xgb.XGBRegressor(
        n_estimators=PRESELECT_N_ESTIMATORS,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(Xs, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)
    shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values
    mean_abs = np.abs(shap_vals).mean(axis=0)
    mean_abs_by_feat = dict(zip(candidates, mean_abs))

    selected_features = []
    group_choice = {}
    for group_name, feats in group_to_features.items():
        best_feat = max(feats, key=lambda f: mean_abs_by_feat.get(f, -np.inf))
        selected_features.append(best_feat)
        group_choice[group_name] = best_feat

    return selected_features, group_choice

def sample_raster_by_lonlat(raster_fp, lon, lat, band_map):
    """按 lon/lat 采样多 band 栅格，返回 dict: band_name -> values(np.array)。band_map 自动截断到实际 band 数。"""
    if not os.path.exists(raster_fp):
        raise FileNotFoundError(f"Raster not found: {raster_fp}")
    ds = gdal.Open(raster_fp)
    if ds is None:
        raise FileNotFoundError(f"无法打开栅格：{raster_fp}")
    gt = ds.GetGeoTransform()
    arrs = {}
    x = ((lon - gt[0]) / gt[1]).astype(int)
    y = ((lat - gt[3]) / gt[5]).astype(int)
    h = ds.RasterYSize
    w = ds.RasterXSize
    # 根据实际 band 数截断
    band_count = ds.RasterCount
    band_map_trunc = {k: v for k, v in band_map.items() if v <= band_count}
    if len(band_map_trunc) < len(band_map):
        print(f"[WARN] {raster_fp} 只有 {band_count} 个 band，已截断映射到 {len(band_map_trunc)} 个可用变量。")
    for name, bidx in band_map_trunc.items():
        band = ds.GetRasterBand(bidx)
        if band is None:
            continue
        arr = band.ReadAsArray()
        vals = np.full(len(lon), np.nan, dtype=float)
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        vals[valid] = arr[y[valid], x[valid]]
        # 单位调整：precip_sum_7d 从米到毫米
        if name == "precip_sum_7d":
            vals = vals * 1000.0
        arrs[name] = vals
    ds = None
    return arrs

def get_daily_band_map_from_descriptions(raster_fp):
    """基于 band 描述动态建立 ERA5 日尺度栅格映射。"""
    ds = gdal.Open(raster_fp)
    if ds is None:
        return {}
    name_map = {
        "temperature_2m_1d": "temp_1d",
        "dewpoint_temperature_2m_1d": "dew_1d",
        "u_component_of_wind_10m_1d": "u10_1d",
        "v_component_of_wind_10m_1d": "v10_1d",
        "total_precipitation_1d": "tp_1d",
        "surface_solar_radiation_downwards_1d": "ssrd_1d",
        "swdown_wm2_1d": "swdown_1d",
        "par_wm2_1d": "par_1d",
        "wind10_1d": "wind10_1d",
        "vpd_kpa_1d": "vpd_1d",
        "temperature_2m_3d": "temp_3d",
        "dewpoint_temperature_2m_3d": "dew_3d",
        "u_component_of_wind_10m_3d": "u10_3d",
        "v_component_of_wind_10m_3d": "v10_3d",
        "total_precipitation_3d": "tp_3d",
        "surface_solar_radiation_downwards_3d": "ssrd_3d",
        "swdown_wm2_3d": "swdown_3d",
        "par_wm2_3d": "par_3d",
        "wind10_3d": "wind10_3d",
        "vpd_kpa_3d": "vpd_3d",
        "precip_sum_7d": "precip_sum_7d",
        "lst_1d": "lst_1d",
        "lst_3d": "lst_3d",
        "lai_1d": "lai_1d",
        "lai_3d": "lai_3d",
        "ndvi_1d": "ndvi_1d",
        "ndvi_3d": "ndvi_3d",
    }
    band_map = {}
    for i in range(1, ds.RasterCount + 1):
        desc = ds.GetRasterBand(i).GetDescription()
        if not desc:
            continue
        key = desc.strip().lower()
        if key in name_map:
            band_map[name_map[key]] = i
    ds = None
    return band_map


def sample_hourly_stack(raster_fp, lon, lat, hours_local=list(range(9, 17))):
    """
    采样小时级栅格（band 描述形如 20220423T00_temperature_2m）。
    返回逐小时特征（var_h09...）以及 9-16 小时的均值/累计（var_hmean_9_16, var_hsum_9_16）。
    """
    if not os.path.exists(raster_fp):
        raise FileNotFoundError(f"Hourly raster not found: {raster_fp}")
    ds = gdal.Open(raster_fp)
    if ds is None:
        raise FileNotFoundError(f"无法打开栅格：{raster_fp}")
    gt = ds.GetGeoTransform()
    x = ((lon - gt[0]) / gt[1]).astype(int)
    y = ((lat - gt[3]) / gt[5]).astype(int)
    h = ds.RasterYSize
    w = ds.RasterXSize

    # 变量名映射
    var_map = {
        "temperature_2m": "temp",
        "dewpoint_temperature_2m": "dew",
        "u_component_of_wind_10m": "u10",
        "v_component_of_wind_10m": "v10",
        "total_precipitation": "tp",
        "surface_solar_radiation_downwards": "ssrd",
        "SWdown_Wm2": "swdown",
        "PAR_Wm2": "par",
        "Wind10": "wind10",
        "VPD_kPa": "vpd",
    }

    per_hour = {}
    per_var_hour_vals = {}

    band_count = ds.RasterCount
    for bidx in range(1, band_count + 1):
        band = ds.GetRasterBand(bidx)
        desc = band.GetDescription()
        # 期望格式：YYYYMMDDTHH_xxx
        parts = desc.split("_", 1)
        if len(parts) < 2 or "T" not in parts[0]:
            continue
        date_part = parts[0]
        rest = parts[1]
        try:
            hour_utc = int(date_part.split("T")[1][:2])
        except Exception:
            continue
        # 转本地时（+8）
        hour_local = (hour_utc + 8) % 24
        if hour_local not in hours_local:
            continue
        var_raw = rest
        var_name = var_map.get(var_raw, var_raw.lower())

        arr = band.ReadAsArray()
        vals = np.full(len(lon), np.nan, dtype=float)
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        vals[valid] = arr[y[valid], x[valid]]

        feat_name = f"{var_name}_h{hour_local:02d}"
        per_hour[feat_name] = vals
        per_var_hour_vals.setdefault(var_name, {})[hour_local] = vals

    # 聚合 9-16：均值与累计（tp/ssrd/swdown/par 用累计，其他用均值；也同时给出均值）
    agg = {}
    for var_name, hv in per_var_hour_vals.items():
        hours_sorted = sorted(hv.keys())
        if not hours_sorted:
            continue
        stack = np.stack([hv[h] for h in hours_sorted], axis=0)
        mean_vals = np.nanmean(stack, axis=0)
        sum_vars = {"tp", "ssrd", "swdown", "par"}
        if var_name in sum_vars:
            agg_vals = np.nansum(stack, axis=0)
        else:
            agg_vals = mean_vals
        agg[f"{var_name}_hmean_9_16"] = mean_vals
        agg[f"{var_name}_hsum_9_16"] = agg_vals

    ds = None
    per_hour.update(agg)
    return per_hour


def load_gee_env_factors(gee_tif_path, ref_fp):
    """
    从 GEE 导出的环境因子 TIF 加载 LST, SM, AOD, Elev, SOC 等因子，
    返回 dict: band_name -> values(np.array)，其中 values 是 2D 数组（与参考栅格形状一致）。
    """
    if not gee_tif_path or not os.path.exists(gee_tif_path):
        print(f"[GEE] GEE environment factor TIF not provided or not found: {gee_tif_path}")
        return {}

    print(f"\n[GEE] Loading GEE environmental factors from: {gee_tif_path}")

    # GEE TIF 的 band 名称映射（按实际导出顺序，需要与 GEE 导出一致）
    GEE_BAND_NAMES = [
        "lst_mean_last_3d",  # LST 3d
        "sm_mean_last_7d",   # SM 7d
        "aod_mean_last_7d",  # AOD 7d
        "elevation",         # Elevation
        "soc",               # SOC
    ]

    try:
        # 读取参考栅格信息
        ref_ds = gdal.Open(ref_fp)
        if ref_ds is None:
            print(f"[GEE] Failed to open reference file: {ref_fp}")
            return {}
        gt, proj = ref_ds.GetGeoTransform(), ref_ds.GetProjection()
        cols, rows = ref_ds.RasterXSize, ref_ds.RasterYSize
        ref_ds = None
        
        # 重投影并重采样到参考栅格
        mem = gdal.Warp(
            "",
            gee_tif_path,
            format="MEM",
            width=cols,
            height=rows,
            dstSRS=proj,
            resampleAlg="near",
        )
        if mem is None:
            print("[GEE] Failed to warp GEE TIF to reference grid")
            return {}
        
        env_factors = {}
        n_bands = mem.RasterCount

        # 读取每个 band（按顺序映射到名称）
        for i, band_name in enumerate(GEE_BAND_NAMES[:n_bands], start=1):
            band = mem.GetRasterBand(i)
            if band is None:
                continue
            arr = band.ReadAsArray().astype(float)
            # 处理无效值
            arr[arr < -9990] = np.nan
            env_factors[band_name.lower()] = arr
        
        mem = None
        print(f"[GEE] Loaded {len(env_factors)} GEE environmental factors: {list(env_factors.keys())}")
        return env_factors
    except Exception as e:
        print(f"[GEE] Failed to load GEE environmental factors: {e}")
        import traceback
        traceback.print_exc()
        return {}

def _find_best_tif_in_folder(folder_path, target_date=None):
    """从目录中选择最匹配的 TIF：优先 static，其次按日期最接近 target_date。"""
    if not folder_path or not os.path.isdir(folder_path):
        return None
    tifs = sorted(glob.glob(os.path.join(folder_path, "*.tif")))
    if not tifs:
        return None
    for fp in tifs:
        if "static" in os.path.basename(fp).lower():
            return fp
    if target_date is not None:
        candidates = []
        for fp in tifs:
            name = os.path.basename(fp)
            m = re.search(r"(\d{8})", name)
            if not m:
                continue
            try:
                d = datetime.strptime(m.group(1), "%Y%m%d").date()
                candidates.append((abs((d - target_date).days), d, fp))
            except Exception:
                continue
        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]))
            return candidates[0][2]
    return tifs[0]

def _find_band_index_by_keywords(ds, keywords):
    for i in range(1, ds.RasterCount + 1):
        desc = (ds.GetRasterBand(i).GetDescription() or "").lower()
        if any(k in desc for k in keywords):
            return i
    return 1

def load_export_env_factors(export_base_dir, target_date, ref_fp):
    """
    从 export_tif 目录加载 AOD/SMAP/DEM/SOIL 等非 ERA5 因子。
    返回 dict: factor_name -> 2D array
    """
    env_factors = {}
    if not export_base_dir or not os.path.isdir(export_base_dir):
        print(f"[EXPORT] export_tif not found: {export_base_dir}")
        return env_factors

    source_map = {
        "aod_mean_last_7d": "ENV_FACTORS_HHH-AOD",
        "sm_mean_last_7d": "ENV_FACTORS_HHH-SMAP",
    }

    # AOD / SMAP 单波段
    for factor_name, folder in source_map.items():
        folder_path = os.path.join(export_base_dir, folder)
        tif_path = _find_best_tif_in_folder(folder_path, target_date)
        if not tif_path:
            print(f"[EXPORT] Missing {factor_name} in {folder_path}")
            continue
        try:
            arr = resample_to_ref(tif_path, ref_fp, band_index=1).astype(float)
            arr[arr < -9990] = np.nan
            env_factors[factor_name] = arr
            print(f"[EXPORT] Loaded {factor_name} from {tif_path}")
        except Exception as exc:
            print(f"[EXPORT] Failed to load {factor_name} from {tif_path}: {exc}")

    # DEM：多波段（elevation, slope, aspect）
    dem_folder = os.path.join(export_base_dir, "ENV_FACTORS_HHH-DEM")
    dem_path = _find_best_tif_in_folder(dem_folder, target_date)
    if dem_path:
        dem_bands = {1: "elevation", 2: "slope", 3: "aspect"}
        for band_index, name in dem_bands.items():
            try:
                arr = resample_to_ref(dem_path, ref_fp, band_index=band_index).astype(float)
                arr[arr < -9990] = np.nan
                env_factors[name] = arr
                print(f"[EXPORT] Loaded {name} from {dem_path} (band {band_index})")
            except Exception as exc:
                print(f"[EXPORT] Failed to load {name} from {dem_path}: {exc}")
    else:
        print(f"[EXPORT] Missing DEM in {dem_folder}")

    # SOIL：多波段（sand, clay, soc, ph, texture）
    soil_folder = os.path.join(export_base_dir, "ENV_FACTORS_HHH-SOIL")
    soil_path = _find_best_tif_in_folder(soil_folder, target_date)
    if soil_path:
        try:
            ds = gdal.Open(soil_path)
            if ds is None:
                raise RuntimeError("could not open soil tif")
            band_map = {}
            for i in range(1, ds.RasterCount + 1):
                desc = (ds.GetRasterBand(i).GetDescription() or "").strip().lower()
                meta_desc = (ds.GetRasterBand(i).GetMetadata().get("DESCRIPTION", "") or "").strip().lower()
                label = desc or meta_desc
                if label:
                    band_map[label] = i
            ds = None

            fallback = {1: "sand", 2: "clay", 3: "soc", 4: "ph", 5: "texture"}
            target_labels = ["sand", "clay", "soc", "ph", "texture"]
            for label in target_labels:
                band_index = band_map.get(label)
                if band_index is None:
                    for idx, name in fallback.items():
                        if name == label:
                            band_index = idx
                            break
                if band_index is None:
                    print(f"[EXPORT] Missing soil band for {label} in {soil_path}")
                    continue
                arr = resample_to_ref(soil_path, ref_fp, band_index=band_index).astype(float)
                arr[arr < -9990] = np.nan
                env_factors[label] = arr
                print(f"[EXPORT] Loaded {label} from {soil_path} (band {band_index})")
        except Exception as exc:
            print(f"[EXPORT] Failed to load soil factors from {soil_path}: {exc}")
    else:
        print(f"[EXPORT] Missing SOIL in {soil_folder}")

    return env_factors

def load_era5_env_factors(
    era5_env_raster,
    era5_hourly_raster,
    lon_grid,
    lat_grid,
    gee_env_tif=None,
    ref_fp=None,
    target_date=None,
    export_env_base_dir=None,
):
    """
    从 ERA5 数据加载环境因子（与 plot_env_factor_correlations.py 一致）
    可选：从 GEE TIF 加载额外的环境因子（LST, SM, AOD, Elev, SOC）
    返回 dict: band_name -> values(np.array)，其中 values 是2D数组（与lon_grid/lat_grid形状相同）
    """
    print(f"\n[ERA5] Loading environmental factors from: {era5_env_raster}")
    
    env_factors = {}
    
    # 采样日尺度数据
    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()
    
    try:
        band_map = get_daily_band_map_from_descriptions(era5_env_raster)
        if not band_map:
            band_map = ERA5_BAND_TO_INDEX
        sampled_daily = sample_raster_by_lonlat(era5_env_raster, lon_flat, lat_flat, band_map)
        for k, v in sampled_daily.items():
            # 将1D数组重塑为2D数组
            env_factors[k] = v.reshape(lon_grid.shape)
        print(f"[ERA5] Loaded {len(sampled_daily)} daily variables: {list(sampled_daily.keys())[:10]}...")
    except Exception as e:
        print(f"[WARN] Failed to load daily ERA5 data: {e}")
    
    # 采样小时级数据
    try:
        sampled_hourly = sample_hourly_stack(era5_hourly_raster, lon_flat, lat_flat, hours_local=list(range(9, 17)))
        for k, v in sampled_hourly.items():
            # 将1D数组重塑为2D数组
            env_factors[k] = v.reshape(lon_grid.shape)
        print(f"[ERA5] Loaded {len(sampled_hourly)} hourly variables")
    except Exception as e:
        print(f"[WARN] Failed to load hourly ERA5 data: {e}")
    
    # 尝试加载 GEE 环境因子（LST, SM, AOD, Elev, SOC）
    if gee_env_tif and ref_fp:
        gee_factors = load_gee_env_factors(gee_env_tif, ref_fp)
        for k, v in gee_factors.items():
            env_factors[k] = v
    else:
        print(f"[GEE] GEE environment factor TIF not provided (gee_env_tif={gee_env_tif})")

    # 尝试加载 export_tif 目录中的额外环境因子（AOD/SMAP/DEM/SOIL）
    if export_env_base_dir and ref_fp:
        export_factors = load_export_env_factors(export_env_base_dir, target_date, ref_fp)
        for k, v in export_factors.items():
            if k not in env_factors:
                env_factors[k] = v
            else:
                existing = env_factors[k]
                if isinstance(existing, np.ndarray) and np.all(~np.isfinite(existing)):
                    env_factors[k] = v
    else:
        print(f"[EXPORT] export_tif not provided (export_env_base_dir={export_env_base_dir})")
    
    # 调试：打印所有已加载的环境因子名称
    print(f"[DEBUG] Total loaded {len(env_factors)} environmental factors")
    print(f"[DEBUG] Factor names: {sorted(env_factors.keys())}")
    
    # 检查期望的因子是否存在
    expected_factors = {
        'lst_3d': ['lst_3d', 'lst_mean_last_3d'],
        'sm_mean_last_7d': ['sm_mean_last_7d', 'sm_7d'],
        'aod_mean_last_7d': ['aod_mean_last_7d', 'aod_7d'],
        'elevation': ['elevation', 'elev'],
        'soc': ['soc', 'soil_organic_carbon'],
    }
    for expected, variants in expected_factors.items():
        found = False
        for variant in variants:
            if variant.lower() in [k.lower() for k in env_factors.keys()]:
                print(f"[DEBUG] ✓ Found {expected} as '{variant}'")
                found = True
                break
        if not found:
            print(f"[DEBUG] ✗ Missing {expected} (searched variants: {variants})")
    
    print(f"[ERA5] Successfully loaded {len(env_factors)} environmental factors")
    return env_factors


# ==================== 🆕 日周期指标计算 ====================
def compute_metrics_from_hourly(hours_list, med_vals):
    hrs = list(hours_list)
    vals = np.array(med_vals, dtype=float)
    
    if np.all(np.isnan(vals)) or len(vals) == 0:
        return {k: np.nan for k in ['MDI','t_peak','A_NIRv','Skew','Recovery_rate',
                                    'NIRv_integral','centroid','centroid_shift']}
    
    def idxs(subhrs):
        return [hrs.index(h) for h in subhrs if h in hrs]
    
    morning_idx = idxs([9, 10, 11])
    noon_idx = idxs([12, 13, 14])
    afternoon_idx = idxs([15, 16])
    
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
        valid_vals = vals[~np.isnan(vals)]
        if len(valid_vals) >= 3:
            skewness = float(skew(valid_vals))
    except Exception:
        pass
    
    recovery_rate = np.nan
    if len(noon_idx) > 0 and len(afternoon_idx) > 0:
        mean_afternoon = np.nanmean(vals[afternoon_idx])
        time_diff = np.mean(afternoon_idx) - np.mean(noon_idx) if len(noon_idx) > 0 else 1.0
        if not np.isnan(mean_afternoon) and not np.isnan(min_noon) and time_diff != 0 and abs(min_noon) > 1e-9:
            recovery_rate = (mean_afternoon - min_noon) / time_diff / abs(min_noon)
    
    try:
        integral = float(np.trapz(vals, hrs))
    except Exception:
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

# ==================== 组装像元级特征表 ====================
def build_pixel_feature_table_gee(brdf_dir, target_dates, era5_env_raster, era5_hourly_raster, par_k, **kwargs):
    brdf_files = sorted(glob.glob(os.path.join(brdf_dir, '*.tif')))
    if not brdf_files:
        raise RuntimeError("FATAL: No BRDF files found in " + brdf_dir)
    ref_fp = brdf_files[0]

    year_match = re.search(r'(\d{4})', brdf_dir)
    if not year_match:
        raise ValueError("FATAL: Cannot determine year from BRDF directory path")
    year_from_dir = int(year_match.group(1))
    print(f"\n[INFO] Year extracted from directory: {year_from_dir}")

    sample_pixels = kwargs.get('sample_pixels', 0)
    drought_tif = kwargs.get('drought_tif', None)
    if DEBUG_MODE:
        print(f">>> DEBUG MODE: Processing only first 5 BRDF files")
        brdf_files = brdf_files[:5]
        if sample_pixels > 0:
            sample_pixels = min(sample_pixels, 500)

    mask_arr = resample_to_ref(WHEAT_MASK_TIF, ref_fp, band_index=1).astype(np.uint8) == 1
    lon_grid, lat_grid = get_geo_coords_from_ref(ref_fp)
    print(f"[INFO] Mask shape: {mask_arr.shape}, Valid pixels: {np.sum(mask_arr)}")

    # 加载干旱分级数据
    drought_class_arr = None
    if drought_tif and os.path.exists(drought_tif):
        try:
            drought_class_arr = resample_to_ref(drought_tif, ref_fp, band_index=1).astype(int)
            print(f"[INFO] Loaded drought classification from {drought_tif}")
            print(f"[INFO] Drought class distribution: {np.unique(drought_class_arr, return_counts=True)}")
        except Exception as e:
            print(f"[WARN] Failed to load drought classification: {e}")
            drought_class_arr = None
    else:
        print(f"[WARN] Drought classification file not provided or not found: {drought_tif}")

    # 加载 ERA5 环境因子数据（与 plot_env_factor_correlations.py 一致）
    # 可选：从 GEE TIF 加载额外的环境因子（LST, SM, AOD, Elev, SOC）
    gee_env_tif = kwargs.get('gee_env_tif', None)
    target_date = target_dates[0] if target_dates else None
    era5_env_factors = load_era5_env_factors(
        era5_env_raster,
        era5_hourly_raster,
        lon_grid,
        lat_grid,
        gee_env_tif=gee_env_tif,
        ref_fp=ref_fp,
        target_date=target_date,
        export_env_base_dir=EXPORT_ENV_BASE_DIR,
    )
    if not era5_env_factors:
        raise RuntimeError("FATAL: Failed to load ERA5 environmental factors")

    print("\n[INFO] Step 1: Collecting hourly NIRv time series for each pixel...")
    pixel_hourly_data = {}
    
    for fp in tqdm(brdf_files, desc='Collecting hourly data'):
        dt = parse_fy4_timestamp_from_name(fp, year_from_dir)
        if dt is None or dt.date() not in target_dates or dt.hour not in DIURNAL_HOURS:
            continue

        ds = gdal.Open(fp)
        if ds is None:
            continue
        nir = ds.GetRasterBand(NIR_BAND).ReadAsArray().astype(float)
        red = ds.GetRasterBand(RED_BAND).ReadAsArray().astype(float)
        ds = None

        valid_pixels_mask = mask_arr & np.isfinite(nir) & np.isfinite(red)
        rows_idx, cols_idx = np.where(valid_pixels_mask)
        
        if len(rows_idx) == 0:
            continue

        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = np.where((nir + red) > 1e-6, (nir - red) / (nir + red), np.nan)
        nirv = nir * ndvi

        for r, c in zip(rows_idx, cols_idx):
            key = (dt.date(), r, c)
            if key not in pixel_hourly_data:
                pixel_hourly_data[key] = {}
            pixel_hourly_data[key][dt.hour] = float(nirv[r, c])

    print(f"[INFO] Collected data for {len(pixel_hourly_data)} pixel-date combinations")

    print("\n[INFO] Step 2: Computing diurnal metrics for each pixel-date...")
    rows = []
    
    for (d, r, c), hourly_dict in tqdm(pixel_hourly_data.items(), desc='Computing metrics'):
        hours_sorted = sorted(hourly_dict.keys())
        vals_sorted = [hourly_dict[h] for h in hours_sorted]
        
        metrics = compute_metrics_from_hourly(hours_sorted, vals_sorted)
        
        rec = {
            'date': d.isoformat(),
            'lon': float(lon_grid[r, c]),
            'lat': float(lat_grid[r, c]),
            'target_nirv': hourly_dict.get(12, np.nan),
            'target_t_peak': metrics['t_peak'],
            'target_mdi': metrics['MDI'],
            'a_nirv': metrics['A_NIRv'],
            'skew': metrics['Skew'],
            'recovery_rate': metrics['Recovery_rate'],
            'nirv_integral': metrics['NIRv_integral'],
            'centroid': metrics['centroid'],
            'centroid_shift': metrics['centroid_shift'],
        }

        for factor_name, factor_arr in era5_env_factors.items():
            try:
                val = float(factor_arr[r, c])
                if val < -9990 or not np.isfinite(val):
                    val = np.nan
                rec[factor_name] = val
            except Exception:
                rec[factor_name] = np.nan

        # 添加干旱等级
        if drought_class_arr is not None:
            try:
                rec['drought_class'] = int(drought_class_arr[r, c])
            except Exception:
                rec['drought_class'] = -1  # 无效值
        else:
            rec['drought_class'] = -1  # 无数据

        rows.append(rec)

    if sample_pixels and 0 < sample_pixels < len(rows):
        print(f"[INFO] Sampling {sample_pixels} from {len(rows)} pixel-dates")
        import random
        random.seed(42)
        rows = random.sample(rows, sample_pixels)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.columns = [c.lower() for c in df.columns]
    
    print(f"[INFO] Built feature table with {len(df)} rows and {len(df.columns)} columns")
    return df

# ==================== 安全写入工具 ====================
def safe_to_csv(df, out_fp):
    """
    将 DataFrame 安全写入 CSV。
    先写入同目录下的临时文件，再用 os.replace 原子替换，避免已有文件被占用导致 PermissionError。
    """
    directory = os.path.dirname(out_fp)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix='.csv', dir=directory if directory else None)
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, out_fp)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

# ==================== 训练 + SHAP (修复版) ====================

# 终极修复函数：通过 JSON 重载清洗 base_score
def get_clean_booster(xgb_model):
    """
    终极方案：将模型保存为 JSON，用正则清洗所有 base_score，再重新加载。
    这样 SHAP 在任何读取路径都会拿到干净的值。
    """
    import tempfile
    import re
    import json
    
    booster = xgb_model.get_booster()
    
    # 创建临时 JSON 文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # 1. 保存为 JSON
        booster.save_model(tmp_path)
        
        # 2. 读取内容
        with open(tmp_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        #诊断：打印所有包含 base_score 的行
        print("[DEBUG] Lines containing 'base_score' in JSON:")
        for i, line in enumerate(content.split('\n'), 1):
            if 'base_score' in line.lower():
                print(f"  Line {i}: {line.strip()[:200]}")  # 只打印前200字符
        
        # 3. 正则替换所有 "base_score":"[...]" 为 "base_score":"..."
        pattern = r'"base_score"\s*:\s*"\[([^\]]+)\]"'
        matches = re.findall(pattern, content)
        
        if matches:
            print(f"[INFO] Cleaned base_score from '[{matches[0]}]' to '{matches[0]}'")
            clean_content = re.sub(pattern, r'"base_score":"\1"', content)
            
            # 4. 写回文件
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(clean_content)
        else:
            print("[WARN] No bracketed base_score found in JSON. Using original content.")
            clean_content = content
        
        # 5. 重新加载
        clean_booster = xgb.Booster()
        clean_booster.load_model(tmp_path)
        
        return clean_booster
        
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def train_and_shap(df, out_dir, feat_cols, targets):
    """XGBoost 训练 + SHAP 分析"""
    os.makedirs(out_dir, exist_ok=True)
    df.columns = [c.lower() for c in df.columns]

    if DEBUG_MODE and len(df) > XGB_DEBUG_SUBSAMPLE_N:
        print(f"\n>>> DEBUG MODE: Subsampling to {XGB_DEBUG_SUBSAMPLE_N} rows")
        df = df.sample(n=XGB_DEBUG_SUBSAMPLE_N, random_state=42)

    print("\n[INFO] Feature columns for training:", feat_cols)
    print(f"[INFO] Training targets: {targets}")

    shap_results = {}
    validation_reports = {}
    for target in targets:
        print(f"\n{'='*20} TRAINING TARGET: {target} {'='*20}")
        
        d = df.dropna(subset=[target]).copy()
        if d.empty:
            print(f"[WARNING] No valid data for target '{target}'. Skipping.")
            continue
        if isinstance(feat_cols, dict):
            feat_cols_target = feat_cols.get(target, [])
        else:
            feat_cols_target = feat_cols
        if not feat_cols_target:
            print(f"[WARNING] No features selected for target '{target}'. Skipping.")
            continue

        X = d[feat_cols_target].copy()
        X.fillna(X.median(numeric_only=True), inplace=True)
        y = d[target].values

        print(f"[INFO] Training samples: {len(X)}, Features: {len(feat_cols_target)}")

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X.values)

        # ========== 5折交叉验证评估 ==========
        print(f"[CV] Performing 5-fold cross-validation for {target}...")
        n_splits = 5
        if len(X) < 50:
            n_splits = min(3, len(X) // 10)  # 样本少时减少折数
            print(f"[CV] Adjusted to {n_splits}-fold CV due to small sample size")
        
        try:
            y_true_cv, y_pred_cv, cv_r2_scores = perform_cross_validation(
                Xs, y, n_splits=n_splits, random_state=42
            )
            
            # 计算整体性能指标
            rmse_cv = np.sqrt(mean_squared_error(y_true_cv, y_pred_cv))
            mae_cv = mean_absolute_error(y_true_cv, y_pred_cv)
            r2_cv = r2_score(y_true_cv, y_pred_cv)
            
            # 计算交叉验证R²的均值和标准差
            cv_r2_mean = np.mean(cv_r2_scores)
            cv_r2_std = np.std(cv_r2_scores)
            
            print(f"[CV] Results for {target}:")
            print(f"  RMSE: {rmse_cv:.4f}")
            print(f"  MAE:  {mae_cv:.4f}")
            print(f"  R²:   {r2_cv:.4f}")
            print(f"  CV R² (mean ± std): {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
            print(f"  CV R² per fold: {[f'{s:.4f}' for s in cv_r2_scores]}")
            
            # 诊断信息
            if r2_cv < 0:
                print(f"  [WARN] ⚠️ Negative R² ({r2_cv:.4f}) indicates model performs WORSE than mean predictor!")
                print(f"  [WARN] Possible causes:")
                print(f"    - Weak feature-target relationship")
                print(f"    - Overfitting or underfitting")
                print(f"    - Data quality issues (outliers, noise, missing patterns)")
                print(f"    - Model hyperparameters need tuning")
            elif r2_cv < 0.1:
                print(f"  [INFO] Low R² ({r2_cv:.4f}) indicates weak predictive power")
            elif r2_cv < 0.3:
                print(f"  [INFO] Moderate R² ({r2_cv:.4f}) - model has some predictive power")
            else:
                print(f"  [INFO] ✓ Good R² ({r2_cv:.4f}) - model has reasonable predictive power")
            
            # 检查CV R²的标准差是否过大（说明模型不稳定）
            if cv_r2_std > abs(cv_r2_mean) * 0.5 and abs(cv_r2_mean) > 0.01:
                print(f"  [WARN] ⚠️ High CV R² std ({cv_r2_std:.4f}) indicates model instability across folds")
            
            # 保存验证结果
            validation_reports[target] = {
                'n_samples': len(y),
                'n_splits': n_splits,
                'rmse': rmse_cv,
                'mae': mae_cv,
                'r2': r2_cv,
                'cv_r2_mean': cv_r2_mean,
                'cv_r2_std': cv_r2_std,
                'cv_r2_scores': cv_r2_scores,
                'y_true': y_true_cv.tolist(),
                'y_pred': y_pred_cv.tolist()
            }
            
        except Exception as exc:
            print(f"[ERROR] Cross-validation failed for {target}: {exc}")
            traceback.print_exc()
            validation_reports[target] = None

        # ========== 全量训练用于SHAP分析 ==========
        model = xgb.XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
            random_state=42, n_jobs=-1
        )
        model.fit(Xs, y)
        print(f"[INFO] Trained on full dataset for SHAP analysis.")

        try:
            # 直接用 model 初始化 TreeExplainer
            # 由于 base_score 已经在训练前设为固定值 0.5，
            # 不会再出现 "[xxx]" 格式的字符串，SHAP 可以正常解析
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(Xs)
            
            if isinstance(shap_values, list):
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values

            mean_abs_shap = pd.DataFrame({
                'feature': feat_cols_target,
                'mean_abs_shap': np.abs(shap_vals).mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)
            shap_results[target] = mean_abs_shap

            shap_csv = os.path.join(out_dir, f'shap_values_{target}.csv')
            mean_abs_shap.to_csv(shap_csv, index=False)
            print(f"[SAVED] SHAP values: {shap_csv}")

            # 注释掉其他绘图部分
            # print(f"[INFO] Generating SHAP beeswarm for {target} ...")
            # plt.figure(figsize=(10, 8))
            # shap.summary_plot(shap_vals, features=X, show=False)
            # plt.tight_layout()
            # plt.savefig(os.path.join(out_dir, f'shap_beeswarm_{target}.png'), 
            #                          dpi=300, bbox_inches='tight')
            # plt.close()

            # depend_dir = os.path.join(out_dir, f'dependence_plots_{target}')
            # os.makedirs(depend_dir, exist_ok=True)
            # for feature_name in mean_abs_shap['feature'].head(15):
            #     ...

        except Exception as exc:
            print(f"[ERROR] SHAP calculation failed for {target}: {exc}")
            traceback.print_exc()

    return shap_results, validation_reports

# ==================== 交叉验证评估函数 ====================
def perform_cross_validation(X, y, n_splits=5, random_state=42):
    """
    执行K折交叉验证，返回每折的预测值和真实值
    
    参数:
        X: 特征矩阵（已标准化）
        y: 目标变量
        n_splits: 折数（默认5折）
        random_state: 随机种子
    
    返回:
        y_true_all: 所有真实值
        y_pred_all: 所有预测值
        cv_scores: 每折的R²分数
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    y_true_all = []
    y_pred_all = []
    cv_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 训练模型
        model = xgb.XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_val)
        
        # 计算R²
        r2 = r2_score(y_val, y_pred)
        cv_scores.append(r2)
        
        # 收集所有预测值和真实值
        y_true_all.extend(y_val)
        y_pred_all.extend(y_pred)
    
    return np.array(y_true_all), np.array(y_pred_all), cv_scores

# ==================== 按干旱分类计算SHAP ====================
def train_and_shap_by_drought_class(df, out_dir, feat_cols, targets, year_tag):
    """
    按四种干旱-年份组合分别计算SHAP值：
    1. 干旱年旱地 (2022年干旱等级2,3,4)
    2. 干旱年湿润地 (2022年干旱等级0,1)
    3. 非干旱年干旱地 (2023年干旱等级2,3,4)
    4. 非干旱年湿润地 (2023年干旱等级0,1)
    
    同时进行交叉验证评估模型性能
    """
    os.makedirs(out_dir, exist_ok=True)
    df.columns = [c.lower() for c in df.columns]
    
    # 获取干旱等级定义
    drought_classes = DROUGHT_CLASSES.get(year_tag, {"drought": {2, 3, 4}, "wet": {0, 1}})
    drought_set = drought_classes["drought"]
    wet_set = drought_classes["wet"]
    
    # 检查是否有drought_class列
    if 'drought_class' not in df.columns:
        print(f"[WARN] No 'drought_class' column found. Using all data without classification.")
        return train_and_shap(df, out_dir, feat_cols, targets)
    
    shap_results_by_class = {}
    validation_reports = {}
    
    # 定义四种组合
    class_groups = {
        'drought_year_drought': {'year': '2022_Drought', 'class_set': drought_set, 'label': 'Drought Year - Dry Land'},
        'drought_year_wet': {'year': '2022_Drought', 'class_set': wet_set, 'label': 'Drought Year - Wet Land'},
        'wet_year_drought': {'year': '2023_Wet', 'class_set': drought_set, 'label': 'Wet Year - Dry Land'},
        'wet_year_wet': {'year': '2023_Wet', 'class_set': wet_set, 'label': 'Wet Year - Wet Land'},
    }
    
    for group_key, group_info in class_groups.items():
        # 只处理匹配的年份
        if year_tag != group_info['year']:
            continue
            
        print(f"\n{'='*20} Processing: {group_info['label']} {'='*20}")
        
        # 筛选数据
        df_group = df[df['drought_class'].isin(group_info['class_set'])].copy()
        
        if df_group.empty:
            print(f"[WARN] No data for {group_info['label']}. Skipping.")
            continue
        
        print(f"[INFO] Samples for {group_info['label']}: {len(df_group)}")
        
        # 对每个目标变量训练和计算SHAP
        shap_results_group = {}
        validation_group = {}
        
        for target in targets:
            print(f"\n[INFO] Training {target} for {group_info['label']}...")
            
            d = df_group.dropna(subset=[target]).copy()
            if d.empty:
                print(f"[WARNING] No valid data for target '{target}'. Skipping.")
                continue

            # 确保只使用feat_cols中存在的列，并且过滤掉不在ENV_CHOICES中的因子
            valid_feat_cols = [f for f in feat_cols if f in d.columns]
            if len(valid_feat_cols) == 0:
                print(f"[WARNING] No valid features found for {target}. Skipping.")
                continue
            
            X = d[valid_feat_cols].copy()
            X.fillna(X.median(numeric_only=True), inplace=True)
            y = d[target].values

            if len(X) < 10:
                print(f"[WARNING] Too few samples ({len(X)}) for {target}. Skipping.")
                continue

            # ========== 数据诊断 ==========
            # 检查目标变量的分布
            y_valid = y[np.isfinite(y)]
            if len(y_valid) == 0:
                print(f"[WARNING] No valid target values for {target}. Skipping.")
                continue
            
            print(f"[DIAG] Target variable statistics for {target}:")
            print(f"  Count: {len(y_valid)}")
            print(f"  Mean:  {np.mean(y_valid):.4f}")
            print(f"  Std:   {np.std(y_valid):.4f}")
            print(f"  Min:   {np.min(y_valid):.4f}")
            print(f"  Max:   {np.max(y_valid):.4f}")
            print(f"  Median: {np.median(y_valid):.4f}")
            
            # 检查是否有异常值（使用IQR方法）
            q1, q3 = np.percentile(y_valid, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            outliers = np.sum((y_valid < lower_bound) | (y_valid > upper_bound))
            print(f"  Outliers (3*IQR): {outliers} ({100*outliers/len(y_valid):.2f}%)")
            
            # 检查特征和目标变量的相关性
            if len(valid_feat_cols) > 0:
                from scipy.stats import spearmanr
                max_corr = -1
                max_corr_feat = None
                for feat in valid_feat_cols[:5]:  # 只检查前5个特征
                    if feat in X.columns:
                        feat_vals = X[feat].values
                        valid_mask = np.isfinite(feat_vals) & np.isfinite(y)
                        if np.sum(valid_mask) > 10:
                            try:
                                corr, _ = spearmanr(feat_vals[valid_mask], y[valid_mask])
                                if not np.isnan(corr) and abs(corr) > abs(max_corr):
                                    max_corr = corr
                                    max_corr_feat = feat
                            except:
                                pass
                if max_corr_feat:
                    print(f"  Max feature correlation: {max_corr_feat} = {max_corr:.4f}")

            scaler = StandardScaler()
            Xs = scaler.fit_transform(X.values)

            # ========== 交叉验证评估 ==========
            print(f"[CV] Performing 5-fold cross-validation for {target}...")
            n_splits = 5
            if len(X) < 50:
                n_splits = min(3, len(X) // 10)  # 样本少时减少折数
                print(f"[CV] Adjusted to {n_splits}-fold CV due to small sample size")
            
            try:
                y_true_cv, y_pred_cv, cv_r2_scores = perform_cross_validation(
                    Xs, y, n_splits=n_splits, random_state=42
                )
                
                # 计算整体性能指标
                rmse_cv = np.sqrt(mean_squared_error(y_true_cv, y_pred_cv))
                mae_cv = mean_absolute_error(y_true_cv, y_pred_cv)
                r2_cv = r2_score(y_true_cv, y_pred_cv)
                
                # 计算交叉验证R²的均值和标准差
                cv_r2_mean = np.mean(cv_r2_scores)
                cv_r2_std = np.std(cv_r2_scores)
                
                print(f"[CV] Results for {target} ({group_info['label']}):")
                print(f"  RMSE: {rmse_cv:.4f}")
                print(f"  MAE:  {mae_cv:.4f}")
                print(f"  R²:   {r2_cv:.4f}")
                print(f"  CV R² (mean ± std): {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
                print(f"  CV R² per fold: {[f'{s:.4f}' for s in cv_r2_scores]}")
                
                # 诊断：如果R²为负，说明模型比简单均值预测还差
                if r2_cv < 0:
                    print(f"  [WARN] ⚠️ Negative R² ({r2_cv:.4f}) indicates model performs WORSE than mean predictor!")
                    print(f"  [WARN] Possible causes:")
                    print(f"    - Weak feature-target relationship")
                    print(f"    - Overfitting or underfitting")
                    print(f"    - Data quality issues (outliers, noise, missing patterns)")
                    print(f"    - Model hyperparameters need tuning")
                elif r2_cv < 0.1:
                    print(f"  [INFO] Low R² ({r2_cv:.4f}) indicates weak predictive power")
                elif r2_cv < 0.3:
                    print(f"  [INFO] Moderate R² ({r2_cv:.4f}) - model has some predictive power")
                else:
                    print(f"  [INFO] ✓ Good R² ({r2_cv:.4f}) - model has reasonable predictive power")
                
                # 检查CV R²的标准差是否过大（说明模型不稳定）
                if cv_r2_std > abs(cv_r2_mean) * 0.5 and abs(cv_r2_mean) > 0.01:
                    print(f"  [WARN] ⚠️ High CV R² std ({cv_r2_std:.4f}) indicates model instability across folds")
                
                # 诊断：如果R²为负，说明模型比简单均值预测还差
                if r2_cv < 0:
                    print(f"  [WARN] Negative R² indicates model performs worse than mean predictor!")
                    print(f"  [WARN] This may indicate:")
                    print(f"    - Poor feature-target relationship")
                    print(f"    - Overfitting or underfitting")
                    print(f"    - Data quality issues (outliers, noise)")
                    print(f"    - Model hyperparameters need tuning")
                
                # 保存验证结果
                validation_group[target] = {
                    'n_samples': len(y),
                    'n_splits': n_splits,
                    'rmse': rmse_cv,
                    'mae': mae_cv,
                    'r2': r2_cv,
                    'cv_r2_mean': cv_r2_mean,
                    'cv_r2_std': cv_r2_std,
                    'cv_r2_scores': cv_r2_scores,
                    'y_true': y_true_cv.tolist(),
                    'y_pred': y_pred_cv.tolist()
                }
            except Exception as exc:
                print(f"[ERROR] Cross-validation failed for {target} in {group_info['label']}: {exc}")
                traceback.print_exc()
                validation_group[target] = None

            # ========== 全量训练用于SHAP分析 ==========
            model = xgb.XGBRegressor(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                random_state=42, n_jobs=-1
            )
            model.fit(Xs, y)

            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(Xs)
                
                if isinstance(shap_values, list):
                    shap_vals = shap_values[0]
                else:
                    shap_vals = shap_values

                mean_abs_shap = pd.DataFrame({
                    'feature': valid_feat_cols,
                    'mean_abs_shap': np.abs(shap_vals).mean(axis=0)
                }).sort_values('mean_abs_shap', ascending=False)
                
                shap_results_group[target] = mean_abs_shap
                
            except Exception as exc:
                print(f"[ERROR] SHAP calculation failed for {target} in {group_info['label']}: {exc}")
                traceback.print_exc()
        
        shap_results_by_class[group_key] = shap_results_group
        validation_reports[group_key] = validation_group
    
    return shap_results_by_class, validation_reports

# ==================== 环境因子分类（三个调节方面）====================
def get_feature_regulation_category(feature_name):
    """
    根据三个调节方面对环境因子进行分类：
    1. 气孔调节（Stomatal Regulation）：VPD, LST, Temp, SW↓等与蒸腾和气孔开闭相关的
    2. 含水量调节（Water Content Regulation）：SM, P (precip)等与水分相关的
    3. 碳调节（Carbon Regulation）：PAR, AOD等与光合作用和光照相关的
    4. 其他（Other）：地形、土壤等静态因子
    """
    feature_lower = feature_name.lower()
    
    # 气孔调节（Stomatal Regulation）- 红色
    stomatal_features = ['vpd', 'lst', 'temp']  # 移除swdown和ssrd（已移除SW↓因子）
    # 含水量调节（Water Content Regulation）- 蓝色
    water_content_features = ['sm', 'precip', 'tp', 'p_']
    # 碳调节（Carbon Regulation）- 绿色（PAR直接驱动光合作用，AOD影响光照）
    carbon_features = ['par', 'aod']  # 移除swdown和ssrd，它们归类为气孔调节
    # 其他（地形/土壤）- 橙色/金色
    other_features = ['elev', 'aspect', 'clay', 'sand', 'soc', 'ph']  # 移除slope
    
    if any(f in feature_lower for f in stomatal_features):
        return 'Stomatal Regulation'
    elif any(f in feature_lower for f in water_content_features):
        return 'Water Content Regulation'
    elif any(f in feature_lower for f in carbon_features):
        return 'Carbon Regulation'
    elif any(f in feature_lower for f in other_features):
        return 'Other'
    else:
        return 'Other'

# ==================== 环境因子名称映射（示例图格式）====================
def map_feature_name_to_display(feature_name):
    """
    将环境因子名称映射为示例图格式
    示例：AOD_7d, LAI, Clay, Sand, Elev, Aspect, LST_3d, VPD_3d, P_7d, SM_7d
    """
    feature_lower = feature_name.lower()
    
    # 映射规则
    name_mapping = {
        # AOD
        'aod_mean_last_7d': 'AOD_7d',
        'aod_7d': 'AOD_7d',
        # 土壤
        'clay': 'Clay',
        'sand': 'Sand',
        # 地形
        'elevation': 'Elev',
        'elev': 'Elev',
        'aspect': 'Aspect',
        # 温度/干旱
        'lst_mean_last_3d': 'LST_3d',
        'lst_3d': 'LST_3d',
        'lst_1d': 'LST_1d',
        # 辐射（注意SW↓_1d有向下箭头，使用Unicode字符）
        'swdown_hmean_9_16': 'SW↓_1d',
        'swdown_j_1d': 'SW↓_1d',
        'swdown_1d': 'SW↓_1d',
        'swdown_j_7d': 'SW_7d',
        'swdown_j_30d': 'SW_30d',
        'ssrd_hmean_9_16': 'SW↓_1d',  # 兼容其他名称
        'par_hmean_9_16': 'PAR_inst',
        'par_inst_umol_s': 'PAR_inst',
        'par_cum_to_obs_mol': 'PAR_cum_obs',
        'par_cum_day_mol': 'PAR_cum_day',
        'par_1d': 'PAR_1d',
        # VPD
        'vpd_mean_last_3d': 'VPD_3d',
        'vpd_3d': 'VPD_3d',
        'vpd_hmean_9_16': 'VPD_hmean',
        # 温度（Temp）
        'temp_1d': 'Temp_1d',
        'temp_3d': 'Temp_3d',
        'temp_hmean_9_16': 'Temp_hmean',
        'temp_hmean': 'Temp_hmean',
        # 降水
        'precip_sum_7d': 'P_7d',
        'precip_sum_last_30d': 'P_30d',
        'tp_7d': 'P_7d',
        'tp_3d': 'P_3d',
        'tp_1d': 'P_1d',
        'tp_30d': 'P_30d',
        # 土壤水分
        'sm_mean_last_7d': 'SM_7d',
        'sm_7d': 'SM_7d',
        # 土壤
        'soc': 'SOC',
        'ph': 'pH',
    }
    
    # 直接匹配
    if feature_lower in name_mapping:
        return name_mapping[feature_lower]
    
    # 部分匹配
    for key, display_name in name_mapping.items():
        if key in feature_lower:
            return display_name
    
    # 默认返回原名称（首字母大写）
    return feature_name.replace('_', ' ').title().replace(' ', '_')

# ==================== 堆叠柱形图（四种干旱-年份组合）====================
def plot_stacked_shap_bars(shap_results_by_class, output_png_path, target, title_tag="", 
                           subplot_label="", target_display_name="", xlim_max=None):
    """
    绘制堆叠柱形图，显示四种干旱-年份组合的SHAP值
    颜色方案：
    - 干旱年旱地和非干旱年湿润地：较深颜色（#F28E2B橙色和#4E79A7蓝色）
    - 干旱年湿润地和非干旱年干旱地：较浅颜色
    在不同地类型的条形中间放上对应环境因子重要性排序的序号
    
    参数:
    - subplot_label: 子图编号，如 "a", "b", "c"
    - target_display_name: 指标显示名称，如 "T_peak", "C_shift", "MDI"
    - xlim_max: 统一的x轴最大值
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.weight"] = "bold"
    
    # 定义四种组合的标签和颜色
    group_info = {
        'drought_year_drought': {'label': 'Drought Year - Dry Land', 'color': '#F28E2B', 'alpha': 1.0},  # 深橙色
        'drought_year_wet': {'label': 'Drought Year - Wet Land', 'color': '#F28E2B', 'alpha': 0.5},      # 浅橙色
        'wet_year_drought': {'label': 'Wet Year - Dry Land', 'color': '#4E79A7', 'alpha': 0.5},         # 浅蓝色
        'wet_year_wet': {'label': 'Wet Year - Wet Land', 'color': '#4E79A7', 'alpha': 1.0},             # 深蓝色
    }
    
    # 收集所有特征
    all_features = set()
    for group_key, shap_results in shap_results_by_class.items():
        if target in shap_results:
            all_features.update(shap_results[target]['feature'].tolist())
    
    if not all_features:
        print(f"[WARN] No SHAP data found for target {target}. Skipping plot.")
        return
    
    all_features = sorted(list(all_features))
    
    # 构建数据矩阵和排序
    data_matrix = {}
    rank_matrix = {}  # 存储每个组合中每个特征的排序
    
    for group_key in group_info.keys():
        if group_key in shap_results_by_class and target in shap_results_by_class[group_key]:
            shap_df = shap_results_by_class[group_key][target].copy()
            shap_df = shap_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
            shap_df['rank'] = range(1, len(shap_df) + 1)  # 从1开始排序
            
            shap_dict = dict(zip(shap_df['feature'], shap_df['mean_abs_shap']))
            rank_dict = dict(zip(shap_df['feature'], shap_df['rank']))
            
            data_matrix[group_key] = [shap_dict.get(f, 0) for f in all_features]
            rank_matrix[group_key] = [rank_dict.get(f, None) for f in all_features]
        else:
            data_matrix[group_key] = [0] * len(all_features)
            rank_matrix[group_key] = [None] * len(all_features)
    
    # 转换为DataFrame便于绘图
    plot_df = pd.DataFrame({
        'feature': all_features,
        'drought_year_drought': data_matrix['drought_year_drought'],
        'drought_year_wet': data_matrix['drought_year_wet'],
        'wet_year_drought': data_matrix['wet_year_drought'],
        'wet_year_wet': data_matrix['wet_year_wet'],
    })
    
    # 添加排序信息
    plot_df['rank_drought_year_drought'] = rank_matrix['drought_year_drought']
    plot_df['rank_drought_year_wet'] = rank_matrix['drought_year_wet']
    plot_df['rank_wet_year_drought'] = rank_matrix['wet_year_drought']
    plot_df['rank_wet_year_wet'] = rank_matrix['wet_year_wet']
    
    # 计算总重要性用于排序
    plot_df['total'] = (plot_df['drought_year_drought'] + plot_df['drought_year_wet'] + 
                        plot_df['wet_year_drought'] + plot_df['wet_year_wet'])
    plot_df = plot_df.sort_values('total', ascending=True).reset_index(drop=True)
    
    # 映射特征名称到显示格式
    plot_df['feature_display'] = plot_df['feature'].apply(map_feature_name_to_display)
    
    # 添加调节类别和颜色
    plot_df['regulation_category'] = plot_df['feature'].apply(get_feature_regulation_category)
    
    # 定义调节类别的颜色（根据例图）
    # 绿色：AOD_7d（碳调节）
    # 橙色/金色：Clay, Sand, Elev, Aspect（其他/地形土壤）
    # 红色：LST_3d, SW↓_1d, VPD_3d（气孔调节）
    # 蓝色：P_7d, SM_7d（含水量调节）
    # 绿色：AOD_7d, PAR（碳调节）
    category_colors = {
        'Stomatal Regulation': '#d32f2f',      # 红色（LST_3d, VPD_3d, SW↓_1d, Temp）
        'Water Content Regulation': '#1976d2', # 蓝色（P_7d, SM_7d）
        'Carbon Regulation': '#388e3c',        # 绿色（AOD_7d, PAR相关）
        'Other': '#f57c00',                    # 橙色/金色（Clay, Sand, Elev, Aspect）
    }
    
    # 绘图
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.unicode_minus"] = False
    # 调整y轴位置，减小柱子之间的间距（间距系数从1.0改为0.7）
    # 相应调整图表高度：高度系数也要乘以0.7
    spacing_factor = 1
    y_pos = np.arange(len(plot_df)) * spacing_factor
    bar_height = 0.3  # 柱形再细一点（从0.5改为0.4）
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(all_features) * 0.4 * spacing_factor)), dpi=300)
    
    # 添加类别背景色（浅色）
    category_bg_colors = {
        'Stomatal Regulation': '#ffebee',      # 浅红色
        'Water Content Regulation': '#e3f2fd', # 浅蓝色
        'Carbon Regulation': '#e8f5e9',        # 浅绿色
        'Other': '#fff3e0',                    # 浅橙色
    }
    
    # 为每个类别添加背景矩形
    current_category = None
    start_idx = 0
    for i, row in plot_df.iterrows():
        category = row['regulation_category']
        if category != current_category:
            if current_category is not None:
                # 绘制前一个类别的背景
                bg_color = category_bg_colors.get(current_category, 'white')
                ax.axhspan(start_idx - 0.5, i - 0.5, color=bg_color, alpha=0.3, zorder=0)
            current_category = category
            start_idx = i
    # 绘制最后一个类别的背景
    if current_category is not None:
        bg_color = category_bg_colors.get(current_category, 'white')
        ax.axhspan(start_idx - 0.5, len(plot_df) - 0.5, color=bg_color, alpha=0.3, zorder=0)
    
    # 堆叠柱形图：从下到上的顺序
    bottom1 = np.zeros(len(plot_df))
    bottom2 = bottom1 + plot_df['drought_year_drought'].values
    bottom3 = bottom2 + plot_df['drought_year_wet'].values
    bottom4 = bottom3 + plot_df['wet_year_drought'].values
    
    ax.barh(y_pos, plot_df['drought_year_drought'].values, bar_height,
            left=bottom1, label=group_info['drought_year_drought']['label'],
            color=group_info['drought_year_drought']['color'],
            alpha=group_info['drought_year_drought']['alpha'])
    
    ax.barh(y_pos, plot_df['drought_year_wet'].values, bar_height,
            left=bottom2, label=group_info['drought_year_wet']['label'],
            color=group_info['drought_year_wet']['color'],
            alpha=group_info['drought_year_wet']['alpha'])
    
    ax.barh(y_pos, plot_df['wet_year_drought'].values, bar_height,
            left=bottom3, label=group_info['wet_year_drought']['label'],
            color=group_info['wet_year_drought']['color'],
            alpha=group_info['wet_year_drought']['alpha'])
    
    ax.barh(y_pos, plot_df['wet_year_wet'].values, bar_height,
            left=bottom4, label=group_info['wet_year_wet']['label'],
            color=group_info['wet_year_wet']['color'],
            alpha=group_info['wet_year_wet']['alpha'])
    
    # 在每个条形中间添加排序序号
    for i, row in plot_df.iterrows():
        y = y_pos[i]
        
        # 干旱年旱地（最底层）
        if row['rank_drought_year_drought'] is not None and row['drought_year_drought'] > 0:
            x_pos = bottom1[i] + row['drought_year_drought'] / 2
            ax.text(x_pos, y, str(int(row['rank_drought_year_drought'])), 
                   ha='center', va='center', fontsize=9, color='black', 
                   fontweight='bold', fontfamily='Times New Roman')
        
        # 干旱年湿润地
        if row['rank_drought_year_wet'] is not None and row['drought_year_wet'] > 0:
            x_pos = bottom2[i] + row['drought_year_wet'] / 2
            ax.text(x_pos, y, str(int(row['rank_drought_year_wet'])), 
                   ha='center', va='center', fontsize=9, color='black', 
                   fontweight='bold', fontfamily='Times New Roman')
        
        # 非干旱年干旱地
        if row['rank_wet_year_drought'] is not None and row['wet_year_drought'] > 0:
            x_pos = bottom3[i] + row['wet_year_drought'] / 2
            ax.text(x_pos, y, str(int(row['rank_wet_year_drought'])), 
                   ha='center', va='center', fontsize=9, color='black', 
                   fontweight='bold', fontfamily='Times New Roman')
        
        # 非干旱年湿润地（最上层）
        if row['rank_wet_year_wet'] is not None and row['wet_year_wet'] > 0:
            x_pos = bottom4[i] + row['wet_year_wet'] / 2
            ax.text(x_pos, y, str(int(row['rank_wet_year_wet'])), 
                   ha='center', va='center', fontsize=9, color='black', 
                   fontweight='bold', fontfamily='Times New Roman')
    
    # 设置y轴标签，根据调节类别着色
    ax.set_yticks(y_pos)
    yticklabels_list = []
    for i, row in plot_df.iterrows():
        category = row['regulation_category']
        color = category_colors.get(category, 'black')
        yticklabels_list.append(row['feature_display'])
    
    # 设置标签文本
    ax.set_yticklabels(yticklabels_list, fontsize=10, fontfamily='Times New Roman', fontweight='bold')
    
    # 为每个标签设置颜色（SW↓现在已归类为气孔调节，不需要特殊处理）
    for i, (tick, row) in enumerate(zip(ax.get_yticklabels(), plot_df.itertuples())):
        category = row.regulation_category
        color = category_colors.get(category, 'black')
        tick.set_color(color)
    ax.set_xlabel('SHAP Value', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    
    # 添加子图编号和指标名（左上角）
    if subplot_label and target_display_name:
        ax.text(0.02, 0.98, f'({subplot_label}) {target_display_name}', 
               transform=ax.transAxes, fontsize=14, fontweight='bold', 
               fontfamily='Times New Roman', va='top', ha='left')
    
    # 半开放图框：只显示左和下的轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    # 注释掉图例
    # ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # 设置x轴格式：保留两位小数，并设置字体
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    ax.tick_params(axis='x', labelsize=10, which='major')
    for label in ax.get_xticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontweight('bold')
    
    # 每张图各自适应x轴范围（移除统一x轴范围限制）
    # 自动调整x轴范围以适应数据
    
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Stacked SHAP bar plot saved: {output_png_path}")

# ==================== 按类别汇总的百分比贡献图 ====================
def plot_category_percentage_contribution(all_shap_by_class, output_png_path):
    """
    绘制按环境因子类别汇总的百分比贡献图
    显示2022和2023两个年份，每个年份的柱形包含四个数据（四种干旱-年份组合）
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.unicode_minus"] = False
    
    # 定义类别和颜色
    category_colors = {
        'Stomatal Regulation': '#d32f2f',      # 红色
        'Water Content Regulation': '#1976d2', # 蓝色
        'Carbon Regulation': '#388e3c',        # 绿色
        'Other': '#f57c00',                    # 橙色/金色
    }
    
    # 定义四种组合的标签和颜色（与plot_stacked_shap_bars一致）
    group_info = {
        'drought_year_drought': {'label': 'Drought Year - Dry Land', 'color': '#F28E2B', 'alpha': 1.0},  # 深橙色
        'drought_year_wet': {'label': 'Drought Year - Wet Land', 'color': '#F28E2B', 'alpha': 0.5},      # 浅橙色
        'wet_year_drought': {'label': 'Wet Year - Dry Land', 'color': '#4E79A7', 'alpha': 0.5},         # 浅蓝色
        'wet_year_wet': {'label': 'Wet Year - Wet Land', 'color': '#4E79A7', 'alpha': 1.0},             # 深蓝色
    }
    
    # 类别顺序（简写，去掉Regulation）
    categories = ['Stomatal Regulation', 'Water Content Regulation', 'Carbon Regulation', 'Other']
    category_labels = {
        'Stomatal Regulation': 'Stomatal',
        'Water Content Regulation': 'Water Content',
        'Carbon Regulation': 'Carbon',
        'Other': 'Other'
    }
    
    # 为每个年份和每个目标变量计算类别贡献
    years = ['2022_Drought', '2023_Wet']
    targets = ['target_centroid_shift', 'target_t_peak', 'target_mdi']
    
    # 计算所有目标变量的平均贡献
    category_data = {}
    for year in years:
        category_data[year] = {}
        for category in categories:
            category_data[year][category] = {
                'drought_year_drought': 0,
                'drought_year_wet': 0,
                'wet_year_drought': 0,
                'wet_year_wet': 0
            }
        
        # 遍历所有目标变量
        for target in targets:
            if year in all_shap_by_class:
                for group_key in group_info.keys():
                    # 确定该组合对应的年份标签
                    if group_key in ['drought_year_drought', 'drought_year_wet']:
                        actual_year = '2022_Drought'
                    else:
                        actual_year = '2023_Wet'
                    
                    if actual_year == year and group_key in all_shap_by_class[actual_year]:
                        if target in all_shap_by_class[actual_year][group_key]:
                            shap_df = all_shap_by_class[actual_year][group_key][target].copy()
                            
                            # 按类别汇总
                            for _, row in shap_df.iterrows():
                                feature = row['feature']
                                shap_value = row['mean_abs_shap']
                                category = get_feature_regulation_category(feature)
                                
                                if category in category_data[year]:
                                    category_data[year][category][group_key] += shap_value
        
        # 计算百分比：每个类别在该年份所有类别总和中的占比
        # 先计算每个组合的总SHAP值（所有类别之和）
        total_by_group = {}
        for group_key in group_info.keys():
            total_by_group[group_key] = sum(
                category_data[year][cat][group_key] for cat in categories
            )
        
        # 计算每个类别在每个组合中的百分比
        for category in categories:
            for group_key in group_info.keys():
                if total_by_group[group_key] > 0:
                    category_data[year][category][group_key] = (
                        category_data[year][category][group_key] / total_by_group[group_key] * 100
                    )
                else:
                    category_data[year][category][group_key] = 0
    
    # 创建图表：合并为一张图（尺寸与abc图一致）
    # d图有4个类别，使用与abc图相同的高度计算方式
    num_categories = len(categories)
    fig, ax = plt.subplots(figsize=(10, max(8, num_categories * 0.4)), dpi=300)
    
    # 绘制所有四个组合的柱形
    y_pos = np.arange(len(categories))
    bar_width = 0.18
    # 四个柱形的x位置偏移：2022的两个组合（左侧），2023的两个组合（右侧）
    x_offset = [-0.27, -0.09, 0.09, 0.27]
    all_group_keys = ['drought_year_drought', 'drought_year_wet', 'wet_year_drought', 'wet_year_wet']
    
    for i, group_key in enumerate(all_group_keys):
        # 确定该组合对应的年份
        if group_key in ['drought_year_drought', 'drought_year_wet']:
            year = '2022_Drought'
        else:
            year = '2023_Wet'
        
        values = [category_data[year][cat][group_key] for cat in categories]
        color = group_info[group_key]['color']
        alpha = group_info[group_key]['alpha']
        
        ax.barh(y_pos + x_offset[i], values, bar_width, 
                label=group_info[group_key]['label'], 
                color=color, alpha=alpha, edgecolor='black', linewidth=0.5)
    
    # 设置y轴
    ax.set_yticks(y_pos)
    ax.set_yticklabels([category_labels[cat] for cat in categories], fontsize=11, fontweight='bold')
    # 名称平行于y轴（垂直显示，旋转90度）
    # 调整标签与y轴的距离（pad参数，单位是点，默认是4）
    ax.tick_params(axis='y', pad=15)  # 增加距离，可以调整这个值
    for label in ax.get_yticklabels():
        label.set_rotation(90)  # 垂直显示，旋转90度
        label.set_ha('left')    # 左对齐
        label.set_va('center')  # 垂直居中
    
    # 设置y轴范围，确保标签位置与abc图一致
    # 计算y轴范围：考虑柱形位置和高度
    y_min = min(y_pos + x_offset) - bar_width / 2 - 0.5
    y_max = max(y_pos + x_offset) + bar_width / 2 + 0.5
    ax.set_ylim(y_min, y_max)
    
    # 为y轴标签设置颜色
    for i, (tick, cat) in enumerate(zip(ax.get_yticklabels(), categories)):
        tick.set_color(category_colors[cat])
    
    # 设置x轴
    # 添加子图编号d（左上角，位置与abc图一致，使用transAxes确保相对位置一致）
    ax.text(0.02, 0.98, '(d)', 
           transform=ax.transAxes, fontsize=14, fontweight='bold', 
           fontfamily='Times New Roman', va='top', ha='left')
    
    # 半开放图框：只显示左和下的轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    ax.set_xlabel('Percentage Contribution (%)', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    # 计算最大值的范围
    max_val = 0
    for year in ['2022_Drought', '2023_Wet']:
        for group_key in all_group_keys:
            if (group_key in ['drought_year_drought', 'drought_year_wet'] and year == '2022_Drought') or \
               (group_key in ['wet_year_drought', 'wet_year_wet'] and year == '2023_Wet'):
                for cat in categories:
                    max_val = max(max_val, category_data[year][cat][group_key])
    ax.set_xlim(0, max_val * 1.2 if max_val > 0 else 100)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # 设置x轴格式
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    for label in ax.get_xticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Category percentage contribution plot saved: {output_png_path}")

# ==================== SHAP 哑铃图 ====================
def plot_advanced_shap_importance(feat_importance_df, output_png_path, 
                                  title_tag, plot_labels):
    """生成分组哑铃图"""
    feature_groups = {
        "Radiation & Energy": [
            'swdown_j_1d', 'swdown_j_7d', 'swdown_j_30d',
            'par_inst_umol_s', 'par_cum_to_obs_mol', 'par_cum_day_mol'
        ],
        "Temperature & Drought": [
            'lst_mean_last_3d', 'vpd_mean_last_3d'
        ],
        "Water & Precipitation": [
            'sm_mean_last_7d', 'precip_sum_last_30d'
        ],
        "Vegetation & Atmosphere": [
            'lai_current', 'aod_mean_last_7d'
        ],
        "Topography": [
            'elevation', 'slope', 'aspect'
        ],
        "Soil Properties": [
            'sand', 'clay', 'soc', 'ph', 'texture'
        ],
    }
    
    colors = {
        "Radiation & Energy": "#f79e7b",
        "Temperature & Drought": "#e48dbc",
        "Water & Precipitation": "#88c0d0",
        "Vegetation & Atmosphere": "#bfdc84",
        "Topography": "#9babd2",
        "Soil Properties": "#8cbeb2",
        "Other": "#d3d3d3"
    }

    def get_group(feature):
        feature_lower = feature.lower()
        for g, f_list in feature_groups.items():
            if feature_lower in f_list:
                return g
        return "Other"

    feat_importance = feat_importance_df.copy()
    feat_importance.columns = ['Feature', 'Importance', 'Importance2']
    feat_importance["Group"] = feat_importance["Feature"].apply(get_group)

    # 归一化
    if feat_importance["Importance"].sum() > 0:
        feat_importance["Importance"] = (
            feat_importance["Importance"] / feat_importance["Importance"].sum() * 100
        )
    if feat_importance["Importance2"].sum() > 0:
        feat_importance["Importance2"] = (
            feat_importance["Importance2"] / feat_importance["Importance2"].sum() * 100
        )

    feat_importance['mean_importance'] = (
        feat_importance['Importance'] + feat_importance['Importance2']
    ) / 2
    feat_importance = feat_importance.sort_values(
        by=["Group", "mean_importance"], ascending=[True, False]
    )
    feat_importance = feat_importance.reset_index(drop=True)
    feat_importance['idx'] = feat_importance.index

    # 分组统计
    group_imp1 = feat_importance.groupby("Group")["Importance"].sum()
    group_imp2 = feat_importance.groupby("Group")["Importance2"].sum()

    # 绘图
    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[4, 1.2], hspace=0.4)
    ax0 = fig.add_subplot(gs[0, :])
    fig.suptitle(f'SHAP Feature Importance ({title_tag})', 
                 fontsize=20, weight='bold')

    # 主图：哑铃图
    for _, row in feat_importance.iterrows():
        ax0.plot([row["Importance"], row["Importance2"]], 
                 [row["idx"], row["idx"]],
                 color="gray", linewidth=4, alpha=0.7, zorder=1)
    
    ax0.scatter(feat_importance["Importance"], feat_importance['idx'],
               color="#db2d26", s=100, zorder=2, label=plot_labels['label1'])
    ax0.scatter(feat_importance["Importance2"], feat_importance['idx'],
               color="#467db6", s=100, zorder=2, label=plot_labels['label2'])

    ax0.set_yticks(feat_importance['idx'])
    ax0.set_yticklabels(feat_importance['Feature'], fontsize=12)
    ax0.set_xlabel("Ranking percentage of variables (%)", fontsize=14)
    ax0.tick_params(axis='x', labelsize=14)
    ax0.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax0.grid(axis='x', linestyle='--', alpha=0.6)
    ax0.invert_yaxis()

    # 分组背景色
    for g in feat_importance["Group"].unique():
        idx_in_group = feat_importance[feat_importance["Group"]==g]['idx'].values
        if len(idx_in_group) > 0:
            ymin, ymax = idx_in_group.min(), idx_in_group.max()
            ax0.axhspan(ymin-0.5, ymax+0.5, 
                        facecolor=colors.get(g, "#d3d3d3"), 
                        alpha=0.2, zorder=0)

    ax0.legend(fontsize=14, loc='lower right')
    ax0.text(0.02, 0.98, "(a)", transform=ax0.transAxes, 
            fontsize=18, va='top', weight='bold')

    # 分组条形图
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.barh(group_imp1.index, group_imp1.values,
            color=[colors.get(g) for g in group_imp1.index], alpha=0.7)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.set_xlabel(f"{plot_labels['label1']} (%)", fontsize=12)
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax1.text(0.05, 0.95, "(b)", transform=ax1.transAxes, 
            fontsize=18, va='top', weight='bold')

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.barh(group_imp2.index, group_imp2.values,
            color=[colors.get(g) for g in group_imp2.index], alpha=0.7)
    ax2.set_yticklabels([])
    ax2.tick_params(axis='x', labelsize=12)
    ax2.set_xlabel(f"{plot_labels['label2']} (%)", fontsize=12)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax2.text(0.05, 0.95, "(c)", transform=ax2.transAxes, 
            fontsize=18, va='top', weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Advanced SHAP plot saved: {output_png_path}")

# ==================== 主流程 ====================
def main():
    """
    主流程：
    - 仍然分别处理两个年份：2022_Drought（干旱年）、2023_Wet（非干旱年）
    - 为每个年份单独构建像元级特征表并保存
    - 不再将不同年份的数据合并，而是为每个年份分别训练一个 XGBoost+SHAP 模型
    - SAMPLE_PIXELS=0 表示“像元全部采样”，尽量使用全部有效像元以获得更稳健的模型
    """
    if DEBUG_MODE:
        print("=" * 50 + "\n=== RUNNING IN DEBUG MODE ===\n" + "=" * 50)
    
    print(f"\n[CONFIG] Target mode: {TARGET_MODE}")
    print(f"[CONFIG] Training targets: {TRAINING_TARGETS}")

    # 辅助函数：在当前 DataFrame 中按优先级选择环境因子列
    def choose_env_columns(df):
        """从 ENV_CHOICES 中按优先级选择存在的列，返回映射 name -> column_name（小写列名）。支持列名变体。"""
        selected = {}
        cols = set([c.lower() for c in df.columns])  # 转换为小写集合
        
        # 扩展的列名变体映射（用于处理 GEE 导出的不同命名）
        env_variants = {
            'LST': ['lst_3d', 'lst_mean_last_3d', 'lst_mean_3d', 'lst'],
            'SM': ['sm_mean_last_7d', 'sm_7d', 'sm_mean_7d', 'sm_mean_last_30d', 'sm'],
            'AOD': ['aod_mean_last_7d', 'aod_7d', 'aod_mean_7d', 'aod'],
            'Elev': ['elevation', 'elev', 'dem', 'elevation_m', 'altitude'],
            'SOC': ['soc', 'soil_organic_carbon', 'organic_carbon', 'soil_carbon'],
        }
        
        for name, options in ENV_CHOICES:
            found = False
            # 首先尝试 ENV_CHOICES 中定义的选项
            for opt in options:
                opt_lower = opt.lower()
                if opt_lower in cols:
                    selected[name] = opt_lower
                    print(f"[INFO] Selected {name} -> {opt_lower}")
                    found = True
                    break
            
            # 如果没找到，尝试变体
            if not found and name in env_variants:
                for variant in env_variants[name]:
                    variant_lower = variant.lower()
                    if variant_lower in cols:
                        selected[name] = variant_lower
                        print(f"[INFO] Selected {name} -> {variant_lower} (variant)")
                        found = True
                        break
            
            if not found:
                print(f"[WARN] No column found for {name} from options: {options}")
        return selected

    # 1. 为每个年份分别构建像元级特征表、选择环境因子并独立建模
    all_validation_reports = {}
    for config in YEARLY_CONFIGS:
        year_tag = config["year_tag"]
        print(f"\n{'=' * 20} PROCESSING: {year_tag} {'=' * 20}")

        year_out_dir = os.path.join(OUTPUT_BASE_DIR, year_tag + "_Results")
        os.makedirs(year_out_dir, exist_ok=True)

        df_pixels = build_pixel_feature_table_gee(
            brdf_dir=config["brdf_dir"],
            target_dates=config["target_dates"],
            era5_env_raster=config["era5_env_raster"],
            era5_hourly_raster=config["era5_hourly_raster"],
            par_k=config["par_k"],
            sample_pixels=SAMPLE_PIXELS,
            drought_tif=config.get("drought_tif", None),
            gee_env_tif=config.get("gee_env_tif", None),  # 传递 GEE 环境因子 TIF 路径
        )

        print(f"[INFO] Feature table shape for {year_tag}: {df_pixels.shape}")
        csv_out = os.path.join(year_out_dir, f"pixel_feature_table_{year_tag}.csv")
        try:
            safe_to_csv(df_pixels, csv_out)
            print(f"[SAVED] {csv_out}")
        except PermissionError as exc:
            print(f"[WARN] 无法写入 {csv_out}（可能被 Excel 占用）：{exc}")
            fallback_csv = os.path.join(
                year_out_dir,
                f"pixel_feature_table_{year_tag}_{datetime.now():%Y%m%d_%H%M%S}.csv",
            )
            safe_to_csv(df_pixels, fallback_csv)
            csv_out = fallback_csv
            print(f"[SAVED] 使用备用文件：{csv_out}")

        if df_pixels.empty:
            print(f"[ERROR] Empty DataFrame for {year_tag}, skipping this year.")
            continue

        # 2. 特征选择：在该年份的数据中按 ENV_CHOICES 选择环境因子列
        df_year = df_pixels.copy()
        df_year.columns = [c.lower() for c in df_year.columns]

        if USE_ALL_ENV_FACTORS:
            available_features = get_all_env_features(df_year, TRAINING_TARGETS)
            print(f"[INFO] [{year_tag}] Using ALL env factors (excluding LAI): {len(available_features)}")
        elif AUTO_SELECT_TIME_SCALE:
            print(f"[INFO] [{year_tag}] Auto-selecting time scales by SHAP importance...")
            feature_map_by_target = {}
            for target in TRAINING_TARGETS:
                selected_features, group_choice = select_best_scales_by_shap(
                    df_year, target, TIME_SCALE_GROUPS
                )
                feature_map_by_target[target] = selected_features
                print(f"[INFO] [{year_tag}] [{target}] Selected scales: {group_choice}")
            available_features = feature_map_by_target
        else:
            env_map = choose_env_columns(df_year)
            env_cols = list(env_map.values())  # 使用实际列名（精确到时间尺度，小写）

            # 注意：ENV_CHOICES 已经包含了所有需要的因子（包括静态因子 Elev, SOC）
            available_features = list(dict.fromkeys(env_cols))  # 去重

            # 确保所有特征都在 DataFrame 中
            missing_features = set(available_features) - set(df_year.columns)
            if missing_features:
                print(f"[WARNING] Missing features in table for {year_tag}: {list(missing_features)}")
                available_features = [f for f in available_features if f in df_year.columns]

            # 最终验证：确保 available_features 只包含 ENV_CHOICES 中定义的因子
            all_env_choice_cols = set()
            for name, options in ENV_CHOICES:
                for opt in options:
                    all_env_choice_cols.add(opt.lower())

            filtered_features = [f for f in available_features if f in all_env_choice_cols]
            if len(filtered_features) < len(available_features):
                removed = set(available_features) - set(filtered_features)
                print(f"[WARNING] Removed features not in ENV_CHOICES for {year_tag}: {sorted(removed)}")
            available_features = filtered_features

            print(f"[INFO] [{year_tag}] Using {len(available_features)} features")
            print(f"[INFO] [{year_tag}] Selected environment factors (columns): {sorted(available_features)}")
            print(f"[INFO] [{year_tag}] Environment factor mapping (logical name -> column): {env_map}")

        # 3. 针对当前年份单独训练 XGBoost + SHAP 模型
        year_model_dir = os.path.join(OUTPUT_BASE_DIR, f"{year_tag}_XGB_SHAP")
        os.makedirs(year_model_dir, exist_ok=True)

        print(f"\n[INFO] Training models for {year_tag} (no cross-year merging)...")
        shap_results, validation_reports = train_and_shap(
            df_year,
            year_model_dir,
            feat_cols=available_features,
            targets=TRAINING_TARGETS,
        )

        all_validation_reports[year_tag] = validation_reports

        # 保存该年份的交叉验证结果
        if validation_reports:
            cv_summary_rows = []
            for target, metrics in validation_reports.items():
                if metrics is not None:
                    cv_summary_rows.append({
                        'year_tag': year_tag,
                        'target': target,
                        'n_samples': metrics['n_samples'],
                        'n_splits': metrics['n_splits'],
                        'RMSE': metrics['rmse'],
                        'MAE': metrics['mae'],
                        'R2': metrics['r2'],
                        'CV_R2_mean': metrics['cv_r2_mean'],
                        'CV_R2_std': metrics['cv_r2_std'],
                        'CV_R2_scores': str(metrics['cv_r2_scores'])
                    })

            if cv_summary_rows:
                cv_summary_df = pd.DataFrame(cv_summary_rows)
                cv_csv = os.path.join(year_model_dir, 'cross_validation_summary.csv')
                safe_to_csv(cv_summary_df, cv_csv)
                print(f"[SAVED] Cross-validation summary for {year_tag}: {cv_csv}")

        # 可选：打印该年份每个目标的前若干重要特征
        for target, df_imp in shap_results.items():
            print(f"\n[SHAP SUMMARY] Top features for {target} ({year_tag}):")
            print(df_imp.head(10))

    print("\n" + "="*60)
    print("[✓] ALL PROCESSING COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_BASE_DIR}")
    print(f"\nTarget mode: {TARGET_MODE}")
    print(f"Training targets: {TRAINING_TARGETS}")
    print("\nGenerated outputs:")
    print("  1. Pixel-level feature tables with diurnal metrics (CSV)")
    print("  2. SHAP importance values (CSV)")
    print("  3. SHAP beeswarm plots (PNG)")
    print("  4. SHAP dependence plots (PNG)")
    print("  5. Within-year multi-target comparison plots (PNG)")
    print("  6. Cross-year comparison plots (PNG)")
    print("  7. Validation metrics summary (CSV + console logs)")
    
    print("\n" + "="*60)
    print("TARGET VARIABLE STATISTICS")
    print("="*60)
    for config in YEARLY_CONFIGS:
        year_tag = config['year_tag']
        csv_path = os.path.join(OUTPUT_BASE_DIR, year_tag + '_Results', 
                               f'pixel_feature_table_{year_tag}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"\n{year_tag}:")
            for target in TRAINING_TARGETS:
                if target in df.columns:
                    print(f"  {target}:")
                    print(f"    Count: {df[target].count()}")
                    print(f"    Mean:  {df[target].mean():.4f}")
                    print(f"    Std:   {df[target].std():.4f}")
                    print(f"    Min:   {df[target].min():.4f}")
                    print(f"    Max:   {df[target].max():.4f}")

    # 打印交叉验证结果汇总
    if validation_reports:
        print("\n" + "="*60)
        print("CROSS-VALIDATION METRICS SUMMARY")
        print("="*60)
        for target, metrics in validation_reports.items():
            if metrics is not None:
                print(f"\n{target}:")
                print(f"  Samples: {metrics['n_samples']}")
                print(f"  CV Folds: {metrics['n_splits']}")
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  MAE:  {metrics['mae']:.4f}")
                print(f"  R²:   {metrics['r2']:.4f}")
                print(f"  CV R² (mean ± std): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
                print(f"  CV R² per fold: {[f'{s:.4f}' for s in metrics['cv_r2_scores']]}")
            else:
                print(f"\n{target}: (validation failed)")

if __name__ == '__main__':
    main()
