"""
Spearman 相关性矩阵绘图（按年份分图）
- 使用 2022（干旱年）和 2023（非干旱年）的像元特征表
- 节律指标与环境因子全量参与
- 输出单独子图：干旱年相关矩阵、非干旱年相关矩阵、两年显著性箭头对比
绘图风格尽量模拟示例图（方块大小代表相关强度，箭头表示显著性与方向）。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.stats import spearmanr
from osgeo import gdal
import traceback

import importlib.util
import sys

# 预先定义占位，避免静态检查警告
YEARLY_CONFIGS = None
build_pixel_feature_table_gee = None
safe_to_csv = None
_IMPORT_ERROR = None


def _load_source_module():
    """
    动态加载带有连字符的源文件 XGBoost_shap_GEE-Adapted.py
    返回模块对象；失败时抛出异常。
    """
    global YEARLY_CONFIGS, build_pixel_feature_table_gee, safe_to_csv, _IMPORT_ERROR
    script_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(script_dir, "XGBoost_shap_GEE-Adapted.py")
    module_name = "xgb_shap_adapted"  # 任意合法名称

    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法创建导入规范：{module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore
    except Exception as e:
        _IMPORT_ERROR = e
        raise

    YEARLY_CONFIGS = getattr(module, "YEARLY_CONFIGS", None)
    build_pixel_feature_table_gee = getattr(module, "build_pixel_feature_table_gee", None)
    safe_to_csv = getattr(module, "safe_to_csv", None)
    if YEARLY_CONFIGS is None or build_pixel_feature_table_gee is None or safe_to_csv is None:
        raise ImportError("源模块缺少必要对象：YEARLY_CONFIGS/build_pixel_feature_table_gee/safe_to_csv")
    _IMPORT_ERROR = None
    return module

# 字体设置
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# ==== 文件路径配置（如有变动请直接修改） ====
BASE_OUT_DIR = r"F:\FY4\outputs_final\pixel_level_xgb_shap_gee"
YEAR_FILES = {
    "2022_Drought": os.path.join(
        BASE_OUT_DIR,
        "2022_Drought_Results",
        "pixel_feature_table_2022_Drought.csv",
    ),
    "2023_Wet": os.path.join(
        BASE_OUT_DIR,
        "2023_Wet_Results",
        "pixel_feature_table_2023_Wet.csv",
    ),
}

# 干旱分级图（SMPct_DroughtClass）路径
DROUGHT_MASKS = {
    "2022_Drought": r"F:\G_disk\FY4\data\Drought_GEE\drive-download-20251227T184021Z-1-001\SMPct_DroughtClass_20220423.tif",
    "2023_Wet": r"F:\G_disk\FY4\data\Drought_GEE\drive-download-20251227T184021Z-1-001\SMPct_DroughtClass_20230425.tif",
}

# 选择的干旱等级（定义旱地和湿地）
DROUGHT_CLASSES = {2, 3, 4}  # 旱地
WET_CLASSES = {0, 1}          # 湿地

# 保留原有的DROUGHT_KEEP用于兼容（但现在我们会保留所有drought_class）
DROUGHT_KEEP = {
    # 保留所有干旱等级，不再在这里过滤
    "2022_Drought": DROUGHT_CLASSES | WET_CLASSES,  # 保留所有
    "2023_Wet": DROUGHT_CLASSES | WET_CLASSES,      # 保留所有
}

# ==== 变量列表 ====
# 将总变量控制在 4 个节律 + 环境因子；若 TOP_ENV 为 None 则使用全部可用环境因子
RHYTHM_VARS = [
    "target_nirv",
    "target_t_peak",
    "target_mdi",
    "centroid_shift",
]

# 新增：ERA5/MODIS 不同时标的环境因子（从 GEE 导出的 Env_daily_1d3d_*.tif 栅格采样）
# 1d/3d（均值）+ 7d 降水，顺序与导出脚本一致
ENV_RASTER_2022 = r"F:\G_disk\FY4\data\ancillary_data\ERA5_hour\Env_daily_1d3d_dry_20220423.tif"
ENV_RASTER_2023 = r"F:\G_disk\FY4\data\ancillary_data\ERA5_hour\Env_daily_1d3d_wet_20230425.tif"
# 小时级栈（band 描述包含 Txx_变量名），需要同时输出逐小时与 9-16 小时累计/均值
HOURLY_RASTER_2022 = r"F:\G_disk\FY4\data\ancillary_data\ERA5_hour\ERA5_hourly_dry_20220423_hourly_stack.tif"
HOURLY_RASTER_2023 = r"F:\G_disk\FY4\data\ancillary_data\ERA5_hour\ERA5_hourly_wet_20230425_hourly_stack.tif"

# 栅格 band 名（1-based 映射在采样时按实际 band 数截断，避免越界）
# 期望顺序：10 个 1d、10 个 3d、降水 7d、LST/LST3、LAI/LAI3、NDVI/NDVI3
ERA5_BANDS = [
    "temp_1d", "dew_1d", "u10_1d", "v10_1d", "tp_1d", "ssrd_1d", "swdown_1d", "par_1d", "wind10_1d", "vpd_1d",
    "temp_3d", "dew_3d", "u10_3d", "v10_3d", "tp_3d", "ssrd_3d", "swdown_3d", "par_3d", "wind10_3d", "vpd_3d",
    "precip_sum_7d",
    "lst_1d", "lst_3d",
    "lai_1d", "lai_3d",
    "ndvi_1d", "ndvi_3d",
]
# 1-based 索引映射
BAND_TO_INDEX = {name: idx + 1 for idx, name in enumerate(ERA5_BANDS)}


def display_name(var):
    """用于绘图显示的通用短名。"""
    mapping = {
        "target_nirv": "NIRv",
        "target_t_peak": "T_peak",
        "target_mdi": "MDI",
        "centroid_shift": "C_shift",
        "swdown_j_1d": "PAR",
        "lst_mean_last_1d": "LST",
        "lst_mean_last_3d": "LST",
        "vpd_mean_last_3d": "VPD",
        "sm_mean_last_1d": "SM",
        "sm_mean_last_7d": "SM",
        "precip_sum_last_30d": "P",
        "elevation": "DEM",
    }
    return mapping.get(var, var.upper())

# 年份样式
YEAR_STYLE = {
    "2022_Drought": {"label": "2022 (Drought)", "color": "#2b83ba"},
    "2023_Wet": {"label": "2023 (Wet)", "color": "#fdae61"},
}


def compute_spearman(df, variables, min_samples=10):
    """
    逐对 Spearman 计算，返回相关系数矩阵与 p 值矩阵。
    强制转为 1D 数值数组，避免返回 2x2 矩阵导致赋值报错。
    """
    r_mat = pd.DataFrame(np.nan, index=variables, columns=variables, dtype=float)
    p_mat = pd.DataFrame(np.nan, index=variables, columns=variables, dtype=float)
    for i in variables:
        for j in variables:
            # 尝试转为数值型，无法转的置为 NaN
            subset = (
                df[[i, j]]
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
            )
            if len(subset) < min_samples:
                continue

            x = subset[i].to_numpy().astype(float).ravel()
            y = subset[j].to_numpy().astype(float).ravel()
            try:
                rho, pval = spearmanr(x, y)
                # 避免返回矩阵类型
                if isinstance(rho, np.ndarray):
                    rho = float(np.nanmean(rho))
                if isinstance(pval, np.ndarray):
                    pval = float(np.nanmean(pval))
            except Exception:
                continue

            r_mat.loc[i, j] = rho
            p_mat.loc[i, j] = pval
    return r_mat, p_mat


def select_top_envs(df_list, env_cols, top_k=None):
    """根据节律指标与环境因子的 |rho| 平均值挑选 top_k 环境因子；top_k=None 时返回全部排序列表。"""
    if not env_cols:
        return []
    all_df = pd.concat(df_list, axis=0, ignore_index=True)
    scores = {}
    for env in env_cols:
        rhos = []
        for tgt in RHYTHM_VARS:
            subset = all_df[[tgt, env]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(subset) < 10:
                continue
            try:
                rho, _ = spearmanr(subset[tgt].to_numpy(), subset[env].to_numpy())
                if isinstance(rho, np.ndarray):
                    rho = float(np.nanmean(rho))
                rhos.append(abs(rho))
            except Exception:
                continue
        if rhos:
            scores[env] = float(np.nanmean(rhos))
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top_k is None:
        return [k for k, _ in ranked]
    return [k for k, _ in ranked[:top_k]]


def compute_env_scores(df_list, env_cols, min_samples=10):
    """
    计算环境因子的综合评分：
    - explain = mean(|rho(env, rhythm)|)  across RHYTHM_VARS
    - collinear = max_{env2!=env} |rho(env, env2)|
    - final_score = explain / (1 + collinear)
    返回排序列表 [ {name, explain, collinear, score} ... ] 按 score 降序。
    """
    if not env_cols:
        return []
    all_df = pd.concat(df_list, axis=0, ignore_index=True)
    # 先计算 env 与 rhythm 的 |rho| 均值
    explain = {}
    for env in env_cols:
        rhos = []
        for tgt in RHYTHM_VARS:
            subset = all_df[[tgt, env]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(subset) < min_samples:
                continue
            try:
                rho, _ = spearmanr(subset[tgt].to_numpy(), subset[env].to_numpy())
                if isinstance(rho, np.ndarray):
                    rho = float(np.nanmean(rho))
                rhos.append(abs(rho))
            except Exception:
                continue
        explain[env] = float(np.nanmean(rhos)) if rhos else 0.0

    # 再计算 env 间的最大 |rho|
    collinear = {}
    for env in env_cols:
        max_r = 0.0
        for env2 in env_cols:
            if env2 == env:
                continue
            subset = all_df[[env, env2]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(subset) < min_samples:
                continue
            try:
                rho, _ = spearmanr(subset[env].to_numpy(), subset[env2].to_numpy())
                if isinstance(rho, np.ndarray):
                    rho = float(np.nanmean(rho))
                max_r = max(max_r, abs(rho))
            except Exception:
                continue
        collinear[env] = max_r

    rows = []
    for env in env_cols:
        exp = explain.get(env, 0.0)
        col = collinear.get(env, 0.0)
        score = exp / (1.0 + col)
        rows.append({"name": env, "explain": exp, "collinear": col, "score": score})
    rows = sorted(rows, key=lambda x: x["score"], reverse=True)
    return rows


def compute_env_rhythm_corr(df, env_cols, rhythm_cols, min_samples=10, year_tag=""):
    """
    逐环境因子×节律指标计算 Spearman 相关，返回列表：
    {year, env, rhythm, rho, pval, n}
    """
    rows = []
    for env in env_cols:
        for rh in rhythm_cols:
            subset = df[[env, rh]].apply(pd.to_numeric, errors="coerce").dropna()
            n = len(subset)
            if n < min_samples:
                rows.append({"year": year_tag, "env": env, "rhythm": rh, "rho": np.nan, "pval": np.nan, "n": n})
                continue
            try:
                rho, pval = spearmanr(subset[env].to_numpy(), subset[rh].to_numpy())
                if isinstance(rho, np.ndarray):
                    rho = float(np.nanmean(rho))
                if isinstance(pval, np.ndarray):
                    pval = float(np.nanmean(pval))
            except Exception:
                rho, pval = np.nan, np.nan
            rows.append({"year": year_tag, "env": env, "rhythm": rh, "rho": rho, "pval": pval, "n": n})
    return rows


def load_year_df(year_key):
    fp = YEAR_FILES[year_key]
    if not os.path.exists(fp):
        print(f"[WARN] 未找到特征表 {fp}，尝试自动重建……")
        fp = regenerate_feature_table(year_key)

    df = pd.read_csv(fp)
    df.columns = [c.lower() for c in df.columns]
    lonlat_cols = [c for c in df.columns if c.lower() in ("lon", "lat", "longitude", "latitude", "x", "y")]
    # 保留原表的所有环境列（DEM、土壤质地、SOC 等），仅统一列名小写
    df = df.copy()

    # 采样新增环境栅格（日尺度）
    raster_fp = ENV_RASTER_2022 if "2022" in year_key else ENV_RASTER_2023
    df, lon_col, lat_col = _extract_lon_lat(df)
    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy()
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy()
    try:
        sampled = sample_raster_by_lonlat(raster_fp, lon, lat, BAND_TO_INDEX)
        for k, v in sampled.items():
            df[k] = v
    except Exception as e:
        print(f"[WARN] {year_key} 栅格采样失败：{e}")

    # 采样小时级栅格，增加逐小时和 9-16 聚合
    hourly_fp = HOURLY_RASTER_2022 if "2022" in year_key else HOURLY_RASTER_2023
    try:
        sampled_h = sample_hourly_stack(hourly_fp, lon, lat, hours_local=list(range(9, 17)))
        for k, v in sampled_h.items():
            df[k] = v
    except Exception as e:
        print(f"[WARN] {year_key} 小时栅格采样失败：{e}")

    env_cols = [c for c in df.columns if c not in RHYTHM_VARS + [lon_col, lat_col]]
    available_vars = RHYTHM_VARS + env_cols
    return df, available_vars


def _extract_lon_lat(df):
    """
    提取/标准化 lon/lat 列，兼容多种命名。返回 (df, lon_col, lat_col)。
    若不存在则抛出 KeyError。
    """
    candidates = [
        ("lon", "lat"),
        ("longitude", "latitude"),
        ("x", "y"),
    ]
    cols = [c.lower() for c in df.columns]
    col_map = dict(zip(df.columns, cols))
    df = df.rename(columns=col_map)

    for lon_c, lat_c in candidates:
        if lon_c in df.columns and lat_c in df.columns:
            return df, lon_c, lat_c

    raise KeyError(
        "DataFrame 中缺少 lon/lat（或 longitude/latitude / x/y）列，"
        "无法按干旱分级筛选。请确认 CSV 是否包含经纬度。"
    )


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


def sample_mask_values(mask_fp, lon, lat):
    """按 lon/lat 从干旱分级图采样，返回同长度的数组。"""
    ds = gdal.Open(mask_fp)
    if ds is None:
        raise FileNotFoundError(f"无法打开干旱分级图：{mask_fp}")
    gt = ds.GetGeoTransform()
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    # 将经纬度转换为像元行列
    x = ((lon - gt[0]) / gt[1]).astype(int)
    y = ((lat - gt[3]) / gt[5]).astype(int)

    h, w = arr.shape
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = np.full(len(lon), np.nan)
    out[valid] = arr[y[valid], x[valid]]
    return out


def add_drought_class_to_df(df, year_key):
    """为DataFrame添加干旱分类列（不删除，保留所有数据）。"""
    # 若缺少经纬度，尝试提示并重建
    try:
        df, lon_col, lat_col = _extract_lon_lat(df)
    except KeyError:
        print(f"[WARN] {year_key} 数据缺少经纬度，尝试自动重建特征表…")
        new_fp = regenerate_feature_table(year_key)
        df_reload = pd.read_csv(new_fp)
        df_reload.columns = [c.lower() for c in df_reload.columns]
        # 保留原有数值列 + 新的经纬度列
        numeric_cols = [c for c in df.columns if c in df_reload.columns]
        df = df_reload[numeric_cols + [c for c in df_reload.columns if c in ("lon","lat","longitude","latitude","x","y")]]
        df, lon_col, lat_col = _extract_lon_lat(df)
    
    # 如果已经有drought_class列，直接返回
    if "drought_class" in df.columns:
        return df
    
    mask_fp = DROUGHT_MASKS[year_key]
    if not os.path.exists(mask_fp):
        print(f"[WARN] 干旱分类图不存在：{mask_fp}，无法添加drought_class")
        return df

    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy()
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy()
    mask_vals = sample_mask_values(mask_fp, lon, lat)

    df = df.copy()
    df["drought_class"] = mask_vals
    return df


def filter_by_drought_mask(df, year_key):
    """根据干旱等级筛选像元（保留此函数用于向后兼容，但建议使用add_drought_class_to_df）。"""
    df = add_drought_class_to_df(df, year_key)
    keep_vals = DROUGHT_KEEP[year_key]
    if "drought_class" in df.columns:
        df = df[df["drought_class"].isin(keep_vals)].copy()
        df = df.drop(columns=["drought_class"])
    return df


def filter_by_dates(df, target_dates):
    """若存在 date 列（或 datetime 列），仅保留目标日期；target_dates 为 {date(...)} 集合。"""
    if not target_dates:
        return df
    cols = [c for c in df.columns if "date" in c.lower()]
    if not cols:
        return df
    col = cols[0]
    try:
        dt = pd.to_datetime(df[col]).dt.date
        return df[dt.isin(target_dates)].copy()
    except Exception:
        return df


def regenerate_feature_table(year_key):
    """当 CSV 缺失或无经纬度时，调用源头构建函数重建特征表。"""
    if build_pixel_feature_table_gee is None or safe_to_csv is None or YEARLY_CONFIGS is None:
        try:
            _load_source_module()
        except Exception as e:
            raise RuntimeError(
                f"无法自动重建特征表，导入 XGBoost_shap_GEE-Adapted 失败：{e}"
            )

    cfg = None
    for c in YEARLY_CONFIGS:
        if c.get("year_tag") == year_key:
            cfg = c
            break
    if cfg is None:
        raise ValueError(f"未在 YEARLY_CONFIGS 中找到 year_tag={year_key}")

    print(f"[INFO] 正在重建特征表：{year_key}，请稍候（可能耗时）...")
    df_pixels = build_pixel_feature_table_gee(
        brdf_dir=cfg["brdf_dir"],
        target_dates=cfg["target_dates"],
        gee_env_tif=cfg["gee_env_tif"],
        par_k=cfg["par_k"],
        sample_pixels=0,
    )

    out_dir = os.path.join(BASE_OUT_DIR, f"{year_key}_Results")
    os.makedirs(out_dir, exist_ok=True)
    out_fp = os.path.join(out_dir, f"pixel_feature_table_{year_key}.csv")
    safe_to_csv(df_pixels, out_fp)
    print(f"[SAVED] 重建完成：{out_fp}")
    return out_fp


def plot_corr_matrix(r_mat, p_mat, year_key, save_path, vmax=0.5, alpha=0.05):
    """绘制单年相关矩阵，仿示例：上三角方块，下三角数值+箭头，对角线变量名。"""
    variables = r_mat.columns.tolist()
    labels = [display_name(v) for v in variables]
    n = len(variables)

    cmap = cm.get_cmap("Greens")
    norm = colors.Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    # 网格
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    # 不再显示轴标签
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", which="both",
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False,
                   length=0, width=0)  # 彻底关闭刻度线
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    blue_base = np.array(colors.to_rgb("#3b6ea5"))
    orange_base = np.array(colors.to_rgb("#f5a623"))

    year_color = YEAR_STYLE[year_key]["color"]

    for i in range(n):
        for j in range(n):
            rho = r_mat.iloc[i, j]
            pval = p_mat.iloc[i, j]
            # 对角线：写变量名，白底
            if i == j:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor="0.55", linewidth=1.1, zorder=1))
                ax.text(
                    j,
                    i,
                    labels[i],
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="black",
                )
                continue

            if np.isnan(rho):
                continue

            if j > i:
                # 上三角：方块大小与色深
                size = np.clip(abs(rho), 0, 1) * 1800
                color = cmap(norm(abs(rho)))
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor="0.55", linewidth=1.0, zorder=1))
                ax.scatter(
                    j,
                    i,
                    s=size,
                    marker="s",
                    color=color,
                    edgecolors="none",
                    linewidths=0,
                    alpha=0.9,  # 柔化方块边缘
                    zorder=2,
                )
            else:
                # 下三角：数值 + 显著性箭头（位置类似示例：数值居中，箭头稍下）
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor="0.55", linewidth=1.0, zorder=1))
                shade = np.clip(abs(rho) / vmax, 0, 1)
                num_color = colors.to_hex(blue_base * (0.5 + 0.5 * shade))
                ax.text(
                    j,
                    i - 0.05,
                    f"{abs(rho):.2f}",
                    ha="center",
                    va="center",
                    fontsize=12,  # 数值字号
                    color=num_color,
                    fontweight="bold",
                    zorder=2,
                )
                if pval < alpha:
                    arrow = "↑" if rho > 0 else "↓"
                    arr_color = colors.to_hex(orange_base * (0.5 + 0.5 * shade))
                    ax.text(
                        j,
                        i + 0.20,
                        arrow,
                        ha="center",
                        va="center",
                        fontsize=12,  # 单年箭头字号
                        color=arr_color,
                        fontweight="bold",
                        zorder=2,
                    )

    # 颜色条（|rho|）
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, aspect=30)
    cbar.set_label("|Spearman ρ|", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    # 无标题
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {save_path}")


def plot_significance_overlay(r_2022, p_2022, r_2023, p_2023, save_path, alpha=0.05):
    """绘制两年显著性方向对比（仅左下三角），对角线标注变量名。"""
    variables = r_2022.columns.tolist()
    labels = [display_name(v) for v in variables]
    n = len(variables)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", which="both",
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False,
                   length=0, width=0)  # 彻底关闭刻度线
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    color_up = "#0057b7"    # 乌克兰蓝
    color_down = "#ffd700"  # 乌克兰黄
    edge_up = "#003f88"
    edge_down = "#ccac00"

    for i in range(n):
        for j in range(n):
            # 对角线写变量名
            if i == j:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor="0.55", linewidth=1.1, zorder=1))
                ax.text(
                    j,
                    i,
                    labels[i],
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="black",
                    zorder=2,
                )
                continue

            if j > i:
                continue  # 只绘制左下三角

            # 格网白底
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor="0.55", linewidth=1.0, zorder=1))

            # 2022箭头（左偏，横向排布）
            rho22 = r_2022.iloc[i, j]
            p22 = p_2022.iloc[i, j]
            sig22 = (not np.isnan(rho22)) and p22 < alpha
            if np.isnan(rho22):
                rho22 = 0.0
            if rho22 > 0:
                start = (j - 0.35, i + 0.12)
                end = (j - 0.05, i - 0.18)  # ↗
                fc, ec = color_up, edge_up
            else:
                start = (j - 0.35, i - 0.18)
                end = (j - 0.05, i + 0.12)  # ↘
                fc, ec = color_down, edge_down
            if not sig22:
                fc, ec, alpha_val = "#d0d0d0", "#a0a0a0", 0.35
            else:
                alpha_val = 0.9
            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle="simple",
                mutation_scale=22,  # 对比图箭头整体尺度
                linewidth=1.4,      # 箭头边框线宽
                facecolor=fc,
                edgecolor=ec,
                alpha=alpha_val,
                zorder=2,
            )
            ax.add_patch(arrow)

            # 2023箭头（右偏，横向排布）
            rho23 = r_2023.iloc[i, j]
            p23 = p_2023.iloc[i, j]
            sig23 = (not np.isnan(rho23)) and p23 < alpha
            if np.isnan(rho23):
                rho23 = 0.0
            if rho23 > 0:
                start = (j + 0.05, i + 0.12)
                end = (j + 0.35, i - 0.18)  # ↗
                fc, ec = color_up, edge_up
            else:
                start = (j + 0.05, i - 0.18)
                end = (j + 0.35, i + 0.12)  # ↘
                fc, ec = color_down, edge_down
            if not sig23:
                fc, ec, alpha_val = "#d0d0d0", "#a0a0a0", 0.35
            else:
                alpha_val = 0.9
            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle="simple",
                mutation_scale=22,  # <- 对比图箭头整体尺度
                linewidth=1.4,      # <- 箭头边框线宽
                facecolor=fc,
                edgecolor=ec,
                alpha=alpha_val,
                zorder=2,
            )
            ax.add_patch(arrow)

    # 图例
    ax.text(
        0.98,
        -0.08,
        f"{YEAR_STYLE['2022_Drought']['label']} (蓝/黄箭头, 左偏)",
        transform=ax.transAxes,
        color="black",
        fontsize=12,
        ha="right",
        fontweight="bold",
    )
    ax.text(
        0.98,
        -0.14,
        f"{YEAR_STYLE['2023_Wet']['label']} (蓝/黄箭头, 右偏)",
        transform=ax.transAxes,
        color="black",
        fontsize=12,
        ha="right",
        fontweight="bold",
    )

    # 无标题
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {save_path}")


def main():
    # 目标日期限制：仅 2022-04-23 和 2023-04-25
    target_dates = {
        "2022_Drought": {pd.to_datetime("2022-04-23").date()},
        "2023_Wet": {pd.to_datetime("2023-04-25").date()},
    }
    # 加载 2022 与 2023
    df_22, vars_22 = load_year_df("2022_Drought")
    df_23, vars_23 = load_year_df("2023_Wet")
    lonlat_cols_22 = [c for c in df_22.columns if c in ("lon", "lat", "longitude", "latitude", "x", "y")]
    lonlat_cols_23 = [c for c in df_23.columns if c in ("lon", "lat", "longitude", "latitude", "x", "y")]

    # 环境列全集（去除节律和经纬度），两年交集
    env_cols_22 = [c for c in df_22.columns if c not in RHYTHM_VARS + lonlat_cols_22]
    env_cols_23 = [c for c in df_23.columns if c not in RHYTHM_VARS + lonlat_cols_23]
    env_common = [c for c in env_cols_22 if c in env_cols_23]
    
    # 强制包含静态环境因子（elevation 和 soc），即使它们不在交集中
    # 这些因子应该在原始特征表中存在（从 GEE 导出）
    static_factors = ['elevation', 'soc']
    for static_factor in static_factors:
        # 检查是否在任一数据集中存在
        if static_factor in df_22.columns or static_factor in df_23.columns:
            # 确保添加到 env_common（如果还没有）
            if static_factor not in env_common:
                env_common.append(static_factor)
                print(f"[INFO] 强制添加静态环境因子到 env_common: {static_factor}")
        
        # 如果某个数据集中缺少该列，用 NaN 填充
        if static_factor not in df_22.columns and static_factor in df_23.columns:
            df_22[static_factor] = np.nan
            print(f"[INFO] 在 df_22 中添加缺失的静态因子列: {static_factor} (填充 NaN)")
        elif static_factor in df_22.columns and static_factor not in df_23.columns:
            df_23[static_factor] = np.nan
            print(f"[INFO] 在 df_23 中添加缺失的静态因子列: {static_factor} (填充 NaN)")

    # 仅保留交集列（节律+环境+经纬度），并输出特征表
    all_vars = RHYTHM_VARS + env_common
    df_22 = df_22[[c for c in df_22.columns if c in all_vars + lonlat_cols_22]].copy()
    df_23 = df_23[[c for c in df_23.columns if c in all_vars + lonlat_cols_23]].copy()

    # 按日期筛选（若存在 date 列）- 只保留2022-04-23和2023-04-25
    df_22 = filter_by_dates(df_22, target_dates.get("2022_Drought", set()))
    df_23 = filter_by_dates(df_23, target_dates.get("2023_Wet", set()))

    # 添加drought_class列（不删除，保留所有数据）
    df_22 = add_drought_class_to_df(df_22, "2022_Drought")
    df_23 = add_drought_class_to_df(df_23, "2023_Wet")

    # 检查drought_class的有效值
    if "drought_class" in df_22.columns:
        print(f"[INFO] 2022年drought_class分布: {df_22['drought_class'].value_counts().to_dict()}")
    if "drought_class" in df_23.columns:
        print(f"[INFO] 2023年drought_class分布: {df_23['drought_class'].value_counts().to_dict()}")

    # 将数据分为四个场景
    print("[INFO] 将数据分为四个场景...")
    # 2022年旱地
    df_22_dry = df_22[df_22["drought_class"].isin(DROUGHT_CLASSES)].copy() if "drought_class" in df_22.columns else pd.DataFrame()
    # 2022年湿地
    df_22_wet = df_22[df_22["drought_class"].isin(WET_CLASSES)].copy() if "drought_class" in df_22.columns else pd.DataFrame()
    # 2023年旱地
    df_23_dry = df_23[df_23["drought_class"].isin(DROUGHT_CLASSES)].copy() if "drought_class" in df_23.columns else pd.DataFrame()
    # 2023年湿地
    df_23_wet = df_23[df_23["drought_class"].isin(WET_CLASSES)].copy() if "drought_class" in df_23.columns else pd.DataFrame()

    print(f"[INFO] 2022年旱地: {len(df_22_dry)} 样本")
    print(f"[INFO] 2022年湿地: {len(df_22_wet)} 样本")
    print(f"[INFO] 2023年旱地: {len(df_23_dry)} 样本")
    print(f"[INFO] 2023年湿地: {len(df_23_wet)} 样本")

    # 导出特征表（可选，保留原始数据用于其他分析）
    os.makedirs("fig_corr", exist_ok=True)
    if len(df_22) > 0:
        out22 = os.path.join("fig_corr", "feature_table_2022_0423.csv")
        df_22_dry_without_class = df_22_dry.drop(columns=["drought_class"]) if "drought_class" in df_22_dry.columns else df_22_dry
        df_22_dry_without_class.to_csv(out22, index=False, mode="w")
        print(f"[SAVE] 2022 特征表: {out22} (rows={len(df_22_dry_without_class)}, cols={len(df_22_dry_without_class.columns)})")
    if len(df_23) > 0:
        out23 = os.path.join("fig_corr", "feature_table_2023_0425.csv")
        df_23_wet_without_class = df_23_wet.drop(columns=["drought_class"]) if "drought_class" in df_23_wet.columns else df_23_wet
        df_23_wet_without_class.to_csv(out23, index=False, mode="w")
        print(f"[SAVE] 2023 特征表: {out23} (rows={len(df_23_wet_without_class)}, cols={len(df_23_wet_without_class.columns)})")

    # 为每个场景分别计算环境因子与节律指标的相关性
    print("[INFO] 为四个场景分别计算相关性...")
    rows_corr = []
    
    # 2022年旱地（干旱年旱地）
    if len(df_22_dry) > 0:
        rows_corr += compute_env_rhythm_corr(df_22_dry, env_common, RHYTHM_VARS, min_samples=10, year_tag="2022_Drought_DryLand")
    
    # 2022年湿地（干旱年湿地）
    if len(df_22_wet) > 0:
        rows_corr += compute_env_rhythm_corr(df_22_wet, env_common, RHYTHM_VARS, min_samples=10, year_tag="2022_Drought_WetLand")
    
    # 2023年旱地（非干旱年旱地）
    if len(df_23_dry) > 0:
        rows_corr += compute_env_rhythm_corr(df_23_dry, env_common, RHYTHM_VARS, min_samples=10, year_tag="2023_Wet_DryLand")
    
    # 2023年湿地（非干旱年湿地）
    if len(df_23_wet) > 0:
        rows_corr += compute_env_rhythm_corr(df_23_wet, env_common, RHYTHM_VARS, min_samples=10, year_tag="2023_Wet_WetLand")
    
    if not rows_corr:
        print("[WARN] 没有计算出任何相关性数据，请检查数据是否完整")
        return
    
    corr_df = pd.DataFrame(rows_corr)
    # 按 env+rhythm+year 排序，便于查看
    corr_df = corr_df.sort_values(by=["year", "env", "rhythm"]).reset_index(drop=True)
    
    # 同时按 |rho| 降序导出一份便于挑选
    corr_sorted = corr_df.copy()
    corr_sorted["abs_rho"] = corr_sorted["rho"].abs()
    corr_sorted = corr_sorted.sort_values(by=["abs_rho"], ascending=False)
    
    # 保存相关性表格
    corr_csv = os.path.join("fig_corr", "env_corr_by_rhythm.csv")
    corr_df.to_csv(corr_csv, index=False, mode="w")
    corr_sorted_csv = os.path.join("fig_corr", "env_corr_by_rhythm_abs_sorted.csv")
    corr_sorted.to_csv(corr_sorted_csv, index=False, mode="w")
    print(f"[SAVE] 环境因子逐节律相关性表: {corr_csv} (rows={len(corr_df)})")
    print(f"[SAVE] 环境因子逐节律相关性表（按|rho|排序）: {corr_sorted_csv} (rows={len(corr_sorted)})")
    
    # 统计每个场景的相关性数量
    print("\n[INFO] 各场景相关性统计:")
    scenarios = ["2022_Drought_DryLand", "2022_Drought_WetLand", "2023_Wet_DryLand", "2023_Wet_WetLand"]
    for scenario in scenarios:
        count = len(corr_df[corr_df["year"] == scenario])
        print(f"  {scenario}: {count} 条相关性记录")
    
    # 计算环境因子之间的相关性（用于补全相关性矩阵）
    print("\n[INFO] 计算环境因子之间的相关性...")
    rows_env_env = []
    
    # 映射场景名称
    scenario_map = {
        "2022_Drought_DryLand": ("2022_0423_dry", df_22_dry),
        "2022_Drought_WetLand": ("2022_0423_wet", df_22_wet),
        "2023_Wet_DryLand": ("2023_0425_dry", df_23_dry),
        "2023_Wet_WetLand": ("2023_0425_wet", df_23_wet),
    }
    
    for scenario_key, (year_tag, df_scenario) in scenario_map.items():
        if len(df_scenario) == 0:
            continue
        
        print(f"  [INFO] 计算 {scenario_key} 的环境因子之间相关性...")
        
        # 确保 DataFrame 包含所有 env_common 中的列
        missing_cols = set(env_common) - set(df_scenario.columns)
        if missing_cols:
            print(f"    [WARN] 缺少列: {missing_cols}，用 NaN 填充")
            for col in missing_cols:
                df_scenario[col] = np.nan
        
        # 计算所有环境因子之间的相关性
        env_count = 0
        for i, env1 in enumerate(env_common):
            for j, env2 in enumerate(env_common):
                if i >= j:  # 只计算上三角（下三角通过对称填充）
                    continue
                
                # 检查列是否存在
                if env1 not in df_scenario.columns or env2 not in df_scenario.columns:
                    continue
                
                subset = df_scenario[[env1, env2]].apply(pd.to_numeric, errors="coerce").dropna()
                n = len(subset)
                if n < 10:
                    continue
                
                try:
                    rho, pval = spearmanr(subset[env1].to_numpy(), subset[env2].to_numpy())
                    if isinstance(rho, np.ndarray):
                        rho = float(np.nanmean(rho))
                    if isinstance(pval, np.ndarray):
                        pval = float(np.nanmean(pval))
                    
                    if np.isfinite(rho) and np.isfinite(pval):
                        rows_env_env.append({
                            "year": year_tag,
                            "env1": env1,
                            "env2": env2,
                            "rho": rho,
                            "pval": pval,
                            "n": n
                        })
                        env_count += 1
                except Exception as e:
                    # 输出详细错误信息以便调试
                    print(f"    [WARN] 计算 {env1} 和 {env2} 的相关性失败: {e}")
                    continue
        
        print(f"    [INFO] {scenario_key}: 计算出 {env_count} 个环境因子之间的相关性对")
    
    # 保存环境因子之间的相关性表
    if rows_env_env:
        env_env_df = pd.DataFrame(rows_env_env)
        env_env_csv = os.path.join("fig_corr", "env_env_corr.csv")
        env_env_df.to_csv(env_env_csv, index=False, mode="w")
        print(f"[SAVE] 环境因子之间相关性表: {env_env_csv} (rows={len(env_env_df)})")
    else:
        print("[WARN] 没有计算出环境因子之间的相关性")
    
    return


if __name__ == "__main__":
    main()

