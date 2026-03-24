# -*- coding: utf-8 -*-
"""
plot_region_dtw_align_monthly_final.py

绘制 DTW 对齐后的 EVI 时间序列图，横坐标为月份（3–7 月），
用于与 SPEI–SM 图上下对照。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator
import dtw
import matplotlib.dates as mdates
import seaborn as sns
import os
import sys
import warnings
from datetime import date, timedelta  # Added for calculating shading dates
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Parameters and Style (Inherit SPEI/SM style) ----------
sns.set(style="whitegrid", context="talk", font_scale=1.1)
# Global font settings: Times New Roman, 9.5 pt (unified)
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "svg.fonttype": "none",  # keep text as text for crisp scaling in Word
})
# New distinct color scheme (Ukrainian flag colors)
COLOR_2022 = "#a2cffe"  # Deep Blue (2022 EVI Reference)
COLOR_2023 = "#FFDD00"  # Bright Yellow/Gold (2023 EVI DTW Aligned)

# --- User File Paths ---
csv_2022 = r'F:/G_disk/FY4/data/ancillary_data/dtw/HHH_WinterWheat_MODIS_2022.csv'
csv_2023 = r'F:/G_disk/FY4/data/ancillary_data/dtw/HHH_WinterWheat_MODIS_2023.csv'
use_index = 'EVI'
OUT_DIR = r'F:\FY4\outputs_paper_figs' # User-specified output directory
os.makedirs(OUT_DIR, exist_ok=True)
out_fig_fp = os.path.join(OUT_DIR, 'Fig_EVI_DTW_Aligned_MarJul.png')
out_fig_zoom_fp = os.path.join(OUT_DIR, 'Fig_EVI_DTW_KeyZoom.png')
out_fig_full_lines_fp = os.path.join(OUT_DIR, 'Fig_EVI_DTW_FullPath.png')
out_fig_fp_svg = os.path.join(OUT_DIR, 'Fig_EVI_DTW_Aligned_MarJul.svg')
out_fig_zoom_fp_svg = os.path.join(OUT_DIR, 'Fig_EVI_DTW_KeyZoom.svg')
out_fig_full_lines_fp_svg = os.path.join(OUT_DIR, 'Fig_EVI_DTW_FullPath.svg')
SHOW_FIG = False  # 避免脚本阻塞，可设为 True 以弹出窗口
DEBUG = True      # 打印调试信息
# ---------- Helper: daily interpolation for plotting ----------
def interpolate_daily(dates, values, start, end):
    """Linearly interpolate to daily resolution for smoother curves in the zoom plot."""
    ser = pd.Series(np.asarray(values), index=pd.to_datetime(dates))
    ser = ser[~ser.index.duplicated(keep="first")]  # guard duplicates
    ser = ser.sort_index()
    daily_index = pd.date_range(start, end, freq="1D")
    ser_interp = ser.reindex(ser.index.union(daily_index)).interpolate("time")
    ser_interp = ser_interp.ffill().bfill()  # fill edges to avoid NaN gaps
    out = ser_interp.reindex(daily_index)
    if DEBUG:
        print(f"[DEBUG] interpolate_daily: in_len={len(ser)}, non_nan={ser.notna().sum()}, out_len={len(out)}, out_non_nan={out.notna().sum()}, out_min/max={out.min() if out.notna().any() else 'nan'}/{out.max() if out.notna().any() else 'nan'}")
        print(f"[DEBUG] interpolate_daily head:\n{out.head(5)}")
    # Fallback: if still all NaN, do manual linear interpolation
    if out.isna().all():
        idx_num = ser.index.view(np.int64)
        tgt_num = daily_index.view(np.int64)
        out_vals = np.interp(tgt_num, idx_num, ser.values)
        out = pd.Series(out_vals, index=daily_index)
        if DEBUG:
            print("[DEBUG] interpolate_daily fallback np.interp used.")
    return out

# ---------- Helper: spline smoothing for plotting ----------
def spline_smooth(dates, values, start, end, freq="1D"):
    """PCHIP spline smoothing on daily grid for plotting (shape-preserving, no overshoot)."""
    idx = pd.to_datetime(dates)
    vals = np.asarray(values)
    df = pd.DataFrame({"x": idx.astype("int64"), "y": vals}).groupby("x", as_index=False).mean()
    x = df["x"].values
    y = df["y"].values
    dense_idx = pd.date_range(start, end, freq=freq)
    xi = dense_idx.astype("int64")
    f = PchipInterpolator(x, y)
    yi = f(xi)
    return pd.Series(yi, index=dense_idx)

# ---------- Helper: map dates to a base year (preserve DOY) ----------
def to_base_year(dates, base_year=2022):
    dates = pd.to_datetime(dates)
    doys = dates.dayofyear
    return pd.to_datetime([date(base_year, 1, 1) + timedelta(days=int(doy) - 1) for doy in doys])


# ---------- Data Loading ----------
def load_modis(csv_fp):
    df = pd.read_csv(csv_fp)
    df["date"] = pd.to_datetime(df["date"])
    # Filter months: Mar-Jul (3-7 month)
    df = df[(df["date"].dt.month >= 3) & (df["date"].dt.month <= 6)]
    # Scale factor correction
    if df[use_index].max() > 1.2:
        df[use_index] *= 0.0001
    df = df.sort_values("date").reset_index(drop=True)
    return df

try:
    df22 = load_modis(csv_2022)
    df23 = load_modis(csv_2023)
except FileNotFoundError as e:
    print(f"[FATAL] Missing input file: {e}")
    sys.exit(1)


vals22, vals23 = df22[use_index].values, df23[use_index].values
dates22, dates23 = df22["date"].values, df23["date"].values

# ---------- Smoothing Function (Savitzky-Golay) ----------
def smooth(vals, window=7, poly=3):
    """Savgol smoothing, dynamically adjusts window size to be odd and less than data length."""
    n = len(vals)
    wl = min(window, n if n % 2 == 1 else n - 1)
    if wl < 3: 
        return vals # Not enough data points to smooth
    return savgol_filter(vals, wl, poly, mode='nearest')

s22, s23 = smooth(vals22), smooth(vals23)
if DEBUG:
    print(f"[DEBUG] raw22 len={len(vals22)}, raw23 len={len(vals23)}")
    print(f"[DEBUG] s22 min/max={s22.min():.4f}/{s22.max():.4f}, s23 min/max={s23.min():.4f}/{s23.max():.4f}")
    print(f"[DEBUG] dates22 range={dates22.min()} -> {dates22.max()}")
    print(f"[DEBUG] dates23 range={dates23.min()} -> {dates23.max()}")

# ---------- DTW Alignment ----------
print("INFO: Performing DTW alignment...")
# dtw 库要求提供距离函数 dist（未提供会抛 TypeError）
euclid = lambda x, y: np.linalg.norm(x - y)
# dtw 返回 (distance, cost_matrix, acc_cost_matrix, path)
dist_val, _, _, path = dtw.dtw(s22.reshape(-1, 1), s23.reshape(-1, 1), euclid)
p22, p23 = np.array(path[0]), np.array(path[1])
print(f"INFO: DTW distance: {dist_val:.4f}")
if DEBUG:
    print(f"[DEBUG] path length={len(p22)}, uniq i22={len(np.unique(p22))}, uniq i23={len(np.unique(p23))}")

# ---------- DTW Mapping (Map 2023 onto 2022 time axis) ----------
aligned_vals23 = []
aligned_dates = []
warp_map = {}
# Collect which index i22 from 2022 corresponds to each index i23 from 2023
for i22, i23 in zip(p22, p23):
    warp_map.setdefault(i23, []).append(i22)

# For each 2023 index, take the median of the corresponding 2022 index as the target map
warp_median = {i23: int(np.median(v)) for i23, v in warp_map.items()}
# 映射表：每个 i23 对应的 2022 日期（用于绘制连线）
mapped_date_by_i23 = {i23: dates22[idx] for i23, idx in warp_median.items()}

# Reconstruct 2023 EVI series onto 2022's time axis
for i23 in sorted(warp_median.keys()):
    aligned_vals23.append(s23[i23])
    # Mapped date: take the date corresponding to the median index in the 2022 date array
    aligned_dates.append(dates22[warp_median[i23]])

aligned_df = pd.DataFrame({'date22_mapped': aligned_dates, 'EVI_2023_mapped': aligned_vals23})
aligned_df['date22_mapped'] = pd.to_datetime(aligned_df['date22_mapped'])
if DEBUG:
    print(f"[DEBUG] aligned_df rows={len(aligned_df)}, date range={aligned_df['date22_mapped'].min()}->{aligned_df['date22_mapped'].max()}")
    print(aligned_df.head())

# ---------- Final Plot (Single Figure) ----------
# Figure: 主图 + 关键期放大子图
fig, ax = plt.subplots(1, 1, figsize=(7, 5))

SHADE_DOY_START = 105
SHADE_DOY_END = 115  # extend 5 more days
BASE_YEAR = 2022 # X-axis is based on 2022 dates
shade_start_date = pd.Timestamp(date(BASE_YEAR, 1, 1) + timedelta(days=SHADE_DOY_START - 1))
shade_end_date = pd.Timestamp(date(BASE_YEAR, 1, 1) + timedelta(days=SHADE_DOY_END - 1))

# 使用 PCHIP 平滑后的日尺度曲线绘制主图
main_start = pd.Timestamp("2022-03-01")
main_end = pd.Timestamp("2022-07-01")
main_daily22 = spline_smooth(dates22, s22, main_start, main_end)
main_daily23 = spline_smooth(aligned_df['date22_mapped'], aligned_df['EVI_2023_mapped'], main_start, main_end)

# 1. Plot 2022 (Reference) - Deep Blue
ax.plot(main_daily22.index, main_daily22.values, color=COLOR_2022, lw=2.5, label='2022 EVI (Reference)')

ax.axvspan(shade_start_date, shade_end_date, color="gray", alpha=0.2, label=f"Key Period (DOY {SHADE_DOY_START}–{SHADE_DOY_END})")
# 2. Plot 2023 (DTW Mapped) - Bright Yellow
# Note: Increasing line visibility for the light yellow color
ax.plot(main_daily23.index, main_daily23.values,
        color=COLOR_2023, lw=3.0, label='2023 EVI (DTW Aligned)')

# Formatting and Style
# X-axis range: Mar 1 to Jul 1 (Mar-Jun)
ax.set_xlim(pd.Timestamp("2022-03-01"), pd.Timestamp("2022-07-01"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# Labels and Title
ax.set_ylabel('EVI', fontweight='bold')
ax.set_xlabel('Month (Mar–Jul)')
ax.tick_params(axis='both', labelsize=16)

plt.tight_layout()
plt.savefig(out_fig_fp, dpi=600, bbox_inches="tight")
plt.savefig(out_fig_fp_svg, format="svg", bbox_inches="tight")
if SHOW_FIG:
    plt.show()
else:
    plt.close(fig)
print(f"✅ Saved aligned EVI figure to: {out_fig_fp} and {out_fig_fp_svg}")

# ---------- Full-season DTW correspondence figure ----------
# 使用全季节曲线 + DTW 连线示意
fig_full, ax_full = plt.subplots(1, 1, figsize=(7, 5))

full_start = pd.Timestamp("2022-03-01")
full_end = pd.Timestamp("2022-07-01")
# PCHIP 平滑后的日尺度曲线（更平滑）
full_daily22 = spline_smooth(dates22, s22, full_start, full_end)
full_daily23 = spline_smooth(aligned_df['date22_mapped'], aligned_df['EVI_2023_mapped'], full_start, full_end)
if DEBUG:
    print(f"[DEBUG] full_daily22 len={len(full_daily22)}, min/max={full_daily22.min():.4f}/{full_daily22.max():.4f}")
    print(f"[DEBUG] full_daily23 len={len(full_daily23)}, min/max={full_daily23.min():.4f}/{full_daily23.max():.4f}")

ax_full.plot(full_daily22.index, full_daily22.values, color=COLOR_2022, lw=2.0, label='2022 EVI (Ref)')
ax_full.plot(full_daily23.index, full_daily23.values, color=COLOR_2023, lw=2.5, label='2023 EVI (DTW)')

# Dense DTW correspondence lines across full season
# Build daily grid and interpolate path mapping to densify lines
path_date22 = dates22[p22]
# 使用 2023 原日期映射到 2022 基准年，保持 DOY 分布，避免多点聚到同一日期
path_date23_mapped = to_base_year(dates23[p23], BASE_YEAR)
mask_path_full = ((path_date22 >= full_start) & (path_date22 <= full_end)) | \
                 ((path_date23_mapped >= full_start) & (path_date23_mapped <= full_end))
path_date22 = path_date22[mask_path_full]
path_date23_mapped = path_date23_mapped[mask_path_full]

# Dense DTW correspondence lines across full season
# Build daily grid and interpolate path mapping to densify lines
path_date22 = dates22[p22]
path_date23_mapped = np.array([mapped_date_by_i23[i23] for i23 in p23])
mask_path_full = ((path_date22 >= full_start) & (path_date22 <= full_end)) | \
                 ((path_date23_mapped >= full_start) & (path_date23_mapped <= full_end))
path_date22 = path_date22[mask_path_full]
path_date23_mapped = path_date23_mapped[mask_path_full]

# use smoothed curves for values
path_val22 = spline_smooth(dates22, s22, full_start, full_end)
path_val23 = spline_smooth(aligned_df['date22_mapped'], aligned_df['EVI_2023_mapped'], full_start, full_end)

path_date22_num = pd.to_datetime(path_date22).astype("int64")
path_date23_num = pd.to_datetime(path_date23_mapped).astype("int64")
daily_grid = pd.date_range(full_start, full_end, freq="1D")
daily_num = daily_grid.astype("int64")

# interpolate mapped dates for each daily 2022 date
mapped_num_dense = np.interp(daily_num, path_date22_num, path_date23_num)
mapped_dates_dense = pd.to_datetime(mapped_num_dense)

# interpolate values on dense grids
val22_dense = np.interp(daily_num, path_val22.index.astype("int64"), path_val22.values)
val23_dense = np.interp(mapped_num_dense, path_val23.index.astype("int64"), path_val23.values)

stride = 2  # draw a line every 2 days
for i in range(0, len(daily_grid), stride):
    ax_full.plot([daily_grid[i], mapped_dates_dense[i]],
                 [val22_dense[i], val23_dense[i]],
                 color="gray", alpha=0.18, lw=0.6)
if DEBUG:
    print(f"[DEBUG] dense full-path lines drawn≈{len(range(0, len(daily_grid), stride))}, stride={stride}, grid_len={len(daily_grid)}")

ax_full.set_xlim(full_start, full_end)
ax_full.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax_full.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax_full.set_ylabel('EVI', fontweight='bold')
ax_full.set_xlabel('Month (Mar–Jul)')
ax_full.tick_params(axis='both', labelsize=16)
ax_full.legend(loc="lower left", frameon=True,
               prop={'family': 'Times New Roman', 'size': 9.5, 'weight': 'bold'})

plt.tight_layout()
plt.savefig(out_fig_full_lines_fp, dpi=450, bbox_inches="tight")
plt.savefig(out_fig_full_lines_fp_svg, format="svg", bbox_inches="tight")
if SHOW_FIG:
    plt.show()
else:
    plt.close(fig_full)
print(f"✅ Saved full-season DTW path figure to: {out_fig_full_lines_fp} and {out_fig_full_lines_fp_svg}")

# ---------- Separate Key Period Zoom Figure (expanded to Apr-May) ----------
zoom_start_date = pd.Timestamp("2022-04-17")
zoom_end_date = pd.Timestamp("2022-04-30")

fig_zoom, ax_zoom = plt.subplots(1, 1, figsize=(6, 4))

# Smooth curves in zoom window
zoom_daily22 = spline_smooth(dates22, s22, zoom_start_date, zoom_end_date)
zoom_daily23 = spline_smooth(aligned_df['date22_mapped'], aligned_df['EVI_2023_mapped'], zoom_start_date, zoom_end_date)

ax_zoom.plot(zoom_daily22.index, zoom_daily22.values, color=COLOR_2022, lw=2.0)
ax_zoom.plot(zoom_daily23.index, zoom_daily23.values, color=COLOR_2023, lw=2.5)

# Correspondence lines in zoom window — 强制输出约 20 条示意线（线性均匀采样）
zoom_grid = pd.date_range(zoom_start_date, zoom_end_date, freq="1D")
grid_num = zoom_grid.astype("int64")

path_val22 = spline_smooth(dates22, s22, zoom_start_date, zoom_end_date)
path_val23 = spline_smooth(aligned_df['date22_mapped'], aligned_df['EVI_2023_mapped'], zoom_start_date, zoom_end_date)

# 均匀选取 ~20 个点
target_lines = 20
stride = max(1, len(zoom_grid) // target_lines)
for i in range(0, len(zoom_grid), stride):
    d22 = zoom_grid[i]
    # 将 22 侧的日期通过 DTW 路径按比例映射到 23 侧：这里用线性比例映射整个窗口作为示意
    ratio = (i + 1) / len(zoom_grid)
    d23 = zoom_grid[0] + (zoom_grid[-1] - zoom_grid[0]) * ratio
    v22 = np.interp(d22.value, path_val22.index.view("int64"), path_val22.values)
    v23 = np.interp(d23.value, path_val23.index.view("int64"), path_val23.values)
    ax_zoom.plot([d22, d23], [v22, v23], color="gray", alpha=0.35, lw=0.8)

ax_zoom.set_xlim(zoom_start_date, zoom_end_date)
ax_zoom.grid(False)
ax_zoom.set_xticks([])
ax_zoom.set_yticks([])
ax_zoom.set_xlabel('')
ax_zoom.set_ylabel('')
ax_zoom.set_title('')
for spine in ax_zoom.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig(out_fig_zoom_fp, dpi=450, bbox_inches="tight")
plt.savefig(out_fig_zoom_fp_svg, format="svg", bbox_inches="tight")
if SHOW_FIG:
    plt.show()
else:
    plt.close(fig_zoom)
print(f"✅ Saved zoomed key-period figure to: {out_fig_zoom_fp} and {out_fig_zoom_fp_svg}")
