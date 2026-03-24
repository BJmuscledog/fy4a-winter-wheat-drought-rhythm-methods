# -*- coding: utf-8 -*-
"""
Combined Drought & DTW Figure
合并drought_pic.py和region_dtw_align.py的绘图
上下排列：(a) SPEI-SM时序图 和 (b) EVI DTW对齐图
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec, rcParams
import statsmodels.api as sm
import seaborn as sns
import numpy as np
import os
from scipy.signal import savgol_filter
import dtw
import matplotlib.dates as mdates
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ============== 统一科研绘图格式配置 ==============
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['font.size'] = 6.5
rcParams['axes.labelsize'] = 6.5
rcParams['axes.titlesize'] = 9
rcParams['xtick.labelsize'] = 6.5
rcParams['ytick.labelsize'] = 6.5
rcParams['legend.fontsize'] = 6.5
rcParams['axes.linewidth'] = 0.25
rcParams['lines.linewidth'] = 0.5
rcParams['patch.linewidth'] = 0.25
rcParams['grid.linewidth'] = 0.25
rcParams['xtick.major.width'] = 0.25
rcParams['ytick.major.width'] = 0.25
rcParams['savefig.dpi'] = 300
rcParams['figure.dpi'] = 300

sns.set(style="whitegrid", context="talk", font_scale=0.8)

# ===================== 配色 =====================
COLOR_SPEI = "#1f77b4"
COLOR_SM = "#ff7f0e"
COLOR_2022 = "#a2cffe"  # Deep Blue
COLOR_2023 = "#FFDD00"  # Bright Yellow

# ===================== 文件路径 =====================
csv_path_drought = r"F:\G_disk\FY4\data\Drought_GEE\HHH_Wheat_SPEI_SM_Drought_2022_2023_Optimizedn.csv"
csv_2022_evi = r'F:/G_disk/FY4/data/ancillary_data/dtw/HHH_WinterWheat_MODIS_2022.csv'
csv_2023_evi = r'F:/G_disk/FY4/data/ancillary_data/dtw/HHH_WinterWheat_MODIS_2023.csv'
OUT_DIR = r"F:\FY4\outputs_paper_figs"
os.makedirs(OUT_DIR, exist_ok=True)
out_fig = os.path.join(OUT_DIR, "Fig_Combined_Drought_DTW.png")

# ===================== 数据加载与处理 =====================
# 1. SPEI-SM数据
df_drought = pd.read_csv(csv_path_drought)
df_drought["date"] = pd.to_datetime(df_drought["date"], errors="coerce")
df_drought["year"] = df_drought["year"].astype(int)
df_drought = df_drought[(df_drought["date"].dt.month >= 3) & (df_drought["date"].dt.month <= 6)]
df_drought = df_drought.sort_values(["year", "date"]).copy()

def smooth_series(values, frac=0.15):
    """LOWESS平滑"""
    x_idx = np.arange(len(values))
    return sm.nonparametric.lowess(values, x_idx, frac=frac, return_sorted=False)

# 2. EVI数据
def load_modis(csv_fp):
    df = pd.read_csv(csv_fp)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"].dt.month >= 3) & (df["date"].dt.month <= 6)]
    if df['EVI'].max() > 1.2:
        df['EVI'] *= 0.0001
    df = df.sort_values("date").reset_index(drop=True)
    return df

df22_evi = load_modis(csv_2022_evi)
df23_evi = load_modis(csv_2023_evi)
vals22, vals23 = df22_evi['EVI'].values, df23_evi['EVI'].values
dates22, dates23 = df22_evi["date"].values, df23_evi["date"].values

# Savgol平滑
def smooth_evi(vals, window=5, poly=2):
    n = len(vals)
    wl = min(window, n if n % 2 == 1 else n - 1)
    if wl < 3:
        return vals
    return savgol_filter(vals, wl, poly, mode='nearest')

s22, s23 = smooth_evi(vals22), smooth_evi(vals23)

# DTW对齐 - 使用简单的手动实现，避免库依赖问题
def simple_dtw(x, y):
    """简单的DTW实现，返回对齐路径"""
    n, m = len(x), len(y)
    
    # 初始化累积距离矩阵
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix[0, :] = np.inf
    dtw_matrix[:, 0] = np.inf
    dtw_matrix[0, 0] = 0
    
    # 填充矩阵
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i-1] - y[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # 插入
                                          dtw_matrix[i, j-1],    # 删除
                                          dtw_matrix[i-1, j-1])  # 匹配
    
    # 回溯路径
    i, j = n, m
    path_x, path_y = [i-1], [j-1]
    
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            min_val = min(dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1])
            if dtw_matrix[i-1, j-1] == min_val:
                i -= 1
                j -= 1
            elif dtw_matrix[i-1, j] == min_val:
                i -= 1
            else:
                j -= 1
        path_x.insert(0, i-1)
        path_y.insert(0, j-1)
    
    return np.array(path_x), np.array(path_y)

p22, p23 = simple_dtw(s22, s23)

aligned_vals23 = []
aligned_dates = []
warp_map = {}
for i22, i23 in zip(p22, p23):
    warp_map.setdefault(i23, []).append(i22)
warp_median = {i23: int(np.median(v)) for i23, v in warp_map.items()}

for i23 in sorted(warp_median.keys()):
    aligned_vals23.append(s23[i23])
    aligned_dates.append(dates22[warp_median[i23]])

aligned_df = pd.DataFrame({'date22_mapped': aligned_dates, 'EVI_2023_mapped': aligned_vals23})
aligned_df['date22_mapped'] = pd.to_datetime(aligned_df['date22_mapped'])

# ===================== 绘图 =====================
fig = plt.figure(figsize=(7, 9))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], 
                       hspace=0.35, wspace=0.30, left=0.10, right=0.95, top=0.96, bottom=0.06)

# ========== (a,b) SPEI-SM时序图（2022和2023并排） ==========
years = [2022, 2023]
for idx, year in enumerate(years):
    ax = fig.add_subplot(gs[0, idx])
    label = '(a)' if idx == 0 else '(b)'
    ax.text(-0.15, 1.05, label, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')
    
    dsub = df_drought[df_drought["year"] == year].reset_index(drop=True)
    spei_smooth = smooth_series(dsub["SPEI"])
    sm_smooth = smooth_series(dsub["SM"])
    
    ax.plot(dsub["DOY"], spei_smooth, color=COLOR_SPEI, lw=0.5, label="SPEI")
    ax.axhline(y=-1.0, color=COLOR_SPEI, linestyle="--", lw=0.5, alpha=0.7)
    ax.set_ylabel("SPEI", color=COLOR_SPEI, fontweight='bold')
    ax.tick_params(axis="y", labelcolor=COLOR_SPEI)
    
    ax2 = ax.twinx()
    ax2.plot(dsub["DOY"], sm_smooth, color=COLOR_SM, lw=0.5, label="Soil Moisture")
    ax2.set_ylabel("Soil Moisture (m³/m³)", color=COLOR_SM, fontweight='bold')
    ax2.tick_params(axis="y", labelcolor=COLOR_SM)
    
    # 灰色阴影区域
    if year == 2022:
        ax.axvspan(105, 110, color="gray", alpha=0.2)
    elif year == 2023:
        ax.axvspan(105, 110, color="gray", alpha=0.2)
    
    ax.set_xlim(60, 180)
    ax.set_xticks([60, 90, 120, 150, 180])
    ax.set_xticklabels(["Mar", "Apr", "May", "Jun", "Jul"])
    ax.set_xlabel("Month (DOY)", fontweight='bold')
    ax.set_title(f"{'Dry' if year==2022 else 'Wet'}", fontweight='bold', pad=8)

# ========== (c) EVI DTW对齐图 ==========
ax3 = fig.add_subplot(gs[1, :])
ax3.text(-0.08, 1.05, '(c)', transform=ax3.transAxes, fontsize=9, fontweight='bold', va='top')

# 2022 EVI参考线
ax3.plot(dates22, s22, color=COLOR_2022, lw=0.5, label='2022 EVI (Reference)')

# 灰色阴影
SHADE_DOY_START, SHADE_DOY_END = 105, 110
BASE_YEAR = 2022
shade_start_date = pd.Timestamp(date(BASE_YEAR, 1, 1) + timedelta(days=SHADE_DOY_START - 1))
shade_end_date = pd.Timestamp(date(BASE_YEAR, 1, 1) + timedelta(days=SHADE_DOY_END - 1))
ax3.axvspan(shade_start_date, shade_end_date, color="gray", alpha=0.2)

# 2023 EVI DTW对齐线
ax3.plot(aligned_df['date22_mapped'], aligned_df['EVI_2023_mapped'], 
         color=COLOR_2023, lw=0.5, label='2023 EVI (DTW Aligned)')

ax3.set_xlim(pd.Timestamp("2022-03-01"), pd.Timestamp("2022-07-01"))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax3.set_ylabel('EVI', fontweight='bold')
ax3.set_xlabel('Month (Mar–Jul)', fontweight='bold')
ax3.set_title('Regional MODIS EVI Aligned Time Series (DTW Mapping)', fontweight='bold', pad=8)
ax3.legend(loc="lower left", frameon=False)

# 保存
fig.savefig(out_fig, dpi=300, bbox_inches="tight")
try:
    fig.savefig(out_fig.replace('.png', '.svg'), bbox_inches="tight")
    fig.savefig(out_fig.replace('.png', '.pdf'), bbox_inches="tight")
except Exception:
    pass
plt.close(fig)
print(f"✅ Saved combined figure to: {out_fig}")

