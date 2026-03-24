"""
基于已有的相关性CSV文件，绘制4个子图的相关性矩阵：
- 左上：干旱年旱地 (2022_Drought_DryLand)
- 左下：干旱年非旱地 (2022_Drought_WetLand)
- 右上：非干旱年旱地 (2023_Wet_DryLand)
- 右下：非干旱年非旱地 (2023_Wet_WetLand)

数据源：
- 环境因子之间相关性：env_env_corr.csv
- 动态环境因子与节律指标相关性：selected.csv
- 静态环境因子（Elev, SOC）与节律指标相关性：env_corr_by_rhythm.csv
- 节律指标之间相关性：从原始特征表计算

环境因子显示名称：使用带时间尺度的名称（如PPT_7d, PAR_hmean_9_16等）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr
from project_config import CORRELATION_CLASS_SCHEME, XGB_OUTPUT_DIR, get_class_scheme

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# 节律指标
RHYTHM_VARS = ["target_nirv", "target_t_peak", "target_mdi", "centroid_shift"]
RHYTHM_LABEL = {
    "target_nirv": "NIRv",
    "target_t_peak": "T_peak",
    "target_mdi": "MDI",
    "centroid_shift": "C_shift",
}

# 环境因子及时间尺度（与XGBoost_shap_GEE-Adapted.py保持一致）
ENV_CHOICES = [
    ("VPD", "vpd_hmean_9_16"),
    ("LST", "lst_3d"),
    ("Temp", "temp_hmean_9_16"),
    ("SM", "sm_mean_last_7d"),
    ("PPT", "precip_sum_7d"),
    ("PAR", "par_hmean_9_16"),
    ("Elev", "elevation"),
    ("SOC", "soc"),
]

ENV_KEYS = [name for name, _ in ENV_CHOICES]
ENV_TO_TIMESCALE = {name: timescale for name, timescale in ENV_CHOICES}

# 变量顺序
VAR_ORDER = RHYTHM_VARS + ENV_KEYS
CLASS_SCHEME = get_class_scheme(CORRELATION_CLASS_SCHEME)
DRY_CLASSES = set(CLASS_SCHEME["drought"])
WET_CLASSES = set(CLASS_SCHEME["wet"])

# 场景映射
SCENARIO_MAP = {
    "2022_dry": ("2022_0423_dry", "2022年旱地"),
    "2022_wet": ("2022_0423_wet", "2022年非旱地"),
    "2023_dry": ("2023_0425_dry", "2023年旱地"),
    "2023_wet": ("2023_0425_wet", "2023年非旱地"),
}

# 目标变量映射
TARGET_VAR_MAP = {
    "C_shift": "centroid_shift",
    "MDI": "target_mdi",
    "NIRv": "target_nirv",
    "T_peak": "target_t_peak",
}


def display_name(var_key):
    """返回变量的显示名称"""
    # 节律指标
    if var_key in RHYTHM_LABEL:
        return RHYTHM_LABEL[var_key]
    # 静态环境因子（Elev, SOC）保持原名称
    elif var_key in ["Elev", "SOC"]:
        return var_key
    # 动态环境因子使用时间尺度名称
    elif var_key in ENV_TO_TIMESCALE:
        return ENV_TO_TIMESCALE[var_key]
    return var_key


def load_env_env_corr(scenario_key):
    """从env_env_corr.csv读取环境因子之间的相关性"""
    csv_path = os.path.join("fig_corr", "env_env_corr.csv")
    if not os.path.exists(csv_path):
        return {}
    
    year_tag, _ = SCENARIO_MAP[scenario_key]
    
    try:
        df = pd.read_csv(csv_path)
        df = df[df['year'] == year_tag].copy()
        df = df[df['rho'].notna()].copy()
        df = df[df['pval'].notna()].copy()
        
        # 环境因子名称映射
        env_name_mapping = {
            'lst_3d': 'LST', 'lst_mean_last_3d': 'LST', 'lst_1d': 'LST', 'lst': 'LST',
            'sm_mean_last_7d': 'SM', 'sm_7d': 'SM', 'sm_mean_7d': 'SM', 'sm_mean_last_30d': 'SM', 'sm': 'SM',
            'elevation': 'Elev', 'elev': 'Elev', 'dem': 'Elev', 'elevation_m': 'Elev', 'altitude': 'Elev',
            'soc': 'SOC', 'soil_organic_carbon': 'SOC', 'organic_carbon': 'SOC', 'soil_carbon': 'SOC',
            'vpd_hmean_9_16': 'VPD', 'vpd_mean_last_3d': 'VPD', 'vpd_3d': 'VPD', 'vpd': 'VPD',
            'temp_hmean_9_16': 'Temp', 'temp_mean_last_3d': 'Temp', 'temp_3d': 'Temp', 'temp': 'Temp',
            'precip_sum_7d': 'PPT', 'tp_7d': 'PPT', 'tp_3d': 'PPT', 'tp_1d': 'PPT', 'precip': 'PPT',
            'par_hmean_9_16': 'PAR', 'par_mean_last_3d': 'PAR', 'par_3d': 'PAR', 'par': 'PAR',
        }
        
        env_env_corr = {}
        for _, row in df.iterrows():
            env1_csv = str(row['env1']).strip().lower()
            env2_csv = str(row['env2']).strip().lower()
            env1_mapped = env_name_mapping.get(env1_csv)
            env2_mapped = env_name_mapping.get(env2_csv)
            
            if env1_mapped and env2_mapped and env1_mapped != env2_mapped:
                rho = float(row['rho'])
                pval = float(row['pval'])
                env_env_corr[(env1_mapped, env2_mapped)] = (rho, pval)
                env_env_corr[(env2_mapped, env1_mapped)] = (rho, pval)
        
        return env_env_corr
    except Exception as e:
        print(f"[WARN] 无法读取环境因子之间相关性: {e}")
        return {}


def load_env_rhythm_corr_dynamic(scenario_key):
    """从selected.csv读取动态环境因子与节律指标的相关性"""
    csv_path = os.path.join("fig_corr", "selected.csv")
    if not os.path.exists(csv_path):
        return {}
    
    year_tag, _ = SCENARIO_MAP[scenario_key]
    
    try:
        df = pd.read_csv(csv_path)
        df = df[df['year'] == year_tag].copy()
        df = df[df['Correlation (ρ)'].notna()].copy()
        df = df[df['Best Time Scale'].notna()].copy()
        
        # 处理Target Variable列
        last_target = None
        for idx in df.index:
            target_val = df.loc[idx, 'Target Variable']
            if pd.isna(target_val) or target_val == '':
                if last_target is not None:
                    df.loc[idx, 'Target Variable'] = last_target
            else:
                last_target = target_val
        
        category_to_env = {
            'Temperature': 'Temp',
            'PAR': 'PAR',
            'VPD': 'VPD',
            'LST': 'LST',
            'Soil Moisture': 'SM',
            'Precip (Cum)': 'PPT',
            'Total Precip': 'PPT',
        }
        
        df['base_env'] = df['Env Factor Category'].map(category_to_env)
        dynamic_base_envs = ['VPD', 'LST', 'Temp', 'SM', 'PPT', 'PAR']
        df = df[df['base_env'].isin(dynamic_base_envs)].copy()
        
        env_rhythm_corr = {}
        for _, row in df.iterrows():
            target_var_csv = str(row['Target Variable']).strip()
            base_env = row['base_env']
            rho = float(row['Correlation (ρ)'])
            rhythm_var = TARGET_VAR_MAP.get(target_var_csv)
            
            if rhythm_var and base_env in ENV_KEYS:
                abs_rho = abs(rho)
                p_val = 0.01 if abs_rho > 0.3 else (0.03 if abs_rho > 0.2 else 0.1)
                env_rhythm_corr[(rhythm_var, base_env)] = (rho, p_val)
        
        return env_rhythm_corr
    except Exception as e:
        print(f"[WARN] 无法读取动态环境因子相关性: {e}")
        return {}


def load_env_rhythm_corr_static(scenario_key):
    """从env_corr_by_rhythm.csv读取静态环境因子（Elev, SOC）与节律指标的相关性"""
    csv_path = os.path.join("fig_corr", "env_corr_by_rhythm.csv")
    if not os.path.exists(csv_path):
        return {}
    
    year_tag, _ = SCENARIO_MAP[scenario_key]
    
    try:
        df = pd.read_csv(csv_path)
        df = df[df['year'] == year_tag].copy()
        df = df[df['env'].isin(['elevation', 'soc'])].copy()
        df = df[df['rho'].notna()].copy()
        df = df[df['pval'].notna()].copy()
        
        env_rhythm_corr = {}
        for _, row in df.iterrows():
            env_name_csv = str(row['env']).strip().lower()
            rhythm_var_csv = str(row['rhythm']).strip()
            rho = float(row['rho'])
            pval = float(row['pval'])
            
            if env_name_csv == 'elevation':
                env_var = 'Elev'
            elif env_name_csv == 'soc':
                env_var = 'SOC'
            else:
                continue
            
            rhythm_var = TARGET_VAR_MAP.get(rhythm_var_csv)
            if rhythm_var is None and rhythm_var_csv in RHYTHM_VARS:
                rhythm_var = rhythm_var_csv
            
            if rhythm_var and env_var in ENV_KEYS:
                env_rhythm_corr[(rhythm_var, env_var)] = (rho, pval)
        
        return env_rhythm_corr
    except Exception as e:
        print(f"[WARN] 无法读取静态环境因子相关性: {e}")
        return {}


def load_rhythm_rhythm_corr(scenario_key):
    """从原始特征表计算节律指标之间的相关性"""
    base_dir = os.fspath(XGB_OUTPUT_DIR)
    year_tag, _ = SCENARIO_MAP[scenario_key]
    
    if "2022" in year_tag:
        fp = os.path.join(base_dir, "2022_Drought_Results", "pixel_feature_table_2022_Drought.csv")
    else:
        fp = os.path.join(base_dir, "2023_Wet_Results", "pixel_feature_table_2023_Wet.csv")
    
    if not os.path.exists(fp):
        return {}
    
    try:
        df = pd.read_csv(fp)
        df.columns = [c.lower() for c in df.columns]
        
        # 根据场景筛选数据
        if "dry" in scenario_key:
            df = df[df['drought_class'].isin(DRY_CLASSES)].copy()
        else:
            df = df[df['drought_class'].isin(WET_CLASSES)].copy()
        
        rhythm_rhythm_corr = {}
        for i, vi in enumerate(RHYTHM_VARS):
            for j, vj in enumerate(RHYTHM_VARS):
                if i >= j:
                    continue
                if vi not in df.columns or vj not in df.columns:
                    continue
                
                subset = df[[vi, vj]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(subset) < 10:
                    continue
                
                try:
                    rho, pval = spearmanr(subset[vi].to_numpy(), subset[vj].to_numpy())
                    if np.isfinite(rho) and np.isfinite(pval):
                        rhythm_rhythm_corr[(vi, vj)] = (rho, pval)
                        rhythm_rhythm_corr[(vj, vi)] = (rho, pval)
                except:
                    continue
        
        return rhythm_rhythm_corr
    except Exception as e:
        print(f"[WARN] 无法计算节律指标之间相关性: {e}")
        return {}


def build_correlation_matrix(scenario_key):
    """构建完整的相关性矩阵"""
    r_mat = pd.DataFrame(np.nan, index=VAR_ORDER, columns=VAR_ORDER, dtype=float)
    p_mat = pd.DataFrame(np.nan, index=VAR_ORDER, columns=VAR_ORDER, dtype=float)
    
    # 1. 节律指标之间的相关性
    rhythm_rhythm = load_rhythm_rhythm_corr(scenario_key)
    for (v1, v2), (rho, pval) in rhythm_rhythm.items():
        if v1 in VAR_ORDER and v2 in VAR_ORDER:
            r_mat.loc[v1, v2] = rho
            r_mat.loc[v2, v1] = rho
            p_mat.loc[v1, v2] = pval
            p_mat.loc[v2, v1] = pval
    
    # 2. 环境因子之间的相关性
    env_env = load_env_env_corr(scenario_key)
    for (env1, env2), (rho, pval) in env_env.items():
        if env1 in VAR_ORDER and env2 in VAR_ORDER:
            r_mat.loc[env1, env2] = rho
            r_mat.loc[env2, env1] = rho
            p_mat.loc[env1, env2] = pval
            p_mat.loc[env2, env1] = pval
    
    # 3. 环境因子与节律指标之间的相关性
    env_rhythm_dynamic = load_env_rhythm_corr_dynamic(scenario_key)
    env_rhythm_static = load_env_rhythm_corr_static(scenario_key)
    env_rhythm = {**env_rhythm_dynamic, **env_rhythm_static}
    
    for (rhythm_var, env_var), (rho, pval) in env_rhythm.items():
        if rhythm_var in VAR_ORDER and env_var in VAR_ORDER:
            r_mat.loc[rhythm_var, env_var] = rho
            r_mat.loc[env_var, rhythm_var] = rho
            p_mat.loc[rhythm_var, env_var] = pval
            p_mat.loc[env_var, rhythm_var] = pval
    
    return r_mat, p_mat


def plot_matrix(ax, r_mat, p_mat, scenario_key, vmax=0.8, alpha=0.05):
    """在指定axes上绘制相关性矩阵"""
    vars_list = r_mat.index.tolist()
    n = len(vars_list)
    
    # 根据场景选择颜色方案
    if "2022" in scenario_key and "dry" in scenario_key:
        cmap = plt.colormaps.get_cmap("Oranges")
    elif "2022" in scenario_key and "wet" in scenario_key:
        cmap = plt.colormaps.get_cmap("YlOrBr")
    elif "2023" in scenario_key and "dry" in scenario_key:
        cmap = plt.colormaps.get_cmap("Blues")
    else:  # 2023_wet
        cmap = plt.colormaps.get_cmap("Blues")
    
    norm = colors.Normalize(vmin=0, vmax=vmax)
    soft_power = 2.0
    inner_ratio = 0.60
    max_alpha = 1.0
    num_fs = 15
    arrow_fs = 26
    color_power = 0.7
    
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", which="both",
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False,
                   length=0, width=0)
    ax.grid(False)
    
    for spine_name, spine in ax.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor("0.55")
    
    blue_base = np.array(colors.to_rgb("#3b6ea5"))
    orange_base = np.array(colors.to_rgb("#f5a623"))
    
    for i, vi in enumerate(vars_list):
        for j, vj in enumerate(vars_list):
            rho = r_mat.loc[vi, vj]
            pval = p_mat.loc[vi, vj]
            
            # 对角线：变量名
            if i == j:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor="white", edgecolor="0.55", linewidth=1.1, zorder=1))
                disp_name = display_name(vi)
                ax.text(j, i, disp_name, ha="center", va="center",
                        fontsize=16, fontweight="bold", color="black")
                continue
            
            if np.isnan(rho):
                continue
            
            if j > i:
                # 右上三角：色块
                color = cmap(norm(abs(rho)))
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor="white", edgecolor="0.55", linewidth=1.0, zorder=1))
                grid = np.linspace(-0.5, 0.5, 41)
                xx, yy = np.meshgrid(grid, grid)
                dist = np.maximum(np.abs(xx), np.abs(yy))
                inner = inner_ratio * 0.5
                alpha_mask = np.where(
                    dist <= inner,
                    1.0,
                    np.clip((0.5 - dist) / (0.5 - inner), 0, 1) ** soft_power,
                )
                alpha_mask = alpha_mask * max_alpha
                rgba = np.zeros((41, 41, 4))
                rgba[..., 0:3] = colors.to_rgb(color)
                rgba[..., 3] = alpha_mask
                ax.imshow(
                    rgba,
                    origin="lower",
                    extent=(j - 0.5, j + 0.5, i - 0.5, i + 0.5),
                    zorder=2,
                    interpolation="bilinear",
                )
            else:
                # 左下三角：数值 + 箭头
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor="white", edgecolor="0.55", linewidth=1.0, zorder=1))
                shade = np.clip((abs(rho) / vmax) ** color_power, 0, 1)
                num_color = colors.to_hex(blue_base * (0.5 + 0.5 * shade))
                ax.text(j, i - 0.05, f"{abs(rho):.2f}", ha="center", va="center",
                        fontsize=num_fs, color=num_color, fontweight="bold", zorder=2)
                arr_color_sig = colors.to_hex(orange_base * (0.5 + 0.5 * shade))
                arr_color_nonsig = "#b0b0b0"
                is_sig = (not np.isnan(pval)) and (pval < alpha)
                arr_color = arr_color_sig if is_sig else arr_color_nonsig
                alpha_arr = 0.95 if is_sig else 0.70
                arrow = "↑" if rho > 0 else "↓"
                ax.text(
                    j,
                    i + 0.20,
                    arrow,
                    ha="center",
                    va="center",
                    fontsize=arrow_fs,
                    color=arr_color,
                    fontweight="bold",
                    zorder=2,
                    alpha=alpha_arr,
                )
    
    # 添加颜色条（只在第一个子图添加，使用统一的colormap）
    if scenario_key == "2022_dry":
        # 使用统一的colormap（Oranges）作为参考
        sm = cm.ScalarMappable(cmap=plt.colormaps.get_cmap("Oranges"), norm=norm)
        sm.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.12)
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label("|Spearman ρ|", fontsize=12, fontweight="bold")
        cbar.ax.tick_params(labelsize=12)


def main():
    """主函数：绘制4个子图"""
    os.makedirs("fig_corr", exist_ok=True)
    
    print("[INFO] 加载相关性数据...")
    
    # 构建4个场景的相关性矩阵
    scenarios = ["2022_dry", "2022_wet", "2023_dry", "2023_wet"]
    matrices = {}
    
    for scenario_key in scenarios:
        print(f"  [INFO] 处理 {scenario_key}...")
        r_mat, p_mat = build_correlation_matrix(scenario_key)
        matrices[scenario_key] = (r_mat, p_mat)
    
    # 创建2x2子图
    print("[INFO] 绘制4个子图...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 20), dpi=300)
    
    # 左上：干旱年旱地
    plot_matrix(axes[0, 0], matrices["2022_dry"][0], matrices["2022_dry"][1], "2022_dry")
    axes[0, 0].set_title("2022 Drought - Dry Land", fontsize=14, fontweight="bold", pad=10)
    
    # 左下：干旱年非旱地
    plot_matrix(axes[1, 0], matrices["2022_wet"][0], matrices["2022_wet"][1], "2022_wet")
    axes[1, 0].set_title("2022 Drought - Wet Land", fontsize=14, fontweight="bold", pad=10)
    
    # 右上：非干旱年旱地
    plot_matrix(axes[0, 1], matrices["2023_dry"][0], matrices["2023_dry"][1], "2023_dry")
    axes[0, 1].set_title("2023 Wet - Dry Land", fontsize=14, fontweight="bold", pad=10)
    
    # 右下：非干旱年非旱地
    plot_matrix(axes[1, 1], matrices["2023_wet"][0], matrices["2023_wet"][1], "2023_wet")
    axes[1, 1].set_title("2023 Wet - Wet Land", fontsize=14, fontweight="bold", pad=10)
    
    plt.tight_layout()
    save_path = os.path.join("fig_corr", "env_corr_matrix_4panels.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {save_path}")


if __name__ == "__main__":
    main()
