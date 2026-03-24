# -*- coding: utf-8 -*-
"""
快速生成 metrics_pix_22.csv 和 metrics_pix_23.csv 缓存文件
用于后续的绘图脚本使用
"""
import os, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
from osgeo import gdal
import geopandas as gpd

# 添加模块路径
PROJ_ROOT = Path.cwd()
MODULE_DIR = PROJ_ROOT / 'FY_p2'
sys.path.append(str(MODULE_DIR))
import plot_metrics as pm

def main():
    print("=" * 70)
    print("生成指标缓存文件 (metrics_pix_22.csv / metrics_pix_23.csv)")
    print("=" * 70)
    
    # 输出目录
    OUT_DIR = Path(pm.OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    CACHE_22 = OUT_DIR / 'metrics_pix_22.csv'
    CACHE_23 = OUT_DIR / 'metrics_pix_23.csv'
    
    # 检查是否已存在
    if CACHE_22.exists() and CACHE_23.exists():
        print(f"\n[✓] 缓存文件已存在:")
        print(f"  - {CACHE_22}")
        print(f"  - {CACHE_23}")
        
        # 显示文件信息
        df22 = pd.read_csv(CACHE_22)
        df23 = pd.read_csv(CACHE_23)
        print(f"\n[INFO] 2022数据: {len(df22)} 像元")
        print(f"[INFO] 2023数据: {len(df23)} 像元")
        print(f"[INFO] 列名: {list(df22.columns)}")
        
        response = input("\n是否重新生成? (y/n): ")
        if response.lower() != 'y':
            print("[EXIT] 保留现有缓存")
            return
    
    print("\n[INFO] 开始生成缓存...")
    
    # 1. 获取参考栅格
    sample_candidates = glob.glob(str(Path(pm.BRDF_DIR_2022) / '*.tif')) + \
                        glob.glob(str(Path(pm.BRDF_DIR_2023) / '*.tif'))
    if not sample_candidates:
        raise FileNotFoundError("未找到BRDF目录中的.tif文件")
    
    sample_fp = sample_candidates[0]
    print(f"[INFO] 使用参考栅格: {os.path.basename(sample_fp)}")
    
    # 2. 获取栅格信息
    gt, proj, nx, ny = pm.get_raster_info(sample_fp)
    print(f"[INFO] 栅格大小: {nx} x {ny}")
    
    # 3. 重采样小麦掩膜
    print("[INFO] 重采样小麦掩膜...")
    mask_arr = pm.resample_mask_to_ref(pm.WHEAT_MASK_TIF, sample_fp)
    print(f"[INFO] 有效像元数: {np.sum(mask_arr)}")
    
    # 4. 构建每日每小时NIRv像元值 - 2022
    print("\n[INFO] 处理2022年数据...")
    print(f"  日期: {pm.DATES_2022}")
    print(f"  时间: {pm.HOURS}")
    df_pixels_22 = pm.build_daily_hourly_pixel_nirv(
        pm.BRDF_DIR_2022, pm.DATES_2022, pm.HOURS, (ny, nx)
    )
    print(f"  构建完成: {len(df_pixels_22)} 记录")
    
    # 5. 构建每日每小时NIRv像元值 - 2023
    print("\n[INFO] 处理2023年数据...")
    print(f"  日期: {pm.DATES_2023}")
    print(f"  时间: {pm.HOURS}")
    df_pixels_23 = pm.build_daily_hourly_pixel_nirv(
        pm.BRDF_DIR_2023, pm.DATES_2023, pm.HOURS, (ny, nx)
    )
    print(f"  构建完成: {len(df_pixels_23)} 记录")
    
    # 6. 加载研究区边界
    print("\n[INFO] 加载研究区边界...")
    gdf_hhh = gpd.read_file(pm.HHH_SHP_PATH)
    gdf_hhh_4326 = gdf_hhh.to_crs(epsg=4326)
    print(f"  边界范围: {gdf_hhh_4326.total_bounds}")
    
    # 7. 处理像元值（应用掩膜和范围）- 2022
    print("\n[INFO] 应用掩膜和边界筛选 (2022)...")
    processed_pix_22 = pm.build_processed_pixel_values_region(
        df_pixels_22, mask_arr, gt, proj, nx, ny, gdf_hhh_4326, pm.HOURS
    )
    print(f"  处理后: {len(processed_pix_22)} 记录")
    
    # 8. 处理像元值（应用掩膜和范围）- 2023
    print("\n[INFO] 应用掩膜和边界筛选 (2023)...")
    processed_pix_23 = pm.build_processed_pixel_values_region(
        df_pixels_23, mask_arr, gt, proj, nx, ny, gdf_hhh_4326, pm.HOURS
    )
    print(f"  处理后: {len(processed_pix_23)} 记录")
    
    # 9. 计算日变化指标 - 2022
    print("\n[INFO] 计算指标 (2022)...")
    metrics_pix_22 = pm.metrics_from_processed_pixels(processed_pix_22, pm.HOURS)
    print(f"  完成: {len(metrics_pix_22)} 像元")
    print(f"  指标列: {list(metrics_pix_22.columns)}")
    
    # 10. 计算日变化指标 - 2023
    print("\n[INFO] 计算指标 (2023)...")
    metrics_pix_23 = pm.metrics_from_processed_pixels(processed_pix_23, pm.HOURS)
    print(f"  完成: {len(metrics_pix_23)} 像元")
    print(f"  指标列: {list(metrics_pix_23.columns)}")
    
    # 11. 保存为CSV
    print("\n[INFO] 保存缓存文件...")
    metrics_pix_22.to_csv(CACHE_22, index=False)
    metrics_pix_23.to_csv(CACHE_23, index=False)
    
    print("\n" + "=" * 70)
    print("[✓] 缓存生成完成！")
    print("=" * 70)
    print(f"\n输出文件:")
    print(f"  - {CACHE_22} ({len(metrics_pix_22)} 像元)")
    print(f"  - {CACHE_23} ({len(metrics_pix_23)} 像元)")
    print(f"\n现在可以运行 Fig_Metrics_Spiral_Maps.ipynb 了！")

if __name__ == '__main__':
    main()





