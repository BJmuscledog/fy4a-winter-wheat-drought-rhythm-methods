"""
FY4 BRDF 校正脚本 - 参考 BRDF1.py 方法改进版
使用 FY4 多时相数据拟合参数，参考几何采用视角校正方法
"""
import os
import glob
import numpy as np
import pandas as pd

from write_tif import write_multiplebands_tiff
from read_data_func import readtiff
from get_kvol_geo_func import rou_li_sparse_k_vol_geo

from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler

# 研究区左上角经纬度（黄淮海平原）
F_LU_LO = 110.35571015  # 经度
F_LU_LA = 42.6154530898  # 纬度


def brdf_fy_func_v2(fy_dir, angle_dir, output_dir,
                     runnumber, runnumber_id, suz_id,
                     xRes, yRes,
                     f_lulo, f_lula):
    """
    FY4 BRDF 校正函数 V2 - 改进的参考几何方法
    
    改进点：
    1. 参考几何使用视角校正方法：Kvol_o, Kgeo_o = rou_li_sparse_k_vol_geo(saa, 0, sua, suz_12)
       - 只将卫星天顶角设为 0（星下点观测）
       - 保留实际的卫星方位角和太阳方位角
       - 使用参考时刻的太阳天顶角
    2. 统一使用 a=1, b=1 参数（稀疏植被/农作物）
    3. 添加除零保护和异常值检测
    """
    # 列举影像与角度文件（按文件名排序以保证对应）
    angle_list = sorted(glob.glob(os.path.join(angle_dir, '*.tif')))
    path_list  = sorted(glob.glob(os.path.join(fy_dir,   '*.tif')))
    fy_band    = [1, 2, 3]

    # 参考太阳天顶角（通常选择正午时刻）
    suz_12 = readtiff(angle_list[5 * suz_id + 4], 1)

    # BRDF 模型参数（统一使用，确保拟合和应用一致）
    A_PARAM = 1  # h/b 比值
    B_PARAM = 1  # b/r 比值（稀疏植被）
    
    print(f"\n{'='*70}")
    print(f"BRDF 校正 V2 - 日期索引: {runnumber_id}")
    print(f"Li-Sparse 模型参数: a={A_PARAM}, b={B_PARAM}")
    print(f"参考几何方法: 视角校正（保留方位角，星下点观测）")
    print(f"处理波段: {fy_band}")
    print(f"{'='*70}\n")
    
    # ========================================================================
    # 第一部分：拟合 BRDF 参数
    # ========================================================================
    parameter_dfs = pd.DataFrame()
    
    for bandid in fy_band:
        final_df = pd.DataFrame()
        
        # 收集一天内所有时相的数据
        for n in range(runnumber * runnumber_id,
                       runnumber * (runnumber_id + 1)):
            # 1. 读取角度
            saa = readtiff(angle_list[n*5 + 0], 1)
            saz = readtiff(angle_list[n*5 + 1], 1)
            sua = readtiff(angle_list[n*5 + 2], 1)
            suz = readtiff(angle_list[n*5 + 4], 1)

            # 2. 计算 Kvol, Kgeo（使用统一参数）
            Kvol, Kgeo = rou_li_sparse_k_vol_geo(saa, saz, sua, suz, a=A_PARAM, b=B_PARAM)

            # 3. 读取大气校正影像
            fy_6s = readtiff(path_list[n], bandid)

            # 4. 裁剪到共同大小
            h = min(fy_6s.shape[0], Kvol.shape[0], Kgeo.shape[0])
            w = min(fy_6s.shape[1], Kvol.shape[1], Kgeo.shape[1])
            if h == 0 or w == 0:
                continue

            fy_crop   = fy_6s[:h, :w]
            Kvol_crop = Kvol[:h, :w]
            Kgeo_crop = Kgeo[:h, :w]

            # 5. 构造 DataFrame 并去 NaN
            df = pd.DataFrame({
                'radiance': fy_crop.ravel(),
                'k_vol':    Kvol_crop.ravel(),
                'k_geo':    Kgeo_crop.ravel()
            }).dropna()

            final_df = pd.concat([final_df, df], ignore_index=True)

        # 6. 回归拟合
        if len(final_df) > 50:
            X = final_df[['k_vol', 'k_geo']].values
            y = final_df['radiance'].values

            scaler = MinMaxScaler().fit(X)
            Xs = scaler.transform(X)
            Xtr, Xte, Ytr, Yte = train_test_split(Xs, y, test_size=0.3, random_state=42)

            reg = LR().fit(Xtr, Ytr)
            f_vol, f_geo = reg.coef_.flatten()
            f_iso        = reg.intercept_

            neg_mse = cross_val_score(
                reg, Xs, y,
                scoring='neg_mean_squared_error',
                cv=10
            ).mean()
            rmse = np.sqrt(-neg_mse)

            y_pred = reg.predict(Xte)
            r2     = np.corrcoef(Yte, y_pred)[0,1]**2

            parameter_dfs = pd.concat([
                parameter_dfs,
                pd.DataFrame([{
                    'bandid':        bandid,
                    'f_vol':         f_vol,
                    'f_geo':         f_geo,
                    'f_iso':         f_iso,
                    'RMSE_CrossVal': rmse,
                    'R2_CrossVal':   r2
                }])
            ], ignore_index=True)
            
            # 输出拟合质量
            if r2 < 0.5:
                print(f"[Warning] 波段 {bandid} 拟合质量较差: R²={r2:.3f}, RMSE={rmse:.4f}")
            else:
                print(f"[Info] 波段 {bandid} 拟合完成: R²={r2:.3f}, RMSE={rmse:.4f}")
        else:
            print(f"[Warn] 波段 {bandid} 样本量不足 ({len(final_df)} 样本)，跳过")

    if len(parameter_dfs) < len(fy_band):
        print(f"\n[Error] 参数拟合失败！只拟合了 {len(parameter_dfs)} 个波段，需要 {len(fy_band)} 个")
        return

    # ========================================================================
    # 第二部分：应用 BRDF 校正
    # ========================================================================
    print(f"\n开始应用 BRDF 校正...")
    
    for n in range(runnumber * runnumber_id,
                   runnumber * (runnumber_id + 1)):
        outband = []
        
        # 1. 读取当前时相的角度信息
        saa = readtiff(angle_list[n*5 + 0], 1)
        saz = readtiff(angle_list[n*5 + 1], 1)
        sua = readtiff(angle_list[n*5 + 2], 1)
        suz = readtiff(angle_list[n*5 + 4], 1)

        # 2. 计算当前观测几何下的核函数（使用统一参数）
        Kvol, Kgeo = rou_li_sparse_k_vol_geo(saa, saz, sua, suz, a=A_PARAM, b=B_PARAM)

        # 3. 计算参考几何下的核函数
        # 【关键改进】视角校正方法：只将卫星天顶角设为0，保留方位角
        # 参考 BRDF1.py 的方法
        Kvol_o, Kgeo_o = rou_li_sparse_k_vol_geo(
            saa,      # 保留实际卫星方位角
            0,        # 星下点观测（卫星天顶角=0）
            sua,      # 保留实际太阳方位角
            suz_12,   # 参考时刻太阳天顶角（正午）
            a=A_PARAM, 
            b=B_PARAM
        )

        # 4. 对每个波段应用 BRDF 校正
        for bandid in fy_band:
            fy_6s = readtiff(path_list[n], bandid)
            h, w  = fy_6s.shape

            # 裁剪核函数到影像大小
            Kv  = Kvol[:h, :w]
            Kg  = Kgeo[:h, :w]
            Kv_o= Kvol_o[:h, :w]
            Kg_o= Kgeo_o[:h, :w]

            # 获取该波段的拟合参数
            row = parameter_dfs[parameter_dfs.bandid == bandid].iloc[0]
            f_vol = row.f_vol
            f_geo = row.f_geo
            f_iso = row.f_iso

            # 5. 计算模拟反射率
            fy_current = f_iso + f_vol * Kv + f_geo * Kg      # 当前几何
            fy_ref     = f_iso + f_vol * Kv_o + f_geo * Kg_o  # 参考几何（星下点）

            # 6. 计算归一化因子（ANTF）
            # 添加除零保护
            epsilon = 1e-6
            fy_safe = np.where(np.abs(fy_current) < epsilon, epsilon, fy_current)
            antf = fy_ref / fy_safe
            
            # 异常值保护：限制 ANTF 的合理范围
            # 通常 BRDF 校正因子在 0.5-2.0 之间
            antf = np.clip(antf, 0.1, 10.0)

            # 7. 应用 BRDF 校正
            brdf_corrected = fy_6s * antf
            
            # NaN 检查
            nan_count = np.isnan(brdf_corrected).sum()
            total_pixels = brdf_corrected.size
            if nan_count > 0:
                nan_percent = nan_count / total_pixels * 100
                print(f"  [Warning] 影像 {n}, 波段 {bandid}: {nan_count} 个 NaN ({nan_percent:.2f}%)")
            
            outband.append(brdf_corrected)

        # 8. 写出 GeoTIFF
        out_name = os.path.basename(path_list[n])
        out_fp   = os.path.join(output_dir, 'Brdf_v2_' + out_name)
        write_multiplebands_tiff(
            out_fp, outband,
            w, h,
            lon_res = xRes,
            lat_res = yRes,
            f_lulo  = f_lulo,
            f_lula  = f_lula
        )
        print(f"  ✓ 处理完成: {out_name}")

    print(f"\n{'='*70}")
    print(f"日期索引 {runnumber_id} 处理完成！")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    fy_dir     = r'F:\风云数据\Fy_p2_data\2023_0423_week\Atmospheric_correction_hhh_n'
    angle_dir  = r'F:\风云数据\Fy_p2_data\2023_0423_week\Angledata_hhh'
    output_dir = r'F:\风云数据\Fy_p2_data\2023_0423_week\Brdf_hhh_v2'
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    xRes, yRes = 0.00897, 0.00897
    days       = 3  # 测试用，处理前3天

    print("\n" + "="*70)
    print("FY4 BRDF 校正 V2 - 改进的参考几何方法")
    print("="*70)
    print(f"输入目录: {fy_dir}")
    print(f"角度目录: {angle_dir}")
    print(f"输出目录: {output_dir}")
    print(f"处理天数: {days}")
    print("="*70 + "\n")

    for i in range(days):
        brdf_fy_func_v2(
            fy_dir, angle_dir, output_dir,
            runnumber    = 9,
            runnumber_id = i,
            suz_id       = 3 + 9*i,
            xRes         = xRes,
            yRes         = yRes,
            f_lulo       = F_LU_LO,
            f_lula       = F_LU_LA
        )
    
    print("\n" + "="*70)
    print("全部处理完成！")
    print("="*70)





