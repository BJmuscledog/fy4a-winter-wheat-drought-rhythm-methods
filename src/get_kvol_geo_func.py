import numpy as np


def rou_li_sparse_k_vol_geo(saa, saz, sua, suz, a=1, b=1):
    """此函数用于计算体散射核及几何散射核，输入各角度矩阵，按照顺序索引
    :parameter a: 函数参数，后期遍历查看效果，None = 1
    :parameter b : 函数参数，遍历查看效果，None = 2"""

    # # 掩膜外区域赋予 空值
    # saa[saa < 1] = np.nan
    # saz[saz < 1] = np.nan
    # sua[sua < 1] = np.nan
    # suz[suz < 1] = np.nan
    # 非空角度改为弧度 radians
    ra_saa = np.radians(saa)
    ra_saz = np.radians(saz)
    ra_sua = np.radians(sua)
    # suz_p = np.full((1252,1377),50)
    ra_suz = np.radians(suz)

    # 计算相对方位角 relative azimuth angle-raa
    ra_raa = np.abs(np.subtract(ra_saa, ra_sua))

    # 计算散射角 Scattering angle
    # 注释角度符号 θ_s:太阳天顶角;θ_v:卫星天顶角；ϕ：相对方位角；ξ：散射角
    # 散射角计算公式为：cos(ξ) = cos(θ_s)*cos(θ_v) + sin(θ_s)*sin(θ_v)*cos(ϕ)
    # 使用Numpy向量级操作直接运算
    ra_sca = np.arccos(np.cos(ra_suz) * np.cos(ra_saz) + np.sin(ra_suz) * np.sin(ra_saz) * np.cos(ra_raa))

    # 计算体散射核volumetric scattering kernel - Kvol和几何光学散射核geometric scattering kernel- Kgeo_s，此处命名对应后面的Modis产品的对应的系数
    # Kvol计算公式：F_1 (θ_s,θ_v,ϕ)=((π/2-ξ)*cos(ξ)+sin(ξ))/(cos(θ_s)θ_s +cos(θ_v))-π/4
    Kvol = ((np.pi / 2 - ra_sca) * np.cos(ra_sca) + np.sin(ra_sca)) / (np.cos(ra_saz) + np.cos(ra_suz)) - np.pi / 4

    # Kgeo计算公式：此处公式较为复杂，涉及三个梯次中间变量，D，t，O，公式依次如下
    # 转换天顶角, 此处a表示b/r
    tr_suz = np.arctan(a*np.tan(ra_suz))
    tr_saz = np.arctan(a*np.tan(ra_saz))
    tr_sca = np.arccos(np.cos(tr_suz) * np.cos(tr_saz) + np.sin(tr_suz) * np.sin(tr_saz) * np.cos(ra_raa))

    # D=(tan^2(θ_s)+tan^2(θ_v)-2 tan(θ_s)tan(θ_v)cos(ϕ))^0.5.
    D = np.sqrt(np.tan(tr_suz) ** 2 + np.tan(tr_saz) ** 2 - 2 * np.tan(tr_suz) * np.tan(tr_saz) * np.cos(ra_raa))
    D_1 = D[150,850]
    # cos(t)=b*(D^2+(tan(θ_s)*tan(θ_v)*sin(ϕ))^2)^0.5/((sec(θ_s)+`sec(θ_v))
    cos_t = b * np.sqrt(D ** 2 + (np.tan(tr_suz) * np.tan(tr_saz) * np.sin(ra_raa)) ** 2) / (
            1 / np.cos(tr_suz) + 1 / np.cos(tr_saz))
    clip_cost = np.clip(cos_t, -1, 1)
    cos_a = cos_t[150,850]
    t = np.where((clip_cost < 1) & (clip_cost > -1), np.arccos(clip_cost), np.nan)

    # O(θ_s,θ_v,ϕ)=1/π*(t-sin(t)cos(t))*(sec(θ_s)+sec(θ_v))
    O = (1 / np.pi) * (t - np.sin(t) * np.cos(t)) * (1 / np.cos(tr_suz) + 1 / np.cos(tr_saz))
    a = O[150,850]

    # 计算Kgeo：F_2 (θ_s,θ_v,ϕ)=O(θ_s,θ_v,ϕ)-sec(θ_s)-sec(θ_v)+1/2*(1+cos(ξ))*sec(θ_s)*sec(θ_v)
    Ksprase = O - 1 / np.cos(tr_suz) - 1 / np.cos(tr_saz) + 0.5 * (1 + np.cos(tr_sca)) * 1 / np.cos(
        tr_suz) * 1 / np.cos(tr_saz)
    return Kvol, Ksprase
def rou_li_dense_k_vol_geo(saa, saz, sua, suz, a=1, b=2):
    """此函数为另一类模型，适用于LAI>4时，用于计算体散射核及几何散射核，输入各角度矩阵，按照顺序索引
    :parameter a: 函数参数，后期遍历查看效果，None = 1
    :parameter b : 函数参数，遍历查看效果，None = 2"""

    # # 掩膜外区域赋予 空值
    # saa[saa < 1] = np.nan
    # saz[saz < 1] = np.nan
    # sua[sua < 1] = np.nan
    # suz[suz < 1] = np.nan
    # 非空角度改为弧度 radians
    ra_saa = np.radians(saa)
    ra_saz = np.radians(saz)
    ra_sua = np.radians(sua)
    ra_suz = np.radians(suz)

    # 计算相对方位角 relative azimuth angle-raa
    ra_raa = np.abs(np.subtract(ra_saa, ra_sua))

    # 计算散射角 Scattering angle
    # 注释角度符号 θ_s:太阳天顶角;θ_v:卫星天顶角；ϕ：相对方位角；ξ：散射角
    # 散射角计算公式为：cos(ξ) = cos(θ_s)*cos(θ_v) + sin(θ_s)*sin(θ_v)*cos(ϕ)
    # 使用Numpy向量级操作直接运算
    ra_sca = np.arccos(np.cos(ra_suz) * np.cos(ra_saz) + np.sin(ra_suz) * np.sin(ra_saz) * np.cos(ra_raa))

    # 计算体散射核volumetric scattering kernel - Kvol和几何光学散射核geometric scattering kernel- Kgeo_s，此处命名对应后面的Modis产品的对应的系数
    # Kvol计算公式：F_1 (θ_s,θ_v,ϕ)=((π/2-ξ)*cos(ξ)+sin(ξ))/(cos(θ_s)θ_s +cos(θ_v))-π/4
    Kvol = ((np.pi / 2 - ra_sca) * np.cos(ra_sca) + np.sin(ra_sca)) / (np.cos(ra_saz) + np.cos(ra_suz)) - np.pi / 4

    # Kgeo计算公式：此处公式较为复杂，涉及三个梯次中间变量，D，t，O，公式依次如下
    # 转换天顶角, 此处a表示b/r
    tr_suz = np.arctan(a * np.tan(ra_suz))
    tr_saz = np.arctan(a * np.tan(ra_saz))
    tr_sca = np.arccos(np.cos(tr_suz) * np.cos(tr_saz) + np.sin(tr_suz) * np.sin(tr_saz) * np.cos(ra_raa))

    # D=(tan^2(θ_s)+tan^2(θ_v)-2 tan(θ_s)tan(θ_v)cos(ϕ))^0.5.
    D = np.sqrt(np.tan(tr_suz) ** 2 + np.tan(tr_saz) ** 2 - 2 * np.tan(tr_suz) * np.tan(tr_saz) * np.cos(ra_raa))

    # cos(t)=b*(D^2+(tan(θ_s)*tan(θ_v)*sin(ϕ))^2)^0.5/((sec(θ_s)+`sec(θ_v))
    cos_t = b * np.sqrt(D ** 2 + (np.tan(tr_suz) * np.tan(tr_saz) * np.sin(ra_raa)) ** 2) / (
            1 / np.cos(tr_suz) + 1 / np.cos(tr_saz))
    clip_cost = np.clip(cos_t, -1, 1)
    # 修复：允许边界值，避免不必要的 NaN
    t = np.arccos(clip_cost)

    # O(θ_s,θ_v,ϕ)=1/π*(t-sin(t)cos(t))*(sec(θ_s)+sec(θ_v))
    O = (1 / np.pi) * (t - np.sin(t) * np.cos(t)) * (1 / np.cos(tr_suz) + 1 / np.cos(tr_saz))

    # 计算Kdense：F_2 (θ_s,θ_v,ϕ)=((1+cosξ)*sec(θ_v)) / (sec(θ_s)+sec(θ_v)-O(θ_s,θ_v,ϕ))-2
    Kdense = ((1 / np.cos(tr_saz)) * (1 + np.cos(tr_sca))) / (1 / np.cos(tr_suz) + 1 / np.cos(tr_saz) - O) - 2
    return Kvol, Kdense
