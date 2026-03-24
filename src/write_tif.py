from osgeo import gdal,osr
import numpy as np

def write_oneband_tiff(out_filename, outband, cols, rows,rows_start=0, cols_start=0,lon_res=0.02128601, lat_res=0.02142906,f_lulo = 110.34429247,f_lula = 36.39371121):


    # 输出计算结果(此处只输出波段1的数据）
    # 创建tif文件
    driver = gdal.GetDriverByName("GTiff")
    out_tiff = driver.Create(out_filename, cols, rows, 1, gdal.GDT_Float32)

    # 设置投影转换信息
    c_lulo = f_lulo + lon_res * cols_start  # 选择区域左上角起始经度
    c_lula = f_lula - lat_res * rows_start  # 选择区域左上角起始纬度
    transform = (
        c_lulo,  # 左上角经度
        lon_res,  # x方向分辨率
        0,  # 旋转角度
        c_lula,  # 左上角纬度
        0,  # 旋转角度
        -lat_res,  # y方向分辨率, 由于自左上角开始(纬度往下逐渐减小), 因此为负;
    )
    out_tiff.SetGeoTransform(transform)
    # 设置投影信息为WGS_1984
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS_1984的EPSG代码是4326
    out_tiff.SetProjection(srs.ExportToWkt())

    band = out_tiff.GetRasterBand(1)
    band.WriteArray(outband)
    out_tiff.ReadAsArray(0, 0, cols, rows)

def write_multiplebands_tiff(out_filename, outband,cols, rows,rows_start=0, cols_start=0,lon_res = 0.02128601,lat_res = 0.02142906,f_lulo = 110.34429247,f_lula = 36.39371121):

    # 输出计算结果(此处只输出波段1的数据）
    # 创建tif文件
    driver = gdal.GetDriverByName("GTiff")
    out_tiff = driver.Create(out_filename, cols, rows, len(outband), gdal.GDT_Float32)

    # 设置投影转换信息
    c_lulo = f_lulo + lon_res * cols_start  # 选择区域左上角起始经度
    c_lula = f_lula - lat_res * rows_start  # 选择区域左上角起始纬度
    transform = (
        c_lulo,  # 左上角经度
        lon_res,  # x方向分辨率
        0,  # 旋转角度
        c_lula,  # 左上角纬度
        0,  # 旋转角度
        -lat_res,  # y方向分辨率, 由于自左上角开始(纬度往下逐渐减小), 因此为负;
    )
    # 设置投影信息为WGS_1984
    out_tiff.SetGeoTransform(transform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS_1984的EPSG代码是4326
    out_tiff.SetProjection(srs.ExportToWkt())

    for i in range(len(outband)):
        out_tiff.GetRasterBand(i + 1).WriteArray(outband[i])  # GetRasterBand()传入的索引从1开始, 而非0
    out_tiff.FlushCache()
def write_maize_tiff(out_filename, outband, rows_start, cols_start, cols, rows):
    # 原始影像左上角经纬度
    f_lulo = 110.03467469
    f_lula = 36.78628578
    # 影像经纬度分辨率
    lon_res = 0.021625761586
    lat_res = 0.018030890121

    # 输出计算结果(此处只输出波段1的数据）
    # 创建tif文件
    driver = gdal.GetDriverByName("GTiff")
    out_tiff = driver.Create(out_filename, cols, rows, 1, gdal.GDT_Float32)

    # 设置投影转换信息
    c_lulo = f_lulo + lon_res * cols_start  # 选择区域左上角起始经度
    c_lula = f_lula - lat_res * rows_start  # 选择区域左上角起始纬度
    transform = (
        c_lulo,  # 左上角经度
        lon_res,  # x方向分辨率
        0,  # 旋转角度
        c_lula,  # 左上角纬度
        0,  # 旋转角度
        -lat_res,  # y方向分辨率, 由于自左上角开始(纬度往下逐渐减小), 因此为负;
    )
    out_tiff.SetGeoTransform(transform)
    # 设置投影信息为WGS_1984
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS_1984的EPSG代码是4326
    out_tiff.SetProjection(srs.ExportToWkt())

    band = out_tiff.GetRasterBand(1)
    band.WriteArray(outband)
    out_tiff.ReadAsArray(0, 0, cols, rows)
