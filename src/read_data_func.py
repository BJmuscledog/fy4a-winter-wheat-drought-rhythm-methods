from osgeo import gdal

def readtiff(path, band):
    '''函数用于读取影像各波段数值并提取为二维矩阵'''
    dataset = gdal.Open(path)
    src_band = dataset.GetRasterBand(band)
    # 波段转数组
    band_arr = src_band.ReadAsArray()
    return band_arr

def readenvi(path,band):
    dataset = gdal.Open(path)
    src_band = dataset.GetRasterBand(band)
    # 波段转数组
    band_arr = src_band.ReadAsArray()
    return band_arr



