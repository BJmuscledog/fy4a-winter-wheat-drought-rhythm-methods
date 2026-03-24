# from fy6s_atc_func import *
# from joblib import Parallel,delayed
# if __name__ == '__main__':
#     start = time.time()
#     formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#     print("Start time for all:", formatted_time)
#     fy_dir = r'D:\Submission_assay\Fy_p2_data\2023_0423_week\Cloud_removal'
#     angle_dir = r'D:\Submission_assay\Fy_p2_data\2023_0423_week\Angledata'
#     out_dir = r'D:\Submission_assay\Fy_p2_data\2023_0423_week\Atmospheric_correction'
#     # months = [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5]
#     # days = [2, 7, 8, 10, 11, 28, 2, 3, 5, 6, 7, 8, 10, 11, 18, 21, 23, 2, 3, 4, 5, 16, 17, 20, 21]
#     xRes = 0.00897
#     yRes = 0.00897
#     year= 2023
#     months= [4,4,4,4,4,4,4]
#     days = [17,19,20,21,22,23,23]
#     for j in range(len(days)):
#         # bandids = [2,3,5,6]
#         bandids = [2, 3]
#         # rows_parts = [0, 40, 80, 120, 160, 200,234]
#         # cols_parts = [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264,288,296]
#         rows_parts = [0, 80, 160, 240,320,400,480,556]
#         cols_parts = [0, 50,100,150,200,250,300,350,400,450,500,550,600,650,702]
#         fy_tifs = glob.iglob(fy_dir + '\\' + '*.tif')
#         fy_list = list(fy_tifs)
#         n_jobs = 3
#         jobs =9
#         batch_size = max(1, (jobs + n_jobs - 1) // n_jobs)
#         tasks = (delayed(fy6s_atc_test)(fy_dir, angle_dir, out_dir,bandids,rows_parts,cols_parts,year,months[j],days[j],i,i+1,lon_res=xRes,lat_res=yRes) for i in range(j*9,(j+1)*9))
#         result = Parallel(n_jobs, batch_size=batch_size)(tasks)
#     print("Time:", (end - start) / 60, "Minutes")

#         # fy6s_atc(fy_dir, angle_dir, out_dir,bandids,rows_parts,cols_parts,2022,3,15,0,4)
#     end = time.time()



from fy6s_atc_func import *
from joblib import Parallel,delayed
if __name__ == '__main__':
    start = time.time()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("Start time for all:", formatted_time)
    fy_dir = r'H:\风云数据\Fy_p2_data\2022_0423_week\Cloud_removal_hhh_n'
    angle_dir = r'H:\风云数据\Fy_p2_data\2022_0423_week\Angledata_hhh'
    out_dir = r'H:\风云数据\Fy_p2_data\2022_0423_week\Atmospheric_correction_hhh_n'
    # months = [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5]
    # days = [2, 7, 8, 10, 11, 28, 2, 3, 5, 6, 7, 8, 10, 11, 18, 21, 23, 2, 3, 4, 5, 16, 17, 20, 21]
    xRes = 0.00897
    yRes = 0.00897
    year= 2022
    months= [4,4,4]
    days = [25,26,27]
    for j in range(len(days)):
        # bandids = [2,3,5,6]
        bandids = [1,2,3]
        # rows_parts = [0, 40, 80, 120, 160, 200,234]
        # cols_parts = [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264,288,296]
        # 河南行列数
        # rows_parts = [0, 80, 160, 240,320,400,480,556]
        # cols_parts = [0, 50,100,150,200,250,300,350,400,450,500,550,600,650,702]
        # 把 1252 像素均分成 7 段，需要 8 个边界点
        rows_parts = list(np.linspace(0, 1252, 8, dtype=int))
        # 把 1377 像素均分成 14 段，需要 15 个边界点
        cols_parts = list(np.linspace(0, 1377, 15, dtype=int))
        fy_tifs = glob.iglob(fy_dir + '\\' + '*.tif')
        fy_list = list(fy_tifs)
        n_jobs = 3
        jobs = 9
        batch_size = max(1, (jobs + n_jobs - 1) // n_jobs)
        tasks = (delayed(fy6s_atc_test)(fy_dir, angle_dir, out_dir,bandids,rows_parts,cols_parts,year,months[j],days[j],i,i+1,lon_res=xRes,lat_res=yRes) for i in range(j*9,(j+1)*9))
        result = Parallel(n_jobs, batch_size=batch_size)(tasks)
        # fy6s_atc(fy_dir, angle_dir, out_dir,bandids,rows_parts,cols_parts,2022,3,15,0,4)
    end = time.time()
    print("Time:", (end - start) / 60, "Minutes")