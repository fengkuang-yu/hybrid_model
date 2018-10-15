# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   batch_test.py
@Time    :   2018/10/12 11:08
@Desc    :
"""

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from statsmodels.nonparametric.smoothers_lowess import lowess


def seg_filter(data, frac_param):
    smoothed_data = lowess(data.flatten(),
                           np.array([x for x in range(len(data.flatten()))]),
                           is_sorted=True,
                           frac=frac_param,
                           it=0)
    return smoothed_data


def data_pro(data, time_steps=None, slide_sep=True):
    """
    数据处理，将列状的数据拓展开为行形式

    :param data: 输入交通数据
    :param time_steps: 分割时间长度
    :return: 处理过的数据
    :param slide_sep: 是否连续切割数据
    """
    if time_steps is None:
        time_steps = 1
    size = data.shape
    if isinstance(data, pd.DataFrame):
        data = np.array(data)
        # 如果输入的数据是矩阵形式的
    if isinstance(data, np.ndarray):
        if slide_sep is True:
            temp = np.zeros((size[0] - time_steps, size[1] * time_steps))
            for i in range(data.shape[0] - time_steps):
                temp[i, :] = data[i:i + time_steps, :].flatten()
            return temp
        else:
            try:
                temp = np.zeros((int(size[0] / time_steps), size[1] * time_steps))
                for i in range(int(size[0] / time_steps)):
                    temp[i, :] = data[i * time_steps:i * time_steps + time_steps, :].flatten()
                return temp
            except Exception:
                print('time_step必须要能整除288')
                raise

    else:
        raise Exception('data_pro数据输入格式错误')


time_step = 100
frac_param = 0.05
# data = np.array(
#     [69, 69, 51, 84, 57, 51, 45, 45, 60, 36, 30, 33, 36, 45, 30, 21, 24, 24, 30, 24, 30, 30, 36, 27, 30, 33, 33, 18, 21,
#      21, 18, 21, 21, 24, 18, 27, 27, 33, 21, 30, 36, 42, 27, 36, 36, 36, 36, 30, 39, 33, 36, 42, 63, 57, 87, 78, 81, 93,
#      96, 69, 90, 111, 96, 123, 159, 174, 177, 195, 177, 201, 192, 177, 189, 207, 204, 207, 234, 234, 285, 270, 282, 297,
#      282, 306, 282, 303, 291, 321, 315, 294, 312, 327, 330, 324, 309, 285, 303, 285, 285, 315, 321, 297, 339, 297, 336,
#      288, 327, 312, 303, 276, 318, 291, 303, 279, 312, 315, 315, 291, 258, 318, 297, 309, 273, 282, 294, 309, 300, 312,
#      285, 306, 324, 279, 294, 297, 321, 315, 330, 339, 258, 342, 348, 345, 315, 318, 303, 339, 333, 354, 366, 345, 327,
#      357, 342, 357, 366, 327, 375, 363, 336, 405, 381, 336, 342, 363, 390, 396, 363, 360, 375, 381, 381, 378, 381, 342,
#      387, 405, 438, 387, 381, 369, 381, 387, 393, 375, 357, 348, 351, 396, 354, 393, 369, 339, 375, 372, 324, 333, 321,
#      369, 354, 339, 366, 351, 330, 315, 294, 285, 300, 315, 336, 330, 357, 324, 351, 333, 276, 276, 297, 291, 303, 297,
#      285, 315, 327, 300, 282, 273, 303, 300, 294, 321, 315, 309, 294, 285, 306, 309, 303, 285, 255, 258, 258, 252, 273,
#      264, 249, 219, 246, 252, 246, 240, 237, 210, 249, 237, 240, 285, 231, 207, 213, 222, 198, 189, 177, 162, 180, 180,
#      171, 165, 171, 144, 141, 156, 126, 108, 129, 87, 105, 120, 111, 108, 123, 108, 84, 111, 72, 87, 87, 96])
pd_data = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\speed_data_59.csv',
                   index_col='Datetime \ Milepost')
flow_demo = np.array(pd_data['20.93'], dtype=float)
data = flow_demo.reshape(-1, 1)
pro_data = data_pro(data, time_steps=288)
smooth_data = []
for i in range(len(pro_data)):
    seg_filted = seg_filter(pro_data[i], frac_param=0.10)
    smooth_data.append(seg_filted[-1, -1])

smooth_data = np.array(smooth_data)
sio.savemat(r'D:\桌面\speed_data_param=0.10', {'smooth_speed': np.array(smooth_data)})











a = np.array([1,2,3])
b = np.array([2,4,6])
print(b/a)









plt.plot(data[-1-288:])
plt.plot(smooth_data[-1-288:])
# 使用整体的平滑结果
filtered_data = lowess(data.flatten(),
                       np.array([x for x in range(len(data.flatten()))]),
                       is_sorted=True, frac=0.05 / 59, it=0)
plt.plot(filtered_data[-1-288:, -1])
plt.legend(['real', 'seg_smooth', 'all_smooth'])
plt.show()


seg_filted[:,-1]