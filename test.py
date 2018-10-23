# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   test.py
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


# 读取数据
real_data = pd.read_csv(r'Data/arima_volatility_data/real_data.csv', index_col=0, header=None)
arima_pred = pd.read_csv(r'Data/arima_volatility_data/ARIMA_prediction.csv', index_col=0, header=None)
arima_resid = pd.read_csv(r'Data/arima_volatility_data/ARIMA_residuals.csv', index_col=0, header=None)
arima_resid.index = real_data.index
arima_smooth = pd.read_csv(r'Data/arima_volatility_data/ARIMA_smoothed.csv', index_col=0, header=None)
arima_smooth.index = real_data.index
res_volatility_GARCH = pd.read_csv(r'Data/arima_volatility_data/res_conditional_volatility_GARCH.csv',
                                   index_col=0, header=None)
res_volatility_GJRARCH = pd.read_csv(r'Data/arima_volatility_data/res_conditional_volatility_GJR_ARCH.csv',
                                     index_col=0, header=None)
pred_volatility_GARCH = pd.read_csv(r'Data/arima_volatility_data/res_conditional_volatility_prediction.csv',
                                    index_col=0)
merged_data = pd.concat([arima_resid, arima_smooth, arima_pred, pred_volatility_GARCH], axis=1)


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




plt.plot(data[-1-288:])
plt.plot(smooth_data[-1-288:])
# 使用整体的平滑结果
filtered_data = lowess(data.flatten(),
                       np.array([x for x in range(len(data.flatten()))]),
                       is_sorted=True, frac=0.05 / 59, it=0)
plt.plot(filtered_data[-1-288:, -1])
plt.legend(['real', 'seg_smooth', 'all_smooth'])
plt.show()
