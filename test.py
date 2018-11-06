# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   test.py
@Time    :   2018/10/12 11:08
@Desc    :
"""

import pandas as pd
import numpy as np
import datetime
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, drange
import matplotlib.ticker as ticker

def mae_mape(pred, real):
    pred = np.array(pred).flatten()
    real = np.array(real).flatten()
    d = abs(pred - real)
    mape = sum(d / real) / real.shape[0]
    mae = sum(d) / real.shape[0]
    return mae, mape

def plot_one_day(data1, data2, y_label='Traffic flow(Vehicles)', x_label='Time', legend=None):
    """
    画出图
    :param data1: 真实数据
    :param data2: 仿真数据
    :param y_label: y轴的坐标
    :param x_label: x轴的坐标
    :param legend: 图例
    :return: 无
    """
    formatter = DateFormatter('%H:%M')  # 时间表现形式，这里只显示了时分
    d1 = datetime.datetime(2018, 2, 10, 0, 0, 0)
    d2 = datetime.datetime(2018, 2, 11, 0, 0, 0)
    delta = datetime.timedelta(minutes=5)  # 以0.5秒为间隔生成时间序列
    x = drange(d1, d2, delta)
    y1 = data1
    y2 = data2
    fig1, ax1 = plt.subplots()
    plt.plot(x, y1)
    plt.plot(x, y2)
    ax1.xaxis.set_major_formatter(formatter)
    tick_spacing = 1 / 6
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax1.xaxis.set_tick_params(rotation=0, labelsize=10)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    if legend is None:
        plt.legend(['real traffic flow', 'simulation result'], loc=1)
    else:
        plt.legend(legend)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()

# real = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\flow_test_real.csv', index_col=0)[-3168:]
#
# # ARIMA
# arima_5 = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\arima_prediction_5min.csv', index_col=0, header=None)[-3168:]
# arima_10 = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\arima_prediction_10min.csv', index_col=0, header=None)[-3168:]
# arima_15 = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\arima_prediction_15min.csv', index_col=0, header=None)[-3168:]
# arima_20 = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\arima_prediction_20min.csv', index_col=0, header=None)[-3168:]
# arima_5_mae, arima_5_mape = mae_mape(arima_5, real)
# arima_10_mae, arima_10_mape = mae_mape(arima_10, real)
# arima_15_mae, arima_15_mape = mae_mape(arima_15, real)
# arima_20_mae, arima_20_mape = mae_mape(arima_20, real)
#
#
# # ARIMA + history_average
# arima_trend_5 = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\arima_trend_prediction_5min.csv', index_col=0, header=None)[-3168:]
# arima_trend_10 = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\arima_trend_prediction_10min.csv', index_col=0, header=None)[-3168:]
# arima_trend_15 = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\arima_trend_prediction_15min.csv', index_col=0, header=None)[-3168:]
# arima_trend_20 = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\arima_trend_prediction_20min.csv', index_col=0, header=None)[-3168:]
# arima_trend_5_mae, arima_trend_5_mape = mae_mape(arima_trend_5, real)
# arima_trend_10_mae, arima_trend_10_mape = mae_mape(arima_trend_10, real)
# arima_trend_15_mae, arima_trend_15_mape = mae_mape(arima_trend_15, real)
# arima_trend_20_mae, arima_trend_20_mape = mae_mape(arima_trend_20, real)
#
#
# flow_data = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\flow_data_59.csv', index_col=0)
# select_segment = ['15.63', '16.12', '16.67', '17.23', '17.99', '18.7', '19.21', '19.71', '20.22', '20.93',
#                   '21.36', '21.83', '22.31', '22.73', '23.51', '23.93', '24.39', '25.17','25.68', '26.16']
# flow_select_data = flow_data[select_segment]
# flow_select_data.index = pd.date_range(start='2016-02-01 00:00:00', periods=16992, freq='5min', normalize=True)
# flow_select_data.to_csv(r'D:\桌面\flow_data_20segments.csv')
merged_data_5min = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\merged_data.csv', index_col=0)
# 真实值
real_data = merged_data_5min['Real_data']
# 历史平均
history_average_roll_5min = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\history_average_roll_5min.csv', index_col=0, header=None)
history_average_roll_10min = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\history_average_roll_10min.csv', index_col=0, header=None)
history_average_roll_15min = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\history_average_roll_15min.csv', index_col=0, header=None)
history_average_roll_20min = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\history_average_roll_20min.csv', index_col=0, header=None)
# 差分
history_diff_5min = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\history_diff_5min.csv', index_col=0, header=None)
history_diff_10min = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\history_diff_10min.csv', index_col=0, header=None)
history_diff_15min = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\history_diff_15min.csv', index_col=0, header=None)
history_diff_20min = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\history_diff_20min.csv', index_col=0, header=None)
# 波动
volatility_pred_5min = np.roll(np.array(pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\volatility.csv', index_col=0, header=None)), -1)
volatility_pred_10min = np.roll(volatility_pred_5min, -1)
volatility_pred_15min = np.roll(volatility_pred_10min, -1)
volatility_pred_20min = np.roll(volatility_pred_15min, -1)
volatility_pred_5min = pd.Series(volatility_pred_5min.flatten())
volatility_pred_5min.index = pd.date_range(start='2016-02-01 00:00:00', periods=16992, freq='5min', normalize=True)
volatility_pred_10min = pd.Series(volatility_pred_10min.flatten())
volatility_pred_10min.index = pd.date_range(start='2016-02-01 00:00:00', periods=16992, freq='5min', normalize=True)
volatility_pred_15min = pd.Series(volatility_pred_15min.flatten())
volatility_pred_15min.index = pd.date_range(start='2016-02-01 00:00:00', periods=16992, freq='5min', normalize=True)
volatility_pred_20min = pd.Series(volatility_pred_20min.flatten())
volatility_pred_20min.index = pd.date_range(start='2016-02-01 00:00:00', periods=16992, freq='5min', normalize=True)
# 确定性
deterministic = merged_data_5min['Smoothed_deterministic']
# ARIMA的预测
arima_trend_prediction_5min = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\arima_trend_prediction_5min.csv', index_col=0, header=None)
arima_trend_prediction_10min = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\arima_trend_prediction_10min.csv', index_col=0, header=None)
arima_trend_prediction_15min = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\arima_trend_prediction_15min.csv', index_col=0, header=None)
arima_trend_prediction_20min = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\arima_trend_prediction_20min.csv', index_col=0, header=None)


merged_data_5min = pd.concat([real_data, history_average_roll_5min, history_diff_5min, volatility_pred_5min, deterministic, arima_trend_prediction_5min], axis=1)
merged_data_5min.columns = ['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6']
merged_data_10min = pd.concat([real_data, history_average_roll_10min, history_diff_10min, volatility_pred_10min, deterministic, arima_trend_prediction_10min], axis=1)
merged_data_10min.columns = ['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6']
merged_data_15min = pd.concat([real_data, history_average_roll_15min, history_diff_15min, volatility_pred_15min, deterministic, arima_trend_prediction_15min], axis=1)
merged_data_15min.columns = ['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6']
merged_data_20min = pd.concat([real_data, history_average_roll_20min, history_diff_20min, volatility_pred_20min, deterministic, arima_trend_prediction_20min], axis=1)
merged_data_20min.columns = ['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6']

merged_data_5min.to_csv(r'D:\桌面\merged_data_5min.csv')
merged_data_10min.to_csv(r'D:\桌面\merged_data_10min.csv')
merged_data_15min.to_csv(r'D:\桌面\merged_data_15min.csv')
merged_data_20min.to_csv(r'D:\桌面\merged_data_20min.csv')