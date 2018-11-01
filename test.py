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

# merged_data = np.array(pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\merged_data.csv', index_col=0).iloc[-3340:]).flatten()
arima_pred = np.array(pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\arima_prediction.csv', index_col=0).iloc[-3340:]).flatten()
arima_trend_pred = np.array(pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\arima_trend_prediction.csv', index_col=0).iloc[-3340:]).flatten()
lstm_pred = np.array(pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\lstm_prediction_0.0947782581867.csv', index_col=0)).flatten()
lstm_pred_hybrid = np.array(pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\prediction_0.0821729604981.csv', index_col=0)).flatten()
test_real = np.array(pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\flow_test_real.csv', index_col=0)).flatten()
print(len(arima_pred),len(arima_trend_pred),len(lstm_pred),len(lstm_pred_hybrid),len(test_real))

arima_pred_mae, arima_pred_mape = mae_mape(arima_pred, test_real)
arima_trend_pred_mae, arima_trend_pred_mape = mae_mape(arima_trend_pred, test_real)
lstm_pred_mae, lstm_pred_mape = mae_mape(lstm_pred, test_real)
lstm_pred_hybrid_mae, lstm_pred_hybrid_mape = mae_mape(lstm_pred_hybrid, test_real)

plot_one_day(test_real[-288:], arima_trend_pred[-288:], legend={'real traffic flow', 'arima result'})
# plt.plot(arima_trend_pred[-288:])
# plt.plot(test_real[-288:])



pred = np.array(pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\prediction_0.0856552987614.csv', index_col=0)).flatten()
real = test_real[1:]
d = abs(pred - real)
mape = sum(d / real) / real.shape[0]
mae = sum(d) / real.shape[0]
