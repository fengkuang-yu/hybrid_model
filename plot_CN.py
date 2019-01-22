# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   plot_CN.py
@Time    :   2019/1/20 14:29
@Desc    :
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as smt


class MergeDataFig:
    PRED_STEP = 1


# 处理输入数据、添加index
plt.style.use('fivethirtyeight')
pd_data = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\flow_data_59.csv')
y = pd.Series(pd_data['20.93'])
y.index = pd.date_range(start='2016-02-01 00:00:00', periods=16992, freq='5min', normalize=True)

# 对输入数据进行历史平均处理,这里构造三个特征：
# 1. 第一个特征（history_average）是历史平均值，使用历史平均值作为周期/季节成分
# 2. 第二个特征（history_diff   ）是使用历史平均值的差分值，相当于下一刻相对于现在的变化量
# 3. 第三个特征（deterministic  ）是实际数据减去历史平均值的每日实时波动部分
demo_array = np.array(y).reshape((-1, 288)).T
demo_average = np.mean(demo_array, axis=1)
demo_average_extend = np.tile(demo_average, int(len(pd_data) / 288))
history_average = pd.Series(demo_average_extend)
history_average.index = pd.date_range(start='2016-02-01 00:00:00', periods=16992, freq='5min', normalize=True)
deterministic = y - history_average  # 减去均值后得到的序列的部分

# 建立ARIMA模型，最优参数根据AIC准则选取（2, 0，2）
mod = sm.tsa.statespace.SARIMAX(deterministic, order=(2, 0, 2), enforce_stationarity=False, enforce_invertibility=False)
results = mod.fit()
residuals = pd.DataFrame(results.resid)  # 对于训练数据的拟合的残差值
residuals = residuals.rename(columns={0: 'Residuals'})  # 改变列的名字
smoothed_deterministic = results.fittedvalues  # deterministic中取出residuals的剩余值


def plot_day_fourlines():
    plt.figure(figsize=(10, 6))
    y.iloc[0:288].plot(label=u'真实数据', linewidth=2, fontsize=16)
    history_average.iloc[0:288].plot(label=u'历史平均', linewidth=2, fontsize=16)
    deterministic.iloc[0:288].plot(label=u'残差部分', linewidth=2, fontsize=16)
    plt.legend(fontsize=16)
    plt.box()
    plt.ylabel(u'交通流量(vehicles)', fontsize=16)
    plt.xlabel(u'时间', fontsize=14)
    plt.show()


# 分析一个月的日内趋势
def intra_day_trend():
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 6))
    ax = Axes3D(fig)
    X = np.arange(1, 30, 1)
    Y = np.arange(288)
    X, Y = np.meshgrid(X, Y)
    Z = demo_array[Y, X]
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    ax.set_title(u"一个月的交通流量", fontsize=16)
    ax.set_xlabel(u"日期", fontsize=16)
    ax.set_ylabel(u"时间", fontsize=16)
    ax.set_zlabel(u"车流量(辆/5分钟)", fontsize=16)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)
    plt.box()
    plt.show()


def arch_effect():
    at = residuals[0:2880]
    at2 = np.square(at)
    fig = plt.figure(figsize=(10, 6))
    layout = (2, 2)
    at2_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    at2_ax.plot(np.array(at2))
    at2_ax.xaxis.set_tick_params(rotation=0, labelsize=16)
    at2_ax.set_title('残差的平方')
    at2_ax.set_xlabel('观测点(5min)', fontsize=16)
    smt.graphics.plot_acf(at2, lags=30, ax=acf_ax, alpha=0.5)
    acf_ax.set_xlabel('时滞(5mins)', fontsize=16)
    acf_ax.set_title(u'自相关')
    smt.graphics.plot_pacf(at2, lags=30, ax=pacf_ax, alpha=0.5)
    pacf_ax.set_xlabel('时滞(5mins)', fontsize=16)
    pacf_ax.set_title(u'偏自相关')
    plt.tight_layout()
    plt.show()


def residuals_acf_pacf_plot():
    # 画出拟合残差的图像并进行ACF和PACF分析
    fig = plt.figure(figsize=(10, 6))
    layout = (2, 2)
    ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    ax.plot(np.array(residuals.iloc[0:2880]))
    ax.set_xlabel(u'观测点(5mins)', fontsize=16)
    ax.set_title(u'残差')
    ax.xaxis.set_tick_params(rotation=0, labelsize=16)
    smt.graphics.plot_acf(residuals, lags=30, ax=acf_ax, alpha=0.5)
    acf_ax.set_xlabel(u'时滞(5mins)', fontsize=16)
    acf_ax.set_title(u'自相关')
    smt.graphics.plot_pacf(residuals, lags=30, ax=pacf_ax, alpha=0.5)
    pacf_ax.set_xlabel(u'时滞(5mins)', fontsize=16)
    pacf_ax.set_title(u'偏自相关')
    plt.tight_layout()
    plt.box()
    plt.show()


# 分析deterministic部分的相关性，自相关和互相关分析得出arima模型的参数
def plot_deterministic():
    fig = plt.figure(figsize=(10, 6))
    layout = (2, 2)
    ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    ax.plot(np.array(deterministic.iloc[0:2880]))
    ax.set_title(u'剩余部分')
    ax.set_xlabel(u'观测点(5min)', fontsize=16)
    ax.xaxis.set_tick_params(rotation=0, labelsize=16)
    smt.graphics.plot_acf(deterministic, lags=30, ax=acf_ax, alpha=0.5)
    acf_ax.set_xlabel(u'时滞(5min)', fontsize=16)
    smt.graphics.plot_pacf(deterministic, lags=30, ax=pacf_ax, alpha=0.5)
    pacf_ax.set_xlabel(u'时滞(5min)', fontsize=16)
    plt.tight_layout()
    plt.show()
