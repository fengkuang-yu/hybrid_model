# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   ARIMAmodule.py
@Time    :   2018/10/18 14:53
@Desc    :
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as scs
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class MergeDataFig:
    pred_step = 1

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
history_diff_extend = np.roll(demo_average_extend, -1) - demo_average_extend  # np.roll()为循环移位函数，这里表示循环向右移动一位
history_diff = pd.Series(history_diff_extend)
history_diff.index = pd.date_range(start='2016-02-01 00:00:00', periods=16992, freq='5min', normalize=True)

# 使用AIC进行模型选择
# p = d = q = range(0, 3)
# pdq = list(itertools.product(p, d, q))
# warnings.filterwarnings("ignore")  # 忽略参数配置时的警告信息
# for param in pdq:
#     try:
#         mod = sm.tsa.statespace.SARIMAX(deterministic, order=param)
#         results = mod.fit()
#         print('ARIMA{}- AIC:{}'.format(param, results.aic))
#     except:
#         continue

# 建立ARIMA模型，最优参数根据AIC准则选取（2, 0，2）
mod = sm.tsa.statespace.SARIMAX(deterministic,
                                order=(2, 0, 2),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary())  # 模型的诊断表
results.plot_diagnostics(figsize=(15, 12))  # 模型的拟合诊断图
plt.show()
residuals = pd.DataFrame(results.resid)  # 对于训练数据的拟合的残差值
residuals = residuals.rename(columns={0: 'Residuals'})  # 改变列的名字
smoothed_deterministic = results.fittedvalues  # deterministic中取出residuals的剩余值
# y == smoothed_deterministic + results.resid + history_average

# 利用建立好的模型进行预测
pred_test = results.forecast()
pred = results.get_prediction(start=pd.to_datetime('2016-02-01 00:00:00'), dynamic=False)

# 计算预测误差
arima_forecasted = pred.predicted_mean + history_average
y_truth = y['2016-02-01 00:00:00':]
mae = ((arima_forecasted[2:] - y_truth[2:]) / y_truth[2:]).abs().mean()
print('The Mean Absolute Error of ARIMA forecasts is {}'.format(round(mae, 5)))


# 建立GJR-ARCH模型
am = arch_model(residuals, vol='Garch', p=1, q=1, dist='t')
res = am.fit(update_freq=5, disp='off')
print(res.summary())
index = residuals.index
start_loc = 0
end_loc = 288
forecasts = {}

# 这个的输出结果基本上和res.conditional_volatility相同
for i in range(len(residuals) - end_loc):
    if i % 1000 == 0:
        print(i)
    res2 = am.fit(first_obs=0, last_obs=i + end_loc, disp='off')
    temp_variance = res2.forecast(horizon=1).variance
    fcast = temp_variance.iloc[i + end_loc - 1]
    forecasts[fcast.name] = fcast

# 画出波动率和方差的图
variance_pred = pd.DataFrame(forecasts).T
variance_pred = pd.concat([temp_variance.iloc[0:287], variance_pred], axis=0)
variance_pred = variance_pred.rename(columns={0: 'variance_pred'})  # 改变列的名字
volatility_pred = np.sqrt(variance_pred)
volatility_pred.plot(label='volatility_prediction')
plt.legend()
plt.show()
res.conditional_volatility.plot(label='volatility_real')
plt.legend()
plt.show()

# 将各特征拼接成一个DataFrame，然后导出为csv文件
merged_feature = pd.concat([y, history_average, smoothed_deterministic, residuals, volatility_pred, history_diff],
                           axis=1)
merged_feature = merged_feature.rename(columns={'20.93': 'Real_data',
                                                0: 'History_average',
                                                1: 'Smoothed_deterministic',
                                                'h.1': 'Volatility_pred',
                                                2: 'History_diff'})
merged_feature = merged_feature.fillna(value=0)

# 检验序列具有ARCH效应
def arch_effect():
    at = residuals[0:2880]
    at2 = np.square(at)
    fig = plt.figure(figsize=(10, 6))
    layout = (2, 2)
    at2_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    at2.plot(ax=at2_ax)
    at2_ax.legend_.remove()
    at2_ax.xaxis.set_tick_params(rotation=0, labelsize=10)
    at2_ax.set_title('Squared Residuals')
    at2_ax.set_xlabel('Date', fontsize=16)
    smt.graphics.plot_acf(at2, lags=30, ax=acf_ax, alpha=0.5)
    acf_ax.set_xlabel('Time Lag(5mins)', fontsize=16)
    smt.graphics.plot_pacf(at2, lags=30, ax=pacf_ax, alpha=0.5)
    pacf_ax.set_xlabel('Time Lag(5mins)', fontsize=16)
    plt.tight_layout()
    plt.show()

# 查看波动率
def plot_volatility():
    merged_feature = pd.read_csv(r'D:\software\pycharm\PycharmProjects\demo\merged_data.csv')
    Volatility_pred = merged_feature['Volatility_pred']
    (Volatility_pred.iloc[-288:]*10).plot()
    (res.conditional_volatility.iloc[-288:]*10).plot()
    y.iloc[-288:].plot()
    plt.show()

# 查看历史平均、实际、趋势和残差部分
def plot_day_fourlines(disp_day):
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    history_average.iloc[disp_day].plot(label='History Average', linewidth=1, fontsize=12)
    y.iloc[disp_day].plot(label='Real', linewidth=1, fontsize=12)
    deterministic.iloc[disp_day].plot(label='Residual Part', linewidth=1, fontsize=12)
    (smoothed_deterministic + history_average).iloc[disp_day].plot(label='somoothed', linewidth=1, fontsize=12)
    plt.legend(fontsize=6)
    plt.ylabel('Traffic Flow(vehicles)', fontsize=14)
    plt.xlabel('Time', fontsize=14)
    plt.show()

# 查看残差的residuals的残差
def residuals_acf_pacf_plot():
    # 画出拟合残差的图像并进行ACF和PACF分析
    residuals.plot()
    plt.ylabel('Residuals(vehicles)', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.show()
    print(residuals.describe())
    # ARIMA的残差
    plot_acf(residuals, lags=30)
    plt.title('')
    plt.ylabel('Residuals ACF', fontsize=16)
    plt.xlabel('Time Lag(5mins)', fontsize=16)
    plt.show()
    plot_pacf(residuals, lags=30)
    plt.title('')
    plt.ylabel('Residuals PACF', fontsize=16)
    plt.xlabel('Time Lag(5mins)', fontsize=16)
    plt.show()

# 分析deterministic部分的相关性，自相关和互相关分析得出arima模型的参数
def plot_deterministic():
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    plot_acf(deterministic, lags=30)
    plt.title('')
    plt.ylabel('ACF', fontsize=16)
    plt.xlabel('Time Lag(5mins)', fontsize=16)
    plt.show()
    plot_pacf(deterministic, lags=30)
    plt.title('')
    plt.ylabel('PACF', fontsize=16)
    plt.xlabel('Time Lag(5mins)', fontsize=16)
    plt.show()

# 分析一个月的日内趋势
def intra_day_trend():
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(1,30,1)
    Y = np.arange(288)
    X, Y = np.meshgrid(X, Y)
    Z = demo_array[Y, X]
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    ax.set_title("Traffic flow over a month", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Time index of each day", fontsize=12)
    ax.set_zlabel("Traffic flow(Vehicle/5 min)", fontsize=12)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.zaxis.set_tick_params(labelsize=10)
    plt.show()




merged_feature.to_csv(r'E:\merged_data.csv')
