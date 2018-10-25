# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   ARIMAmodule.py
@Time    :   2018/10/18 14:53
@Desc    :
"""

import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 处理输入数据
plt.style.use('fivethirtyeight')
pd_data = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\flow_data_59.csv')
y = pd.Series(pd_data['20.93'])
y.index = pd.date_range(start='2016-02-01 00:00:00', periods=16992,freq='5min',normalize=True)


# 对输入数据进行历史平均处理,这里构造三个特征：
# 1. 第一个特征（history_average）是历史平均值，使用历史平均值作为周期/季节成分
# 2. 第二个特征（history_diff   ）是使用历史平均值的差分值，相当于下一刻相对于现在的变化量
# 3. 第三个特征（deterministic  ）是实际数据减去历史平均值的每日实时波动部分
demo_array = np.array(y).reshape((-1, 288)).T
demo_average = np.mean(demo_array, axis=1)
demo_average_extend = np.tile(demo_average, int(len(pd_data)/288))
history_average = pd.Series(demo_average_extend)
history_average.index = pd.date_range(start='2016-02-01 00:00:00', periods=16992,freq='5min',normalize=True)
deterministic = y - history_average  # 减去均值后得到的序列的部分
history_diff_extend = np.roll(demo_average_extend, -1) - demo_average_extend  # np.roll()为循环移位函数，这里表示循环向右移动一位
history_diff = pd.Series(history_diff_extend)
history_diff.index = pd.date_range(start='2016-02-01 00:00:00', periods=16992,freq='5min',normalize=True)
# 以2月1日为例画出分析后的特征图
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
history_average.iloc[0:288].plot(label='History Average', linewidth=1, fontsize=12)
y.iloc[0:288].plot(label='Real', linewidth=1, fontsize=12)
deterministic.iloc[0:288].plot(label='Residual Part', linewidth=1, fontsize=12)
plt.legend(fontsize=12)
plt.ylabel('Traffic Flow(vehicles)', fontsize=14)
plt.xlabel('Time', fontsize=14)
plt.show()


# 分析deterministic部分的相关性，自相关和互相关分析得出arima模型的参数
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


# 使用AIC进行模型选择
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
warnings.filterwarnings("ignore")  # 忽略参数配置时的警告信息
for param in pdq:
    try:
        mod = sm.tsa.statespace.SARIMAX(deterministic, order=param)
        results = mod.fit()
        print('ARIMA{}- AIC:{}'.format(param, results.aic))
    except:
        continue

# 建立ARIMA模型，最优参数根据AIC准则选取（2, 0，2）
mod = sm.tsa.statespace.SARIMAX(deterministic,
                                order=(2, 0, 2),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit(disp=0)
print(results.summary().tables[1])  # 模型的诊断表
results.plot_diagnostics(figsize=(15, 12))  # 模型的拟合诊断图
plt.show()
residuals = pd.DataFrame(results.resid)  # 对于训练数据的拟合的残差值
residuals = residuals.rename(columns={0:'Residuals'})  # 改变列的名字

# 画出拟合残差的图像并进行ACF和PACF分析
residuals.plot()
plt.ylabel('Residuals(vehicles)', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.show()
print(residuals.describe())
# ARIMA的残差
residuals = results.forecasts_error
plot_acf(residuals.flatten(), lags=576)
plt.title('')
plt.ylabel('Residuals ACF', fontsize=16)
plt.xlabel('Time Lag(5mins)', fontsize=16)
plt.show()
plot_pacf(residuals.flatten(), lags=576)
plt.title('')
plt.ylabel('Residuals PACF', fontsize=16)
plt.xlabel('Time Lag(5mins)', fontsize=16)
plt.show()


pred = results.get_prediction(start=pd.to_datetime('2016-02-01 00:00:00'), dynamic=False)  # 利用建立好的模型进行预测
pred_ci = pred.conf_int()  # 置信度区间计算

ax = y['2016-02-01 00:00:00':].plot(label='observed')  # 画出真实值
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)  # 画出预测的值
ax.fill_between(pred_ci.index,  # 画出置信区间，其中pred_ci.iloc[:,0]表示的是下界，另外一个表示的是上界
                pred_ci.iloc[:, 0],  # 下界
                pred_ci.iloc[:, 1],  # 上界
                color='k',
                alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('traffic flow')
plt.legend()
plt.show()
y_forecasted = pred.predicted_mean
y_truth = deterministic['2016-02-01 00:00:00':]

# 预测结果的ARIMA的残差分析
residuals = results.forecasts_error
plot_acf(residuals.flatten(), lags=576)
plt.title('')
plt.ylabel('Residuals ACF', fontsize=16)
plt.xlabel('Time Lag(5mins)', fontsize=16)
plt.show()
plot_pacf(residuals.flatten(), lags=576)
plt.title('')
plt.ylabel('Residuals PACF', fontsize=16)
plt.xlabel('Time Lag(5mins)', fontsize=16)
plt.show()

# 计算预测误差
mae = ((y_forecasted - y_truth) / y_truth).abs().mean()
print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 5)))

# ARIMA的残差
residuals = results.forecasts_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(residuals.flatten(), lags=10)
plt.show()
plot_pacf(residuals.flatten(), lags=10)
plt.show()

# 检验序列具有ARCH效应
at = residuals.flatten()[0:2880]
at2 = np.square(at)
plt.figure(figsize=(10,6))
plt.subplot(211)
plt.plot(at,label = 'at')
plt.legend()
plt.subplot(212)
plt.plot(at2,label='at^2')
plt.legend(loc=0)
plt.show()
# 检查残差平方项的ACF和PACF效应
plot_acf(at2, lags=30)
plt.title('', fontsize=16)
plt.ylabel('Squared Value of Residuals ACF', fontsize=16)
plt.xlabel('Time Lag(5mins)', fontsize=16)
plt.show()
plot_pacf(at2, lags=30)
plt.title('', fontsize=16)
plt.ylabel('Squared Value of Residuals PACF', fontsize=16)
plt.xlabel('Time Lag(5mins)', fontsize=16)
plt.show()

# 建立GJR-ARCH模型
returns = pd.Series(residuals.flatten(),
                    index=pd.date_range(start='2016-02-01 00:00:00',
                                        periods=16992,
                                        freq='5min',
                                        normalize=True
                                        )
                    )
am = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
res = am.fit(update_freq=5, disp='off')
print(res.summary())

index = returns.index
start_loc = 0
end_loc = 20
forecasts = {}
import sys
for i in range(len(returns)-20):
    sys.stdout.write('.')
    sys.stdout.flush()
    res2 = am.fit(first_obs=i, last_obs=i+end_loc, disp='off')
    temp = res.forecast(horizon=1).variance
    fcast = temp.iloc[i+end_loc-1]
    forecasts[fcast.name] = fcast

res_conditional_volatility_prediction = pd.DataFrame(forecasts).T
res_conditional_volatility_prediction.plot(label='volatility_perdiction')
plt.legend()
plt.show()
res.conditional_volatility.plot(label='volatility_real')
plt.legend()
plt.show()


# 原始序列构造完整序列处理过程与原始序列进行对比
# hybird_data = smoothed_data + res.resid
# mae2 = np.mean(np.abs(((hybird_data - real_data) / real_data)[1:]))
# print('The Mean Absolute Error of our forecasts is {}'.format(round(mae2, 5)))
#
# plt.plot(hybird_data[288:576], label='hybrid_data', linewidth=1)
# plt.plot(real_data[288:576], label='real_data', linewidth=1)
# plt.show()
# pd.Series(res.conditional_volatility).to_csv(r"D:\桌面\res_conditional_volatility.csv")


mae = (np.abs(np.array(history_average) - np.array(y_truth))/np.array(y_truth)).mean()
print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 5)))