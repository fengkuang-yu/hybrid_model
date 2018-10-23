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

# 处理输入数据
plt.style.use('fivethirtyeight')
pd_data = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\flow_data_59.csv')
y = pd.Series(pd_data['20.93'])
y.index = pd.date_range(start='2018-08-01 00:00:00', periods=16992,freq='5min',normalize=True)

# 对输入数据进行历史平均处理
demo_array = np.array(y).reshape((-1, 288)).T
demo_average = np.mean(demo_array, axis=1)
history_average = pd.Series(demo_average)
history_average.index = pd.date_range(start='0', periods=288, freq='1s',normalize=True)

# 建立ARIMA模型
mod = sm.tsa.statespace.SARIMAX(y, order=(5, 2, 1), enforce_stationarity=False, enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(15, 12))  # 模型的拟合诊断图
plt.show()
pred = results.get_prediction(start=pd.to_datetime('2018-08-01 00:00:00'), dynamic=False)  # 利用建立好的模型进行预测
pred_ci = pred.conf_int()  # 置信度区间计算

ax = y['2018-08-01 00:00:00':].plot(label='observed')  # 画出真实值
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
y_truth = y['2018-08-01 00:00:00':]

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
# plt.plot(results.forecasts_error.flatten()[0:288])
# plt.rcParams['savefig.dpi'] = 600  # 图片像素
# plt.rcParams['figure.dpi'] = 600  # 分辨率
# plt.show()

# 输出平滑后的曲线
real_data = np.array(y)
smoothed_data = real_data - residuals.flatten()
plt.plot(smoothed_data[0:288], linewidth=1, label='smoothed_data')
plt.plot(real_data[0:288], linewidth=1, label='real_data')
plt.rcParams['savefig.dpi'] = 600  # 图片像素
plt.rcParams['figure.dpi'] = 600  # 分辨率
plt.legend()
plt.show()

# 检验序列具有ARCH效应
# at = residuals.flatten()[0:2880]
# at2 = np.square(at)
# plt.figure(figsize=(10,6))
# plt.subplot(211)
# plt.plot(at,label = 'at')
# plt.legend()
# plt.subplot(212)
# plt.plot(at2,label='at^2')
# plt.legend(loc=0)
# plt.show()

# 建立GJR-ARCH模型
returns = pd.Series(residuals.flatten(),
                    index=pd.date_range(start='2018-08-01 00:00:00',
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