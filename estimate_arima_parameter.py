# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   estimate_arima_parameter.py
@Time    :   2018/10/21 21:09
@Desc    :
"""

import warnings
import itertools
import pandas as pd
import numpy as npb
import statsmodels.api as sm
import matplotlib.pyplot as plt
from arch import arch_model

# 处理输入数据
plt.style.use('fivethirtyeight')
pd_data = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\flow_data_59.csv')
y = pd.Series(pd_data['20.93'])
y.index = pd.date_range(start='2018-08-01 00:00:00', periods=16992,freq='5min',normalize=True)

# 建立ARIMA模型
mod = sm.tsa.statespace.SARIMAX(y, order=(18, 2, 1), enforce_stationarity=False, enforce_invertibility=False)
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

residuals = results.forecasts_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(residuals.flatten(), lags=576)
plt.show()
plot_pacf(residuals.flatten(), lags=300)
plt.show()



p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
warnings.filterwarnings("ignore") # specify to ignore warning messages
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue