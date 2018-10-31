# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   test.py
@Time    :   2018/10/12 11:08
@Desc    :
"""

import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt


merged_data = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\merged_data.csv', index_col=0).iloc[-3340:]
arima_pred = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\arima_prediction.csv', index_col=0).iloc[-3340:]
arima_trend_pred = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\arima_trend_prediction.csv', index_col=0).iloc[-3340:]
lstm_pred = sio.loadmat(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\prediction_lstm.mat').get('pred')
lstm_pred_hybrid = sio.loadmat(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\prediction_lstm_hybrid.mat').get('pred')
test_real = sio.loadmat(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result\test_real.mat').get('real').flatten()
print(len(arima_pred),len(arima_trend_pred),len(lstm_pred),len(lstm_pred_hybrid),len(test_real))


