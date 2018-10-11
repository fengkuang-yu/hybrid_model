# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 20:41:46 2018

@author: yyh
"""

import pandas as pd

if __name__ == '__main__':
    speed_file_dir = r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\405_month2_4_speed.xlsx'
    flow_file_dir = r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\405_month2_4_flow.xlsx'
    speed_data = pd.read_excel(speed_file_dir, sheetname=0)
    flow_data = pd.read_excel(flow_file_dir, sheetname=0)
    flow_data.to_csv(r'Data\flow_data_60.csv', index_label='Datetime \ Milepost')
    speed_data.to_csv(r'Data\speed_data_60.csv', index_label='Datetime \ Milepost')
