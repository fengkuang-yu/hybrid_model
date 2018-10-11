# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 20:41:46 2018

@author: yyh
"""

import pandas as pd


def generate_data(speed_file_dir, flow_file_dir):
    speed_data = pd.read_excel(speed_file_dir, sheetname=[1])
    flow_data = pd.read_excel(flow_file_dir, sheetname=[1])

    """
    sheetname:
        0: speed
        1: volumn per lane per Hour
        2: volumn all lanes
        3: frequence of congestion
        4: lane number
        5: Dates with available data
    """

    flow_data.to_csv('flow_data_60.csv', index_label='Datetime \ Milepost')
    speed_data.to_csv('speed_data_60.csv', index_label='Datetime \ Milepost')

if __name__ == '__main__':
    speed_file_dir = r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\405_month2_4_speed.xlsx'
    flow_file_dir = r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\405_month2_4_flow.xlsx'
    generate_data(speed_file_dir, flow_file_dir)