# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 20:41:46 2018

@author: yyh
"""

import pandas as pd

def generate_data(filedir):
    
    data = pd.read_excel(filedir, sheetname=[0,2])
    speed_data = data[0]
    flow_data = data[2]
    
    """
    sheetname:
        0: speed
        1: volumn per lane per Hour
        2: volumn all lanes
        3: frequence of congestion
        4: lane number
        5: Dates with available data
    """
    
    
    flow_data.to_csv('flow_data.csv', index_label='Datetime \ Milepost')
    speed_data.to_csv('speed_data.csv', index_label='Datetime \ Milepost')
    