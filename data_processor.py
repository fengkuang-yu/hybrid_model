# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 20:41:46 2018

@author: yyh
"""

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess


class DataProcessConfig(object):
    """
    道路可选择的检测器编号：
    'Datetime \ Milepost', '0.11', '0.44', '1.39', '1.71', '2.21', '2.43',
       '3.42', '3.91', '6.08', '7', '8.03', '8.4', '8.9', '9.75', '10.31',
       '10.79', '11.28', '11.78', '12.28', '13.31', '13.92', '14.27', '15.08',
       '15.63', '16.12', '16.67', '17.23', '17.99', '18.7', '19.21', '19.71',
       '20.22', '20.93', '21.36', '21.83', '22.31', '22.73', '23.51', '23.93',
       '24.39', '25.17', '25.68', '26.16', '26.62', '27.44', '27.96', '28.53',
       '28.98'

    影响方差计算的参数:
    平滑百分比：frac_param
    方差计算的长度：var_calc_step
    选取的路段编号：data_select
    误差计算选用的方案：error_tend
    """
    FLOW_DIR = r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\flow_data.csv'
    SPEED_DIR = r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\speed_data.csv'
    flow_frac_param = 0.05  # 用于平滑流量的参数
    speed_frac_param = 0.05  # 用于平滑速度的参数
    var_calc_step = 2  # 使用之前多长的时滞来计算当前的方差
    slide_slect = True  # 构造数据是否使用滑动选取数据
    data_select = ['17.99']  # 仿真使用的数据是哪个路口的
    disp_day = 2  # 画图展示的日期
    disp_seg = [x for x in range(288 * (disp_day - 1), 288 * disp_day)]


def csv_file_generator():
    """
    由采集的Excel文件生成.CSV文件
    :return: None
    """
    speed_file_dir = r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\405_month2_4_speed.xlsx'
    flow_file_dir = r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\405_month2_4_flow.xlsx'
    speed_data = pd.read_excel(speed_file_dir, sheetname=0)
    flow_data = pd.read_excel(flow_file_dir, sheetname=0)
    flow_data.to_csv(r'Data\flow_data_60.csv', index_label='Datetime \ Milepost')
    speed_data.to_csv(r'Data\speed_data_60.csv', index_label='Datetime \ Milepost')
    return


def data_pro(data, time_steps=None, slide_sep=True):
    """
    数据处理，将列状的数据拓展开为行形式

    :param data: 输入交通数据
    :param time_steps: 分割时间长度
    :return: 处理过的数据
    :param slide_sep: 是否连续切割数据
    """
    if time_steps is None:
        time_steps = 1
    size = data.shape
    if isinstance(data, pd.DataFrame):
        data = np.array(data)
        # 如果输入的数据是矩阵形式的
    if isinstance(data, np.ndarray):
        if slide_sep is True:
            temp = np.zeros((size[0] - time_steps, size[1] * time_steps))
            for i in range(data.shape[0] - time_steps):
                temp[i, :] = data[i:i + time_steps, :].flatten()
            return temp
        else:
            try:
                temp = np.zeros((int(size[0] / time_steps), size[1] * time_steps))
                for i in range(int(size[0] / time_steps)):
                    temp[i, :] = data[i * time_steps:i * time_steps + time_steps, :].flatten()
                return temp
            except Exception:
                print('time_step必须要能整除288')
                raise

    else:
        raise Exception('data_pro数据输入格式错误')


def var_calc(seq, ref_seq=None, calc_step=1, slide_select=True):
    """
    计算输出序列seq的方差并返回，返回的方差长度会比原始的序列短calc_step的长度
    :param seq: 输入原始序列
    :param ref_seq: 输入的参照序列
    :param calc_step: 方差计算所采用的步长
    :param slide_select: 是否连续取值
    :return: res_var: 返回的seq的方差,按照列向量的形式
    """
    seq = np.array(seq)
    ref_seq = np.array(ref_seq)
    if ref_seq is None:
        ref_seq = np.zeros([len(seq), 1])
    res_error = seq - ref_seq
    res_error_pro = data_pro(res_error, time_steps=calc_step, slide_sep=slide_select)
    # 计算与总体趋势的误差
    #    error_tend = res_error_pro.var(axis=1).reshape(-1, 1)  # 使用方差来定义误差的大小
    error_tend = np.mean(abs(res_error_pro), axis=1).reshape(-1, 1)
    res_var = np.concatenate((np.zeros((calc_step, 1)), error_tend), axis=0)
    return res_var


def normal_data(data, reverse=False):
    """
    输入数据按列去均值，只看趋势
    :param data: numpy矩阵
    :param reverse: 反标准化
    :return data: 去均值后的数组
    """
    global normal_data_min  # 将归一化的数据供后面的程序调用，每列的最小值
    global normal_data_gap  # 最大值减去最小值
    if reverse is False:
        data = np.array(data)
        normal_data_min = data.min(axis=0)
        normal_data_gap = data.max(axis=0) - data.min(axis=0)
        mean_ = data - normal_data_min
        data = mean_ / normal_data_gap
    else:
        data = data * normal_data_gap
        data = data + normal_data_min
    return data


def history_average(data):
    """
    这里是计算data数据中的历史平均值
    data的数据是以列形式给出，返回的数据是将列数据裁剪开平均
    """
    demo_array = np.array(data).reshape((-1, 288)).T
    demo_average = np.mean(demo_array, axis=1)
    return demo_average


def smooth_data(data, frac_param=None):
    """
    这个模块是为了给历史数据lowess平滑操作，输入为任意长的列型数据，
    返回同样维度的平滑后的数据
    :param data:输入的列型带噪声数据
    :param frac_param:lowess的参数
    :return: 返回同样维度的列型平滑处理后的数据
    """
    data = np.array(data)
    if frac_param is None:
        frac_param = 0.08 / int(len(data.flatten()) / 288)
    filtered = lowess(data.flatten(),
                      np.array([x for x in range(len(data.flatten()))]),
                      is_sorted=True, frac=frac_param, it=0)
    return filtered[:, 1].reshape(-1, 1)


if __name__ == '__main__':
    csv_file_generator()
