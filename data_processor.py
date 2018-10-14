# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 20:41:46 2018

@author: yyh
"""

import datetime

import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, drange
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
    FLOW_DIR = r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\flow_data_59.csv'
    SPEED_DIR = r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\speed_data_59.csv'
    flow_frac_param = 0.05  # 用于平滑流量的参数
    speed_frac_param = 0.05  # 用于平滑速度的参数
    var_calc_step = 2  # 使用之前多长的时滞来计算当前的方差
    slide_slect = True  # 构造数据是否使用滑动选取数据
    data_select = ['17.23', '17.99', '18.7', '19.21', '19.71',
                   '20.22', '20.93', '21.36', '21.83', '22.31',
                   '22.73', '23.51', '23.93']  # 仿真使用的数据是哪个路口的
    label_select = ['20.93']  # 预测流量是使用的哪个路口
    disp_day = 2  # 画图展示的日期


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


def merge_data(file_config=None):
    """
    神经网络的训练和测试数据集制作
    :param file_config: 数据集中的参数
    :return: 返回做好的数据集
    """
    flow_data = pd.read_csv(file_config.FLOW_DIR, index_col='Datetime \ Milepost')
    speed_data = pd.read_csv(file_config.SPEED_DIR, index_col='Datetime \ Milepost')

    flow_demo = np.array(flow_data[file_config.data_select], dtype=float)
    speed_demo = np.array(speed_data[file_config.data_select], dtype=float)
    flow_label = np.array(flow_data[file_config.label_select], dtype=float)
    speed_label = np.array(speed_data[file_config.label_select], dtype=float)

    smoothed_flow = smooth_data(flow_label, frac_param=file_config.flow_frac_param / (len(flow_demo) / 288))
    flow_demo_var = var_calc(flow_label, smoothed_flow, calc_step=file_config.var_calc_step)
    smoothed_speed = smooth_data(speed_label, frac_param=file_config.speed_frac_param / (len(flow_demo) / 288))
    speed_demo_var = var_calc(speed_label, smoothed_speed, calc_step=file_config.var_calc_step)
    flow_demo_var = normal_data(flow_demo_var)
    speed_demo_var = normal_data(speed_demo_var)
    flow_demo_var = flow_demo_var.clip(0, 0.5) * 2  # 除去流量的奇异值
    speed_demo_var = speed_demo_var.clip(0, 0.5) * 2  # 除去速度的奇异值
    lstm_data = np.concatenate([speed_demo, flow_demo], axis=1)
    hybrid_data= np.concatenate([speed_demo,
                                 flow_demo,
                                 flow_demo_var,
                                 speed_demo_var,
                                 smoothed_flow,
                                 smoothed_speed], axis=1)
    lstm_norm_data = normal_data(lstm_data)[file_config.var_calc_step:, :]  # 去除用于计算方差的那一部分
    hybrid_norm_data = normal_data(hybrid_data)[file_config.var_calc_step:, :]  # 去除用于计算方差的那一部分
    return lstm_norm_data, hybrid_norm_data, flow_label


def plot_one_day(data1, data2, y_label='Traffic flow(Vehicles)', x_label='Time', legend=None):
    """
    画出图
    :param data1: 真实数据
    :param data2: 仿真数据
    :param y_label: y轴的坐标
    :param x_label: x轴的坐标
    :param legend: 图例
    :return: 无
    """
    formatter = DateFormatter('%H:%M')  # 时间表现形式，这里只显示了时分
    d1 = datetime.datetime(2018, 2, 10, 0, 0, 0)
    d2 = datetime.datetime(2018, 2, 11, 0, 0, 0)
    delta = datetime.timedelta(minutes=5)  # 以0.5秒为间隔生成时间序列
    x = drange(d1, d2, delta)
    y1 = data1
    y2 = data2
    fig1, ax1 = plt.subplots()
    plt.plot(x, y1)
    plt.plot(x, y2)
    ax1.xaxis.set_major_formatter(formatter)
    tick_spacing = 1 / 6
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax1.xaxis.set_tick_params(rotation=0, labelsize=10)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    if legend is None:
        plt.legend(['real traffic flow', 'simulation result'], loc=1)
    else:
        plt.legend(legend)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()


if __name__ == '__main__':
    # 第一张图片，没有发生拥塞时的
    def without_blocking(disp_day=2):
        """
        画出没有拥塞的图
        :param disp_day: 默认展示第二天的情况
        :return:
        """
        file_config = DataProcessConfig()
        file_config.flow_frac_param = 0.05
        file_config.var_calc_step = 12
        file_config.disp_day = 2
        disp_seg = [x for x in range(288 * (file_config.disp_day - 1) - file_config.var_calc_step,
                                     288 * file_config.disp_day - file_config.var_calc_step)]
        merged_data = merge_data(file_config)
        formatter = DateFormatter('%H:%M')  # 时间表现形式，这里只显示了时分
        d1 = datetime.datetime(2018, 2, 10, 0, 0, 0)
        d2 = datetime.datetime(2018, 2, 11, 0, 0, 0)
        delta = datetime.timedelta(minutes=5)  # 以0.5秒为间隔生成时间序列
        x = drange(d1, d2, delta)
        y = merged_data[disp_seg, 0:4]
        fig1, ax1 = plt.subplots()
        plt.plot(x, y)
        ax1.xaxis.set_major_formatter(formatter)
        tick_spacing = 1 / 6
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax1.xaxis.set_tick_params(rotation=0, labelsize=10)
        plt.rcParams['savefig.dpi'] = 300  # 图片像素
        plt.rcParams['figure.dpi'] = 300  # 分辨率
        plt.legend(['speed', 'flow', 'flow variance', 'speed variance'], loc=1)
        plt.ylabel('The magnitude of normalized data')
        plt.xlabel('Time')
        plt.savefig('Data/figures/without_bolck.png')
        plt.show()


    # 发生拥塞时的四种关系17,18
    def blocking(disp_day=20):
        """
        画图程序，画出拥塞时的图
        :param disp_day:
        :return:
        """
        file_config = DataProcessConfig()
        file_config.flow_frac_param = 0.05
        file_config.speed_frac_param = 0.05
        file_config.var_calc_step = 12
        file_config.disp_day = disp_day
        disp_seg = [x for x in range(288 * (file_config.disp_day - 1) - file_config.var_calc_step,
                                     288 * file_config.disp_day - file_config.var_calc_step)]
        merged_data = merge_data(file_config)
        formatter = DateFormatter('%H:%M')  # 时间表现形式，这里只显示了时分
        d1 = datetime.datetime(2018, 2, 10, 0, 0, 0)
        d2 = datetime.datetime(2018, 2, 11, 0, 0, 0)
        delta = datetime.timedelta(minutes=5)  # 以0.5秒为间隔生成时间序列
        x = drange(d1, d2, delta)
        y = merged_data[disp_seg, 0:4]
        fig2, ax2 = plt.subplots()
        plt.plot(x, y)
        ax2.xaxis.set_major_formatter(formatter)
        tick_spacing = 1 / 6
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax2.xaxis.set_tick_params(rotation=0, labelsize=10)
        plt.rcParams['savefig.dpi'] = 300  # 图片像素
        plt.rcParams['figure.dpi'] = 300  # 分辨率
        plt.legend(['speed', 'flow', 'flow variance', 'speed variance'], loc=4)
        plt.ylabel('The magnitude of normalized data')
        plt.xlabel('Time')
        plt.savefig('Data/figures/with_bolck.png')
        plt.show()
