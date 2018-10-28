# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   test.py
@Time    :   2018/10/12 11:08
@Desc    :
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


class LstmConfig(object):
    INPUT_NODE_VAR = 6  # 手动提取特征的数量
    TIME_STEPS = 8  # 用于计算的时滞
    SPACE_STEPS = 1  # LSTM输入图片的空间维度
    OUTPUT_NODE = 1  # 输出节点个数
    HIDDEN_NODE = 128  # LSTM隐含层的神经元个数
    STACKED_LAYERS = 2  # LSTM堆叠层数
    FC1_HIDDEN = 64  # 聚合特征回归网络的神经元个数
    FC2_HIDDEN = 32  # 同上
    BATCH_SIZE = 100  # batchsize数
    LEARNING_RATE_BASE = 1e-4  # 初始学习率
    LEARNING_RATE_DECAY = 0.99  # 衰减
    REGULARIZATION_RATE = 1e-4  # 正则化系数
    TRAINING_STEPS = 25000  # 迭代次数
    DISP_PER_TIMES = 1000  # 间隔多少次显示预测效果
    MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减
    QUEUE_CAPACITY = 10000 + BATCH_SIZE * 3
    MIN_AFTER_DEQUEUE = 5000
    EPOCHS = 150
    MODEL_SAVE_PATH = r'D:\Users\yyh\Pycharm_workspace\hybrid_model\model_saver'
    RESULT_SAVE_PATH = r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\predicton_result'
    MODEL_NAME = 'model'


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
    smoothed_flow_data_dir = r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\flow_data_param=0.10.mat'
    smoothed_speed_data_dir = r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\speed_data_param=0.10.mat'
    flow_frac_param = 0.05  # 用于平滑流量的参数
    speed_frac_param = 0.05  # 用于平滑速度的参数
    var_calc_step = 12  # 使用之前多长的时滞来计算当前的方差
    slide_slect = True  # 构造数据是否使用滑动选取数据
    data_select = ['15.08', '15.63', '16.12', '16.67', '17.23', '17.99',
                   '18.7', '19.21', '19.71', '20.22', '20.93', '21.36']  # 仿真使用的数据是哪个路口的
    label_select = ['17.99']  # 预测流量是使用的哪个路口
    disp_day = 2  # 画图展示的日期


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
            temp = np.zeros((size[0] - time_steps + 1, size[1] * time_steps))
            for i in range(data.shape[0] - time_steps + 1):
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


def data_generate():
    path_test = r'D:\桌面\allnodeidceshishuju\allnodeidceshishuju'
    path_train = r'D:\桌面\allnodeidshujuchuli2\allnodeidshujuchuli2'
    test_files = os.listdir(path_test)
    test_data = pd.DataFrame()
    for i in test_files:
        temp_path = os.path.join(path_test, i)
        temp_data = pd.read_excel(temp_path, sheetname=0, header=None)
        temp_data = temp_data.rename(columns={0: i})
        test_data = pd.concat([test_data, temp_data[i]], axis=1)
    test_data.to_csv('D:\桌面\zhuowei_test.csv')

    train_files = os.listdir(path_train)
    train_data = pd.DataFrame()
    for j in train_files:
        temp_path = os.path.join(path_train, j)
        temp_data = pd.read_excel(temp_path, sheetname=0, header=None)
        temp_data = temp_data.rename(columns={0: j})
        train_data = pd.concat([train_data, temp_data[j]], axis=1)
    train_data.to_csv('D:\桌面\zhuowei_train.csv')


def get_weight_variable(shape, regularizer=None):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def get_bais_variable(shape):
    biases = tf.get_variable("biases", shape, initializer=tf.constant_initializer(0.1))
    return biases


def lstm_train(data, label):
    """
    神经网络部分
    :param data: 输入的数据
    :param label: 对应的标签
    :return: 神经网络的输出结果
    """
    global y_test
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=False)
    x_1 = tf.placeholder(tf.float32, [None, nn_config.TIME_STEPS * nn_config.SPACE_STEPS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, nn_config.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(nn_config.REGULARIZATION_RATE)
    input_tensor_image = tf.reshape(x_1, [-1, nn_config.TIME_STEPS, nn_config.SPACE_STEPS])

    def lstm():
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(nn_config.HIDDEN_NODE, forget_bias=1.0,
                                                    state_is_tuple=True,
                                                    reuse=tf.get_variable_scope().reuse)
        return lstm_fw_cell

    with tf.variable_scope(None, default_name="Rnn"):
        #    cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2])
        cell = tf.contrib.rnn.MultiRNNCell([lstm() for _ in range(nn_config.STACKED_LAYERS)], state_is_tuple=True)
        output, _ = tf.nn.dynamic_rnn(cell, input_tensor_image, dtype=tf.float32)
        y_lstm = tf.transpose(output, [1, 0, 2])

    # # 不使用手动提取的特征
    with tf.variable_scope('fc_2'):
        fc2_weights = get_weight_variable([nn_config.HIDDEN_NODE, nn_config.OUTPUT_NODE], regularizer=regularizer)
        fc2_biases = get_bais_variable([nn_config.OUTPUT_NODE])
        y = tf.matmul(y_lstm[-1], fc2_weights) + fc2_biases

    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(nn_config.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy_mean = tf.reduce_sum(tf.square(y_ - y)) / nn_config.BATCH_SIZE
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    train_step = tf.train.AdamOptimizer(nn_config.LEARNING_RATE_BASE).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(nn_config.TRAINING_STEPS):
            sample1 = np.random.randint(0, x_train.shape[0], size=(1, nn_config.BATCH_SIZE))
            train_datas = x_train[sample1].reshape(-1, nn_config.TIME_STEPS * nn_config.SPACE_STEPS)
            train_label = y_train[sample1].reshape(-1, 1)

            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x_1: train_datas, y_: train_label})

            if i % nn_config.DISP_PER_TIMES == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
        print("Optimization Finished!")
        # test
        global prediction
        test_data = x_test.reshape(-1, nn_config.TIME_STEPS * nn_config.SPACE_STEPS)
        test_label = y_test.reshape(-1, 1)
        prediction = sess.run(y, feed_dict={x_1: test_data, y_: test_label})


if __name__ == '__main__':
    nn_config = LstmConfig()  # 配置LSTM网络参数
    file_config = DataProcessConfig()  # 配置文件输入参数

    input_data = pd.read_csv(r'D:\software\pycharm\PycharmProjects\demo\merged_data.csv', index_col=0).iloc[288:, :]
    flow_label_real = np.array(merged_data['Real_data']).reshape(-1, 1)
    hybrid_data = np.array(merged_data)
    hybrid_data_normal = normal_data(hybrid_data)  # 输入DNN的特征数据
    flow_label = normal_data(flow_label_real)  # 输入占位符y的标签数据
    lstm_data = flow_label  # 输入lstm网络的数据

    # 仅使用流量作为输入
    lstm_input = data_pro(lstm_data, nn_config.TIME_STEPS, True)[:-1, :]
    hybrid_input = hybrid_data_normal[nn_config.TIME_STEPS - 1:-1, :]
    label_ = flow_label[nn_config.TIME_STEPS:, 0]  # 构造训练测试数据

    # 将处理好的数据加入神经网络训练
    lstm_train(lstm_input, hybrid_input, label_)
