# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   LSTM_no_hybrid.py
@Time    :   2018/10/10 16:47
@Desc    :
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from statsmodels.nonparametric.smoothers_lowess import lowess


class LstmConfig(object):
    INPUT_NODE_VAR = 6  # 手动提取特征的数量
    TIME_STEPS = 8  # 用于计算的时滞
    SPACE_STEPS = 2  # LSTM输入图片的空间维度
    WHITCH_FEATURE = [0, 1]  # LSTM神经网络的输入数据
    OUTPUT_NODE = 1  # 输出节点个数
    HIDDEN_NODE = 128  # LSTM隐含层的神经元个数
    STACKED_LAYERS = 2  # LSTM堆叠层数
    FC1_HIDDEN = 64  # 聚合特征回归网络的神经元个数
    FC2_HIDDEN = 32  # 同上
    BATCH_SIZE = 100  # batchsize数
    LEARNING_RATE_BASE = 1e-4  # 初始学习率
    LEARNING_RATE_DECAY = 0.99  # 衰减
    REGULARIZATION_RATE = 1e-4  # 正则化系数
    TRAINING_STEPS = 50000  # 迭代次数
    DISP_PER_TIMES = 1000  # 间隔多少次显示预测效果
    MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减
    MODEL_SAVE_PATH = r"D:\Users\yyh\Pycharm_workspace\hybrid_model\model_saver"
    MODEL_NAME = "model"


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
    var_calc_step = 4  # 使用之前多场的时滞来计算当前的方差
    slide_slect = True  # 构造数据是否使用滑动选取数据
    data_select = ['17.99']  # 仿真使用的数据是哪个路口的
    disp_day = 1  # 画图展示的日期
    disp_seg = [x for x in range(288 * (disp_day - 1), 288 * disp_day)]


def get_weight_variable(shape, regularizer=None):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def get_bais_variable(shape):
    biases = tf.get_variable("biases", shape, initializer=tf.constant_initializer(0.1))
    return biases


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
    global normal_data_min
    global normal_data_gap
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
        fc2_weights = get_weight_variable([nn_config.HIDDEN_NODE, nn_config.OUTPUT_NODE],
                                          regularizer=regularizer)
        fc2_biases = get_bais_variable([nn_config.OUTPUT_NODE])
        y = tf.matmul(y_lstm[-1], fc2_weights) + fc2_biases

    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(nn_config.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy_mean = tf.reduce_sum(tf.square(y_ - y)) / nn_config.BATCH_SIZE
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # learning_rate = tf.train.exponential_decay(
    #     nn_config.LEARNING_RATE_BASE,
    #     global_step,
    #     x_train.shape[0] / nn_config.BATCH_SIZE,
    #     nn_config.LEARNING_RATE_DECAY,
    #     staircase=True)
    #
    # train_step = tf.train.GradientDescentOptimizer(
    #     learning_rate).minimize(loss, global_step=global_step)
    train_step = tf.train.AdamOptimizer(nn_config.LEARNING_RATE_BASE).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(nn_config.TRAINING_STEPS):
            sample1 = np.random.randint(0, x_train.shape[0], size=(1, nn_config.BATCH_SIZE))
            train_datas = x_train[sample1].reshape(-1, nn_config.TIME_STEPS * nn_config.SPACE_STEPS)
            train_label = y_train[sample1].reshape(-1, 1)

            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x_1: train_datas, y_: train_label})

            if i % nn_config.DISP_PER_TIMES == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
        saver.save(sess, os.path.join(nn_config.MODEL_SAVE_PATH, nn_config.MODEL_NAME), global_step=global_step)
        print("Optimization Finished!")
        # test
        global prediction
        test_data = x_test.reshape(-1, nn_config.TIME_STEPS * nn_config.SPACE_STEPS)
        test_label = y_test.reshape(-1, 1)
        prediction = sess.run(y, feed_dict={x_1: test_data, y_: test_label})


def cnn_train(data, label):
    """
    CNN神经网络
    :param data:
    :param label:
    :return:
    """
    global y_test
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=False)
    x_1 = tf.placeholder(tf.float32, [None, nn_config.TIME_STEPS * nn_config.SPACE_STEPS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, nn_config.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(nn_config.REGULARIZATION_RATE)
    with tf.variable_scope('conv1'):
        conv1_weights = get_weight_variable([2, 2, 1, 32])
        conv1_biases = get_bais_variable([32])
        x_image = tf.reshape(x_1, [-1, nn_config.TIME_STEPS, nn_config.SPACE_STEPS, 1])
        conv1 = tf.nn.conv2d(x_image, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv2'):
        conv2_weights = get_weight_variable([2, 2, 32, 16])
        conv2_biases = get_bais_variable([16])
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    reshaped = tf.reshape(pool2, [-1, 32])
    with tf.variable_scope('fc_1'):
        fc1_weights = get_weight_variable([32, 16], regularizer=regularizer)
        fc1_biases = get_bais_variable([16])
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
    with tf.variable_scope('fc_2'):
        fc2_weights = get_weight_variable([16, 1], regularizer=regularizer)
        fc2_biases = get_bais_variable([nn_config.OUTPUT_NODE])
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

        error = tf.reduce_sum(tf.square(logit - y_)) / nn_config.BATCH_SIZE
        #    softMaxError = -tf.reduce_sum(y_ * tf.log(logit))
        regularizer = tf.contrib.layers.l2_regularizer(nn_config.REGULARIZATION_RATE)
        regularization = regularizer(fc1_weights) + regularizer(fc2_weights)
        loss = error + regularization
        train_step = tf.train.AdamOptimizer(nn_config.LEARNING_RATE_BASE).minimize(loss)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(nn_config.TRAINING_STEPS):
            sample1 = np.random.randint(0, x_train.shape[0], size=(1, nn_config.BATCH_SIZE))
            train_datas = x_train[sample1].reshape(-1, nn_config.TIME_STEPS * nn_config.SPACE_STEPS)
            train_label = y_train[sample1].reshape(-1, 1)

            if i % nn_config.DISP_PER_TIMES == 0:
                a = error.eval(feed_dict={x_1: train_datas, y_: train_label})
                print("After %d training step(s), loss on training batch is %g." % (i, a))
            train_step.run(feed_dict={x_1: train_datas, y_: train_label})

        print("Optimization Finished!")
        # test
        global prediction
        test_data = x_test.reshape(-1, nn_config.TIME_STEPS * nn_config.SPACE_STEPS)
        test_label = y_test.reshape(-1, 1)
        prediction = sess.run(logit, feed_dict={x_1: test_data, y_: test_label})


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

    smoothed_flow = smooth_data(flow_demo, frac_param=file_config.flow_frac_param / (len(flow_demo) / 288))
    flow_demo_var = var_calc(flow_demo, smoothed_flow, calc_step=file_config.var_calc_step)
    smoothed_speed = smooth_data(speed_demo, frac_param=file_config.speed_frac_param / (len(flow_demo) / 288))
    speed_demo_var = var_calc(speed_demo, smoothed_speed, calc_step=file_config.var_calc_step)
    flow_demo_var = normal_data(flow_demo_var)
    speed_demo_var = normal_data(speed_demo_var)
    flow_demo_var = flow_demo_var.clip(0, 0.5) * 2  # 除去流量的奇异值
    speed_demo_var = speed_demo_var.clip(0, 0.5) * 2  # 除去速度的奇异值
    merge_data = np.concatenate([speed_demo,
                                 flow_demo,
                                 flow_demo_var,
                                 speed_demo_var,
                                 smoothed_flow,
                                 smoothed_speed], axis=1)
    norm_data = normal_data(merge_data)
    norm_data = norm_data[file_config.var_calc_step:, :]  # 去除用于计算方差的那一部分
    return norm_data


if __name__ == '__main__':
    nn_config = LstmConfig()
    file_config = DataProcessConfig()
    merged_data = merge_data(file_config)
    # 仅使用流量作为输入
    data_ = data_pro(merged_data[:, nn_config.WHITCH_FEATURE], nn_config.TIME_STEPS, True)
    label_ = merged_data[nn_config.TIME_STEPS:, 1]
    lstm_train(data_, label_)

    # x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    plt.plot(y_test[:288], color="blue", linewidth=1, linestyle="-", label="real")
    plt.plot(prediction[:288], color="red", linewidth=1, linestyle="-", label="simulation")
    plt.xlabel('Time (per 5 min)')
    plt.ylabel('Totle traffic flow (vehicles)')
    plt.legend(loc='upper right')
    plt.show()
    d = abs(y_test - prediction.flatten())
    mape = sum(d / y_test) / y_test.shape[0]
    mae = sum(d) / y_test.shape[0]
    print('MAPE=', mape, '\nMAE=', mae)

    plt.plot(y_test[:288], color="green", linewidth=1, linestyle="-", label="real")
    plt.plot(prediction[:288], color="red", linewidth=1, linestyle="-", label="simulation")
    plt.xlabel('Time (per 5 min)')
    plt.ylabel('Total traffic flow (vehicles)')
    plt.legend(loc='upper right')
    plt.show()
