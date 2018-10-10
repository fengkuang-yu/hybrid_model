# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   demo.py
@Time    :   2018/10/5 11:05
@Desc    :
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def get_weight_variable(shape, regularizer=None):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def get_bais_variable(shape):
    biases = tf.get_variable("biases", shape, initializer=tf.constant_initializer(0.1))
    return biases


class LstmConfig(object):
    INPUT_NODE_VAR = 2
    TIME_STEPS = 8
    SPACE_STEPS = 20
    WHITCH_FEATURE = [1]
    OUTPUT_NODE = 1
    HIDDEN_NODE = 128
    STACKED_LAYERS = 2
    FC_HIDDEN = 128
    BATCH_SIZE = 20
    LEARNING_RATE_BASE = 1e-3
    LEARNING_RATE_DECAY = 0.99
    REGULARIZATION_RATE = 0.0001
    TRAINING_STEPS = 20000
    DISP_PER_TIMES = 1000
    MOVING_AVERAGE_DECAY = 0.99
    MODEL_SAVE_PATH = r"D:\Users\yyh\Pycharm_workspace\hybrid_model\model_saver"
    MODEL_NAME = "model"


def lstm_train(data, label, test_data, test_label):
    """
    神经网络部分
    :param data: 输入的数据
    :param label: 对应的标签
    :return: 神经网络的输出结果
    """
    global y_test
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0, shuffle=True)
    x_test = test_data
    y_test = test_label
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

    cross_entropy_mean = tf.reduce_sum(tf.square(y_ - y)) / nn_config.BATCH_SIZE
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    train_step = tf.train.AdamOptimizer(nn_config.LEARNING_RATE_BASE).minimize(loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(nn_config.TRAINING_STEPS):
            sample1 = np.random.randint(0, x_train.shape[0], size=(1, nn_config.BATCH_SIZE))
            train_datas = x_train[sample1].reshape(-1, nn_config.TIME_STEPS * nn_config.SPACE_STEPS)
            train_label = y_train[sample1].reshape(-1, 1)

            sess.run(train_step, feed_dict={x_1: train_datas, y_: train_label})

            if i % nn_config.DISP_PER_TIMES == 0:
                loss_value = sess.run(cross_entropy_mean, feed_dict={x_1: train_datas, y_: train_label})
                print("After %d training step(s), loss on training batch is %g." % (i, loss_value))
        print("Optimization Finished!")
        # test
        global prediction
        test_data = x_test.reshape(-1, nn_config.TIME_STEPS * nn_config.SPACE_STEPS)
        test_label = y_test.reshape(-1, 1)
        prediction = sess.run(y, feed_dict={x_1: test_data, y_: test_label})


if __name__ == '__main__':
    nn_config = LstmConfig()
    data = pd.read_excel('DATA/input_data.xlsx', Sheetname='Sheet1', header=None)
    labels = pd.read_excel('DATA/input_labels.xlsx', Sheetname='Sheet1', header=None)
    test_data = pd.read_excel('DATA/test_data_10days.xlsx', Sheetname='Sheet1', header=None)
    test_labels = pd.read_excel('DATA/test_data_labels_10days.xlsx', Sheetname='Sheet1', header=None)

    # 将数据处理为所需类型
    input_data = data.as_matrix()
    Imagesize = nn_config.TIME_STEPS * nn_config.SPACE_STEPS
    ColumnNum = len(input_data)
    SampleNum = ColumnNum - nn_config.TIME_STEPS

    # 生成训练样本
    input_data1 = np.zeros((SampleNum, Imagesize))
    for i in range(SampleNum):
        temp = input_data[i:i + nn_config.TIME_STEPS, :]
        input_data1[i, :] = temp.reshape(1, Imagesize)

    # 输入训练样本labels
    input_data2 = labels.as_matrix()[nn_config.TIME_STEPS:, :]

    # 输入10天的测试数据
    test_data_10days = test_data.as_matrix()
    TestSampleNum = len(test_data_10days) - nn_config.TIME_STEPS
    input_data3 = np.zeros((TestSampleNum, Imagesize))
    for i in range(TestSampleNum):
        temp = test_data_10days[i:i + nn_config.TIME_STEPS, :]
        input_data3[i, :] = temp.reshape(1, Imagesize)

    # 输入10天的测试数据的标签
    input_data4 = test_labels.as_matrix()[nn_config.TIME_STEPS:, :]
    lstm_train(input_data1, input_data2, input_data3, input_data4)

    d = abs(y_test - input_data4)
    mape = sum(d / y_test) / len(input_data4)
    mae = sum(d) / len(input_data4)
    print('Ten days MAPE=', mape, '\nTen days MAE=', mae)

    plt.plot(prediction[0:300])
    plt.plot(input_data4[0:300])
    plt.show()
