# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   hybrid_lstm.py
@Time    :   2018/10/4 14:37
@Desc    :
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_processor import *


class LstmConfig(object):
    PRED_STEP = 2  # 预测步数，1表示5分钟交通流量预测234分别表示10-20分钟
    INPUT_NODE_VAR = 6  # 手动提取特征的数量
    TIME_STEPS = 8  # 用于计算的时滞
    SPACE_STEPS = 1  # LSTM输入图片的空间维度
    OUTPUT_NODE = 1  # 输出节点个数
    HIDDEN_NODE = 32  # LSTM隐含层的神经元个数
    STACKED_LAYERS = 2  # LSTM堆叠层数
    FC1_HIDDEN = 64  # 聚合特征回归网络的神经元个数
    FC2_HIDDEN = 32  # 同上
    BATCH_SIZE = 100  # batchsize数
    LEARNING_RATE_BASE = 1e-4  # 初始学习率
    LEARNING_RATE_DECAY = 0.99  # 衰减
    REGULARIZATION_RATE = 1e-4  # 正则化系数
    TRAINING_STEPS = 40000  # 迭代次数
    DISP_PER_TIMES = 1000  # 间隔多少次显示预测效果
    MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减
    QUEUE_CAPACITY = 1000 + BATCH_SIZE * 3
    MIN_AFTER_DEQUEUE = 500


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

    # 生成batch的数据配置
    input_queue = tf.train.slice_input_producer([x_train, y_train])
    lstm_data = input_queue[0]
    label_data = input_queue[1]

    train_datas1, train_label1 = tf.train.shuffle_batch([lstm_data, label_data],
                                                        batch_size=nn_config.BATCH_SIZE,
                                                        num_threads=32,
                                                        capacity=nn_config.QUEUE_CAPACITY,
                                                        min_after_dequeue=nn_config.MIN_AFTER_DEQUEUE)
    train_datas = tf.reshape(train_datas1, [-1, nn_config.TIME_STEPS * nn_config.SPACE_STEPS])
    train_label = tf.reshape(train_label1, [-1, nn_config.OUTPUT_NODE])

    # 训练开始
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(nn_config.TRAINING_STEPS):
            cur_lstm_data, cur_label = sess.run([train_datas, train_label])
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x_1: cur_lstm_data, y_: cur_label})

            if i % nn_config.DISP_PER_TIMES == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
        coord.request_stop()
        coord.join(threads)
        print("Optimization Finished!")
        # test
        global prediction
        test_data = x_test.reshape(-1, nn_config.TIME_STEPS * nn_config.SPACE_STEPS)
        test_label = y_test.reshape(-1, nn_config.SPACE_STEPS)
        prediction = sess.run(y, feed_dict={x_1: test_data, y_: test_label})


def lstm_train_hybrid(data1, data2, label):
    """
    神经网络部分
    :param data1: 输入给LSTM神经网络的数据
    :param label: 网络的输出对应的标签即预测时刻的交通流量
    :param data2: 手工提取特征的输入的数据
    :return: 神经网络的输出结果
    """
    # 数据测试集和训练集划分
    global y_test
    x_train, x_test, y_train, y_test = train_test_split(data1, label, test_size=0.2, shuffle=False)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(data2, label, test_size=0.2, shuffle=False)

    # 声明数据的占位符
    x_1 = tf.placeholder(tf.float32, [None, nn_config.TIME_STEPS * nn_config.SPACE_STEPS], name='x-input1')
    x_2 = tf.placeholder(tf.float32, [None, nn_config.INPUT_NODE_VAR], name='x-input2')
    y_ = tf.placeholder(tf.float32, [None, nn_config.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(nn_config.REGULARIZATION_RATE)  # 使用L2正则化
    input_tensor_image = tf.reshape(x_1, [-1, nn_config.TIME_STEPS, nn_config.SPACE_STEPS])

    # 标准的LSTM模块
    def lstm():
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(nn_config.HIDDEN_NODE,
                                                    forget_bias=1.0,
                                                    state_is_tuple=True,
                                                    reuse=tf.get_variable_scope().reuse)
        return lstm_fw_cell

    with tf.variable_scope(None, default_name="Rnn1"):
        cell = tf.contrib.rnn.MultiRNNCell([lstm() for _ in range(nn_config.STACKED_LAYERS)], state_is_tuple=True)
        output, _ = tf.nn.dynamic_rnn(cell, input_tensor_image, dtype=tf.float32)
        y_lstm = tf.transpose(output, [1, 0, 2])

    # 处理手工提取的特征，将神经网络提取的特征和手工提取的特征合并
    y_concat = tf.concat([y_lstm[-1], x_2], axis=1)

    # 定义DNN神经网络处理混合特征数据
    with tf.variable_scope('fc_1'):
        fc1_weights = get_weight_variable([nn_config.HIDDEN_NODE + nn_config.INPUT_NODE_VAR, nn_config.FC1_HIDDEN],
                                          regularizer=regularizer)
        fc1_biases = get_bais_variable([nn_config.FC1_HIDDEN])
        fc1 = tf.nn.relu(tf.matmul(y_concat, fc1_weights) + fc1_biases)

    with tf.variable_scope('fc_2'):
        fc2_weights = get_weight_variable([nn_config.FC1_HIDDEN, nn_config.FC2_HIDDEN],
                                          regularizer=regularizer)
        fc2_biases = get_bais_variable([nn_config.FC2_HIDDEN])
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)

    with tf.variable_scope('fc_3'):
        fc3_weights = get_weight_variable([nn_config.FC2_HIDDEN, nn_config.OUTPUT_NODE],
                                          regularizer=regularizer)
        fc3_biases = get_bais_variable([nn_config.OUTPUT_NODE])
        y = tf.matmul(fc2, fc3_weights) + fc3_biases

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(nn_config.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cost = tf.reduce_sum(tf.square(y_ - y)) / nn_config.BATCH_SIZE
    loss = cost + tf.add_n(tf.get_collection('losses'))
    train_step = tf.train.AdamOptimizer(nn_config.LEARNING_RATE_BASE).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 生成batch的数据配置
    input_queue = tf.train.slice_input_producer([x_train, x_train2, y_train])
    lstm_data = input_queue[0]
    hybrid_data = input_queue[1]
    label_data = input_queue[2]

    train_datas1, train_datas2, train_label = tf.train.shuffle_batch([lstm_data, hybrid_data, label_data],
                                                                     batch_size=nn_config.BATCH_SIZE,
                                                                     num_threads=32,
                                                                     capacity=nn_config.QUEUE_CAPACITY,
                                                                     min_after_dequeue=nn_config.MIN_AFTER_DEQUEUE)
    train_datas1 = tf.reshape(train_datas1, [-1, nn_config.TIME_STEPS * nn_config.SPACE_STEPS])
    train_datas2 = tf.reshape(train_datas2, [-1, nn_config.INPUT_NODE_VAR])
    train_label = tf.reshape(train_label, [-1, nn_config.OUTPUT_NODE])
    # 初始化TensorFlow持久化类。
    # saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(nn_config.TRAINING_STEPS):
            cur_lstm_data, cur_hybird_data, cur_label = sess.run([train_datas1, train_datas2, train_label])
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x_1: cur_lstm_data,
                                                      x_2: cur_hybird_data,
                                                      y_: cur_label})

            if i % nn_config.DISP_PER_TIMES == 0:  # 每隔多少次显示一次计算结果
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
        coord.request_stop()
        coord.join(threads)
        # saver.save(sess, os.path.join(nn_config.MODEL_SAVE_PATH, nn_config.MODEL_NAME), global_step=global_step)
        print("Optimization Finished!")
        # 测试集数据验证
        global prediction
        test_datas1 = x_test.reshape(-1, nn_config.TIME_STEPS * nn_config.SPACE_STEPS)
        test_datas2 = x_test2.reshape(-1, nn_config.INPUT_NODE_VAR)
        test_label = y_train.reshape(-1, nn_config.OUTPUT_NODE)
        prediction = sess.run(y, feed_dict={x_1: test_datas1, x_2: test_datas2, y_: test_label})


if __name__ == '__main__':
    nn_config = LstmConfig()
    file_config = DataProcessConfig()
    # lstm_data, hybrid_data, flow_label_real = merge_data(file_config)
    merged_data = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\merged_data.csv', index_col=0).iloc[288:, :]
    merged_data_5min = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\merged_data_5min.csv', index_col=0).iloc[288:, :]
    merged_data_10min = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\merged_data_10min.csv', index_col=0).iloc[288:, :]
    merged_data_15min = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\merged_data_15min.csv', index_col=0).iloc[288:, :]
    merged_data_20min = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\merged_data_20min.csv', index_col=0).iloc[288:, :]

    flow_st_data = pd.read_csv(r'D:\Users\yyh\Pycharm_workspace\hybrid_model\Data\flow_data_20segments.csv', index_col=0).iloc[288:, :]

    flow_label_real = np.array(merged_data['Real_data']).reshape(-1, 1)
    if nn_config.PRED_STEP is 1:
        merged_data = merged_data_5min
    elif nn_config.PRED_STEP is 2:
        merged_data = merged_data_10min
    elif nn_config.PRED_STEP is 3:
        merged_data = merged_data_15min
    elif nn_config.PRED_STEP is 4:
        merged_data = merged_data_20min

    hybrid_data = np.array(merged_data)
    st_data = np.array(flow_st_data)
    hybrid_data_normal = normal_data(hybrid_data)  # 输入DNN的特征数据
    st_data_normal = normal_data(st_data)  # 输入的时空交通流量数据
    flow_label = normal_data(flow_label_real)  # 输入占位符y的标签数据
    lstm_data = flow_label  # 输入lstm网络的数据

    # 将输入数据处理为标签和数据
    lstm_singe_input = data_pro(lstm_data, nn_config.TIME_STEPS, True)[:-nn_config.PRED_STEP, :]
    lstm_st_input = data_pro(st_data_normal, nn_config.TIME_STEPS, True)[:-nn_config.PRED_STEP, :]
    hybrid_input = hybrid_data_normal[nn_config.TIME_STEPS - 1:-nn_config.PRED_STEP, :]
    label_ = flow_label[nn_config.TIME_STEPS + nn_config.PRED_STEP - 1:, 0]  # 构造训练测试数据

    # 将处理好的数据加入神经网络训练
    lstm_train_hybrid(lstm_singe_input, hybrid_input, label_)
    # lstm_train(lstm_singe_input, label_)

    # 还原数据
    normal_data_min = flow_label_real.min(axis=0)
    normal_data_gap = flow_label_real.max(axis=0) - flow_label_real.min(axis=0)
    flow_test_real = y_test * normal_data_gap + normal_data_min
    prediction_real = prediction * normal_data_gap + normal_data_min

    # 训练程序结束，开始画图可视化
    plot_one_day(flow_test_real[-288:], prediction_real[-288:])
    d = abs(flow_test_real - prediction_real.flatten())
    mape = sum(d[-3186:] / flow_test_real.flatten()[-3186:]) / 3168
    mae = sum(d[-3186:]) / 3168
    print('MAPE=', mape, '\nMAE=', mae)
    pd.Series(prediction_real.flatten()).to_csv(r'D:\桌面\prediction{}min_mape{}.csv'.format(nn_config.PRED_STEP * 5, mape))
