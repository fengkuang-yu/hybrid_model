# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 16:51:21 2018

@author: yyh
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# 设置对应的训练参数
learning_rate = 1e-5
max_samples = 300
batch_size = 20
display_step = 100
Datasize = 20
TimeIntervals = 2
predIntervals = 0  # 0 denotes 5min prediction
Imagesize = Datasize * TimeIntervals
NUM_LAYERS = 2

n_input = 20
n_steps = TimeIntervals
n_hidden = 1024
n_classes = 1


def dataProcess():
    # 读入训练和测试数据
    data = pd.read_excel('DATA/input_data.xlsx', Sheetname='Sheet1', header=None)
    labels = pd.read_excel('DATA/input_labels.xlsx', Sheetname='Sheet1', header=None)
    test_data = pd.read_excel('DATA/test_data_10days.xlsx', Sheetname='Sheet1', header=None)
    test_labels = pd.read_excel('DATA/test_data_labels_10days.xlsx', Sheetname='Sheet1', header=None)

    input_data = data.as_matrix()
    ColumnNum = len(input_data)
    SampleNum = ColumnNum - TimeIntervals

    # 生成训练样本
    input_data1 = np.zeros((SampleNum, Imagesize))
    for i in range(SampleNum):
        temp = input_data[i:i + TimeIntervals, :].transpose()
        input_data1[i, :] = temp.reshape(1, Imagesize)

    # 输入训练样本labels
    input_data2 = labels.as_matrix()
    input_data2 = input_data2[TimeIntervals + predIntervals:, :]

    # 输入10天的测试数据
    test_data_10days = test_data.as_matrix()
    TestSampleNum = len(test_data_10days) - TimeIntervals
    input_data3 = np.zeros((TestSampleNum, Imagesize))
    for i in range(TestSampleNum):
        temp = test_data_10days[i:i + TimeIntervals, :].transpose()
        input_data3[i, :] = temp.reshape(1, Imagesize)

    # 输入10天的测试数据的标签
    input_data4 = test_labels.as_matrix()
    input_data4 = input_data4[TimeIntervals + predIntervals:, :]

    return input_data1, input_data2, input_data3, input_data4


x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = tf.Variable(tf.random_normal([n_hidden, n_classes], seed=1))
biases = tf.Variable(tf.random_normal([n_classes], seed=2))


def lstm():
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0,
                                                state_is_tuple=True,
                                                reuse=tf.get_variable_scope().reuse)
    return lstm_fw_cell


with tf.variable_scope(None, default_name="Rnn"):
    #    cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2])
    cell = tf.contrib.rnn.MultiRNNCell([lstm() for _ in range(NUM_LAYERS)], state_is_tuple=True)
    output, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    output = tf.transpose(output, [1, 0, 2])

pred = tf.matmul(output[-1], weights) + biases
cost = tf.reduce_sum(tf.square(pred - y)) / batch_size
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
input_data1, input_data2, input_data3, input_data4 = dataProcess()
testSampleNum = input_data4.shape[0]
step = 1
while step < max_samples:
    Sample1 = np.random.randint(1, 12000, size=(1, batch_size))
    train_datas = input_data1[Sample1].reshape(batch_size, Imagesize)
    train_label = input_data2[Sample1].reshape(batch_size, 1)

    train_datas = train_datas.reshape((batch_size, n_steps, n_input))
    sess.run(optimizer, feed_dict={x: train_datas, y: train_label})
    if step % display_step == 0:
        # acc = sess.run(accuracy, feed_dict = {x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict={x: train_datas, y: train_label})
        print("Iter" + str(step) + ", Loss = " + "{:.6f}".format(loss))
    step += 1
print("Optimization Finished!")

test_data = input_data3.reshape((-1, n_steps, n_input))
test_label = input_data4.reshape(-1, 1)
prediction = sess.run(pred, feed_dict={x: test_data, y: test_label})

# 计算十天的MAPE和MAE
# Prediction=Pred.reshape(TestSampleNum,-1)
d = abs(input_data4 - prediction)
mape = sum(d / input_data4) / testSampleNum
mae = sum(d) / testSampleNum
print('Ten days MAPE=', mape, '\nTen days MAE=', mae)

# 计算一天的MAPE和MAE值
startTime = 2304 - TimeIntervals - predIntervals
real_data = input_data4[startTime: startTime + 288, :]
Prediction1 = prediction[startTime: startTime + 288, :]
AbsoluteE2 = real_data - Prediction1
d = abs(AbsoluteE2)

mape1 = sum(d / real_data) / 288
mae1 = sum(d) / 288
print('One day MAPE=', mape1, '\nOne day MAE=', mae1)
# tensorboard --logdir logs

# 画图开始
# 1.选择出画图的一天，[startTime:startTime+288,:]表示的是画图的那一天。
# 2.Predictin是预测的画图当天的结果，减TimeIntervals的原因是ten_test_label是从0开
#   始到len(test_data)，而Prediction是从0开始到len(test_data-TimeIntervals)

params = {
    'axes.labelsize': '16',
    'xtick.labelsize': '16',
    'ytick.labelsize': '16',
    'lines.linewidth': '4',
    'legend.fontsize': '16',
    'figure.figsize': '8, 8'  # set figure size
}
pylab.rcParams.update(params)

plt.figure()
plt.figure(figsize=(8, 6), dpi=60)

# 第一幅图
# plt.subplot(121)
plt.plot(real_data, color="blue", linewidth=1, linestyle="-", label="real")
plt.plot(Prediction1, color="red", linewidth=1, linestyle="-", label="emulation")
plt.xlabel('Time (per 5 min)')
plt.ylabel('Totle traffic flow (vehicles)')
plt.legend(loc='upper right')

text = "LR:%f    batch:%d    Maxiteration:%d    TimeIntervals:%d\
    StackNum:%d    MAPE:%f" % (learning_rate,
                               batch_size,
                               max_samples,
                               TimeIntervals,
                               NUM_LAYERS,
                               mape)

plt.text(-40, 630, text, fontsize=10, style='italic', ha='left', wrap=False)
plt.savefig('figures/LR%f_batch%d_Maxiteration%d_TimeIntervals%d_StackNum%d.png' % (
    learning_rate,
    batch_size,
    max_samples,
    TimeIntervals,
    NUM_LAYERS), dpi=200)
