# -*- coding: utf-8 -*-

"""
@Author  :   {Yu Yinghao}
@Software:   PyCharm
@File    :   batch_test.py
@Time    :   2018/10/12 11:08
@Desc    :
"""

import numpy as np
import tensorflow as tf

...

if __name__ == '__main__':
    data1 = np.array([x for x in range(1, 101)]).reshape(10, 10)
    data2 = np.array([x for x in range(101, 201)]).reshape(10, 10)
    label = np.array([x for x in range(1, 11)]).reshape(10, 1)
    batch_size = 3
    capacity = 100 + 3 * batch_size
    input_queue = tf.train.slice_input_producer([data1, data2, label])
    label = input_queue[1]
    image1 = input_queue[0]  # read img from a queue
    image2 = input_queue[2]
    image1_batch, image2_batch, label_batch = tf.train.shuffle_batch([image1, image2, label],
                                                      batch_size=batch_size,
                                                      num_threads=32,
                                                      capacity=capacity,
                                                      min_after_dequeue=50,
                                                      allow_smaller_final_batch=True)
    # 重新排列label，行数为[batch_size]
    # label_batch = tf.reshape(label_batch, [batch_size])
    # image_batch = tf.cast(image_batch, tf.float32)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(2):
            cur_data1_batch, cur_data2_batch, cur_label_batch = sess.run([image1_batch, image2_batch, label_batch])
            print(cur_data1_batch, cur_data2_batch, cur_label_batch)
        coord.request_stop()
        coord.join(threads)
