#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


inputs = tf.placeholder(tf.float32, [None,784])
input_reshaped = tf.reshape(inputs, [-1,28,28,1])


conv1_weights = tf.Variable(tf.truncated_normal([5,5,1,8], stddev=0.1))
conv1_bias = tf.Variable(tf.zeros([8]))
conv1 = tf.nn.conv2d(input_reshaped, conv1_weights, strides=[1,1,1,1], padding='SAME')
conv1_relu = tf.nn.relu(conv1 + conv1_bias)

max_pool1 = tf.nn.max_pool(conv1_relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 8, 16], stddev=0.1))
conv2_bias = tf.Variable(tf.zeros([16]))
conv2 = tf.nn.conv2d(max_pool1, conv2_weights, strides=[1,1,1,1], padding='SAME')
conv2_relu = tf.nn.relu(conv2 + conv2_bias)

max_pool2 = tf.nn.max_pool(conv2_relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


flat_output = tf.reshape(max_pool2, [-1, 7*7*16])

fully_connected_weights = tf.Variable(tf.truncated_normal([7*7*16, 128], stddev=0.1))
fully_connected_bias = tf.Variable(tf.zeros([128]))
fully_connected = tf.nn.relu(tf.matmul(flat_output, fully_connected_weights) + fully_connected_bias)


output_weights =tf.Variable(tf.truncated_normal([128, 10], stddev=0.1))
output_bias = tf.Variable(tf.zeros([10]))
output_layer = tf.nn.softmax(tf.matmul(fully_connected, output_weights) + output_bias)


label_data = tf.placeholder(tf.float32, [None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(label_data* tf.log(output_layer), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.0003).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(output_layer,1), tf.argmax(label_data,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

####### Do Not Touch This Block.
####### "TEST" accuracy of your CNN model have to be higher than 0.9 
####### If your model have higher "TEST" accuracy than 0.95 you can get an additional point.
####### If your model have higher "TEST" accuracy than 0.95 you can get an additional point.

####### 이 블럭을 수정하지 마세요.
####### 작성하신 모델의 "TEST" 정확도가 0.9 보다 높아야 합니다.
####### 작성하신 모델의 "TEST" 정확도가 0.95 보다 높은 경우 추가 점수를 얻습니다.
####### 작성하신 모델의 "TEST" 정확도가 0.98 보다 높은 경우 추가 점수를 얻습니다.
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    if i%100 == 0:
        pass
        train_accuracy = accuracy.eval(session=sess, feed_dict={inputs:batch_xs, label_data: batch_ys})
        print("step %d, training ACC %.3f"%(i, train_accuracy))
    sess.run(train_step, feed_dict={inputs:batch_xs, label_data: batch_ys})
print("test ACC %g"% accuracy.eval(session=sess, feed_dict={inputs: mnist.test.images, label_data:mnist.test.labels}))





