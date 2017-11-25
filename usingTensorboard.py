# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def add_layer(inputs,in_size,out_size,activation_function=None): #先不定义激励函数
    #初始权重为随机变量要比全零要好;矩阵一般开头大写，后面的向量biase一般小写
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weight = tf.Variable(tf.random_normal([in_size,out_size]))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1)
        #未激活的预测值
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weight)+biases
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        return outputs

# linspace(x1,x2,N) 用于产生x1,x2之间的N点行线性的矢量。
# 其中x1、x2、N分别为起始值、终止值、元素个数
x_data = np.linspace(-1,1,300)[:,np.newaxis]
# numpy.random.normal(loc=0.0, scale=1.0, size=None),均值，标准差，形状
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
# 定义传入
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_input")
    ys = tf.placeholder(tf.float32, [None, 1], name="y_input")
# 定义输入层1个，隐藏层10个，输出1个神经元的神经网络
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.train.SummeryWriter('logs/',sess.graph)
sess.run(init)
