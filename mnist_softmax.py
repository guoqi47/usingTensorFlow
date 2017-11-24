## -*- coding: utf-8 -*-
#"""A very simple MNIST classifier."""
##sigmoid将一个real value映射到（0,1）的区间（当然也可以是（-1,1）），这样可以用来做二分类。 
##而softmax把一个k维的real value向量（a1,a2,a3,a4….）映射成一个（b1,b2,b3,b4….）
##其中bi是一个0-1的常数，然后可以根据bi的大小来进行多分类的任务，如取权重最大的一维。 
#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#
##import data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#x = tf.placeholder(tf.float32,[None,784]) #Here None means that a dimension can be of any length
#W = tf.Variable(tf.zeros([784,10])) #初始化，随意值，此处初始为0
#b = tf.Variable(tf.zeros([10]))
#y = tf.nn.softmax(tf.matmul(x,W)+b) #model
#y_ = tf.placeholder(tf.float32,[None, 10]) #correct answers
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # learning rate: 0.5
#sess = tf.InteractiveSession()
#tf.global_variables_initializer().run()
#for _ in range(1000):
#    batch_xs, batch_ys = mnist.train.next_batch(100)
#    sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
##Evaluating Our Model
#correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))


#2.git源码
#
## Copyright 2015 The TensorFlow Authors. All Rights Reserved.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
## ==============================================================================
#
#"""A very simple MNIST classifier.
#See extensive documentation at
#https://www.tensorflow.org/get_started/mnist/beginners
#"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#
#import argparse
#import sys
#
#from tensorflow.examples.tutorials.mnist import input_data
#
#import tensorflow as tf
#
#FLAGS = None
#
#
#def main(_):
#  # Import data
#  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
#
#  # Create the model
#  x = tf.placeholder(tf.float32, [None, 784])
#  W = tf.Variable(tf.zeros([784, 10]))
#  b = tf.Variable(tf.zeros([10]))
#  y = tf.matmul(x, W) + b
#
#  # Define loss and optimizer
#  y_ = tf.placeholder(tf.float32, [None, 10])
#
#  # The raw formulation of cross-entropy,
#  #
#  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#  #                                 reduction_indices=[1]))
#  #
#  # can be numerically unstable.
#  #
#  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
#  # outputs of 'y', and then average across the batch.
#  cross_entropy = tf.reduce_mean(
#      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
#  sess = tf.InteractiveSession()
#  tf.global_variables_initializer().run()
#  # Train
#  for _ in range(1000):
#    batch_xs, batch_ys = mnist.train.next_batch(100)
#    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#
#  # Test trained model
#  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
#                                      y_: mnist.test.labels}))
#
#if __name__ == '__main__':
#  parser = argparse.ArgumentParser()
#  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
#                      help='Directory for storing input data')
#  FLAGS, unparsed = parser.parse_known_args()
#  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



#3.csdn
# encoding: utf-8   
  
import tensorflow as tf  
from tensorflow.examples.tutorials.mnist import input_data  
  
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
print("完成")  
  
x = tf.placeholder(tf.float32, [None, 784])  
  
# paras  
W = tf.Variable(tf.zeros([784, 10]))  
b = tf.Variable(tf.zeros([10]))  
  
y = tf.nn.softmax(tf.matmul(x, W) + b)  
y_ = tf.placeholder(tf.float32, [None, 10])  
  
# loss func  
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  
  
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  
  
# init  
init = tf.initialize_all_variables()  
  
sess = tf.Session()  
sess.run(init)  
  
# train  
for i in range(1000):  
    batch_xs, batch_ys = mnist.train.next_batch(100)  
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  
  
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  

print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))