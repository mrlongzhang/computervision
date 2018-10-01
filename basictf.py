# -*- coding: utf-8 -*-
"""
Created on Wed Aug 6 13:31:49 2018

@author: mr_lo
"""

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True) 

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name = name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W, name):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1],padding='SAME', name=name)

def max_pool2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='name')

###############################################################################
# 1. placeholders
x=tf.placeholder(dtype=tf.float32, shape=[None, 28*28], name='input')
y_=tf.placeholder(dtype=tf.float32, shape=[None, 10],name='label')
keepprob = tf.placeholder(tf.float32, name='keepprob')

###############################################################################
# 2. network graph
x_image=tf.reshape(x,shape=[-1,28,28,1], name='image')

# layer 1, convolution with 5x5, 1 channel, 32 feathers, relu activation + 2x2 maxpool
# note: defined the filtration bank, nothing with image!
wconv = weight_variable([5,5,1,32],'cw1') #5x5 kernel size 1 plane 32 kernel
bconv = bias_variable([32],'cb1')
hconv=tf.nn.relu(conv2d(x_image, wconv, 'conv1') + bconv) # image convolved for 32 times
hpool = max_pool2x2(hconv,'pool1') # image now 14x14, 32 images

################################################################################
# layer 2, convolution layer 5x5, 64 feathers, , relu activation + 2x2 maxpool
#32 filtered images becomes to 64 filtred images
wconv = weight_variable([5,5,32,64],'cw2')
bconv = bias_variable([64],'cb2')
hconv = tf.nn.relu(conv2d(hpool, wconv, 'conv2')+bconv) # image convoled for 64 times
hpool = max_pool2x2(hconv,'pool2') # image now 7x7, 64 images

################################################################################
# layer 3, full layer with stretched the vector, 1024 nodes
wfull = weight_variable([7*7*64,1024],'fw1')
bfull = bias_variable([1024],'fb1')
hpool = tf.reshape(hpool, [-1,7*7*64], name='full_in')
hfull = tf.nn.relu(tf.matmul(hpool, wfull)+bfull)

# dropout after fc layer
hfull = tf.nn.dropout(hfull, keepprob)

################################################################################
# output layer, usually not treated as a layer. mapping 1024 to 10 labels
ow = weight_variable([1024, 10],'ow')
ob = bias_variable([10],'ob')
pred_logits = tf.matmul(hfull, ow) + ob
pred = tf.nn.softmax(pred_logits)

################################################################################
# 3. loss and opt
# cross entropy = -y_true * log(y_estimated)
loss = tf.reduce_mean(tf.reduce_sum(-y_*tf.log(pred), reduction_indices=[1])) # 0 index is image indices
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_logits, labels = y_))
opt  = tf.train.AdamOptimizer(1e-4).minimize(loss)
tf.summary.scalar('loss', loss)
summary = tf.summary.merge_all()

###############################################################################
# 4 accuracy eval
accuracy = tf.equal(tf.argmax(y_,axis=1), tf.argmax(pred,axis=1))
accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

################################################################################
# 5 train
cfgProto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

with tf.Session(config = cfgProto) as sess:
    tf.global_variables_initializer().run()
    for epochs in range(5):
        batch = mnist.train.next_batch(50)
        label = batch[1].astype(np.float32)
        for _ in range(1000):
            _, lossval = sess.run([opt, loss],{x:batch[0], y_:batch[1], keepprob:0.5})
            batch = mnist.train.next_batch(50)

        print('loss is {}'.format(lossval) )
        print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                       y_: mnist.test.labels,
                                       keepprob:0.5}))
        acc = sess.run([accuracy], {x:batch[0], y_:batch[1], keepprob:0.5})
        print('accuracy is {}'.format(acc[0]))
    


