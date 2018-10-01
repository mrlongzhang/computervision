# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:26:27 2018

@author: mr_lo
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages
from basicCnn import BasicCnn
import parse

class ResNET(BasicCnn):
    def __init__(self, images, labels, num_residule_unit, leaky, use_bottleneck = True, optimizer='mom'):
        self.flags = parse.ArgParser().get_flags()
        super(ResNET,self).__init__(self.flags)

        self.leaky = leaky
        self.use_bottleneck = use_bottleneck
        self.num_residule_unit = num_residule_unit

        self._images = images
        self.labels = labels

        self.build_model(optimizer = optimizer)

    def build_model(self, optimizer):
        ################################################################################
        # 2. build network
        self.init_global_step()
        self._build_model(self._images, self.labels, self.num_labels)
        
        ################################################################################
        # 3 loss and summary
        self.build_lossop(decay = True)

        if self.training:
            self.build_trainop(optimizer = optimizer)
        
        self.build_summaryop()

        ###############################################################################
        # 4 evaluation
        self.output_label = tf.argmax(tf.nn.softmax(logits = self.output, name='output_softmax'),axis =1)
        is_correct = tf.equal(self.output_label, tf.argmax(tf.nn.softmax(self.labels), axis = 1))
        self.accuracy = tf.reduce_mean(tf.to_float(is_correct))
        tf.summary.scalar('accuracy', self.accuracy)

    def build_trainop(self, optimizer = 'Adam'):
        self.learning_rate = tf.constant(self.learning_rate, tf.float32)
        if optimizer == 'sgd':
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif optimizer == 'mom':
            opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        else:
            opt = tf.train.AdamOptimizer(self.learning_rate)

        trainable = tf.trainable_variables()
        gradients = tf.gradients(self.loss_op, trainable)
        clipped,_ = tf.clip_by_global_norm(gradients, self.max_grad)
        apply_op = opt.apply_gradients(zip(clipped, trainable),global_step=self.global_step,name='train.step')

        allopt_op = [apply_op] +  self.opt_op_extra
        self.opt_op = tf.group(*allopt_op)

    def _build_model(self, input_image, input_labels, num_labels):
        with tf.variable_scope('init'):
            x = input_image #tf.reshape(input_image,shape=[-1, self.image_dim, self.image_dim, 3], name='image')
            x,_ = self.new_conv(x, 3, 16, 3, 1, # filter
                1,1,need_bias = False, name = 'init_conv') # pool pksize = 1 means no pool

        num_filters = []
        strides = [1,2,2]
        act_before = [True,False, False]

        if self.use_bottleneck:
            unit_builder = self._bottleneck
            num_filters = [16,64,128,256]
        else:
            unit_builder = self._residual
            num_filters = [16,16,32,64]

        stage = 1
        with tf.variable_scope('stage_1'):
            x = unit_builder(x, num_filters[stage-1],num_filters[stage], strides[stage-1], act_before[stage-1])
        for i in range(1,self.num_residule_unit):
            with tf.variable_scope('stage_1_%d'%i):
                x = unit_builder(x, num_filters[stage],num_filters[stage], 1, False)

        stage += 1
        with tf.variable_scope('stage_%d'%stage):
            x = unit_builder(x, num_filters[stage-1], num_filters[stage], strides[stage-1],act_before[stage-1])
        for i in range(1,self.num_residule_unit):
            with tf.variable_scope('stage_%d_%d'%(stage,i)):
                x = unit_builder(x, num_filters[stage],num_filters[stage], 1, False)

        stage += 1
        with tf.variable_scope('stage_3'):
            x = unit_builder(x, num_filters[stage-1], num_filters[stage], strides[stage-1],act_before[stage-1])
        for i in range(1,self.num_residule_unit):
            with tf.variable_scope('stage_3_%d'%i):
                x = unit_builder(x, num_filters[stage],num_filters[stage], 1, False)


        with tf.variable_scope('stage_last'):
            x = self._batch_norm('bn',x)
            x = self.leaky_relu(x, self.leaky)
            x = self.global_avg_pool(x)

        with tf.variable_scope('logit'):
            logits,_ = self.new_fc(x, num_labels, 1.0, 'logit_fc')
            self.output = logits
            #self.output_label = tf.nn.softmax(logits)

    def step(self, batch):
        feed_dict = {}
        oplist = None

        if self.training:
            feed_dict = {
                self.input : batch.input_image,
                self.labels : batch.labels,
                self.leaky : 0.5
            }
            oplist = [self.loss_op, self.opt_op, self.summary_op]
        else:
            feed_dict = {
                self.input : batch.input_image,
                self.keepprob : 1.0
            }
            oplist = [self.output_label, self.accuracy]

        return oplist, feed_dict

    def _batch_norm(self, name, input):
        with tf.variable_scope(name):
            params_shape = [input.get_shape()[-1]]

            beta = tf.get_variable('beta', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0,tf.float32))

            gamma = tf.get_variable('gamma', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0,tf.float32))

            if self.training:
                mean, variance = tf.nn.moments(input, [0,1,2], name='moments')

                moving_mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable = False)

                moving_variance = tf.get_variable('moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable = False)

                self.opt_op_extra.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9
                ))

                self.opt_op_extra.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9
                ))
            else:
                mean = tf.get_variable('mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable = False)

                variance = tf.get_variable('variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable = False)

                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)

            # epsilon used to be 1e-5, maybe 0.001 solve NaN problem
            output = tf.nn.batch_normalization(input, mean, variance, beta, gamma, 0.001)
            output.set_shape(input.get_shape())

            return output

    def _residual(self, input, num_filter_in, num_filter_out, stride, act_before_residual = False):
        if act_before_residual:
            with tf.variable_scope('shared_act'):
                input = self._batch_norm('init_bn', input)
                input = self.leaky_relu(input, self.leaky)
                original_input = input
        else:
            with tf.variable_scope('only_act'):
                original_input = input
                input = self._batch_norm('init_bn', input)
                input = self.leaky_relu(input, self.leaky)

        with tf.variable_scope('sub_layer_1'):
            output,_ = self.new_conv(input, 
            num_filter_in, num_filter_out, 3, stride, # filter
            1,1, # pool pksize == 1 means no pool
            need_bias = False, name = 'conv')

        with tf.variable_scope('sub_layer_2'):
            output = self._batch_norm('bn', output)
            output = self.leaky_relu(output, self.leaky)
            output,_ = self.new_conv(output,
            num_filter_out, num_filter_out, 3, 1, # the stride is 1!
            1, 1,
            need_bias = False, name = 'conv')

        with tf.variable_scope('sub_layer_add'):
            if num_filter_in != num_filter_out:
                stride_lst = self.ks_list(stride)
                original_input = tf.nn.avg_pool(original_input, stride_lst,stride_lst, 'VALID')
                original_input = tf.pad(
                    original_input,[[0,0],[0,0],[0,0],
                    [(num_filter_out - num_filter_in)//2, (num_filter_out - num_filter_in)//2]]
                )

            output += original_input

        return output

    def _bottleneck(self, input, num_filter_in, num_filter_out, stride, act_before_residual = False):
        # bottleneck has 3 sublayers
        if act_before_residual:
            with tf.variable_scope('shared_act'):
                input = self._batch_norm('init_bn', input)
                input = self.leaky_relu(input, self.leaky)
                original_input = input
        else:
            with tf.variable_scope('only_act'):
                original_input = input
                input = self._batch_norm('init_bn', input)
                input = self.leaky_relu(input, self.leaky)

        with tf.variable_scope('sub_layer_1'):
            output,_ = self.new_conv(input, 
            num_filter_in, num_filter_out/4, 1, stride, # filter
            1,1, # pool pksize == 1 means no pool
            need_bias = False, name = 'conv')

        with tf.variable_scope('sub_layer_2'):
            output = self._batch_norm('bn', output)
            output = self.leaky_relu(output, self.leaky)
            output,_ = self.new_conv(output, 
            num_filter_out/4, num_filter_out/4, 3, 1,
            1,1,need_bias = False, name = 'conv')

        with tf.variable_scope('sub_layer_3'):
            output = self._batch_norm('bn', output)
            output = self.leaky_relu(output, self.leaky)
            output,_ = self.new_conv(output, 
            num_filter_out/4, num_filter_out, 1, 1,
            1,1,'conv')

        with tf.variable_scope('sub_layer_add'):
            if num_filter_in != num_filter_out:
                original_input,_ = self.new_conv(original_input,
                num_filter_in, num_filter_out, 1, stride,
                1,1,need_bias = False, name = 'project')

                output += original_input

        return output            
        
    def new_fc(self, input, 
                num_outputs,
                weight_stddev,
                name):
        with tf.name_scope(name) as scope:
            w = tf.get_variable('weight', [input.get_shape()[1], num_outputs],
                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
            b = tf.get_variable('bias', [num_outputs],
                initializer=tf.constant_initializer())

        return tf.nn.xw_plus_b(input, w, b),w


