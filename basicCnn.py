# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:37:23 2018

@author: mr_lo
"""
import tensorflow as tf
import sys
import numpy as np

class BaseNeural(object):
    def print_model_states(self):
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)

        sys.stdout.write('Total params: %d\n' % param_stats.total_parameters)

        tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

class BasicCnn(BaseNeural):
    ''' basic cnn defined a basic CNN network
    '''
    def __init__(self, flags):
        super(BasicCnn, self).__init__()
        
        self.loss_op = []
        self.opt_op =[]
        self.opt_op_extra = []
        self.summary_op = []

        self.input = []
        self.labels = []
        self.output = None
        self.output_label = None

        self.accuracy = None
        self.global_step = None

        self.max_grad = flags.max_grad #args
        self.weight_decay = flags.weight_decay_rate #args
        self.keepprob = flags.keep_prob # args
        self.learning_rate = flags.learning_rate #args
        self.image_dim = flags.image_size
        self.num_labels= flags.num_labels

        if flags.mode == 'train':
            self.training = True #args
        else:
            self.training = False

        self.flags = flags
        #self.build_model()
    
    # interface function
    def build_model(self):
        pass

    def init_global_step(self):
        self.global_step = tf.train.get_or_create_global_step()

    def step(self, batch):
        pass
    

    def build_lossop(self, decay = False):
        with tf.variable_scope('loss'):
            self.loss_op = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.output, labels=self.labels
            )
            self.loss_op = tf.reduce_mean(self.loss_op, name = 'cross_entropy')
           
            if decay:
                self.loss_op += self.decay()

            tf.summary.scalar('loss', self.loss_op)

    def build_summaryop(self):
        self.summary_op = tf.summary.merge_all()
    
    def ks_list(self, ksize):
        return [1, ksize, ksize, 1]

    # utils to build network
    def new_weight(self, shape, name, uniform = False, stddev = 0.1):
        if not uniform:
            initial = tf.random_normal_initializer(stddev = stddev)
        else:
            initial = tf.uniform_unit_scaling_initializer(factor = stddev)
        return tf.get_variable(name = name, shape = shape, initializer = initial)

    def new_bias(self, shape, name, initval = 0.0):
        initial = tf.constant_initializer(value = initval)
        return tf.get_variable(name=name, shape=shape, initializer= initial)
    
    # TODO: resnet need no bias
    def new_conv(self, input, 
                num_channels,   # channel of images
                filter_num,     # num features to produce 
                filter_size,    # kernel size
                filter_strides, # filter strides
                pool_ksize,     # pool ksize (as long as pool_ksize > 1)
                pool_strides,    # pool strides
                need_bias = True,
                name = 'conv'):
        with tf.name_scope(name) as scope:
            shape = [filter_size, filter_size, num_channels, filter_num]
            n = filter_size*num_channels*filter_num
            stddev = np.sqrt(2.0/n)
            weight = self.new_weight(shape = shape, name = 'weight', stddev = stddev)

            #conv_strides = [1, filter_strides, filter_strides, 1]
            out = self.conv2d(x = input, W = weight, strides = filter_strides, name = 'conv')

            if need_bias:
                bias = self.new_bias(shape=[filter_num], name='bias')
                out += bias

            if pool_ksize > 1:
                poolk = [1, pool_ksize, pool_ksize, 1]
                pools = [1, pool_strides, pool_strides, 1]
                out = self.maxpool(x = out, ksize = poolk, strides = pools, name = 'pool')
            
            if need_bias:
                out = tf.nn.relu(out, name = scope)

        return out, weight

    def new_fc(self, input, 
                num_outputs,
                name,
                use_relu, # True/False mean use standard relu, 0.0 - 1.0 means use leaky relu
                need_flatten = False,
                keep_prob = 1.0):
        with tf.name_scope(name) as scope:
            input_shape = input.get_shape()
            num_input_features = input_shape[1:4].num_elements()
            if need_flatten:
                input = tf.reshape(input, [-1, num_input_features])
            
            weight = self.new_weight(shape=[num_input_features, num_outputs], name = 'weight', stddev=0.01)
            bias = self.new_bias(shape=[num_outputs], name='bias')

            out = tf.matmul(input, weight) + bias

            if isinstance(use_relu, bool) and use_relu:
                out = tf.nn.relu(out, name=scope)
            if ~isinstance(use_relu, bool) and use_relu > 0.0:
                out = self.leaky_relu(out, use_relu)

            if keep_prob < 1.0:
                out = tf.nn.dropout(out,keep_prob = keep_prob, name='drop')

        return out,weight
    
    def leaky_relu(self, x, leak = 0.0):
        return tf.where(tf.less(x, 0.0), leak*x, x, name ='leaky_relu')
    
    def global_avg_pool(self,x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1,2])

    def conv2d(self, x, W, strides, name):
        conv_strides = [1, strides, strides, 1]
        return tf.nn.conv2d(x, W, strides=conv_strides,padding='SAME', name=name)

    def maxpool(self, x, ksize, strides, name):
        ksize = [1, ksize, ksize, 1]
        strides = [1, strides, strides, 1]
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding='SAME', name='name')
    
    # utils for all the rest
    
    # a) weight decay
    def decay(self):
        # L2 weight decay loss
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'weight') > 0:
                costs.append(tf.nn.l2_loss(var))

        return tf.multiply(self.weight_decay, tf.add_n(costs))

def print_activations(t):
    print(t.op.name, ' ',t.get_shape().as_list())
