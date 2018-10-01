# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:26:27 2018

@author: mr_lo
"""

import tensorflow as tf

def Singleton(cls):
    instances = {}
    def _Singleton(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]
    return _Singleton

@Singleton
class ArgParser:
    def __init__(self):
        self.flags = tf.app.flags.FLAGS

        tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100 or mnist')
        tf.app.flags.DEFINE_string('mode','train', 'train or eval')
        tf.app.flags.DEFINE_string('data_path','./cifar_data/cifar-10-batches-bin/data_batch_*', 'path to training data')
        tf.app.flags.DEFINE_string('save_dir', './save', 'path to save model and outputs')
        tf.app.flags.DEFINE_string('optimizer', 'mom', 'the optimization algorithm to use')

        tf.app.flags.DEFINE_integer('image_size', 32, 'Image size')
        tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
        tf.app.flags.DEFINE_integer('epochs', 30, 'epochs to run')
        tf.app.flags.DEFINE_integer('num_labels', 10, 'image classes')

        tf.app.flags.DEFINE_float('weight_decay_rate', 0.002, 'decay rate of weights')
        tf.app.flags.DEFINE_float('learning_rate', 0.1, 'initial learning rate')
        tf.app.flags.DEFINE_float('max_grad', 5.0, 'max graddients for gradient clip')
        tf.app.flags.DEFINE_float('keep_prob', 0.5, 'keep probability of dropout')
 
    def get_flags(self):
        return self.flags
