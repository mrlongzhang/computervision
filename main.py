# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:26:27 2018

@author: mr_lo
"""

import time
import sys
import six

import numpy as np
import tensorflow as tf
import resnet
import alexnet
import cifar_data
import parse
import hookhelper
import Batch

def train(images, labels, num_labels):
    model = resnet.ResNET(images, labels, num_residule_unit = 5, leaky = 0.1, use_bottleneck=False)
    model.print_model_states()

    #tf.global_variables_initializer().run()
    hookhelper.SessionWithHooksHelper(model)


def main(_):
    flags = parse.ArgParser().get_flags()
    num_labels = 10
    if flags.dataset == 'cifar100':
        num_labels = 100
    
    images, labels = cifar_data.build_input(
      flags.dataset, flags.data_path, flags.batch_size, flags.mode)
    
    if flags.mode == 'train':
        train(images, labels, num_labels)
    

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()