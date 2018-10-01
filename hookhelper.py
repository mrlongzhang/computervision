# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:26:27 2018

@author: mr_lo
"""

import tensorflow as tf
import parse

class LearningRateHook(tf.train.SessionRunHook):
    def __init__(self, model, steps, lrates):
        super(LearningRateHook, self).__init__()
        self.model = model
        self.steps = steps
        self.lrates = lrates

    def begin(self):
        self.learning_rate = 0.1
    
    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            self.model.global_step,
            feed_dict = {self.model.learning_rate:self.learning_rate}
        )

    def after_run(self, run_context, run_values):
        train_step = run_values.results
        if train_step < self.steps[0]:
            self.learning_rate = self.lrates[0]
        else:
            for i in range(1, len(self.steps)-1):
                if self.steps[i] < train_step < self.steps[i+1]:
                    self.learning_rate = self.lrates[i+1]

class SessionWithHooksHelper:
    def __init__(self, model, steps=[20000,40000,80000,120000,160000], lrates=[0.1,0.02, 0.01, 0.001, 0.0001]):
        flags = parse.ArgParser().get_flags()
        summary_hook = tf.train.SummarySaverHook(
            save_steps = 1000,
            output_dir = flags.save_dir,
            summary_op=tf.summary.merge([model.summary_op, 
                tf.summary.scalar('accuracy', model.accuracy)])
        )
        logging_hook = tf.train.LoggingTensorHook(
            tensors = {
                'step' : model.global_step,
                'loss' : model.loss_op,
                'accuracy' : model.accuracy
            },
            every_n_iter = 100
        )
        learningrate_hook = LearningRateHook(model, steps, lrates)
        with tf.train.MonitoredTrainingSession(checkpoint_dir=flags.save_dir,
            hooks = [logging_hook, learningrate_hook], chief_only_hooks=[summary_hook],
            save_summaries_steps=0,
            config = tf.ConfigProto(allow_soft_placement=True)) as sess:
            while not sess.should_stop():
                sess.run(model.opt_op)

    def get_session(self):
        return self.sess




