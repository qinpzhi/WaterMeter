# -*- coding: utf-8 -*-
'''
@Time    : 18-10-22 下午1:57
@Author  : qinpengzhi
@File    : base_model.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
import tensorflow as tf
import os
class BaseModel(object):
    def __init__(self,config):
        self.config=config
        self.init_global_step()
        self.init_cur_epoch()

    # save function thet save the checkpoint in the path defined in configfile
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, os.path.join(self.config.checkpoint_dir, self.config.exp_name,self.config.exp_name), self.global_step_tensor)
        print "save model",os.path.join(self.config.checkpoint_dir, self.config.exp_name)
        print("Model saved")

    # load lateset checkpoint from the experiment path defined in config_file
    def load(self, sess, variables_to_restore=None):
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.config.checkpoint_dir, self.config.exp_name))
        if latest_checkpoint:
            print("Loading model checkpoint {:s} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")
            return True
        return False
    # just inialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    # just inialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)
    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError