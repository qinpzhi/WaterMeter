# -*- coding: utf-8 -*-
'''
@Time    : 18-10-22 下午2:06
@Author  : qinpengzhi
@File    : wm_model.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.D_config import  D_model_cfg
from base.base_model import BaseModel
import utils.TensorflowUtils as utils

class D_Model(BaseModel):
    def __init__(self,config):
        super(D_Model,self).__init__(config)
        self._scope='wm_model'
        self.learing_rate=config.learning_rate
        self.build_model()
        self.init_saver()

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.saver_max_to_keep)

    def build_model(self):
        self.data = tf.placeholder(tf.float32, shape=[None, 36, 24, 1])
        self.real = tf.placeholder(tf.float32, shape=[None, 10])
        # used for dropout to alleviate over-fitting issue
        self.keep_prob = tf.placeholder(tf.float32)
        with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, slim.fully_connected], \
                            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01), \
                            weights_regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay), \
                            biases_initializer=tf.constant_initializer(0.0)):
            with tf.variable_scope(self._scope):
                conv1 = slim.repeat(self.data, 2, slim.conv2d, 32, [5, 5], padding='SAME', scope='conv1')
                print "conv1:", conv1.get_shape()
                pool1 = slim.max_pool2d(conv1, [2, 2], padding='VALID', scope='pool1')
                print "pool1:", pool1.get_shape()
                conv2 = slim.repeat(pool1, 2, slim.conv2d, 64, [5, 5], padding='SAME', scope='conv2')  # stride=2
                pool2 = slim.max_pool2d(conv2, [2, 2], padding='VALID', scope='pool2')
                print "pool2:", pool2.get_shape()
                conv3=tf.reshape(pool2, [-1, 9*6*64])
                W_1 = utils.weight_variable([9*6*64, 1024])
                b_1 = utils.bias_variable([1024])
                conv4 = tf.nn.relu(tf.matmul(conv3, W_1) + b_1)
                drop1=tf.nn.dropout(conv4, keep_prob=self.keep_prob)

                W_2 = utils.weight_variable([1024, 10])
                b_2 = utils.bias_variable([10])

                self.output = tf.nn.softmax(tf.matmul(drop1, W_2) + b_2)

                print "output:", self.output.get_shape()
                self._build_loss()

    def _build_loss(self):
        with tf.variable_scope('loss') as scope:
            number_output = self.output.get_shape()[0]
            print number_output
            self.loss = -tf.reduce_sum(self.real*tf.log(self.output+1e-10))
            # self.lr = tf.train.exponential_decay(self.config.learning_rate, self.global_step_tensor, \
            #                                      self.config.step_size, 0.1, staircase=True)
            #
            # self.optimizer = tf.train.MomentumOptimizer(self.lr, self.config.momentum).minimize(self.loss, \
            #                                                                                     global_step=self.global_step_tensor)
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss, \
                                                                                        global_step=self.global_step_tensor)

# aaa=WM_Model(model_cfg)