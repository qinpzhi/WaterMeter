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
from utils.WM_config import  WM_model_cfg
from base.base_model import BaseModel
import utils.TensorflowUtils as utils

class WM_Model(BaseModel):
    def __init__(self,config):
        super(WM_Model,self).__init__(config)
        self._scope='wm_model'
        self.learing_rate=config.learning_rate
        self.build_model()
        self.init_saver()

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.saver_max_to_keep)

    def build_model(self):
        self.data = tf.placeholder(tf.float32, shape=[None, 88, 160, 1])
        self.real = tf.placeholder(tf.float32, shape=[None, 8])
        # used for dropout to alleviate over-fitting issue
        self.keep_prob = tf.placeholder(tf.float32)
        with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, slim.fully_connected], \
                            activation_fn=self.leaky_relu(0.1),
                            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01), \
                            weights_regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay), \
                            # biases_initializer=tf.constant_initializer(0.0)
                            ):
            with tf.variable_scope(self._scope):
                conv1 = slim.repeat(self.data, 3, slim.conv2d, 64, [3, 3], padding='SAME', scope='conv1')
                conv1=tf.layers.batch_normalization(conv1)
                pool1 = slim.max_pool2d(conv1, [2, 2], padding='VALID', scope='pool1')

                conv2 = slim.repeat(pool1, 3, slim.conv2d, 128, [3, 3], padding='SAME', scope='conv2')  # stride=2
                conv2 = tf.layers.batch_normalization(conv2)
                pool2 = slim.max_pool2d(conv2, [2, 2], padding='VALID', scope='pool2')

                conv3 = slim.repeat(pool2, 3, slim.conv2d, 256, [3, 3], padding='SAME', scope='conv3')  # stride=2
                conv3 = tf.layers.batch_normalization(conv3)
                pool3 = slim.max_pool2d(conv3, [2, 2], padding='VALID', scope='pool3')

                conv4 = slim.repeat(pool3, 3, slim.conv2d, 256, [3, 3], padding='SAME', scope='conv4')  # stride=2
                conv4 = tf.layers.batch_normalization(conv4)
                pool4 = slim.max_pool2d(conv4, [2, 2], padding='VALID', scope='pool4')

                conv5 = slim.repeat(pool4, 3, slim.conv2d, 512, [3, 3], padding='SAME', scope='conv5')  # stride=2
                conv5 = tf.layers.batch_normalization(conv5)
                pool5 = slim.max_pool2d(conv5, [2, 2], padding='VALID', scope='pool5')


                conv6=slim.fully_connected(slim.flatten(pool5), 1024)
                drop1=tf.nn.dropout(conv6, keep_prob=self.keep_prob)

                conv7 = slim.fully_connected(drop1, 4096)
                drop2 = tf.nn.dropout(conv7, keep_prob=self.keep_prob)

                self.output=slim.fully_connected(drop2,8,activation_fn=None)
                # print "output:", self.output.get_shape()
                self._build_loss()

    def _build_loss(self):
        with tf.variable_scope('loss') as scope:
            # number_output = self.output.get_shape()[0]
            # print number_output
            self.iou_predict_truth=tf.reduce_sum(self.calculateIoU(self.output,self.real))
            self.loss = self.iou_predict_truth+tf.reduce_sum(tf.square(self.output-self.real))+1e-10
            # self.loss=tf.reduce_sum(tf.square(self.output-self.real))+1e-10
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss, \
                                                                                        global_step=self.global_step_tensor)

    def calculateIoU(self,candidateBound, groundTruthBound):

        cx1 = candidateBound[:, 0]
        cy1 = candidateBound[:,1]

        cx3 = candidateBound[:, 2]
        cy3 = candidateBound[:, 3]

        cx4 = candidateBound[:, 4]
        cy4 = candidateBound[:, 5]

        cx2 = candidateBound[:,6]
        cy2 = candidateBound[:,7]

        gx1 = groundTruthBound[:,0]
        gy1 = groundTruthBound[:,1]

        gx3 = groundTruthBound[:, 2]
        gy3 = groundTruthBound[:, 3]

        gx4 = groundTruthBound[:, 4]
        gy4 = groundTruthBound[:, 5]

        gx2 = groundTruthBound[:,6]
        gy2 = groundTruthBound[:,7]

        h = tf.sqrt(
            tf.multiply((cx2 - cx3) , (cx2 - cx3)) +  tf.multiply((cy1 - cy3) ,(cy1 - cy3)))
        h = tf.maximum(h, tf.sqrt(tf.multiply((cx4 - cx2) , (cx4 - cx2)) +  tf.multiply( (cy4 - cy2) ,(cy4 - cy2))))

        w = tf.sqrt(
            tf.multiply((cx1 - cx4) , (cx1 - cx4) ) +  tf.multiply((cy1 - cy4) , (cy1 - cy4)))
        w =tf.maximum(w, tf.sqrt(tf.multiply((cx3 - cx2) ,(cx3 - cx2) ) +  tf.multiply( (cy3 - cy2) ,(cy3 - cy2))))

        h1 = tf.sqrt(
            tf.multiply((gx2 - gx3) , (gx2 - gx3)) +  tf.multiply((gy1 - gy3) , (gy1 - gy3)))
        h1 = tf.maximum(w, tf.sqrt(tf.multiply((gx4 - gx2) ,(gx4 - gx2) +  tf.multiply((gy4 - gy2) , (gy4 - gy2)))))

        w1 = tf.sqrt(
            tf.multiply((gx1 - gx4) , (gx1 - gx4)  +  tf.multiply((gy1 - gy4) ,(gy1 - gy4))))
        w1 =tf.maximum(w1, tf.sqrt(tf.multiply((gx3 - gx2) , (gx3 - gx2))  +  tf.multiply(  (gy3 - gy2) , (gy3 - gy2))))

        # carea = tf.maximum(0.0,(cx2 - cx1) * (cy2 - cy1))  # C的面积
        carea=tf.multiply(w,h)
        # garea = tf.maximum(0.0,(gx2 - gx1) * (gy2 - gy1))  # G的面积
        garea = tf.multiply(w1, h1)
        x1 = tf.maximum(cx1, gx1)
        y1 = tf.maximum(cy1, gy1)
        x2 = tf.minimum(cx2, gx2)
        y2 = tf.minimum(cy2, gy2)
        w = tf.maximum(0.0, x2 - x1)
        h = tf.maximum(0.0, y2 - y1)
        area = w * h  # C∩G的面积

        self.cx1 = carea

        # C∩G的面积
        union_square = tf.maximum(carea + garea - area, 1e-10)
        self.cx2 = garea
        iou=tf.clip_by_value(1-area / union_square, 0.0, 1.0)*100
        # iou = 1-area / tf.maximum(carea + garea - area,1e-8)
        return iou

    def leaky_relu(self,alpha):
        def op(inputs):
            return utils.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
        return op
# aaa=WM_Model(model_cfg)