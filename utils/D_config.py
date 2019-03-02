# -*- coding: utf-8 -*-
'''
@Time    : 18-10-17 下午2:41
@Author  : qinpengzhi
@File    : config.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
from easydict import EasyDict as edict

#设置数据相关的config
D_data_cfg=edict()
D_data_cfg.train_path="../pic"
D_data_cfg.test_path=""

#设置model相关参数
D_model_cfg=edict()
D_model_cfg.exp_name='D_model'
D_model_cfg.checkpoint_dir="../checkpoints"
D_model_cfg.train_from_pretrained=False
D_model_cfg.saver_max_to_keep = 5
D_model_cfg.weight_decay = 0.0004
D_model_cfg.saver_max_to_keep = 1
D_model_cfg.learning_rate = 0.0001
D_model_cfg.step_size = 5000
D_model_cfg.momentum = 0.9

#设置train相关参数
D_train_cfg=edict()
D_train_cfg.summary_dir = '../summaries/seg'
D_train_cfg.num_iter_per_epoch=100
D_train_cfg.num_epochs=200