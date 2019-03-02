# -*- coding: utf-8 -*-
'''
@Time    : 18-10-17 下午2:41
@Author  : qinpengzhi
@File    : config.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
from easydict import EasyDict as edict

#设置数据相关的config,train_path是类似的带有位置标注的图片
WM_data_cfg=edict()
WM_data_cfg.train_path="../newdatares"
WM_data_cfg.test_path=""

#设置model相关参数
WM_model_cfg=edict()
WM_model_cfg.exp_name='WM_model'
WM_model_cfg.checkpoint_dir="../checkpoints"
WM_model_cfg.train_from_pretrained=False
WM_model_cfg.saver_max_to_keep = 5
WM_model_cfg.weight_decay = 0.0005
WM_model_cfg.saver_max_to_keep = 1
WM_model_cfg.learning_rate = 0.0001
WM_model_cfg.step_size = 5000
WM_model_cfg.momentum = 0.9

#设置train相关参数
WM_train_cfg=edict()
WM_train_cfg.summary_dir = '../summaries/WM'
WM_train_cfg.num_iter_per_epoch=100
WM_train_cfg.num_epochs=500