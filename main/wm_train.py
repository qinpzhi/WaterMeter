# -*- coding: utf-8 -*-
'''
@Time    : 18-9-5 下午6:41
@Author  : qinpengzhi
@File    : WM_train.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
from utils.WM_config import WM_data_cfg,WM_train_cfg,WM_model_cfg
from pprint import pprint
import os
import tensorflow as tf
from utils.WM_data_provider import WM_DataProvider
from utils.logger import TfLogger
from main.wm_model import WM_Model
from main.wm_trainer import WM_Trainer

def create_dirs(dirs):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
def main():
    print('---------------------- data config ------------------------')
    pprint(WM_data_cfg)

    print('---------------------- model config -------------------')
    pprint(WM_model_cfg)

    print('creating dirs for saving model weights, logs ...')
    checkpoint_dir = os.path.join(
        WM_model_cfg.checkpoint_dir, WM_model_cfg.exp_name)
    create_dirs([checkpoint_dir, WM_train_cfg.summary_dir])

    print('initializing train data provider....')
    data_provider = WM_DataProvider(WM_data_cfg)

    config=tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=0.4
    sess=tf.InteractiveSession(config=config)

    # sess = tf.Session()
    print('creating tensorflow log for summaries...')
    tf_logger = TfLogger(sess, WM_train_cfg)
    print('creating seg models ...')
    train_model = WM_Model(WM_model_cfg)
    if WM_model_cfg.train_from_pretrained:
        train_model.load(sess)

    print('creating seg trainer...')
    trainer = WM_Trainer(sess, train_model,
                         data_provider, WM_train_cfg, tf_logger)

    print('start trainning...')
    trainer.train()

    sess.close()


if __name__ == '__main__':
    main()
