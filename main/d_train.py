# -*- coding: utf-8 -*-
'''
@Time    : 18-9-5 下午6:41
@Author  : qinpengzhi
@File    : WM_train.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
from utils.D_config import D_data_cfg,D_train_cfg,D_model_cfg
from pprint import pprint
import os
import tensorflow as tf
from utils.D_data_provider import D_DataProvider
from utils.logger import TfLogger
from main.d_model import D_Model
from main.d_trainer import D_Trainer

def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
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
    pprint(D_data_cfg)

    print('---------------------- model config -------------------')
    pprint(D_model_cfg)

    print('creating dirs for saving model weights, logs ...')
    checkpoint_dir = os.path.join(
        D_model_cfg.checkpoint_dir, D_model_cfg.exp_name)
    create_dirs([checkpoint_dir, D_train_cfg.summary_dir])

    print('initializing train data provider....')
    data_provider = D_DataProvider(D_data_cfg)

    config=tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=0.4
    sess=tf.InteractiveSession(config=config)

    # sess = tf.Session()
    print('creating tensorflow log for summaries...')
    tf_logger = TfLogger(sess, D_train_cfg)
    print('creating seg models ...')
    train_model = D_Model(D_model_cfg)
    if D_model_cfg.train_from_pretrained:
        train_model.load(sess)

    print('creating seg trainer...')
    trainer = D_Trainer(sess, train_model,
                         data_provider, D_train_cfg, tf_logger)

    print('start trainning...')
    trainer.train()

    sess.close()


if __name__ == '__main__':
    main()
