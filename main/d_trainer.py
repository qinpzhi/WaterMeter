# -*- coding: utf-8 -*-
'''
@Time    : 18-10-22 下午3:47
@Author  : qinpengzhi
@File    : wm_trainer.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np

class D_Trainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(D_Trainer,self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop= tqdm(range(self.config.num_iter_per_epoch))
        wm_losses = []
        for it in loop:
            loss = self.train_step()
            wm_losses.append(loss)
        agv_wm_loss=np.mean(wm_losses)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        print 'wm_losses=%f;itr=%d' % (
            agv_wm_loss, cur_it)

        summaries_dict = {}
        summaries_dict['wm_losses'] = agv_wm_loss
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        im_data, im_real=self.data.next_batch()
        feed_dict={self.model.data:im_data, self.model.real:im_real,self.model.keep_prob:0.85}
        seg_loss, _ =self.sess.run([self.model.loss,self.model.optimizer],feed_dict = feed_dict)
        return seg_loss