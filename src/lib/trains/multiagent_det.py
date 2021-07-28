'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-07-26 09:42:05
Description: 
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import inf
from typing_extensions import OrderedDict

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer
import time
import os
import cv2,kornia

def _reorganize_batch(batch):
    reorg_batch = OrderedDict()
    for k, info in batch.items():
        if 'meta' in k:
            reorg_batch[k] = info
        else:
            reorg_batch[k] = info.flatten(0, 1)
    return reorg_batch

class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt

    def forward(self, outputs, ori_batch):
        opt = self.opt
        batch = _reorganize_batch(ori_batch)
        hm_loss, wh_loss, off_loss, angle_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            if opt.eval_oracle_hm:
                output['hm'] = batch['hm']
            if opt.eval_oracle_wh:
                output['wh'] = torch.from_numpy(gen_oracle_map(
                    batch['wh'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
            if opt.eval_oracle_offset:
                output['reg'] = torch.from_numpy(gen_oracle_map(
                    batch['reg'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            
            # import ipdb; ipdb.set_trace()
            # if not os.path.exists('train_vis'):
            #     os.makedirs('train_vis')
            # cur_time = time.time()
            # c_b = output['hm'][0,0].detach().cpu().numpy()
            # c_b = (c_b / (c_b.max()-c_b.min()) * 255.0).astype('uint8')[:,:,None]
            # c_r = (batch['hm'][0,0]*255).detach().cpu().numpy().astype('uint8')[:,:,None]
            # c_g = np.zeros_like(c_b)
            # heatmap = np.concatenate([c_b, c_g, c_r], axis=-1)
            # cv2.imwrite('train_vis/{}_pred_hm.png'.format(int(cur_time)), heatmap)

            # save_img = (batch['input'][0].detach().cpu().numpy()*255).astype('uint8').transpose(1,2,0)
            # cv2.imwrite('train_vis/{}_ori.png'.format(int(cur_time)), save_img)
            # worldgrid2worldcoord_mat = np.array([[500/800.0, 0, -200], [0, 500/448.0, -250], [0, 0, 1]])
            # cur_trans_mats = np.linalg.inv(batch['trans_mats'][0].detach().cpu().numpy() @ worldgrid2worldcoord_mat)
            # data = kornia.image_to_tensor(save_img, keepdim=False)
            # data_warp = kornia.warp_perspective(data.float(),
            #                                     torch.tensor(cur_trans_mats).repeat([1, 1, 1]).float(),
            #                                     dsize=(448, 800))
            # # convert back to numpy
            # img_warp = kornia.tensor_to_image(data_warp.byte())
            # img_warp = cv2.resize(img_warp, dsize=(400, 224))
            # img_warp = cv2.addWeighted(heatmap, 0.5, img_warp, 1-0.5, 0)
            # cv2.imwrite('train_vis/{}_ori_warp.png'.format(int(cur_time)), img_warp)

            
            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (
                                       self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                                    batch['dense_wh'] * batch['dense_wh_mask']) /
                                       mask_weight) / opt.num_stacks
                elif opt.cat_spec_wh:
                    wh_loss += self.crit_wh(
                        output['wh'], batch['cat_spec_mask'],
                        batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
                else:
                    wh_loss += self.crit_reg(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks
            if opt.polygon and (opt.angle_weight > 0):
                angle_loss += self.crit_reg(
                        output['angle'], batch['reg_mask'],
                        batch['ind'], batch['angle']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

        if opt.polygon and (opt.angle_weight > 0):
            loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
                    opt.off_weight * off_loss + opt.angle_weight * angle_loss
            loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                    'wh_loss': wh_loss, 'off_loss': off_loss, 'angle_loss': angle_loss}
        else:
            loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
                    opt.off_weight * off_loss
            loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                        'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats


class MultiAgentDetTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MultiAgentDetTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        if opt.polygon and (opt.angle_weight > 0):
            loss_states.append('angle_loss')
        loss = CtdetLoss(opt)
        return loss_states, loss

    def debug(self, ori_batch, output, iter_id):
        opt = self.opt
        batch = _reorganize_batch(ori_batch)
        reg = output['reg'] if opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=opt.cat_spec_wh, K=opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        for i in range(1):
            debugger = Debugger(
                dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((
                                   img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                           dets[i, k, 4], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                           dets_gt[i, k, 4], img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
