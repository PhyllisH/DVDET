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
from numpy.core.fromnumeric import put

import torch
import numpy as np

from models.losses import FocalLoss, RateDistortionLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, ZFocalLoss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer
import time
import os
from torch import stack 
import torch.nn as nn

def _reorganize_batch(batch):
    reorg_batch = OrderedDict()
    for k, info in batch.items():
        if 'meta' in k:
            reorg_batch[k] = info
        else:
            reorg_batch[k] = info.flatten(0, 1)
    return reorg_batch


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def rotation_2d_torch(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.
    
    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = angles[:,0]
    rot_cos = angles[:,1]
    rot_mat_T = torch.stack(
        [stack([rot_cos, -rot_sin]),
        stack([rot_sin, rot_cos])])
    return torch.einsum('aij,jka->aik', (points, rot_mat_T))

def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
      Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    b, k, _, _ = regr.shape
    regr = regr.view(b,k,-1)
    gt_regr = gt_regr.view(b,k,-1)
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss
class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.crit_z = ZFocalLoss()
        self.crit_comp = RateDistortionLoss()
        self.opt = opt
        self.acc_z = []
        self.count = 0

    def _acc_z(self, output_z, z, ind, reg_mask):
        z_pred = _transpose_and_gather_feat(output_z, ind)
        z_pred_cls = z_pred.argmax(-1)
        z_cls = z.argmax(-1)
        correct = ((z_pred_cls == z_cls)*1*reg_mask).sum()
        amount = reg_mask.sum()
        return correct/(amount+1e-6)

    
    def _angle_loss(self, output_angle, output_wh, output_reg, wh, reg, angle, reg_mask, ind):
        weights = torch.Tensor([-0.5,-0.5,-0.5,0.5,0.5,0.5,0.5,-0.5]).to(output_angle.device)
        weights = weights.reshape([4,2])

        def _get_corners(wh, reg, angle):
            corners = wh.detach().unsqueeze(2).expand(-1,-1,4,-1)
            cur_weights = weights.unsqueeze(0).expand(corners.shape[1],-1,-1).unsqueeze(0).expand(corners.shape[0],-1,-1,-1)
            corners = corners * cur_weights
            corners = corners + reg.detach().unsqueeze(2)    # (b, k, 4, 2)
            b, k, _ = angle.shape
            corners = corners.view(b*k, 4, 2)
            angle = angle.view(b*k, 2)
            corners = rotation_2d_torch(corners, angle)
            return corners.view(b,k,4,2).contiguous()

        pred_angle = _transpose_and_gather_feat(output_angle, ind)
        pred_wh = _transpose_and_gather_feat(output_wh, ind)
        pred_reg = _transpose_and_gather_feat(output_reg, ind)

        # get the corners
        pred_corners = _get_corners(pred_wh, pred_reg, pred_angle)
        gt_corners = _get_corners(wh, reg, angle)

        angle_loss = _reg_loss(pred_corners, gt_corners, reg_mask)
        return angle_loss

    def forward(self, outputs, ori_batch):
        opt = self.opt
        batch = _reorganize_batch(ori_batch)
        hm_loss, wh_loss, off_loss, angle_loss = 0, 0, 0, 0
        hm_loss_early, wh_loss_early, off_loss_early, angle_loss_early = 0, 0, 0, 0
        hm_loss_fused, wh_loss_fused, off_loss_fused, angle_loss_fused = 0, 0, 0, 0
        hm_loss_i, wh_loss_i, off_loss_i = 0, 0, 0
        z_loss = 0
        acc_z = 0
        comp_loss = 0
        comp_aux_loss = 0
        single_loss = {}
        for i in range(opt.round):
            single_loss['hm_single_r{}_loss'.format(i)] = 0
            single_loss['wh_single_r{}_loss'.format(i)] = 0
            single_loss['off_single_r{}_loss'.format(i)] = 0
            single_loss['angle_single_r{}_loss'.format(i)] = 0
        
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])
                if (opt.coord in ['Global', 'Joint']) and (opt.feat_mode=='fused'):
                    output['hm_early'] = _sigmoid(output['hm_early'])
                    output['hm_fused'] = _sigmoid(output['hm_fused'])
                if opt.coord == 'Joint':
                    output['hm_i'] = _sigmoid(output['hm_i'])
                
                for i in range(opt.round):
                    output['hm_single_r{}'.format(i)] = _sigmoid(output['hm_single_r{}'.format(i)])
                    single_loss['hm_single_r{}_loss'.format(i)] += self.crit(output['hm_single_r{}'.format(i)], batch['hm']) / opt.num_stacks

            if opt.eval_oracle_hm:
                output['hm'] = batch['hm']
                if opt.coord == 'Joint':
                    output['hm_i'] = batch['hm_i']
            if opt.eval_oracle_wh:
                output['wh'] = torch.from_numpy(gen_oracle_map(
                    batch['wh'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
                if opt.coord == 'Joint':
                    output['wh_i'] = torch.from_numpy(gen_oracle_map(
                        batch['wh_i'].detach().cpu().numpy(),
                        batch['ind_i'].detach().cpu().numpy(),
                        output['wh_i'].shape[3], output['wh_i'].shape[2])).to(opt.device)
            if opt.eval_oracle_offset:
                output['reg'] = torch.from_numpy(gen_oracle_map(
                    batch['reg'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)
                if opt.coord == 'Joint':
                    output['reg_i'] = torch.from_numpy(gen_oracle_map(
                        batch['reg_i'].detach().cpu().numpy(),
                        batch['ind_i'].detach().cpu().numpy(),
                        output['reg_i'].shape[3], output['reg_i'].shape[2])).to(opt.device)

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if (opt.coord in ['Global', 'Joint']) and (opt.feat_mode=='fused'):
                hm_loss_early += self.crit(output['hm_early'], batch['hm']) / opt.num_stacks
                hm_loss_fused += self.crit(output['hm_fused'], batch['hm']) / opt.num_stacks
            if opt.coord == 'Joint':
                hm_loss_i += self.crit(output['hm_i'], batch['hm_i']) / opt.num_stacks
            
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
                    if opt.coord == 'Joint':
                        mask_weight = batch['dense_wh_mask_i'].sum() + 1e-4
                        wh_loss_i += (
                                        self.crit_wh(output['wh_i'] * batch['dense_wh_mask_i'],
                                                        batch['dense_wh_i'] * batch['dense_wh_mask_i']) /
                                        mask_weight) / opt.num_stacks
                elif opt.cat_spec_wh:
                    wh_loss += self.crit_wh(
                        output['wh'], batch['cat_spec_mask'],
                        batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
                    if opt.coord == 'Joint':
                        wh_loss_i += self.crit_wh(
                            output['wh_i'], batch['cat_spec_mask_i'],
                            batch['ind_i'], batch['cat_spec_wh_i']) / opt.num_stacks
                else:
                    wh_loss += self.crit_reg(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks
                    
                    for i in range(opt.round):
                        single_loss['wh_single_r{}_loss'.format(i)] += self.crit_reg(
                                            output['wh_single_r{}'.format(i)], batch['reg_mask'],
                                            batch['ind'], batch['wh']) / opt.num_stacks
                    if (opt.coord in ['Global', 'Joint']) and (opt.feat_mode=='fused'):
                        wh_loss_early += self.crit_reg(
                            output['wh_early'], batch['reg_mask'],
                            batch['ind'], batch['wh']) / opt.num_stacks
                        wh_loss_fused += self.crit_reg(
                            output['wh_fused'], batch['reg_mask'],
                            batch['ind'], batch['wh']) / opt.num_stacks
                    if opt.coord == 'Joint':
                        wh_loss_i += self.crit_reg(
                            output['wh_i'], batch['reg_mask_i'],
                            batch['ind_i'], batch['wh_i']) / opt.num_stacks
                    if ('z' in output) and (opt.coord in ['Global', 'Joint']):
                        z_loss += self.crit_z(
                            output['z'], batch['reg_mask'],
                            batch['ind'], batch['cat_depth']) / opt.num_stacks
                        acc_z += self._acc_z(output['z'], batch['cat_depth'], batch['ind'], batch['reg_mask'])
            if opt.polygon and (opt.angle_weight > 0):
                if opt.coord in ['Global', 'Joint']:
                    angle_loss += self.crit_reg(
                            output['angle'], batch['reg_mask'],
                            batch['ind'], batch['angle']) / opt.num_stacks
                    for i in range(opt.round):
                        single_loss['angle_single_r{}_loss'.format(i)] += self.crit_reg(
                                            output['angle_single_r{}'.format(i)], batch['reg_mask'],
                                            batch['ind'], batch['angle']) / opt.num_stacks
                    # angle_loss += self._angle_loss(output['angle'], output['wh'], output['reg'],
                    #                                  batch['wh'], batch['reg'], batch['angle'], batch['reg_mask'], batch['ind'])
                    if opt.feat_mode == 'fused':
                        angle_loss_early += self.crit_reg(
                            output['angle_early'], batch['reg_mask'],
                            batch['ind'], batch['angle']) / opt.num_stacks
                        angle_loss_fused += self.crit_reg(
                            output['angle_fused'], batch['reg_mask'],
                            batch['ind'], batch['angle']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                        batch['ind'], batch['reg']) / opt.num_stacks
                for i in range(opt.round):
                    single_loss['off_single_r{}_loss'.format(i)] += self.crit_reg(
                                        output['reg_single_r{}'.format(i)], batch['reg_mask'],
                                        batch['ind'], batch['reg']) / opt.num_stacks
                if (opt.coord in ['Global', 'Joint']) and (opt.feat_mode=='fused'):
                    off_loss_early += self.crit_reg(output['reg_early'], batch['reg_mask'],
                                            batch['ind'], batch['reg']) / opt.num_stacks
                    off_loss_fused += self.crit_reg(output['reg_fused'], batch['reg_mask'],
                                            batch['ind'], batch['reg']) / opt.num_stacks
                if opt.coord == 'Joint':
                    off_loss_i += self.crit_reg(output['reg_i'], batch['reg_mask_i'],
                                            batch['ind_i'], batch['reg_i']) / opt.num_stacks

        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss 
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss}
        if opt.polygon and (opt.angle_weight > 0):
            loss = loss + opt.angle_weight * angle_loss
            loss_stats.update({'angle_loss': angle_loss})
        if opt.feat_mode in ['fused']:
            loss = loss + opt.hm_weight * (hm_loss_early + hm_loss_fused) + \
                    opt.wh_weight * (wh_loss_early + wh_loss_fused) + \
                    opt.off_weight * (off_loss_early + off_loss_fused) 
            loss_stats.update({'loss': loss, 
                    'hm_loss_early': hm_loss_early, 'wh_loss_early': wh_loss_early, 'off_loss_early': off_loss_early,
                    'hm_loss_fused': hm_loss_fused, 'wh_loss_fused': wh_loss_fused, 'off_loss_fused': off_loss_fused})
            if opt.polygon and (opt.angle_weight > 0):
                loss = loss + opt.angle_weight * (angle_loss_early + angle_loss_fused)
                loss_stats.update({'angle_loss_early': angle_loss_early, 'angle_loss_fused': angle_loss_fused})
        if opt.coord == 'Joint':
            loss = loss + opt.hm_weight *  hm_loss_i + \
                    opt.wh_weight * wh_loss_i + \
                    opt.off_weight * off_loss_i
            loss_stats.update({'loss': loss,'hm_loss_i': hm_loss_i, 'wh_loss_i': wh_loss_i, 'off_loss_i': off_loss_i})
        if ('z' in output) and (opt.depth_mode == 'Weighted'):
            loss = loss + opt.wh_weight * z_loss
            loss_stats.update({'loss': loss, 'z_loss': z_loss, 'acc_z': acc_z})
            self.acc_z.append(acc_z)
        if len(single_loss)>0:
            for i in range(opt.round):
                loss = loss + opt.hm_weight * single_loss['hm_single_r{}_loss'.format(i)] \
                            + opt.wh_weight * single_loss['wh_single_r{}_loss'.format(i)] \
                            + opt.off_weight * single_loss['off_single_r{}_loss'.format(i)]
                if opt.polygon and (opt.angle_weight > 0):
                    loss = loss + opt.angle_weight * single_loss['angle_single_r{}_loss'.format(i)]
            loss_stats.update(single_loss)
        
        if opt.train_mode in ['compressor', 'full']:
            comp_loss += self.crit_comp(output['comp_out'], output['comp_gt'])["loss"]
            comp_aux_loss += output['comp_aux_loss']
            loss_stats.update({
                'comp_loss': comp_loss,
                'comp_aux_loss': comp_aux_loss
            })
            return loss, comp_loss, comp_aux_loss, loss_stats
        return loss, loss, loss, loss_stats


class MultiAgentDetTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MultiAgentDetTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        if opt.feat_mode in ['fused']:
            loss_states.extend(['hm_loss_early', 'wh_loss_early', 'off_loss_early', \
                            'hm_loss_fused', 'wh_loss_fused', 'off_loss_fused'])
        if opt.depth_mode == 'Weighted':
            loss_states.extend(['z_loss', 'acc_z'])
        if opt.polygon and (opt.angle_weight > 0):
            loss_states.append('angle_loss')
            if opt.feat_mode == 'fused':
                loss_states.extend(['angle_loss_early', 'angle_loss_fused'])
        if opt.coord == 'Joint':
            loss_states.extend(['hm_loss_i', 'wh_loss_i', 'off_loss_i'])
        if opt.round >=1:
            for i in range(opt.round):
                loss_states.append('hm_single_r{}_loss'.format(i))
                loss_states.append('wh_single_r{}_loss'.format(i))
                loss_states.append('off_single_r{}_loss'.format(i))
                if opt.polygon and (opt.angle_weight > 0):
                    loss_states.append('angle_single_r{}_loss'.format(i))
        if opt.train_mode in ['compressor', 'full']:
            loss_states.extend(['comp_loss', 'comp_aux_loss'])
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
