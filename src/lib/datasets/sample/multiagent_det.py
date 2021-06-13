'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-06-13 15:40:48
Description: 
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from itertools import count
import random

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math


class MultiAgentDetDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i
    
    def get_aug(self, height, width):
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(height, width) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w
        
        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, width)
                h_border = self._get_border(128, height)
                c[0] = np.random.randint(low=w_border, high=width - w_border)
                c[1] = np.random.randint(low=h_border, high=height - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:
                flipped = True
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        
        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])
        
        return c, s, flipped, trans_input, trans_output, input_h, input_w, output_h, output_w

    def load_sample(self, sample, num_images=8):
        images_key = 'image' if self.opt.coord_mode == 'local' else 'image_g'
        vehicles_key = 'vehicles_i' if self.opt.coord_mode == 'local' else 'vehicles_g'
        
        images = []
        vehicles = []
        category_idx = []
        trans_mats_list = []
        if num_images == 1:
            images.append(cv2.imread(os.path.join(self.img_dir, sample[images_key])))
            vehicles.append(sample[vehicles_key])
            category_idx.append(sample['category_id'])
            trans_mats_list.append(sample['trans_mat'])
        else:
            # cam_id = np.random.randint(low=0, high=5)
            cam_list = random.sample([x for x in sample.keys() if not x.startswith('vehicles')], num_images)
            for cam, info in sample.items():
                if cam.startswith('vehicles'):
                    continue
                # else:
                # elif cam.startswith('F'):
                # elif cam.endswith(str(cam_id)):
                if cam in cam_list:
                    images.append(cv2.imread(os.path.join(self.img_dir, info[images_key])))
                    vehicles.append(info[vehicles_key])
                    category_idx.append(info['category_id'])
                    trans_mats_list.append(info['trans_mat'])
        trans_mats_list = np.concatenate([x[None,:,:] for x in trans_mats_list], axis=0)
        height, width = images[0].shape[0], images[0].shape[1]
        # Use the same aug for the images in the same sample
        c, s, flipped, trans_input, trans_output, input_h, input_w, output_h, output_w = self.get_aug(height, width)

        num_classes = self.num_classes
        hm = np.zeros((num_images, num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((num_images, self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((num_images, 2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((num_images, self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((num_images, self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((num_images, self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((num_images, self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((num_images, self.max_objs, num_classes * 2), dtype=np.uint8)
        aug_imgs = np.zeros((num_images, 3, input_h, input_w), dtype=np.float32)
        
        # ----------- TransMat ------------- #
        trans_mats = np.zeros((num_images, 3, 3), dtype=np.float32)
        trans_mats[:min(num_images, len(trans_mats_list))] = trans_mats_list[:min(num_images, len(trans_mats_list))]

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_dets = []
        for index, info in enumerate(zip(images, vehicles, category_idx)):
            img, objs, category_ids = info
            if flipped:
                img = img[:, ::-1, :]
            inp = cv2.warpAffine(img, trans_input,
                                (input_w, input_h),
                                flags=cv2.INTER_LINEAR)
            # print(inp.shape, img.shape, input_w, input_h)
            inp = (inp.astype(np.float32) / 255.)
            if self.split == 'train' and not self.opt.no_color_aug:
                color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
            inp = (inp - self.mean) / self.std
            inp = inp.transpose(2, 0, 1)
            aug_imgs[index] = inp

            gt_det = []
            num_objs = len(objs)
            for k in range(num_objs):
                bbox = self._coco_box_to_bbox(objs[k])
                cls_id = int(self.cat_ids[category_ids[k]])
                if flipped:
                    bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                bbox[:2] = affine_transform(bbox[:2], trans_output)
                bbox[2:] = affine_transform(bbox[2:], trans_output)
                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                    ct = np.array(
                        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    draw_gaussian(hm[index, cls_id], ct_int, radius)
                    wh[index, k] = 1. * w, 1. * h
                    ind[index, k] = ct_int[1] * output_w + ct_int[0]
                    reg[index, k] = ct - ct_int
                    reg_mask[index, k] = 1
                    cat_spec_wh[index, k, cls_id * 2: cls_id * 2 + 2] = wh[index, k]
                    cat_spec_mask[index, k, cls_id * 2: cls_id * 2 + 2] = 1
                    if self.opt.dense_wh:
                        draw_dense_reg(dense_wh[index], hm.max(axis=1), ct_int, wh[index, k], radius)
                    cur_gt_det = [ct[0] - w / 2, ct[1] - h / 2,
                                ct[0] + w / 2, ct[1] + h / 2, 1, cls_id]
                    gt_det.append(cur_gt_det)
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1, 6), dtype=np.float32)
            gt_dets.append(gt_det)
        return c, s, aug_imgs, trans_mats, hm, wh, dense_wh, reg_mask, ind, cat_spec_wh, cat_spec_mask, reg, gt_dets

    def __getitem__(self, index):
        sample = self.samples[index]
        c, s, aug_imgs, trans_mats, hm, wh, dense_wh, reg_mask, ind, cat_spec_wh, cat_spec_mask, reg, gt_det = self.load_sample(sample, num_images=self.num_agents)
        ret = {'input': aug_imgs, 'trans_mats': trans_mats, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        if self.opt.dense_wh:
            hm_a = hm.max(axis=1, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=1)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = gt_det if len(gt_det) > 0 else [np.zeros((1, 6), dtype=np.float32)]
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'sample_id': index}
            ret['meta'] = meta
        return ret
