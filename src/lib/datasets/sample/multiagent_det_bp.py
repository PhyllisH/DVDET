'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-07-26 23:25:16
Description: 
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from itertools import count
import random
from numpy.core.numeric import ones
from numpy.lib.function_base import angle

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian, draw_polygon_gaussian
from utils.image import draw_dense_reg
import math
import kornia


class MultiAgentDetDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox
    
    def _coco2polygon(self, bbox):
        x, y, w, h = bbox
        left_up = [x, y]
        left_bottom = [x, y+h]
        right_up = [x+w, y]
        right_bottom = [x+w, y+h]
        return np.array([left_up, left_bottom, right_bottom, right_up]).reshape([4,2]).T

    def _2d_bounding_box(self, cords):
        """
        transform the 3D bounding box to 2D
        :param cords: <3, 8> the first channel: x, y, z; the second channel is the points amount
        :return <4, > 2D bounding box (x, y, w, h)
        """
        x_min = min(cords[0])
        x_max = max(cords[0])
        y_min = min(cords[1])
        y_max = max(cords[1])
        bbox = np.array([x_min, y_min, x_max, y_max],
                        dtype=np.float32)
        return bbox
    
    def _get_polygon_gt(self, polygon):
        def _nor_theta(theta):
            if theta > math.radians(45):
                theta -= math.radians(90)
                theta = _nor_theta(theta)
            elif theta <= math.radians(-45):
                theta += math.radians(90)
                theta = _nor_theta(theta)
            return theta

        def _calc_bearing(point1, point2):
            x1, y1 = point1
            x2, y2 = point2
            theta = math.atan2(y2 - y1, x2 - x1)
            theta = _nor_theta(theta)
            return theta
        
        def _get_corners(polygon):
            x_sort = np.argsort(polygon[0])
            lefts = polygon[:, x_sort[:2]]
            rights = polygon[:, x_sort[2:]]
            if abs(lefts[1,0] - lefts[1,1]) > 1e-6:
                lefts = lefts[:, np.argsort(lefts[1,:])]
            if abs(rights[1,0] - rights[1,1]) > 1e-6:
                rights = rights[:, np.argsort(rights[1,:])[::-1]]
            ordered_polygon = list(np.concatenate([lefts, rights], axis=-1).reshape(-1,))
            return ordered_polygon
        
        corners = np.array(_get_corners(polygon)).reshape(2,4).T    # (4,2)
        # print('corners: ', corners)
        center = np.mean(np.array(corners), 0)
        theta = _calc_bearing(corners[0], corners[1])
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
        out_points = np.matmul(corners - center, rotation) + center
        x, y = list(out_points[0, :])
        w, h = list(out_points[2, :] - out_points[0, :])
        # print(h, w)
        # h, w = np.sort(np.sqrt(np.square(corners[1:] - corners[0:1].repeat(3, axis=0)).sum(axis=-1)), axis=-1)[:2]
        # print(h, w)
        # print([center[0], center[1], w, h, np.sin(tsheta), np.cos(theta)])
        return [center[0], center[1], w, h, np.sin(theta), np.cos(theta)]

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
        # if self.split == 'train' and self.opt.coord != 'Global':
        #     if not self.opt.not_rand_crop:
        #         s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        #         w_border = self._get_border(128, width)
        #         h_border = self._get_border(128, height)
        #         c[0] = np.random.randint(low=w_border, high=width - w_border)
        #         c[1] = np.random.randint(low=h_border, high=height - h_border)
        #     else:
        #         sf = self.opt.scale
        #         cf = self.opt.shift
        #         c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
        #         c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
        #         s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        #     if np.random.random() < self.opt.flip:
        #         flipped = True
        #         c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        
        output_h = int(input_h // self.opt.down_ratio)
        output_w = int(input_w // self.opt.down_ratio)
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])
        
        return c, s, flipped, trans_input, trans_output, input_h, input_w, output_h, output_w

    def load_sample(self, sample, img_dir, num_images=8):
        images_key = 'image'
        vehicles_key = 'vehicles_g_corners' 
        
        images = []
        vehicles = []
        vehicles_i = []
        category_idx = []
        trans_mats_list = []
        if num_images == 1:
            images.append(cv2.imread(os.path.join(img_dir, sample[images_key])))
            vehicles.append(sample[vehicles_key])
            category_idx.append(sample['category_id'])
            trans_mats_list.append(sample['trans_mat'])
            vehicles_i.append(sample['vehicles_i'])
        else:
            # cam_id = np.random.randint(low=0, high=5)
            # cam_list = random.sample(set([x for x in sample.keys() if not x.startswith('vehicles')]), random.randint(2, num_images))
            picked_view = list(np.random.choice(['FRONT', 'BOTTOM', 'LEFT', 'RIGHT', 'BACK'], 5))
            cam_id = list(range(5))
            cam_list = ['{}_{}'.format(x, y) for x, y in zip(picked_view, cam_id)]
            # print(cam_list)
            for cam, info in sample.items():
                if cam.startswith('vehicles'):
                    continue
                # else:
                # elif cam.startswith('F'):
                # elif cam.endswith(str(cam_id)):
                if cam in cam_list:
                    images.append(cv2.imread(os.path.join(img_dir, info[images_key])))
                    vehicles.append(info[vehicles_key])
                    category_idx.append(info['category_id'])
                    trans_mats_list.append(info['trans_mat'])
                    vehicles_i.append(info['vehicles_i'])
        trans_mats_list = np.concatenate([x[None,:,:] for x in trans_mats_list], axis=0)
        height, width = images[0].shape[0], images[0].shape[1]
        # Use the same aug for the images in the same sample
        c, s, flipped, trans_input, trans_output, input_h, input_w, output_h, output_w = self.get_aug(height, width)

        num_classes = self.num_classes
        hm = np.zeros((num_images, num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((num_images, self.max_objs, 2), dtype=np.float32)
        angle = np.zeros((num_images, self.max_objs, 2), dtype=np.float32)
        angle[:,:,1] = 1.0
        dense_wh = np.zeros((num_images, 2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((num_images, self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((num_images, self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((num_images, self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((num_images, self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((num_images, self.max_objs, num_classes * 2), dtype=np.uint8)
        aug_imgs = np.zeros((num_images, 3, input_h, input_w), dtype=np.float32)
        
        # ----------- TransMat ------------- #
        trans_mats = np.eye(3, dtype=np.float32)[None,].repeat(num_images, axis=0)
        trans_mats[:min(num_images, len(trans_mats_list))] = trans_mats_list[:min(num_images, len(trans_mats_list))]

        # draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
        #     draw_umich_gaussian
        draw_gaussian = draw_msra_gaussian

        gt_dets = []
        for index, info in enumerate(zip(images, vehicles, vehicles_i, category_idx)):
            img, objs, objs_i, category_ids = info
            if flipped:
                img = img[:, ::-1, :]
            inp = cv2.warpAffine(img, trans_input,
                                (input_w, input_h),
                                flags=cv2.INTER_LINEAR)
            # print(inp.shape, img.shape, input_w, input_h)
            inp = (inp.astype(np.float32) / 255.)
            if self.split == 'train' and not self.opt.no_color_aug:
                color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
            # inp = (inp - self.mean) / self.std
            inp = inp.transpose(2, 0, 1)
            aug_imgs[index] = inp

            gt_det = []
            num_objs = len(objs)
            # worldgrid2worldcoord_mat = np.array([[500/input_w, 0, -200], [0, 500/input_h, -250], [0, 0, 1]])
            worldgrid2worldcoord_mat = np.array([[1/input_w, 0, 0], [0, 1/input_h, 0], [0, 0, 1]])
            for k in range(num_objs):
                cls_id = int(self.cat_ids[category_ids[k]])
                if self.opt.coord == 'Global':
                    # # ########## Get BEV Global coord from the transformation of UAV rectangle #####
                    # print('----------------- UAV2BEV ----------------')
                    # bbox = self._coco2polygon(objs_i[k])  # (2, 4)
                    # if flipped:
                    #     bbox[0] = width - bbox[0] - 1
                    # coord = np.concatenate([bbox, np.ones([1, bbox.shape[-1]])], axis=0)  # (3, 4)
                    # cur_trans_mats = np.linalg.inv(trans_mats[index] @ worldgrid2worldcoord_mat)
                    # coord_warp = cur_trans_mats @ coord
                    # coord_warp = coord_warp / coord_warp[2, :]
                    # # print('UAV2BEV: ', coord_warp)
                    # coord_warp = trans_output @ coord_warp  # (3, 4)
                    # # print('UAV2BEV image: ', coord_warp)

                    # ########## Get BEV Global coord #####
                    # print('----------------- BEV ----------------')
                    # print(objs[k])
                    coord = np.array(objs[k]).reshape([4,2])
                    # print('BEV: ', coord)
                    coord = np.concatenate([coord.T, np.ones([1, coord.shape[0]])], axis=0)
                    coord_warp = trans_output @ coord
                    # print('BEV image: ', coord_warp)
                    if self.opt.polygon:
                        bbox = self._get_polygon_gt(coord_warp[:2]) # (x,y,w,h,sin,cos)
                        gt_polygon = coord_warp[:2]
                    else:
                        bbox = self._2d_bounding_box(coord_warp)
                else:
                    bbox = self._coco_box_to_bbox(objs[k])
                    if flipped:
                        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                    bbox[:2] = affine_transform(bbox[:2], trans_output)
                    bbox[2:] = affine_transform(bbox[2:], trans_output)
                
                if self.opt.polygon:
                    bbox[0] = np.clip(bbox[0], 0, output_w - 1)
                    bbox[1] = np.clip(bbox[1], 0, output_h - 1)
                    h, w = bbox[3], bbox[2]
                    ct = np.array(
                        [bbox[0], bbox[1]], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                else:
                    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
                    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
                    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                    ct = np.array(
                        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    # print(ct_int, h, w)
                if h > 0 and w > 0:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)), min_overlap=0.2)
                    radius = max(1, int(radius))
                    radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                    # print('h, w: ', h, w)
                    # print('radius: ', radius)
                    # if self.opt.polygon:
                    #     draw_polygon_gaussian(hm[index, cls_id], ct_int, radius, bbox[4:])
                    # else:
                    draw_gaussian(hm[index, cls_id], ct_int, radius)
                    wh[index, k] = 1. * w, 1. * h
                    ind[index, k] = ct_int[1] * output_w + ct_int[0]
                    reg[index, k] = ct - ct_int
                    reg_mask[index, k] = 1
                    cat_spec_wh[index, k, cls_id * 2: cls_id * 2 + 2] = wh[index, k]
                    cat_spec_mask[index, k, cls_id * 2: cls_id * 2 + 2] = 1
                    if self.opt.dense_wh:
                        draw_dense_reg(dense_wh[index], hm.max(axis=1), ct_int, wh[index, k], radius)
                    
                    if self.opt.polygon:
                        angle[index, k] = bbox[4:]
                        # print('gt: ', bbox)
                        # print('polygon: ', gt_polygon)
                        cur_gt_det = [gt_polygon[0,0], gt_polygon[1,0], gt_polygon[0,1], gt_polygon[1,1],
                                      gt_polygon[0,2], gt_polygon[1,2], gt_polygon[0,3], gt_polygon[1,3], 1, cls_id]
                    else:
                        cur_gt_det = [ct[0] - w / 2, ct[1] - h / 2,
                                    ct[0] + w / 2, ct[1] + h / 2, 1, cls_id]
                    gt_det.append(cur_gt_det)
            if len(gt_det) > 0:
                gt_det = np.array(gt_det, dtype=np.float32)
            elif self.opt.polygon:
                gt_det = np.zeros((1, 10), dtype=np.float32)
            else:
                gt_det = np.zeros((1, 6), dtype=np.float32)
            gt_dets.append(gt_det)

            if len(np.where(hm[index,0])[0]) > 0:
                cv2.imwrite('hm.png', (hm[index,0]*255).astype('uint8'))
                # print(np.where(hm[index,0]))
                save_img = (aug_imgs[index]*255).astype('uint8').transpose(1,2,0)
                cv2.imwrite('ori.png', save_img)
                # cur_trans_mats = np.linalg.inv(trans_mats[0] @ worldgrid2worldcoord_mat @ np.linalg.inv(np.concatenate([trans_output, np.array([0, 0, 1]).reshape([1,3])], axis=0)))
                cur_trans_mats = np.linalg.inv(trans_mats[index] @ worldgrid2worldcoord_mat)
                data = kornia.image_to_tensor(save_img, keepdim=False)
                data_warp = kornia.warp_perspective(data.float(),
                                                    torch.tensor(cur_trans_mats).repeat([1, 1, 1]).float(),
                                                    dsize=(input_h, input_w))
                # convert back to numpy
                img_warp = kornia.tensor_to_image(data_warp.byte())
                img_warp = cv2.resize(img_warp, dsize=(output_w, output_h))
                heatmap = (hm[index,0].reshape([output_h, output_w, 1]).repeat(3, axis=-1)*255).astype('uint8')
                img_warp = cv2.addWeighted(heatmap, 0.5, img_warp, 1-0.5, 0)
                cv2.imwrite('ori_warp.png', img_warp)

        return c, s, aug_imgs, trans_mats, hm, wh, angle, dense_wh, reg_mask, ind, cat_spec_wh, cat_spec_mask, reg, gt_dets

    def __getitem__(self, index):
        sample = self.samples[index]
        img_dir = self.img_dir if isinstance(self.img_dir, str) else self.img_dir[index]
        c, s, aug_imgs, trans_mats, hm, wh, angle, dense_wh, reg_mask, ind, cat_spec_wh, cat_spec_mask, reg, gt_det = self.load_sample(sample, img_dir, num_images=self.num_agents)
        ret = {'input': aug_imgs, 'trans_mats': trans_mats, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'angle': angle}
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
