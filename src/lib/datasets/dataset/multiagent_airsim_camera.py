'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-07-25 20:21:05
Description: 
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing_extensions import OrderedDict

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import pickle as pkl

import torch.utils.data as data
import sys
from collections import OrderedDict

sys.path.append(__file__)
from .test_eval import run_polygon_eval

sys.path.append(os.path.join(__file__, '..'))
from utils.eval_utils import eval_map


class MULTIAGENTAIRSIMCAM(data.Dataset):
    num_classes = 1
    default_resolution = [448, 800]
    mean = np.array([0.58375601, 0.54399371, 0.47015152],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.25869511, 0.24342069, 0.23500774],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(MULTIAGENTAIRSIMCAM, self).__init__()

        if opt.real:
            self.data_dir = '/DB/public/uav_dataset'
            self.img_dir = os.path.join(self.data_dir, 'images')
        else:
            self.data_dir = ['/GPFS/data/yhu/Dataset/airsim_camera/airsim_camera_seg_15', \
                            '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town6_v2', \
                            '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_40m', \
                            '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_60m', \
                            '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_80m'] if opt.input_dir is '' else opt.input_dir
            # if split == 'val':
                # self.data_dir = '/GPFS/data/yhu/Dataset/airsim_camera/airsim_camera_seg_15'
                # self.data_dir = '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town6_v2'
                # self.data_dir = '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_40m/'
                # self.data_dir = ['/GPFS/data/yhu/Dataset/airsim_camera/airsim_camera_seg_15',
                #                 '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town6_v2']    
            self.img_dir = self.data_dir
        
        self.annot_path = self._get_path(self.data_dir, opt.uav_height, split, 'instances_sample.pkl')
        self.annot_path_cocoformat = self._get_path(self.data_dir, opt.uav_height, split, 'instances_global_crop_woignoredbox.json')
        self.annot_path_cocoformat_uav = self._get_path(self.data_dir, opt.uav_height, split, 'instances_woignoredbox.json')
            
        self.max_objs = 128
        # self.class_name = [
        #     'car', 'car_overlook']
        # self._valid_ids = [1, 2]
        self.class_name = ['car']
        self._valid_ids = [1]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        print('==> initializing multiagent airsim_camera {} data.'.format(split))
        print(self.annot_path)
        print(self.annot_path_cocoformat)
        print(self.annot_path_cocoformat_uav)
        if isinstance(self.annot_path, str):
            self.samples = pkl.load(open(self.annot_path, 'rb'))['samples']
            self.img_dir = self.data_dir
        else:
            self.samples = []
            self.img_dir = []
            self.img_idx_mapping = []
            sample_counts = [0]
            for i, annot_path in enumerate(self.annot_path):
                cur_img_idx_mapping = OrderedDict()
                cur_sample = pkl.load(open(annot_path, 'rb'))['samples']
                sample_counts.append(len(cur_sample)*25)
                for sample_id, sample in enumerate(cur_sample):
                    cams = [x for x in sample.keys() if not x.startswith('vehicles')]
                    sorted(cams)
                    # assert len(cams) == 25
                    for cam_id, cam in enumerate(cams):
                        updated_img_idx = sum(sample_counts[:-1]) + sample_id * 25 + cam_id
                        cur_img_idx_mapping[sample[cam]['image_id']] = updated_img_idx
                        sample[cam]['image_id'] = updated_img_idx
                self.samples.extend(cur_sample)
                cur_data_dir = annot_path.split('multiagent_annotations')[0]
                self.img_dir.extend([cur_data_dir]*len(cur_sample))
                self.img_idx_mapping.append(cur_img_idx_mapping)
        if 'NO_MESSAGE' in opt.message_mode:
            self.num_agents = 1
            samples = []
            if isinstance(self.annot_path, str):
                img_dir = self.img_dir
            else:
                img_dir = []
            for i, sample in enumerate(self.samples):
                for k, data in sample.items():
                    if k.startswith('vehicles'):
                        continue
                    samples.append(data)
                    if not isinstance(self.annot_path, str):
                        img_dir.append(self.img_dir[i])
            self.samples = samples
            self.img_dir = img_dir
            # if split in ['train', 'val']:
            if split in ['train']:
                samples = []
                if isinstance(self.annot_path, str):
                    img_dir = self.img_dir
                else:
                    img_dir = []
                for i, sample in enumerate(self.samples):
                    # print(len(sample['vehicles_i']))
                    thre = 0 if self.opt.real else 20
                    if len(sample['vehicles_i']) > thre:  # keep samples which contains more than 10 vehicles
                        samples.append(sample)
                        if not isinstance(self.annot_path, str):
                            img_dir.append(self.img_dir[i])
                self.samples = samples
                self.img_dir = img_dir
            self.num_samples = len(self.samples)
        else:
            self.num_agents = int(opt.num_agents)
        self.num_samples = len(self.samples)
        print('Loaded {} {} samples'.format(split, self.num_samples))
        with open('gts.pkl', 'wb') as f:
            pkl.dump(self.samples, f)
        with open('imgs_dir.pkl', 'wb') as f:
            pkl.dump(self.img_dir, f)
    
    def _get_path(self, data_dirs, uav_heights='40', split='train', tail='instances_sample.pkl'):
        paths = []
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        if isinstance(uav_heights, str):
            uav_heights = [uav_heights]
        
        paths = [os.path.join(data_dir, 'multiagent_annotations', 'Collaboration', '{}_{}_{}'.format(uav_height, split, tail)) for data_dir in data_dirs for uav_height in uav_heights]
        valid_paths = [x for x in paths if os.path.exists(x)]

        if len(valid_paths) == 1:
            valid_paths = valid_paths[0]
        
        return valid_paths

    def _to_float(x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    if len(bbox) > 5:
                        bbox_out = [float("{:.2f}".format(bbox[i])) for i in range(len(bbox-1))]
                        score = bbox[-1]
                    else:
                        # print(bbox)
                        # print(type(bbox))
                        # print(bbox[0])
                        bbox[2] -= bbox[0]
                        bbox[3] -= bbox[1]
                        score = bbox[4]
                        # bbox_out = list(map(self._to_float, list(bbox[0:4])))
                        bbox_out = [float("{:.2f}".format(bbox[i])) for i in range(4)]

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_path):
        json.dump(self.convert_eval_format(results), open(save_path, 'w'))

    def run_eval(self, results, save_dir, eval_mode='Global'):
        if eval_mode == 'Global':
            annot_path_cocoformat = self.annot_path_cocoformat
        else:
            annot_path_cocoformat = self.annot_path_cocoformat_uav
        if isinstance(annot_path_cocoformat, str):
            self.coco = coco.COCO(annot_path_cocoformat)
        else:
            annot_cocoformat = {'images': [], "type": "instances", 'annotations': []}
            annot_g_cocoformat = {'images': [], "type": "instances", 'annotations': []}
            box_count = 0
            for i, (annot_path, annot_g_path) in enumerate(zip(self.annot_path_cocoformat_uav, self.annot_path_cocoformat)):
                cur_sample = json.load(open(annot_path, 'r'))
                cur_g_sample = json.load(open(annot_g_path, 'r'))
                for image, image_g in zip(cur_sample['images'], cur_g_sample['images']):
                    image['id'] = self.img_idx_mapping[i][image['id']]
                    image_g['id'] = self.img_idx_mapping[i][image_g['id']]

                for anno, anno_g in zip(cur_sample['annotations'], cur_g_sample['annotations']):
                    anno['image_id'] = self.img_idx_mapping[i][anno['image_id']]
                    anno_g['image_id'] = self.img_idx_mapping[i][anno_g['image_id']]
                    anno['id'] = box_count
                    anno_g['id'] = box_count
                    box_count += 1
                
                annot_cocoformat['categories'] = cur_sample['categories']
                annot_cocoformat['images'].extend(cur_sample['images'])
                annot_cocoformat['annotations'].extend(cur_sample['annotations'])

                annot_g_cocoformat['categories'] = cur_g_sample['categories']
                annot_g_cocoformat['images'].extend(cur_g_sample['images'])
                annot_g_cocoformat['annotations'].extend(cur_g_sample['annotations'])
            
            with open('{}/gts_{}.json'.format(save_dir, eval_mode), 'w') as f:
                json.dump(annot_cocoformat, f)
            with open('{}/gts_{}.json'.format(save_dir, 'Global'), 'w') as f:
                json.dump(annot_g_cocoformat, f)
            self.coco = coco.COCO('{}/gts_{}.json'.format(save_dir, eval_mode))
        
        save_path = '{}/results_{}.json'.format(save_dir, eval_mode)
        self.save_results(results, save_path)
        coco_dets = self.coco.loadRes(save_path)
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    
    def run_polygon_eval(self, results, save_dir, eval_mode):
        if eval_mode == 'Global':
            annot_path_cocoformat = self.annot_path_cocoformat
        else:
            annot_path_cocoformat = self.annot_path_cocoformat_uav
        save_path = '{}/results_{}.json'.format(save_dir, eval_mode)

        self.save_results(results, save_path)

        if isinstance(annot_path_cocoformat, str):
            run_polygon_eval(annot_path_cocoformat, save_path)
        else:
            annot_cocoformat = {'images': [], "type": "instances", 'annotations': []}
            box_count = 0
            for i, annot_path in enumerate(annot_path_cocoformat):
                cur_sample = json.load(open(annot_path, 'r'))
                for image in cur_sample['images']:
                    image['id'] = self.img_idx_mapping[i][image['id']]
                for anno in cur_sample['annotations']:
                    anno['image_id'] = self.img_idx_mapping[i][anno['image_id']]
                    anno['id'] = box_count
                    box_count += 1
                
                annot_cocoformat['categories'] = cur_sample['categories']
                annot_cocoformat['images'].extend(cur_sample['images'])
                annot_cocoformat['annotations'].extend(cur_sample['annotations'])
            
            with open('{}/gts_{}.json'.format(save_dir, eval_mode), 'w') as f:
                json.dump(annot_cocoformat, f)
            
            run_polygon_eval('{}/gts_{}.json'.format(save_dir, eval_mode), '{}/results_{}.json'.format(save_dir, eval_mode))