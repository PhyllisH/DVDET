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
        # self.data_dir = os.path.join(opt.data_dir, 'airsim_camera_10scene')
        # self.data_dir = ['/DATA7_DB7/data/shfang/airsim_camera_seg_15']
        # self.data_dir = ['/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town6_v2']
        # self.data_dir = ['/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_80m']
        # self.data_dir = '/DATA7_DB7/data/shfang/airsim_camera_seg_15'
        self.data_dir = ['/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_40m/'] if opt.input_dir is '' else opt.input_dir
        # self.data_dir = ['/DATA7_DB7/data/shfang/airsim_camera_seg_15', \
        #               '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town6_v2', \
        #                 '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_80m', \
        #                     '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_40m/'] if opt.input_dir is '' else opt.input_dir
        # self.data_dir = ['/DATA7_DB7/data/shfang/airsim_camera_seg_15'] if opt.input_dir is '' else opt.input_dir
        # print('Data dir: {}'.format(self.data_dir))
        # self.img_dir = os.path.join(self.data_dir, 'images')
        self.img_dir = self.data_dir
        if split == 'val':
            # self.annot_path = os.path.join(
            #     self.data_dir, 'multiagent_annotations', 'val_instances_sample.pkl')
            if isinstance(self.data_dir, list):
                self.annot_path = [os.path.join(data_dir, 'multiagent_annotations', '{}_val_instances_sample.pkl'.format(opt.uav_height)) \
                                    for data_dir in self.data_dir  if os.path.exists(os.path.join(data_dir, 'multiagent_annotations', '{}_val_instances_sample.pkl'.format(opt.uav_height)))]
            else:
                self.annot_path = os.path.join(self.data_dir, 'multiagent_annotations', '{}_val_instances_sample.pkl'.format(opt.uav_height))
            if opt.coord == 'Global':
                # self.annot_path_cocoformat = os.path.join(
                #     self.data_dir, 'multiagent_annotations', 'val_instances_global.json')
                if isinstance(self.data_dir, list):
                    self.annot_path_cocoformat = [os.path.join(data_dir, 'multiagent_annotations', '{}_val_instances_global.json'.format(opt.uav_height)) \
                                                        for data_dir in self.data_dir if os.path.exists(os.path.join(data_dir, 'multiagent_annotations', '{}_val_instances_global.json'.format(opt.uav_height)))]
                else:
                    self.annot_path_cocoformat = os.path.join(self.data_dir, 'multiagent_annotations', '{}_val_instances_global.json'.format(opt.uav_height))
            else:
                self.annot_path_cocoformat = os.path.join(
                    self.data_dir, 'multiagent_annotations', 'val_instances.json')
        else:
            # self.annot_path = os.path.join(
            #     self.data_dir, 'multiagent_annotations', 'train_instances_sample.pkl')
            self.annot_path = [os.path.join(data_dir, 'multiagent_annotations', '{}_train_instances_sample.pkl'.format(opt.uav_height)) \
                                for data_dir in self.data_dir  if os.path.exists(os.path.join(data_dir, 'multiagent_annotations', '{}_train_instances_sample.pkl'.format(opt.uav_height)))]
            if opt.coord == 'Global':
                # self.annot_path_cocoformat = os.path.join(
                #     self.data_dir, 'multiagent_annotations', 'train_instances_global.json')
                self.annot_path_cocoformat = [os.path.join(data_dir, 'multiagent_annotations', '{}_train_instances_global.json'.format(opt.uav_height)) \
                                                    for data_dir in self.data_dir if os.path.exists(os.path.join(data_dir, 'multiagent_annotations', '{}_train_instances_global.json'.format(opt.uav_height)))]
            else:
                self.annot_path_cocoformat = os.path.join(
                    self.data_dir, 'multiagent_annotations', 'train_instances.json')
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
                    assert len(cams) == 25
                    for cam_id, cam in enumerate(cams):
                        updated_img_idx = sum(sample_counts[:-1]) + sample_id * 25 + cam_id
                        cur_img_idx_mapping[sample[cam]['image_id']] = updated_img_idx
                        sample[cam]['image_id'] = updated_img_idx
                self.samples.extend(cur_sample)
                self.img_dir.extend([self.data_dir[i]]*len(cur_sample))
                self.img_idx_mapping.append(cur_img_idx_mapping)
        self.num_samples = len(self.samples)
        print('Loaded {} {} samples'.format(split, self.num_samples))
        if 'NO_MESSAGE' in opt.message_mode:
            self.num_agents = 1
            samples = []
            img_dir = []
            for i, sample in enumerate(self.samples):
                for k, data in sample.items():
                    if k.startswith('vehicles'):
                        continue
                    samples.append(data)
                    img_dir.append(self.img_dir[i])
            self.samples = samples
            self.img_dir = img_dir
            self.num_samples = len(self.samples)
        else:
            self.num_agents = int(opt.num_agents)

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

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        if isinstance(self.annot_path_cocoformat, str):
            self.coco = coco.COCO(self.annot_path_cocoformat)
        else:
            annot_cocoformat = {'images': [], "type": "instances", 'annotations': []}
            box_count = 0
            for i, annot_path in enumerate(self.annot_path_cocoformat):
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
            
            with open('{}/gts.json'.format(save_dir), 'w') as f:
                json.dump(annot_cocoformat, f)
            self.coco = coco.COCO('{}/gts.json'.format(save_dir))
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        # self.coco = coco.COCO(self.annot_path_cocoformat)
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    
    def run_polygon_eval(self, results, save_dir):
        self.save_results(results, save_dir)

        if isinstance(self.annot_path_cocoformat, str):
            run_polygon_eval(self.annot_path_cocoformat, '{}/results.json'.format(save_dir))
        else:
            annot_cocoformat = {'images': [], "type": "instances", 'annotations': []}
            for i, annot_path in enumerate(self.annot_path_cocoformat):
                cur_sample = json.load(open(annot_path, 'r'))
                for image in cur_sample['images']:
                    image['id'] = self.img_idx_mapping[i][image['id']]
                for anno in cur_sample['annotations']:
                    anno['image_id'] = self.img_idx_mapping[i][anno['image_id']]
                
                annot_cocoformat['categories'] = cur_sample['categories']
                annot_cocoformat['images'].extend(cur_sample['images'])
                annot_cocoformat['annotations'].extend(cur_sample['annotations'])
            
            with open('{}/gts.json'.format(save_dir), 'w') as f:
                json.dump(annot_cocoformat, f)
            
            run_polygon_eval('{}/gts.json'.format(save_dir), '{}/results.json'.format(save_dir))