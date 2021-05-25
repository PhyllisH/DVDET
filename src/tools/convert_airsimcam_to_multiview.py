'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-05-25 11:14:06
Description: Convert the single-view dataformat to multi-view dataformat
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
from typing_extensions import OrderedDict
import numpy as np
import math
import cv2
import os
import random
import pickle as pkl
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import pycocotools.coco as coco

# DATA_PATH = '../../data/kitti/'

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix
import sys
sys.path.append(os.path.dirname(__file__))
from transformation import *

'''
GT:
'samples': [{
    'SENSOR': 
        {
            'trans_mat': (3, 4) # the UAV coordinate transform matrix
            'image':  # the RGB image
            'image_id': # the image id
            'vehicles_i': (n, 4), np.array # the box in current coordinate (2D)
            'vehicles_g': (n, 4), np.array # the box in global coordinate (2D)
            'category_id': (n,), # the box category
        },
    'vehicles_s': (n, 4), np.array # all the boxes in the global coordinate (2D)
}],
'images': []
'annos': []

And COCO Annotation Format: single-view in image or global coordinates


Dataloader Output:
Single view (local coords):
    Item:
        Training:
            image: [1, C, H, W] # UAV image
            trans_mats: [1, 3, 4] # UAV coordinate
            boxes: [1, MAX_BOXES, 4] # box in current coordinate
        Eval:
            image: [1, C, H, W] # UAV image
            transform_matrix: [1, 2, 3] # UAV coordinate
            boxes: [1, BOXES, 4] # box in current coordinate
            all_boxes: [1, ALL_BOXES, 4] # all the boxes in the scene in global coordinate

Single / Multi view (global coords):
    Item: 
        Training:
            g_image: [1, MAX_UAV, C, H, W] # UAV image (global)
            g_boxes: [1, MAX_UAV, MAX_BOXES, 4] # box in global coordinate
            all_boxes: [1, MAX_UAV, MAX_BOXES, 4] # all the boxes in the scene in global coordinate
        Eval:
            g_image: [1, MAX_UAV, C, H, W] # UAV image (global)
            g_boxes: [1, MAX_UAV, MAX_BOXES, 4] # box in global coordinate
            all_boxes: [1, MAX_UAV, MAX_BOXES, 4] # all the boxes in the scene in global coordinate

Multi view:
    Item: 
        Training:
            image: [1, MAX_UAV, C, H, W] # UAV image
            transform_matrix: [1, MAX_UAV, 2, 3] # UAV coordinate
            boxes: [1, MAX_UAV, MAX_BOXES, 4] # box in each coordinate
            g_image: [1, MAX_UAV, C, H, W] # UAV image (global)
            g_boxes: [1, MAX_UAV, MAX_BOXES, 4] # box in global coordinate
            all_boxes: [1, MAX_UAV, MAX_BOXES, 4] # all the boxes in the scene in global coordinate
        Eval:
            image: [1, MAX_UAV, C, H, W] # UAV image
            transform_matrix: [1, MAX_UAV, 2, 3] # UAV coordinate
            boxes: [1, MAX_UAV, MAX_BOXES, 4] # box in each coordinate
            g_boxes: [1, MAX_UAV, MAX_BOXES, 4] # box in global coordinate
            all_boxes: [1, MAX_UAV, MAX_BOXES, 4] # all the boxes in the scene in global coordinate
            all_boxes: [1, ALL_BOXES, 4] # all the boxes in the scene in global coordinate

def TRtoTMat(translation, rotation):
    # translation: camera parameter
    # rotation: camera parameter
    return T

def UAVtoGlobal(UAV_I, T):
    # UAV_I: (C, H, W) Image in UAV Coordinates
    # T: Transform Matrix from UAV Coord to Global Coord
    return Global_I

def UAVtoUAV(UAV_I1, T1, T2):
    # UAV_I1: (C, H, W) Image in UAV1 Coordinates
    # T1: Transform Matrix from UAV1 Coord to Global Coord
    # T2: Transform Matrix from UAV2 Coord to Global Coord
    return UAV_I2
'''


train_split = ['scene_0', 'scene_1', 'scene_2', 'scene_3', 'scene_4', 'scene_5',
               'scene_6', 'scene_8', 'scene_9', 'scene_10', 'scene_11', 'scene_12',
               'scene_13', 'scene_14', 'scene_16', 'scene_17', 'scene_18', 'scene_19',
               'scene_20', 'scene_21', 'scene_22', 'scene_23', 'scene_24', 'scene_26',
               'scene_28', 'scene_29', 'scene_30', 'scene_31', 'scene_32', 'scene_33',
               'scene_34', 'scene_35', 'scene_36', 'scene_37', 'scene_38', 'scene_39',
               'scene_40', 'scene_42', 'scene_44', 'scene_45', 'scene_46', 'scene_47',
               'scene_48', 'scene_49', 'scene_50', 'scene_51', 'scene_52', 'scene_53',
               'scene_55', 'scene_56', 'scene_57', 'scene_61', 'scene_62', 'scene_63',
               'scene_65', 'scene_66', 'scene_67', 'scene_68', 'scene_69', 'scene_70',
               'scene_71', 'scene_72', 'scene_73', 'scene_75', 'scene_76', 'scene_77',
               'scene_78', 'scene_79', 'scene_80', 'scene_81', 'scene_82', 'scene_83',
               'scene_84', 'scene_87', 'scene_88', 'scene_90', 'scene_92', 'scene_94',
               'scene_95', 'scene_97', 'scene_98', 'scene_99', 'scene_100', 'scene_101',
               'scene_102', 'scene_103', 'scene_104', 'scene_105', 'scene_106', 'scene_107',
               'scene_108', 'scene_109', 'scene_110', 'scene_111', 'scene_112', 'scene_113',
               'scene_114', 'scene_116', 'scene_118', 'scene_119']

val_split = ['scene_7', 'scene_15', 'scene_25', 'scene_27', 'scene_41', 'scene_43',
             'scene_54', 'scene_58', 'scene_59', 'scene_60', 'scene_64', 'scene_74',
             'scene_85', 'scene_86', 'scene_89', 'scene_91', 'scene_93', 'scene_96',
             'scene_115', 'scene_117']


def convert_multiview_coco():
    # data_dir = 'C:/Users/35387/Desktop/airsim_camera_demo'
    data_dir = '/DATA5_DB8/data/public/airsim_camera/airsim_camera_10scene'
    DEBUG = False
    nusc = NuScenes(version='v1.0-mini', dataroot=data_dir, verbose=True)

    cats = ['car', 'car_overlook']
    splits = ['train', 'val']
    scene_split = {'train': train_split, 'val': val_split}
    cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}

    F = 400  # focal
    H = 450  # height
    W = 800  # width
    camera_intrinsic = [[400.0, 0.0, 400.0],
                        [0.0, 400.0, 225.0],
                        [0.0, 0.0, 1.0]]

    cat_info = []
    for i, cat in enumerate(cats):
        cat_info.append({'supercategory': 'vehicle', 'name': cat, 'id': i + 1})
    image_id = 0
    bbox_id = 0
    for split in splits:
        ret_i = {'images': [], "type": "instances", 'annotations': [], 'categories': cat_info}
        ret_g = {'images': [], "type": "instances", 'annotations': [], 'categories': cat_info}
        ret_s = {"samples": [], "type": "sample", "categories": cat_info}
        for scene in nusc.scene:
            if not scene["name"] in scene_split[split]:
                continue
            scene_token = scene['token']
            cur_sample_token = scene['first_sample_token']
            while cur_sample_token != "":
                cur_sample_info = OrderedDict()
                print(cur_sample_token)
                cur_sample = nusc.get("sample", cur_sample_token)
                # =======================
                # execute the current sample data
                anno_tokens = cur_sample["anns"]
                # get the vehicle coords in global frame
                vehicle_cords = []
                vehicle_2ds = []
                for anno_token in anno_tokens:
                    anno_data = nusc.get("sample_annotation", anno_token)
                    vehicle_cord = get_vehicle_coord(anno_data)
                    vehicle_cords.append(vehicle_cord)
                    vehicle_2ds.append(get_2d_bounding_box(vehicle_cord))

                cur_sample_info['vehicles_s'] = np.array(vehicle_2ds)

                sample_data = cur_sample["data"]
                sensors = list(sample_data.keys())
                UAV_idx = list(set([sensor.split('_')[-1] for sensor in sensors]))
                UAV_cams = list(set([sensor.split('_')[1] for sensor in sensors]))
                for UAV_id in UAV_idx:
                    for UAV_cam in UAV_cams:
                        image_id += 1
                        cur_UAV_sample = OrderedDict()
                        # camera info
                        sensor = 'CAM_{}_id_{}'.format(UAV_cam, UAV_id)
                        sensor_record = nusc.get("sample_data", sample_data[sensor])
                        calibrated_record = nusc.get("calibrated_sensor", sensor_record["calibrated_sensor_token"])
                        im_position = calibrated_record["translation"]
                        im_position[2] = -im_position[2]
                        im_rotation = calibrated_record["rotation"]
                        im_rotation[3] = -im_rotation[3]
                        im_rotation = Quaternion(im_rotation)

                        # local image
                        image_info = {'file_name': sensor_record['filename'],
                                        'id': image_id,
                                        'height': H,
                                        'width': W}
                        ret_i['images'].append(image_info)
                        # global image
                        # /DATA5_DB8/data/public/airsim_camera/airsim_camera_10scene/global_sweeps
                        # image_g = image_points_to_global()
                        image_g = np.zeros([H, W, 3], dtype=np.uint8)
                        # save global image
                        sensor_g_path = os.path.join(data_dir, 'global_sweeps', sensor)
                        if not os.path.exists(sensor_g_path):
                            os.makedirs(sensor_g_path)
                        image_g_path = os.path.join(sensor_g_path, os.path.basename(sensor_record['filename']))
                        if not os.path.exists(image_g_path):
                            cv2.imwrite(image_g_path, image_g)    # BGR
                        image_g_info = {'file_name': '{}/{}/{}'.format('global_sweeps', sensor, os.path.basename(sensor_record['filename'])),
                                        'id': image_id,
                                        'height': H,
                                        'width': W}
                        ret_g['images'].append(image_g_info)

                        # organize sample
                        cur_UAV_sample['image'] = sensor_record['filename']
                        cur_UAV_sample['image_g']= image_g_path
                        cur_UAV_sample['image_id'] = image_id
                        cur_UAV_sample['trans_mat'] = np.zeros([3, 4])
                        
                        # vehicle info
                        cat_id = 2 if UAV_cam == 'BOTTOM' else 1
                        vehicles_i = []
                        vehicles_g = []
                        category_id = []
                        for vehicle_cord in vehicle_cords:
                            # get bbox from vehicle_cord
                            vehicle_cord_ = np.array(vehicle_cord)
                            vehicle_cord_ = vehicle_cord_[:3, :]
                            for j in range(3):
                                vehicle_cord_[j, :] = vehicle_cord_[j, :] - im_position[j]
                            vehicle_cord_[:3, :] = np.dot(im_rotation.rotation_matrix, vehicle_cord_[:3, :])
                            vehicle_cord_[:3, :] = np.dot(Quaternion([0.5, -0.5, 0.5, -0.5]).rotation_matrix.T,
                                                        vehicle_cord_[:3, :])
                            # Check depth
                            flag = False if min(vehicle_cord_[2, :]) < 0 else True
                            if not flag:
                                continue
                            vehicle_points = view_points(vehicle_cord_[:3, :], np.array(camera_intrinsic), normalize=True)
                            x, y, w, h = get_2d_bounding_box(vehicle_points)
                            # Check box
                            if x < 0 or y < 0 or (x + w) > W or (y + h) > H:
                                flag = False
                            if not flag:
                                continue
                            bbox_id += 1
                            ann = { 'area': w * h,
                                    'iscrowd': 0,
                                    'image_id': image_id,
                                    'bbox': [W - x - w, y, w, h],
                                    'category_id': cat_id,
                                    'id': bbox_id,
                                    'ignore': 0,
                                    'segmentation': []}
                            ret_i['annotations'].append(ann)
                            vehicles_i.append([W - x - w, y, w, h])
                            category_id.append(cat_id)

                            x, y, w, h = get_2d_bounding_box(vehicle_cord_[:3, :])
                            ann_g = {   'area': w * h,
                                        'iscrowd': 0,
                                        'image_id': image_id,
                                        'bbox': [W - x - w, y, w, h],
                                        'category_id': cat_id,
                                        'id': bbox_id,
                                        'ignore': 0,
                                        'segmentation': []}
                            ret_g['annotations'].append(ann_g)
                            vehicles_g.append([W - x - w, y, w, h])
                        cur_UAV_sample['vehicles_i'] = np.array(vehicles_i)
                        cur_UAV_sample['vehicles_g'] = np.array(vehicles_g)
                        cur_UAV_sample['category_id'] = np.array(category_id)
                        cur_sample_info['{}_{}'.format(UAV_cam, UAV_id)] = cur_UAV_sample
                ret_s['samples'].append(cur_sample_info)
                # =======================
                cur_sample_token = cur_sample['next']
        
        print("# samples: ", len(ret_s['samples']))
        print("# images: ", len(ret_i['images']))
        print("# annotations: ", len(ret_i['annotations']))
        # out_path = 'C:/Users/35387/Desktop/airsim_camera_demo/airsim_instances_{}.json'.format(split)
        out_path = '/DATA5_DB8/data/public/airsim_camera/airsim_camera_10scene/multiagent_annotations/{}_instances.json'.format(split)
        out_global_path = '/DATA5_DB8/data/public/airsim_camera/airsim_camera_10scene/multiagent_annotations/{}_instances_global.json'.format(split)
        out_sample_path = '/DATA5_DB8/data/public/airsim_camera/airsim_camera_10scene/multiagent_annotations/{}_instances_sample.pkl'.format(split)
        json.dump(ret_i, open(out_path, 'w'))
        json.dump(ret_g, open(out_global_path, 'w'))
        pkl.dump(ret_s, open(out_sample_path, 'wb')) 

if __name__ == '__main__':
    convert_multiview_coco()
