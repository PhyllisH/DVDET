'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-07-26 00:47:54
Description: Convert the single-view dataformat to multi-view dataformat
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
from typing_extensions import OrderedDict
from ipdb.__main__ import set_trace
import numpy as np
import math
import cv2
import os
import random
import pickle as pkl
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import pycocotools.coco as coco
from tqdm import tqdm

# DATA_PATH = '../../data/kitti/'

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix
import sys
sys.path.append(os.path.dirname(__file__))
from transformation import *
from visualize_coco_result import CoordTrans
from copy import deepcopy

'''
GT:
'samples': [{
    'SENSOR': 
        {
            'trans_mat': (3, 4) # the UAV coordinate transform matrix
            'image':  # the RGB image
            'image_id': # the image id
            'translation':
            'rotation':
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


# train_split = ['scene_0', 'scene_1', 'scene_2', 'scene_3', 'scene_4', 'scene_5',
#                'scene_6', 'scene_8', 'scene_9', 'scene_10', 'scene_11', 'scene_12',
#                'scene_13', 'scene_14', 'scene_16', 'scene_17', 'scene_18', 'scene_19',
#                'scene_20', 'scene_21', 'scene_22', 'scene_23', 'scene_24', 'scene_26',
#                'scene_28', 'scene_29', 'scene_30', 'scene_31', 'scene_32', 'scene_33',
#                'scene_34', 'scene_35', 'scene_36', 'scene_37', 'scene_38', 'scene_39',
#                'scene_40', 'scene_42', 'scene_44', 'scene_45', 'scene_46', 'scene_47',
#                'scene_48', 'scene_49', 'scene_50', 'scene_51', 'scene_52', 'scene_53',
#                'scene_55', 'scene_56', 'scene_57', 'scene_61', 'scene_62', 'scene_63',
#                'scene_65', 'scene_66', 'scene_67', 'scene_68', 'scene_69', 'scene_70',
#                'scene_71', 'scene_72', 'scene_73', 'scene_75', 'scene_76', 'scene_77',
#                'scene_78', 'scene_79', 'scene_80', 'scene_81', 'scene_82', 'scene_83',
#                'scene_84', 'scene_87', 'scene_88', 'scene_90', 'scene_92', 'scene_94',
#                'scene_95', 'scene_97', 'scene_98', 'scene_99', 'scene_100', 'scene_101',
#                'scene_102', 'scene_103', 'scene_104', 'scene_105', 'scene_106', 'scene_107',
#                'scene_108', 'scene_109', 'scene_110', 'scene_111', 'scene_112', 'scene_113',
#                'scene_114', 'scene_116', 'scene_118', 'scene_119']

# val_split = ['scene_7', 'scene_15', 'scene_25', 'scene_27', 'scene_41', 'scene_43',
#              'scene_54', 'scene_58', 'scene_59', 'scene_60', 'scene_64', 'scene_74',
#              'scene_85', 'scene_86', 'scene_89', 'scene_91', 'scene_93', 'scene_96',
#              'scene_115', 'scene_117']

# train_split = ['scene_0', 'scene_1', 'scene_2', 'scene_3', 'scene_4', 
#                'scene_6', 'scene_8', 'scene_9', 'scene_10', 'scene_11', 'scene_12',
#                'scene_13', 'scene_14']
# val_split = [ 'scene_5']

train_split = ['scene_2', 'scene_4', 'scene_5', 'scene_7', 'scene_8', 'scene_10', 
               'scene_11', 'scene_12', 'scene_13', 'scene_14', 'scene_15', 'scene_18', 
               'scene_19', 'scene_21', 'scene_22', 'scene_23', 'scene_24', 'scene_25', 
               'scene_26', 'scene_27', 'scene_28', 'scene_30', 'scene_32', 'scene_33',
               'scene_0', 'scene_6', 'scene_17', 'scene_20', 'scene_29']

val_split = ['scene_1', 'scene_3', 'scene_9', 'scene_16', 'scene_31']

town_config = OrderedDict()
town_config[0] = {
    'world_X': 500,
    'world_Y': 500,
    'world_X_left': 200,
    'world_Y_left': 250
}
town_config[1] = {
    'world_X': 1200,
    'world_Y': 600,
    'world_X_left': 400,
    'world_Y_left': 200
}
town_config[2] = {
    'world_X': 900,
    'world_Y': 900,
    'world_X_left': 400,
    'world_Y_left': 500
}


def convert_multiview_coco(town_id=1, height=40):
    ##################### Get category info ###############
    cats = ['car']
    cat_info = []
    for i, cat in enumerate(cats):
        cat_info.append({'supercategory': 'vehicle', 'name': cat, 'id': i + 1})
    #######################################################

    ########## Camera intrinsic and image settings ########
    F = 400  # focal
    H = 450  # height
    W = 800  # width
    camera_intrinsic = [[400.0, 0.0, 400.0],
                        [0.0, 400.0, 225.0],
                        [0.0, 0.0, 1.0]]
    world_X = town_config[town_id]['world_X']
    world_Y = town_config[town_id]['world_Y']
    image_H = 450
    image_W = 800
    world_X_left = town_config[town_id]['world_X_left']
    world_Y_left = town_config[town_id]['world_Y_left']
    worldgrid2worldcoord_mat = np.array([[1, 0, -world_X_left], [0, 1, -world_Y_left], [0, 0, 1]])
    # worldgrid2worldcoord_mat = np.array([[world_X, 0, -world_X_left], [0, world_Y, -world_Y_left], [0, 0, 1]])
    # worldgrid2worldcoord_mat = np.array([[world_X/image_W, 0, -world_X_left], [0, world_Y/image_H, -world_Y_left], [0, 0, 1]])
    # worldgrid2worldcoord_mat = np.array([[1, 0, -200], [0, 1, -250], [0, 0, 1]])
    #######################################################

    # data_dir = 'C:/Users/35387/Desktop/airsim_camera_demo'
    # data_dir = '/DATA5_DB8/data/public/airsim_camera/airsim_camera_10scene'
    # data_dir = '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town6_v2/'
    # data_dir = '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_60m/'
    # data_dir = '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_80m/'
    # data_dir = '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_40m/'
    # nusc = NuScenes(version='v1.0-{}m-group'.format(height), dataroot=data_dir, verbose=True)
    # data_dir = '/DATA7_DB7/data/shfang/airsim_camera_seg_15/'
    # data_dir = '/GPFS/data/yhu/Dataset/airsim_camera/airsim_camera_seg_15/'
    # nusc = NuScenes(version='v1.0-mini', dataroot=data_dir, verbose=True)
    if town_id == 0:
        data_dir = '/GPFS/data/yhu/Dataset/airsim_camera/airsim_camera_seg_15'
        nusc = NuScenes(version='v1.0-mini', dataroot=data_dir, verbose=True)
    else:
        if town_id == 1:
            data_dir = '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town6_v2'
        elif town_id == 2:
            data_dir = '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_{}m'.format(height)
        nusc = NuScenes(version='v1.0-{}m-group'.format(height), dataroot=data_dir, verbose=True)

    if not os.path.exists(os.path.join(data_dir, 'multiagent_annotations', 'Collaboration')):
        os.makedirs(os.path.join(data_dir, 'multiagent_annotations', 'Collaboration'))

    save_dir = data_dir
    splits = ['train', 'val']
    scene_split = {'train': train_split, 'val': val_split}
    image_id = 0
    bbox_id = 0

    for split in splits:
        ret_i = {'images': [], "type": "instances", 'annotations': [], 'categories': cat_info}
        ret_g = {'images': [], "type": "instances", 'annotations': [], 'categories': cat_info}
        ret_s = {"samples": [], "type": "sample", "categories": cat_info}
        ret_g_crop = {'images': [], "type": "instances", 'annotations': [], 'categories': cat_info}
        for scene in tqdm(nusc.scene):
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
                UAV_idx.sort()
                UAV_cams = list(set([sensor.split('_')[1] for sensor in sensors]))
                UAV_cams.sort()
                for UAV_id in UAV_idx:
                    for UAV_cam in UAV_cams:
                        # if UAV_cam != 'BOTTOM':
                        #     continue
                        image_id += 1
                        cur_UAV_sample = OrderedDict()
                        # camera info
                        sensor = 'CAM_{}_id_{}'.format(UAV_cam, UAV_id)
                        sensor_record = nusc.get("sample_data", sample_data[sensor])
                        calibrated_record = nusc.get("calibrated_sensor", sensor_record["calibrated_sensor_token"])

                        cur_UAV_sample['translation'] = calibrated_record["translation"].copy()
                        cur_UAV_sample['rotation'] = calibrated_record["rotation"].copy()
                        cur_UAV_sample['trans_mat'] = get_imgcoord_matrices(calibrated_record["translation"].copy(),
                                                      calibrated_record["rotation"].copy(),
                                                      camera_intrinsic) @ worldgrid2worldcoord_mat
                        # for z0 in [0.5, 1.0, 1.5]:
                        for z0 in [-1.0, -0.5, 0.5, 0.75, 1.0, 1.5, 2.0, 8.0]:
                            if z0 < 0:
                                z_key = 'trans_mat_n{:03d}'.format(int((-1)*z0*10))
                            else:
                                z_key = 'trans_mat_p{:03d}'.format(int(z0*10))
                            cur_UAV_sample[z_key] = get_imgcoord_matrices(calibrated_record["translation"].copy(),
                                                        calibrated_record["rotation"].copy(),
                                                        camera_intrinsic, z0=z0) @ worldgrid2worldcoord_mat
                        im_position = calibrated_record["translation"].copy()
                        im_position[2] = -im_position[2]
                        im_rotation = calibrated_record["rotation"].copy()
                        im_rotation[3] = -im_rotation[3]
                        im_rotation = Quaternion(im_rotation)

                        # local image
                        image_info = {'file_name': sensor_record['filename'],
                                        'id': image_id,
                                        'height': H,
                                        'width': W}
                        ret_i['images'].append(image_info)
                        ret_g['images'].append(image_info)
                        ret_g_crop['images'].append(image_info)

                        # global image
                        # /DATA5_DB8/data/public/airsim_camera/airsim_camera_10scene/global_sweeps
                        # check warp func
                        # ori_img = cv2.imread(os.path.join('/DATA5_DB8/data/public/airsim_camera/airsim_camera_10scene', sensor_record['filename']))
                        # cv2.imwrite('ori.png', ori_img)
                        # # vis warp
                        # image_g = CoordTrans(ori_img.copy(), cur_UAV_sample['translation'], cur_UAV_sample['rotation'])
                        # cv2.imwrite('img_warp.png', image_g)
                        # # vis warp back
                        # img_warp_back = CoordTrans(image_g.copy(), cur_UAV_sample['translation'], cur_UAV_sample['rotation'], 'G2L')
                        # cv2.imwrite('img_warp_back.png', img_warp_back)
                        # save global image
                        # sensor_g_path = os.path.join(data_dir, 'global_sweeps', sensor)
                        # if not os.path.exists(sensor_g_path):
                        #     os.makedirs(sensor_g_path)
                        # image_g_path = os.path.join(sensor_g_path, os.path.basename(sensor_record['filename']))
                        # if not os.path.exists(image_g_path):
                        # cv2.imwrite(image_g_path, image_g)    # BGR
                        # image_g_info = {'file_name': '{}/{}/{}'.format('global_sweeps', sensor, os.path.basename(sensor_record['filename'])),
                        #                 'id': image_id,
                        #                 'height': image_g.shape[0],
                        #                 'width': image_g.shape[1]}
                        # ret_g['images'].append(image_g_info)

                        # organize sample
                        cur_UAV_sample['image'] = sensor_record['filename']
                        # cur_UAV_sample['image_g']= image_g_path
                        cur_UAV_sample['image_id'] = image_id

                        shift_mats = OrderedDict()
                        for scale in [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]:
                            cur_shift_mat = get_crop_shift_mat(tranlation=calibrated_record["translation"].copy(), \
                                                            rotation=calibrated_record["rotation"].copy(), \
                                                            sensor_type=UAV_cam, map_scale_w=scale, map_scale_h=scale, \
                                                            world_X_left=world_X_left, world_Y_left=world_Y_left)
                            shift_mats[scale] = cur_shift_mat
                        cur_UAV_sample['shift_mats'] = shift_mats

                        # vehicle info
                        # cat_id = 2 if UAV_cam == 'BOTTOM' else 1
                        cat_id = 1
                        vehicles_i = []
                        vehicles_g = []
                        vehicles_g_corners = []
                        vehicles_z = []
                        category_id = []
                        seg_img = cv2.imread(os.path.join(data_dir, os.path.dirname(sensor_record['filename']), os.path.basename(sensor_record['filename']).split('.')[0]+'_seg.png'))
                        for vehicle_cord in vehicle_cords:
                            # get bbox from vehicle_cord
                            vehicle_cord_img = global_points_to_image(vehicle_cord.copy()[:3,], cur_UAV_sample['translation'].copy(), cur_UAV_sample['rotation'].copy(), camera_intrinsic)
                            if vehicle_cord_img.shape[-1] == 0:
                                continue
                            x, y, w, h = get_2d_bounding_box(deepcopy(vehicle_cord_img))
                            # Check box
                            if x < 0 or y < 0 or (x + w) > W or (y + h) > H:
                                continue
                            ignore = 0 if seg_img is None or list(seg_img[int(y+h/2),int(x+w/2)]) == [229, 90, 95] else 1
                            # print(seg_img[int(y+h/2),int(x+w/2)], ignore)
                            bbox_id += 1
                            ann = { 'area': w * h,
                                    'iscrowd': 0,
                                    'image_id': image_id,
                                    'bbox': [x, y, w, h],
                                    'category_id': cat_id,
                                    'id': bbox_id,
                                    'ignore': ignore,
                                    'segmentation': []}
                            ret_i['annotations'].append(ann)
                            if not ignore:
                                vehicles_i.append([x, y, w, h])
                                category_id.append(cat_id)

                            # img_coords = global_points_to_image(vehicle_cord.copy()[:3], cur_UAV_sample['translation'].copy(), cur_UAV_sample['rotation'].copy(), camera_intrinsic)
                            # vehicle_cord_r = image_points_to_global(img_coords.copy(), cur_UAV_sample['translation'].copy(), cur_UAV_sample['rotation'].copy(), camera_intrinsic, z0=vehicle_cord.copy()[2:3])
                            # print('Diff: ', np.abs(vehicle_cord_r[:2] - vehicle_cord[:2]).sum())
                            # img_coords2 = get_imagecoord_from_worldcoord(vehicle_cord.copy()[:3], cur_UAV_sample['translation'].copy(), cur_UAV_sample['rotation'].copy(), camera_intrinsic)
                            # vehicle_cord_r2 = get_worldcoord_from_imagecoord(img_coords.copy(), cur_UAV_sample['translation'].copy(), cur_UAV_sample['rotation'].copy(), camera_intrinsic, z0=vehicle_cord.copy()[2:3])
                            # print('Diff: ', np.abs(vehicle_cord_r2[:2] - vehicle_cord[:2]).sum())
                            # print('Diff: ', np.abs(img_coords2[:2] - img_coords[:2]).sum())

                            # vehicle_grid = WorldCoord2WorldGrid(vehicle_cord[:3, :], scale_w=image_W/world_X, scale_h=image_H/world_Y, world_X_left=world_X_left, world_Y_left=world_Y_left)
                            vehicle_grid = WorldCoord2WorldGrid(vehicle_cord[:3, :], scale_w=1, scale_h=1, world_X_left=world_X_left, world_Y_left=world_Y_left)
                            vehicle_z = [vehicle_cord[2].min(), vehicle_cord[2].max()]
                            x, y, w, h = get_2d_bounding_box(vehicle_grid[:3])
                            polygon_xywhcs, corners = get_angle_polygon(deepcopy(vehicle_grid[:2,:4]))
                            # print(corners)
                            # print(polygon_xywhcs)
                            # print(x, y, w, h)
                            # print(p_x, p_y, p_w, p_h, p_cos, p_sin)
                            ann = { 'area': w * h,
                                    'iscrowd': 0,
                                    'image_id': image_id,
                                    'bbox': [x, y, w, h],
                                    'polygon': polygon_xywhcs,
                                    'corners': corners,
                                    'category_id': cat_id,
                                    'id': bbox_id,
                                    'ignore': ignore,
                                    'segmentation': []}
                            ret_g['annotations'].append(ann)

                            if not ignore:
                                vehicles_g.append([x, y, w, h])
                                vehicles_g_corners.append(corners)
                                vehicles_z.append(vehicle_z)

                            vehicle_grid_crop = get_shift_coord(vehicle_grid.copy(), shift_mats[1])
                            x_crop, y_crop, w_crop, h_crop = get_2d_bounding_box(vehicle_grid_crop[:3])
                            polygon_xywhcs_crop, corners_crop = get_angle_polygon(deepcopy(vehicle_grid_crop[:2,:4]))
                            ann_crop = { 'area': w_crop * h_crop,
                                    'iscrowd': 0,
                                    'image_id': image_id,
                                    'bbox': [x_crop, y_crop, w_crop, h_crop],
                                    'polygon': polygon_xywhcs_crop,
                                    'corners': corners_crop,
                                    'category_id': cat_id,
                                    'id': bbox_id,
                                    'ignore': ignore,
                                    'segmentation': []}
                            ret_g_crop['annotations'].append(ann_crop)

                        cur_UAV_sample['vehicles_i'] = np.array(vehicles_i)
                        cur_UAV_sample['vehicles_g'] = np.array(vehicles_g)
                        cur_UAV_sample['vehicles_g_corners'] = np.array(vehicles_g_corners)
                        cur_UAV_sample['vehicles_z'] = np.array(vehicles_z)
                        cur_UAV_sample['category_id'] = np.array(category_id)
                        cur_sample_info['{}_{}'.format(UAV_cam, UAV_id)] = cur_UAV_sample
                ret_s['samples'].append(cur_sample_info)
                # =======================
                cur_sample_token = cur_sample['next']
        
        print("# samples: ", len(ret_s['samples']))
        print("# images: ", len(ret_i['images']))
        print("# annotations: ", len(ret_i['annotations']))
        # out_path = 'C:/Users/35387/Desktop/airsim_camera_demo/airsim_instances_{}.json'.format(split)
        out_path = os.path.join(data_dir, 'multiagent_annotations/Collaboration/{}_{}_instances.json'.format(height, split))
        out_global_path = os.path.join(data_dir, 'multiagent_annotations/Collaboration/{}_{}_instances_global.json'.format(height, split))
        out_sample_path = os.path.join(data_dir, 'multiagent_annotations/Collaboration/{}_{}_instances_sample.pkl'.format(height, split))
        out_global_crop_path = os.path.join(data_dir, 'multiagent_annotations/Collaboration/{}_{}_instances_global_crop.json'.format(height, split))
        json.dump(ret_i, open(out_path, 'w'))
        json.dump(ret_g, open(out_global_path, 'w'))
        json.dump(ret_g_crop, open(out_global_crop_path, 'w'))
        pkl.dump(ret_s, open(out_sample_path, 'wb')) 

def pop_ignored_box(data_dir='/DATA7_DB7/data/shfang/airsim_camera_seg_15/', ignore_flag=0, height=40):
    splits = ['train', 'val']
    for split in splits:
        out_path = os.path.join(data_dir, 'multiagent_annotations/Collaboration/{}_{}_instances.json'.format(height, split))
        out_global_path = os.path.join(data_dir, 'multiagent_annotations/Collaboration/{}_{}_instances_global.json'.format(height, split))
        out_global_crop_path = os.path.join(data_dir, 'multiagent_annotations/Collaboration/{}_{}_instances_global_crop.json'.format(height, split))
        for anno_path in [out_path, out_global_path, out_global_crop_path]:
            with open(anno_path, 'r') as f:
                annos = json.load(f)
            annos_noignore = []
            for anno in tqdm(annos['annotations']):
                if anno['ignore'] == ignore_flag:
                    annos_noignore.append(anno)
            annos['annotations'] = annos_noignore
            tail = '_ignoredbox' if ignore_flag else '_woignoredbox'
            save_path = os.path.join(os.path.dirname(anno_path), os.path.basename(anno_path).split('.')[0]+'{}.json'.format(tail))
            print(save_path)
            with open(save_path, 'w') as f:
                json.dump(annos, f)

if __name__ == '__main__':
    # convert_multiview_coco(town_id=0, height=40)
    # convert_multiview_coco(town_id=1, height=40)
    # convert_multiview_coco(town_id=1, height=60)
    # convert_multiview_coco(town_id=1, height=80)
    # convert_multiview_coco(town_id=1, height=100)
    # convert_multiview_coco(town_id=2, height=40)
    # convert_multiview_coco(town_id=2, height=60)
    # convert_multiview_coco(town_id=2, height=80)
    # convert_multiview_coco()
    town_ids = [0,1,2]
    for town_id in town_ids:
        if town_id == 0:
            data_dir = '/GPFS/data/yhu/Dataset/airsim_camera/airsim_camera_seg_15'
        elif town_id == 1:
            data_dir = '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town6_v2'
        elif town_id == 2:
            data_dir = '/GPFS/data/shfang/dataset/airsim_camera/airsim_camera_seg_town4_v2_40m'

        pop_ignored_box(data_dir, ignore_flag=0)